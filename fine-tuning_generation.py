import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Config, GPT2LMHeadModel


def load_label(dataset):
    data_path = os.path.join("config/tasks", "{}.json".format(dataset))

    with open(data_path, "r") as f:
        for line in f:
            dp = json.loads(line)
            label = dp["options"]
            label_list = {k: v for v, k in enumerate(label)}

    return label_list


class ICLData(Dataset):
    def __init__(self, data_path):

        self.texts = []

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                concatenate_text = dp["input"] + ' ' + dp["output"]
                self.texts.append(concatenate_text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item]


class Gpt2ClassificationCollator(object):
    def __init__(self, tokenizer, max_sequence_len=None):
        self.tokenizer = tokenizer
        self.max_sequence_len = self.tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

    def __call__(self, sequences):

        texts = [sequence['text'] for sequence in sequences]
        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)

        return inputs


def train(model, dataloader, optimizer, scheduler, device, max_grad_norm=1.0):
    model.train()
    true_labels = []
    predictions_labels = []
    total_loss = 0

    for batch in dataloader:

        input_ids = batch[0].to(device)
        labels = input_ids
        masks = batch[1].to(device)

        model.zero_grad()
        outputs = model(input_ids, labels=labels, attention_mask=masks, token_type_ids=None)
        loss, logits = outputs[:2]

        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        logits = logits.detach().cpu().numpy()

        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        true_labels.append(labels[-1])
    avg_epoch_loss = total_loss / len(dataloader)

    return avg_epoch_loss


def test(model, dataloader, device):

    model.eval()
    true_labels = []
    predictions_labels = []
    total_loss = 0

    for batch in dataloader:

        input_ids = batch[0].to(device)
        labels = input_ids
        masks = batch[1].to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=masks, token_type_ids=None)

            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()

            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

        true_labels.append(labels[-1])
    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


def save(args, model, seed):
    # check if path exist
    path = os.path.join(args.out_dir, args.gpt2, args.dataset)
    is_exit = os.path.exists(path)
    if is_exit:
        torch.save(model.state_dict(), os.path.join(path, 'model_{}_{}_correct_{}.pt'.format(args.dataset, args.correct, seed)))
    else:
        os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model_{}_{}_correct_{}.pt'.format(args.dataset, args.correct, seed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--seeds", type=str, default="100, 13, 21, 42, 87")
    parser.add_argument("--dataset", type=str, default="SST-2")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--correct", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--gpt2", type=str, default="gpt2-large")
    parser.add_argument("--para_dir", type=str, default="hyperparameter")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--result_dir", type=str, default="supervised_learning_results")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    seeds = args.seeds.split(",")

    # get tuned hyperparameter
    para_path = os.path.join(args.para_dir, "{}.json".format(args.dataset))
    with open(para_path, "r") as f:
        para = json.load(f)

    for seed in seeds:
        seed = int(seed.strip())
        # random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)

        model_config = GPT2Config.from_pretrained(args.gpt2, output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained(args.gpt2, config=model_config)
        model.config.pad_token_id = model.config.eos_token_id
        model.to(device)

        collator = Gpt2ClassificationCollator(tokenizer=tokenizer, max_sequence_len=args.max_len)

        if args.correct == 100:
            train_data_path = os.path.join("data", args.dataset, "{}_{}_{}_train.jsonl".format(args.dataset, args.k, seed))
        else:
            train_data_path = os.path.join("data", "{}_{}_correct".format(args.dataset, args.correct),
                                           "{}_{}_correct_{}_{}_train.jsonl".format(args.dataset, args.correct, args.k, seed))
        print(train_data_path)
        train_dataset = ICLData(train_data_path)
        print('Created `train_dataset` with %d examples!' % len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=para["bs"], shuffle=True, collate_fn=collator)

        optimizer = AdamW(model.parameters(), lr=para["lr"], eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=para["steps"])

        all_loss = {'train_loss': [], 'test_loss': []}
        all_acc = {'train_acc': [], 'test_f1': []}

        for epoch in tqdm(range(para["steps"])):
            train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
            train_acc = accuracy_score(train_labels, train_predict)
            print("-Epoch: %.5f  - train_loss: %.5f  - train_acc: %.5f " % (epoch, train_loss, train_acc))

            all_loss['train_loss'].append(train_loss)
            all_acc['train_acc'].append(train_acc)

        save(args, model, seed)

        print("Starting testing!")
        if args.correct == 100:
            test_data_path = os.path.join("data", args.dataset, "{}_{}_{}_test.jsonl".format(args.dataset, args.k, seed))
        else:
            test_data_path = os.path.join("data", "{}_{}_correct".format(args.dataset, args.correct),
                                           "{}_{}_correct_{}_{}_test.jsonl".format(args.dataset, args.correct, args.k, seed))

        test_dataset = ICLData(test_data_path)
        test_dataloader = DataLoader(test_dataset, batch_size=para["bs"], shuffle=True, collate_fn=collator)

        test_true_labels, predictions_labels, avg_epoch_loss = test(model, test_dataloader, device)
        f1 = f1_score(test_true_labels, predictions_labels, average='macro')

        all_loss['test_loss'].append(avg_epoch_loss)
        all_acc['train_acc'].append(f1)

        print("Macro-F1 of %s at seed %d: %.1f " % (args.dataset, seed, f1*100))
        result = {"dataset": args.dataset, "result": f1}
        save_path = os.path.join(args.result_dir, "{}".format(args.dataset),
                                 "{}_{}_correct".format(args.dataset, args.correct))
        is_exit = os.path.exists(save_path)
        if is_exit:
            pass
        else:
            os.makedirs(save_path)
        save_result_path = os.path.join(save_path, "{}_{}_correct_{}_{}.json".format(args.dataset, args.correct, args.k, seed))
        with open(save_result_path, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
