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
from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Config, GPT2ForSequenceClassification


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
        self.labels = []

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                self.texts.append(dp["input"])
                self.labels.append(dp["output"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'label': self.labels[item]}


class Gpt2ClassificationCollator(object):
    def __init__(self, tokenizer, labels_encoder, max_sequence_len=None):
        self.tokenizer = tokenizer
        self.max_sequence_len = self.tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]
        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        inputs.update({'labels': torch.tensor(labels)})

        return inputs


def train(model, dataloader, optimizer, scheduler, device, max_grad_norm=1.0):
    model.train()
    true_labels = []
    predictions_labels = []
    total_loss = 0

    for batch in dataloader:

        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]

        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        logits = logits.detach().cpu().numpy()

        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


def test(model, dataloader, device):

    model.eval()
    true_labels = []
    predictions_labels = []
    total_loss = 0

    for batch in dataloader:

        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()

            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

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

    parser.add_argument('--imbalance_level', type=str, default='low',
                        help="imbalance level of labels, choosing from low, medium, high")
    parser.add_argument('--label_imbalance', type=bool, default=False)

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
    if not args.label_imbalance:
        para_path = os.path.join(args.para_dir, "noisy_label", "{}.json".format(args.dataset))
    else:
        para_path = os.path.join(args.para_dir, "label_imbalance", "{}.json".format(args.dataset))

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

        label_ids = load_label(args.dataset)
        num_label = len(label_ids)

        model_config = GPT2Config.from_pretrained(args.gpt2, output_hidden_states=False, num_labels=num_label)
        model = GPT2ForSequenceClassification.from_pretrained(args.gpt2, config=model_config)
        model.config.pad_token_id = model.config.eos_token_id
        model.to(device)

        collator = Gpt2ClassificationCollator(tokenizer=tokenizer, labels_encoder=label_ids, max_sequence_len=args.max_len)

        if not args.label_imbalance:
            if args.correct == 100:
                train_data_path = os.path.join("data", args.dataset, "{}_{}_{}_train.jsonl".format(args.dataset, args.k, seed))
            else:
                train_data_path = os.path.join("data", "{}_{}_correct".format(args.dataset, args.correct),
                                               "{}_{}_correct_{}_{}_train.jsonl".format(args.dataset, args.correct, args.k, seed))
        else:
            train_data_path = os.path.join("data_imbalance", "{}_{}".format(args.dataset, args.imbalance_level),
                                           "{}_{}_{}_train.jsonl".format(args.dataset, args.k, seed))

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

        # save(args, model, seed)

        print("Starting testing!")
        if not args.label_imbalance:
            if args.correct == 100:
                test_data_path = os.path.join("data", args.dataset, "{}_{}_{}_test.jsonl".format(args.dataset, args.k, seed))
            else:
                test_data_path = os.path.join("data", "{}_{}_correct".format(args.dataset, args.correct),
                                              "{}_{}_correct_{}_{}_test.jsonl".format(args.dataset, args.correct, args.k, seed))
        else:
            test_data_path = os.path.join("data_imbalance", "{}_{}".format(args.dataset, args.imbalance_level),
                                          "{}_{}_{}_test.jsonl".format(args.dataset, args.k, seed))

        test_dataset = ICLData(test_data_path)
        test_dataloader = DataLoader(test_dataset, batch_size=para["bs"], shuffle=True, collate_fn=collator)

        test_true_labels, predictions_labels, avg_epoch_loss = test(model, test_dataloader, device)
        f1 = f1_score(test_true_labels, predictions_labels, average='macro')

        all_loss['test_loss'].append(avg_epoch_loss)
        all_acc['train_acc'].append(f1)

        print("Macro-F1 of %s at seed %d with %s imbalance level: %.1f " % (args.dataset, seed, args.imbalance_level, f1*100))
        result = {"dataset": args.dataset, "result": f1}
        if not args.label_imbalance:
            save_path = os.path.join(args.result_dir, "{}".format(args.dataset),
                                     "{}_{}_correct".format(args.dataset, args.correct))
        else:
            save_path = os.path.join(args.result_dir, "{}".format(args.dataset),
                                     "{}_{}".format(args.dataset, args.imbalance_level))

        is_exit = os.path.exists(save_path)
        if is_exit:
            pass
        else:
            os.makedirs(save_path)
        save_result_path = os.path.join(save_path, "{}_{}_{}_{}.json".format(args.dataset, args.k, seed, args.imbalance_level))
        with open(save_result_path, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
