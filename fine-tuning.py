import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Config, GPT2ForSequenceClassification


# def load_data(dataset, k, seed):
#     data = []
#     data_path = os.path.join("data", dataset, "{}_{}_{}_train.jsonl".format(dataset, k, seed))
#
#     with open(data_path, "r") as f:
#         for line in f:
#             dp = json.loads(line)
#             data.append(dp)
#     return data


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

        if not os.path.isdir(data_path):
            raise ValueError('Invalid `path` variable! Needs to be a directory')

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


def save(args, model):
    # check if path exist
    path = os.path.join(args.out_dir, args.gpt2, args.dataset)
    is_exit = os.path.exists(path)
    if is_exit:
        torch.save(model.state_dict(), os.path.join(path, 'model_{}_{}.pt'.format(args.dataset, args.seed)))
    else:
        os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model_{}_{}.pt'.format(args.dataset, args.seed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="SST-2")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_training_steps", type=int, default=30000)

    parser.add_argument("--gpt2", type=str, default="gpt2-large")
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=model_config)
    model.to(device)

    label_ids = load_label(args.dataset)
    collator = Gpt2ClassificationCollator(tokenizer=tokenizer, labels_encoder=label_ids, max_sequence_len=args.max_len)

    data_path = os.path.join("data", args.dataset, "{}_{}_{}_train.jsonl".format(args.dataset, args.k, args.seed))
    print(data_path)
    train_dataset = ICLData(data_path)
    print('Created `train_dataset` with %d examples!' % len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_training_steps)

    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}

    for epoch in tqdm(range(args.num_training_steps)):
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)
        print("-Epoch: %.5f  - train_loss: %.5f  - train_acc: %.5f " % (epoch, train_loss, train_acc))

        all_loss['train_loss'].append(train_loss)
        all_acc['train_acc'].append(train_acc)

    save(args, model)


if __name__ == "__main__":
    main()
