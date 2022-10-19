import argparse
import os
import json


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="ag_news", type=str)
    parser.add_argument('--dataset', default="ag_news", type=str)
    parser.add_argument('--seeds', default=[100,13,21,42,87], type=list)
    args = parser.parse_args()

    ratios = []
    with open(os.path.join("config", args.task + ".json"), "r") as f:
        config = json.load(f)
        labels = config["options"]

    for seed in args.seeds:
        train_labels = dict.fromkeys(labels, 0)
        data_dir = "data_noisy_label"
        data_path = os.path.join(data_dir, args.dataset, "{}_16_{}_train.jsonl".format(args.dataset, seed))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                if dp["output"] in train_labels:
                    train_labels[dp["output"]] += 1
        num_labels = list(train_labels.values()).sort()
        largest = num_labels[-1]
        smallest = num_labels[0]
        imbalance_ratio = smallest / largest
    ratios.append(imbalance_ratio)
    avg_ratio = sum(ratios) / len(ratios)
    print("The imbalance ratio for {} is {}".format(args.dataset, avg_ratio))


if __name__ == "__main__":
    main()
