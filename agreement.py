import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=str, default="100, 13, 21, 42, 87")
    parser.add_argument("--correct", type=int, default=100)

    parser.add_argument("--icl_dir", type=str, default="/data/v-xindiwang/ICL_LL/out/")
    parser.add_argument("--sl_dir", type=str, default="/data/v-xindiwang/ICL_LL/supervised_learning_predictions/")

    args = parser.parse_args()

    total_ratio = []
    for seed in args.seeds:
        icl_dir = os.path.join(args.icl_dir, args.gpt2, "{}-test-direct-k=16-s={}-no-newlines.txt".format(args.dataset, seed))
        icl_predictions = []
        with open(icl_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                icl_predictions.append(line.strip())

        sl_dir = os.path.join(args.icl_dir, args.gpt2, args.dataset, "{}_{}_correct".format(args.dataset, args.correct),
                              "{}_16_{}.json".format(args.dataset, seed))
        with open(sl_dir, "r") as f_sl:
            sl_predictions = json.load(f_sl)

        assert len(icl_predictions) == len(sl_predictions)

        num_agreement = 0
        for label1, label2 in zip(icl_predictions, sl_predictions):
            if label1.strip() == label2.strip():
                num_agreement += 1
            else:
                continue

        agreement_ratio = num_agreement / len(icl_predictions)
        total_ratio.append(agreement_ratio)

    ratio = sum(total_ratio) / len(total_ratio)
    print("ratio is:", ratio)


if __name__ == "__main__":
    main()