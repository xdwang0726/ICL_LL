# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset


class TweetEval(FewshotGymClassificationDataset):
    def __init__(self, subset_name):
        self.hf_identifier = "tweet_eval-" + subset_name
        self.subset_name = subset_name
        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        if subset_name == "hate":
            self.label = {
                0: "non-hate",
                1: "hate",
            }
        else:
            self.label = {
                0:"none",
                1:"against",
                2:"favor",
            }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if len(datapoint["text"].replace("\n", " ")):
                lines.append((datapoint["text"].replace("\n", " "), self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('tweet_eval', self.subset_name)


def main():
    for subset_name in ['hate', 'stance_atheism', 'stance_feminist']:
        dataset = TweetEval(subset_name)

        for seed in [100, 13, 21, 42, 87]:
            train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data_distribution/")


if __name__ == "__main__":
    main()
