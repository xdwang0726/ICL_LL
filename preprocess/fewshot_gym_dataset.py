# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random

import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict
from random import randint

from utils import load_configs, load_prompts, apply_prompt, map_hf_dataset_to_list, preprocess


def randomList(m, n):

    # Create an array of size m where
    # every element is initialized to 0
    arr = [0] * m

    # To make the sum of the final list as n
    for i in range(n):

        # Increment any random element
        # from the array by 1
        arr[randint(0, n) % m] += 1

    return arr

parser = argparse.ArgumentParser()
parser.add_argument('--inst', action='store_true',
                    help="Construct data from hg datasets.")
parser.add_argument('--do_train', action='store_true',
                    help="Verify the datafiles with pre-computed MD5")
parser.add_argument('--do_test', action='store_true',
                    help="Run 2 tasks per process to test the code")
parser.add_argument('--train_k', type=int, default=16384, help="k for meta-training tasks")
parser.add_argument('--test_k', type=int, default=16, help="k for target tasks")

args = parser.parse_args()

use_instruct = args.inst
do_train = args.do_train
do_test = args.do_test
if args.do_train and args.do_test:
    raise NotImplementedError("You should specify one of `--do_train` and `--do_test`, not both")
if not args.do_train and not args.do_test:
    raise NotImplementedError("You should specify one of `--do_train` and `--do_test`")

config_dict = load_configs()
if use_instruct:
    prompt_names_per_task, prompt_dict = load_prompts(do_train)


class FewshotGymDataset():

    def get_map_hf_dataset_to_list(self):
        if use_instruct:
            def _map_hf_dataset_to_list(dataset, split):
                return map_hf_dataset_to_list(self.hf_identifier, dataset, split, do_train=do_train)
            return _map_hf_dataset_to_list
        return None

    def get_train_test_lines(self, dataset):
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        train_lines = map_hf_dataset_to_list(dataset, "train")
        test_lines = map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def save(self, path, k, seed, k_shot_train, k_shot_dev, k_shot_test, imbalance_level, label_imbalance=False):
        # save to path

        def _apply_prompt(example):
            return apply_prompt(self.hf_identifier, example, do_train=do_train, prompt_names_per_task=prompt_names_per_task, prompt_dict=prompt_dict)

        if do_train and use_instruct:
            # let's save k_shot_train only

            grouped_k_shot_train = defaultdict(list)
            for line in tqdm(k_shot_train):
                line = _apply_prompt(line)
                assert type(line)==dict
                assert len(set(line.keys())-set(["inst:"+self.hf_identifier+":"+name for name in prompt_names_per_task[self.hf_identifier]]))==0

                for key, value in line.items():
                    grouped_k_shot_train[key].append(json.dumps(value))

            for key, lines in grouped_k_shot_train.items():
                hf_identifier = key
                if path:
                    if not label_imbalance:
                        os.makedirs(os.path.join(path, hf_identifier), exist_ok=True)
                        prefix = os.path.join(path, hf_identifier, "{}_{}_{}".format(hf_identifier, k, seed))
                    else:
                        os.makedirs(os.path.join(path, "{}_{}".format(hf_identifier, imbalance_level)), exist_ok=True)
                        prefix = os.path.join(path, "{}_{}".format(hf_identifier, imbalance_level),
                                              "{}_{}_{}".format(hf_identifier, k, seed))
                    self.write(lines, prefix + "_train.jsonl")

        elif use_instruct:
            k_shot_train = [_apply_prompt(example) for example in k_shot_train]
            k_shot_dev = [_apply_prompt(example) for example in k_shot_dev]
            k_shot_test = [_apply_prompt(example) for example in k_shot_test]

            hf_identifier = "inst:"+self.hf_identifier if use_instruct else self.hf_identifier
            if path:
                if not label_imbalance:
                    os.makedirs(os.path.join(path, hf_identifier), exist_ok=True)
                    prefix = os.path.join(path, hf_identifier,
                                          "{}_{}_{}".format(hf_identifier, k, seed))
                else:
                    os.makedirs(os.path.join(path, "{}_{}".format(hf_identifier, imbalance_level)), exist_ok=True)
                    prefix = os.path.joi(path, "{}_{}".format(hf_identifier, imbalance_level),
                                         "{}_{}_{}".format(hf_identifier, k, seed))

                self.write(k_shot_train, prefix + "_train.jsonl")
                self.write(k_shot_dev, prefix + "_dev.jsonl")
                self.write(k_shot_test, prefix + "_test.jsonl")

        else:
            config = config_dict[self.hf_identifier]
            k_shot_train = [preprocess(self.hf_identifier, example, config) for example in k_shot_train]
            if do_test:
                k_shot_dev = [preprocess(self.hf_identifier, example, config) for example in k_shot_dev]
                k_shot_test = [preprocess(self.hf_identifier, example, config) for example in k_shot_test]

            if path:
                if not label_imbalance:
                    os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
                    prefix = os.path.join(path, self.hf_identifier,
                                          "{}_{}_{}".format(self.hf_identifier, k, seed))
                else:
                    os.makedirs(os.path.join(path, "{}_{}".format(self.hf_identifier, imbalance_level)), exist_ok=True)
                    prefix = os.path.join(path, "{}_{}".format(self.hf_identifier, imbalance_level),
                                          "{}_{}_{}".format(self.hf_identifier, k, seed))
                self.write(k_shot_train, prefix + "_train.jsonl")
                if do_test:
                    self.write(k_shot_dev, prefix + "_dev.jsonl")
                    self.write(k_shot_test, prefix + "_test.jsonl")

    def write(self, lst, out_file):
        with open(out_file, "w") as fout:
            for line in lst:
                if line is not None:
                    fout.write(line+"\n")


class FewshotGymClassificationDataset(FewshotGymDataset):

    def generate_k_shot_data(self, k, seed, path=None, label_imbalance=True, imbalance_level='high'):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        if self.hf_identifier not in config_dict:
            return None, None, None

        if use_instruct and self.hf_identifier not in prompt_names_per_task:
            return None, None, None

        if do_train:
            if seed<100:
                return None, None, None
            k = args.train_k
        elif do_test:
            k = args.test_k

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)
        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # label imbalance
        if label_imbalance:

            label_list = {}
            for line in train_lines:
                label = line[1]
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)
            labels = list(label_list.keys())
            print('There are %s labels in the training set' % len(labels))

            sorted_keys = sorted(label_list, key=lambda k: len(label_list[k]))

            if imbalance_level == 'low':
                # all class are even split
                if len(labels) == 2:
                    split_list = [8, 8]  # if binary follow split 8:8
                elif len(labels) == 3:
                    split_list = [5, 5, 6]  # if 3-class classification follow split 4:6:8
                elif len(labels) == 4:
                    split_list = [4, 4, 4, 4]  # if 4-class classification follow split 3:3:4:6
                elif len(labels) == 6:
                    split_list = [2, 2, 3, 3, 3, 3]  # if 6-class classification follow split 2:2:2:3:3:4

                k_shot_train = []
                for i, key in enumerate(sorted_keys):
                    for line in label_list[key][:split_list[i]]:
                        k_shot_train.append(line)
                np.random.shuffle(k_shot_train)

                k_shot_dev = []
                for i, key in enumerate(sorted_keys):
                    for line in label_list[key][split_list[i]:2 * split_list[i]]:
                        k_shot_dev.append(line)
                np.random.shuffle(k_shot_dev)

                k_shot_test = test_lines

            elif imbalance_level == 'medium':

                if len(labels) == 2:
                    split_list = [5, 11]  # if binary follow split 6:12
                elif len(labels) == 3:
                    split_list = [3, 6, 7]  # if 3-class classification follow split 4:6:8
                elif len(labels) == 4:
                    split_list = [3, 4, 3, 6]  # if 4-class classification follow split 3:3:4:6
                elif len(labels) == 6:
                    split_list = [2, 2, 2, 2, 2, 6]  # if 6-class classification follow split 2:2:2:3:3:4
                elif len(labels) == 14:
                    split_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]

                k_shot_train = []
                for i, key in enumerate(sorted_keys):
                    for line in label_list[key][:split_list[i]]:
                        k_shot_train.append(line)
                np.random.shuffle(k_shot_train)

                k_shot_dev = []
                for i, key in enumerate(sorted_keys):
                    for line in label_list[key][split_list[i]:2 * split_list[i]]:
                        k_shot_dev.append(line)
                np.random.shuffle(k_shot_dev)

                k_shot_test = test_lines

            elif imbalance_level == 'high':

                if len(labels) == 2:
                    split_list = [1, 15]  # if binary follow split 6:12
                elif len(labels) == 3:
                    split_list = [1, 6, 9]  # if 3-class classification follow split 4:6:8
                elif len(labels) == 4:
                    split_list = [1, 2, 4, 9]  # if 4-class classification follow split 3:3:4:6
                elif len(labels) == 6:
                    split_list = [1, 1, 1, 1, 2, 10]  # if 6-class classification follow split 2:2:2:3:3:4
                elif len(labels) == 14:
                    split_list = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5]

                k_shot_train = []
                for i, key in enumerate(sorted_keys):
                    for line in label_list[key][:split_list[i]]:
                        k_shot_train.append(line)
                np.random.shuffle(k_shot_train)

                k_shot_dev = []
                for i, key in enumerate(sorted_keys):
                    for line in label_list[key][split_list[i]:2 * split_list[i]]:
                        k_shot_dev.append(line)
                np.random.shuffle(k_shot_dev)

                k_shot_test = test_lines

        else:
            label_list = {}
            for line in train_lines:
                label = "all"
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            k_shot_train = []
            for label in label_list:
                for line in label_list[label][:k]:
                    k_shot_train.append(line)

            k_shot_dev = []
            for label in label_list:
                for line in label_list[label][k:2*k]:
                    k_shot_dev.append(line)

            k_shot_test = test_lines

        # Get label list for balanced sampling
        # if random:
        #     label_list = {}
        #     for line in train_lines:
        #         label = "all"
        #         # label = line[1]
        #         if label not in label_list:
        #             label_list[label] = [line]
        #         else:
        #             label_list[label].append(line)
        #
        #     k_shot_train = []
        #     for label in label_list:
        #         for line in label_list[label][k:2*k]:
        #             k_shot_train.append(line)
        #
        #     k_shot_dev = []
        #     for label in label_list:
        #         for line in label_list[label][:k]:
        #             k_shot_dev.append(line)
        # else:
        #     # get balanced split of demonstration examples
        #     label_list = {}
        #     for line in train_lines:
        #         label = line[1]
        #         if label not in label_list:
        #             label_list[label] = [line]
        #         else:
        #             label_list[label].append(line)
        #     # number of labels in train
        #     labels = list(label_list.keys())
        #     print('There are %s labels in the training set' % len(labels))
        #     idx = int(np.floor(k / len(labels)))
        #     last_idx = int(k - len(labels) * idx + idx)
        #
        #     # make train, dev, test data
        #     if even_split:
        #         k_shot_train = []
        #         for i, label in enumerate(labels):
        #             if i == 0:
        #                 for line in label_list[label][:last_idx]:
        #                     k_shot_train.append(line)
        #             else:
        #                 for line in label_list[label][:idx]:
        #                     k_shot_train.append(line)
        #         np.random.shuffle(k_shot_train)
        #
        #         k_shot_dev = []
        #         for i, label in enumerate(labels):
        #             if i == 0:
        #                 for line in label_list[label][last_idx:2*last_idx]:
        #                     k_shot_dev.append(line)
        #             else:
        #                 for line in label_list[label][idx:2*idx]:
        #                     k_shot_dev.append(line)
        #         np.random.shuffle(k_shot_dev)
        #
        #     else:
        #         k_shot_train = []
        #         k_shot_dev = []
        #         num_random = k - len(labels)
        #         for label in labels:
        #             k_shot_train.append(label_list[label][0])
        #             k_shot_dev.append(label_list[label][1])
        #             del label_list[label][0], label_list[label][1]
        #         remain_examples = list(label_list.values())
        #         flat_examples = [x for xs in remain_examples for x in xs]
        #         np.random.shuffle(flat_examples)
        #         k_shot_train.extend(flat_examples[:num_random])
        #         k_shot_dev.extend(flat_examples[num_random:2*num_random])
        #         np.random.shuffle(k_shot_train)
        #         np.random.shuffle(k_shot_dev)
        #
        # # balanced test examples:
        # if test_balance:
        #     test_list = {}
        #     for line in test_lines:
        #         # label = "all"
        #         label = line[1]
        #         if label not in test_list:
        #             test_list[label] = [line]
        #         else:
        #             test_list[label].append(line)
        #     lables = list(test_list.keys())
        #     min_examples = []
        #     for l in lables:
        #         min_examples.append(len(test_list[l]))
        #     min_examples.sort()
        #     min = next(filter(lambda x: x!=0, min_examples))
        #
        #     k_shot_test = []
        #     for l in lables:
        #         for line in test_list[l][:min]:
        #             k_shot_test.append(line)
        # else:
        #     k_shot_test = test_lines

        # Turn multi-class into binary classification
        # label_list = {}
        # for line in train_lines:
        #     # label = "all"
        #     label = line[1]
        #     if label not in label_list:
        #         label_list[label] = [line]
        #     else:
        #         label_list[label].append(line)
        #
        # num_examples = {}
        # for l in list(label_list.keys()):
        #     num_examples[l] = len(label_list[l])
        # sorted_examples = dict(sorted(num_examples.items(), key=lambda item: item[1], reverse=True))
        # selected_class = [list(sorted_examples.keys())[0], list(sorted_examples.keys())[1]]
        #
        # k_shot_train = []
        # for label in selected_class:
        #     for line in label_list[label][:int(k/2)]:
        #         k_shot_train.append(line)
        # np.random.shuffle(k_shot_train)
        #
        # k_shot_dev = []
        # for label in selected_class:
        #     for line in label_list[label][int(k/2):k]:
        #         k_shot_dev.append(line)
        # np.random.shuffle(k_shot_dev)
        #
        # test_list = {}
        # for line in test_lines:
        #     label = line[1]
        #     if label not in test_list:
        #         test_list[label] = [line]
        #     else:
        #         test_list[label].append(line)
        #
        # k_shot_test = []
        # for label in selected_class:
        #     for line in test_list[label]:
        #         k_shot_test.append(line)
        # np.random.shuffle(k_shot_test)

        # Choose k demonstration examples following the same label distribution as training set
        # label_list = {}
        # for line in tqdm(train_lines):
        # # for line in tqdm(test_lines):
        #     label = line[1]
        #     if label not in label_list:
        #         label_list[label] = [line]
        #     else:
        #         label_list[label].append(line)
        # # calculate the label distribution in training set
        # labels = {key: None for key in list(label_list.keys())}
        # for key in list(labels.keys()):
        #     labels[key] = len(label_list[key])
        # labels = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True))
        #
        # label_distribution = {}
        # sum_value = 0
        # total = sum(labels.values(), 0.0)
        #
        # for key, v in labels.items():
        #     if key == list(labels.keys())[-1]:
        #         label_distribution[key] = k - sum_value
        #     else:
        #         value = round(k * v / total)
        #         label_distribution[key] = value
        #         sum_value += value
        #
        # print(label_distribution)
        # k_shot_train = []
        # for i, label in enumerate(list(label_distribution.keys())):
        #     for line in label_list[label][:label_distribution[label]]:
        #                     k_shot_train.append(line)
        # np.random.shuffle(k_shot_train)
        #
        # k_shot_dev = []
        # for i, label in enumerate(list(label_distribution.keys())):
        #     for line in label_list[label][label_distribution[label]: 2*label_distribution[label]]:
        #         k_shot_dev.append(line)
        # np.random.shuffle(k_shot_dev)


            # # choose random label distribution for demonstration examples
            # label_list = {}
            # for line in tqdm(train_lines):
            #     # for line in tqdm(test_lines):
            #     label = line[1]
            #     if label not in label_list:
            #         label_list[label] = [line]
            #     else:
            #         label_list[label].append(line)
            #
            # random_num = randomList(len(label_list.keys()), k)
            # num_examples = {key: random_num[i] for i, key in enumerate(label_list.keys())}
            # print(num_examples)
            #
            # k_shot_train = []
            # for label in num_examples.keys():
            #     for line in label_list[label][:num_examples[label]]:
            #         k_shot_train.append(line)
            # np.random.shuffle(k_shot_train)
            #
            # k_shot_dev = []
            # for label in num_examples.keys():
            #     for line in label_list[label][num_examples[label]: 2 * num_examples[label]]:
            #         k_shot_dev.append(line)
            # np.random.shuffle(k_shot_dev)
            #
            # k_shot_test = test_lines


    # save to path
        self.save(path, k, seed, k_shot_train, k_shot_dev, k_shot_test, imbalance_level, label_imbalance)
        return k_shot_train, k_shot_dev, k_shot_test
        # return labels


class FewshotGymTextToTextDataset(FewshotGymDataset):

    def generate_k_shot_data(self, k, seed, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        if self.hf_identifier not in config_dict:
            return None, None, None

        if use_instruct and self.hf_identifier not in prompt_names_per_task:
            return None, None, None

        if do_train:
            if seed<100:
                return None, None, None
            k = args.train_k
        elif do_test:
            k = args.test_k

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # make train, dev, test data
        k_shot_train = []
        for line in train_lines[:k]:
            k_shot_train.append(line)

        k_shot_dev = []
        for line in train_lines[k:2*k]:
            k_shot_dev.append(line)

        k_shot_test = test_lines

        self.save(path, k, seed, k_shot_train, k_shot_dev, k_shot_test)
        return k_shot_train, k_shot_dev, k_shot_test
