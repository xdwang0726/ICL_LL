import argparse
import os

from transformers import AutoModelForCausalLM, AutoModel
from transformers import GPT2Tokenizer, AutoTokenizer
import torch
import json
import random
import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpt2", type=str, default="gpt2-large")
    parser.add_argument("--correct", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes")

    parser.add_argument("--dir", type=str, default="/data/v-xindiwang/ICL_LL/data_noisy_label/")



    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random select demonstration examples
    correct_demonstration_data_dir = os.path.join(args.dir, args.dataset, "{}_16_100_train.jsonl".format(args.dataset))
    corrputed_demonstration_data_dir = os.path.join(args.dir, "{}_{}_correct".format(args.dataset, args.correct), "{}_{}_16_21_train.jsonl".format(args.dataset, args.correct))
    correct_examples = []
    corrupted_examples = []
    with open(correct_demonstration_data_dir, "r") as f:
        for line in f:
            dp = json.loads(line)
            correct_examples.append(dp["input"] + " \n " + dp["output"] + " \n\n ")

    with open(corrputed_demonstration_data_dir, "r") as f:
        for line in f:
            dp = json.loads(line)
            corrupted_examples.append(dp["input"] + " \n " + dp["output"] + " \n\n ")

    examples = []
    test_data_dir = os.path.join(args.dir, args.dataset, "{}_16_100_test.jsonl".format(args.dataset))
    with open(test_data_dir, "r") as f:
        for line in f:
            dp = json.loads(line)
            examples.append(dp["input"])
    test_examples = random.choice(examples)
# test_examples = "compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children . \n"

    if args.gpt2.startswith("gpt2"):
        model = AutoModelForCausalLM.from_pretrained(args.gpt2, output_attentions=True)
    else:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", output_attentions=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.to(device)

    # all gold labels
    scores = []

    i = 0
    while i < 50:
        # select one positive and one negative example from the demonstration examples
        correct_demonstration = random.sample(correct_examples, k=5)
        correct = "".join(correct_demonstration)
        input_string = correct + test_examples
        correct_length_1 = len(tokenizer.encode(correct_demonstration[0], return_tensors='pt')[0])
        correct_length_2 = len(tokenizer.encode(correct_demonstration[1], return_tensors='pt')[0])
        correct_length_3 = len(tokenizer.encode(correct_demonstration[2], return_tensors='pt')[0])
        correct_length_4 = len(tokenizer.encode(correct_demonstration[3], return_tensors='pt')[0])
        correct_length_5 = len(tokenizer.encode(correct_demonstration[4], return_tensors='pt')[0])
        # print(correct_length_1, correct_length_2, correct_length_3, correct_length_4, correct_length_5)
        correct_position_1 = correct_length_1 - 1 - 3
        correct_position_2 = correct_length_1 + correct_length_2 - 2 - 3
        correct_position_3 = correct_length_1 + correct_length_2 + correct_length_3 - 3 - 3
        correct_position_4 = correct_length_1 + correct_length_2 + correct_length_3 + correct_length_4 - 4 - 3
        correct_position_5 = correct_length_1 + correct_length_2 + correct_length_3 + correct_length_4 + correct_length_5 - 5 - 3
        # print(correct_position_1, correct_position_2, correct_position_3, correct_position_4, correct_position_5)

        inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        outputs = model(inputs)
        attention = outputs[-1]

        sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
        correct_prob_1 = sum_attention[:, correct_position_1]
        correct_prob_2 = sum_attention[:, correct_position_2]
        correct_prob_3 = sum_attention[:, correct_position_3]
        correct_prob_4 = sum_attention[:, correct_position_4]
        correct_prob_5 = sum_attention[:, correct_position_5]
        prob = correct_prob_1 + correct_prob_2 + correct_prob_3 + correct_prob_4 + correct_prob_5

        scores.append(prob)

        i += 1

    print("all gold labels", sum(scores) / len(scores))

    # gold labels : corrupted labels = 4：1
    scores = []
    correct_scores = []
    corrupted_scores = []

    i = 0
    while i < 50:
        corrupted_index = [4]
        # select one positive and one negative example from the demonstration examples
        correct_demonstration = random.sample(correct_examples, k=4)
        corrupted_demonstration = random.sample(corrupted_examples, k=1)
        examples = correct_demonstration + corrupted_demonstration
        demonstration = {}
        for j, example in enumerate(examples):
            demonstration[j] = example
        item = list(demonstration.items())
        random.shuffle(item)
        demonstration = dict(item)
        demonstraion_values = list(demonstration.values())
        all_demonstraion = "".join(demonstraion_values)
        random_corrupted_index = []
        for idx in corrupted_index:
            if idx in demonstration:
                random_corrupted_index.append(list(demonstration).index(idx))
        #     print(demonstration)
        #     print(random_corrupted_index)
        input_string = all_demonstraion + test_examples
        length_1 = len(tokenizer.encode(demonstraion_values[0], return_tensors='pt')[0])
        length_2 = len(tokenizer.encode(demonstraion_values[1], return_tensors='pt')[0])
        length_3 = len(tokenizer.encode(demonstraion_values[2], return_tensors='pt')[0])
        length_4 = len(tokenizer.encode(demonstraion_values[3], return_tensors='pt')[0])
        length_5 = len(tokenizer.encode(demonstraion_values[4], return_tensors='pt')[0])
        # print(correct_length_1, correct_length_2, correct_length_3, correct_length_4, correct_length_5)
        position_1 = length_1 - 1 - 3
        position_2 = length_1 + length_2 - 2 - 3
        position_3 = length_1 + length_2 + length_3 - 3 - 3
        position_4 = length_1 + length_2 + length_3 + length_4 - 4 - 3
        position_5 = length_1 + length_2 + length_3 + length_4 + length_5 - 5 - 3
        # print(correct_position_1, correct_position_2, correct_position_3, correct_position_4, correct_position_5)

        inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        outputs = model(inputs)
        attention = outputs[-1]

        prob_list = []
        sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
        prob_1 = sum_attention[:, position_1]
        prob_list.append(prob_1)
        prob_2 = sum_attention[:, position_2]
        prob_list.append(prob_2)
        prob_3 = sum_attention[:, position_3]
        prob_list.append(prob_3)
        prob_4 = sum_attention[:, position_4]
        prob_list.append(prob_4)
        prob_5 = sum_attention[:, position_5]
        prob_list.append(prob_5)
        prob = prob_1 + prob_2 + prob_3 + prob_4 + prob_5

        correct_prob = 0
        corrupted_prob = 0
        for k in range(5):
            if k not in random_corrupted_index:
                correct_prob += prob_list[k]
            else:
                corrupted_prob += prob_list[k]

        correct_prob = correct_prob / 4

        scores.append(prob)
        correct_scores.append(correct_prob)
        corrupted_scores.append(corrupted_prob)
        i += 1

    print("gold labels : corrupted labels = 4：1", sum(scores) / len(scores))
    print(sum(correct_scores) / len(correct_scores))
    print(sum(corrupted_scores) / len(corrupted_scores))

    # gold labels : corrupted labels = 3：2
    scores = []
    correct_scores = []
    corrupted_scores = []

    i = 0
    while i < 50:
        corrupted_index = [3, 4]
        # select one positive and one negative example from the demonstration examples
        correct_demonstration = random.sample(correct_examples, k=3)
        corrupted_demonstration = random.sample(corrupted_examples, k=2)
        examples = correct_demonstration + corrupted_demonstration
        demonstration = {}
        for j, example in enumerate(examples):
            demonstration[j] = example
        item = list(demonstration.items())
        random.shuffle(item)
        demonstration = dict(item)
        demonstraion_values = list(demonstration.values())
        all_demonstraion = "".join(demonstraion_values)
        random_corrupted_index = []
        for idx in corrupted_index:
            if idx in demonstration:
                random_corrupted_index.append(list(demonstration).index(idx))
        input_string = all_demonstraion + test_examples
        length_1 = len(tokenizer.encode(demonstraion_values[0], return_tensors='pt')[0])
        length_2 = len(tokenizer.encode(demonstraion_values[1], return_tensors='pt')[0])
        length_3 = len(tokenizer.encode(demonstraion_values[2], return_tensors='pt')[0])
        length_4 = len(tokenizer.encode(demonstraion_values[3], return_tensors='pt')[0])
        length_5 = len(tokenizer.encode(demonstraion_values[4], return_tensors='pt')[0])
        # print(correct_length_1, correct_length_2, correct_length_3, correct_length_4, correct_length_5)
        position_1 = length_1 - 1 - 3
        position_2 = length_1 + length_2 - 2 - 3
        position_3 = length_1 + length_2 + length_3 - 3 - 3
        position_4 = length_1 + length_2 + length_3 + length_4 - 4 - 3
        position_5 = length_1 + length_2 + length_3 + length_4 + length_5 - 5 - 3
        # print(correct_position_1, correct_position_2, correct_position_3, correct_position_4, correct_position_5)

        inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        outputs = model(inputs)
        attention = outputs[-1]

        prob_list = []
        sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
        prob_1 = sum_attention[:, position_1]
        prob_list.append(prob_1)
        prob_2 = sum_attention[:, position_2]
        prob_list.append(prob_2)
        prob_3 = sum_attention[:, position_3]
        prob_list.append(prob_3)
        prob_4 = sum_attention[:, position_4]
        prob_list.append(prob_4)
        prob_5 = sum_attention[:, position_5]
        prob_list.append(prob_5)
        prob = prob_1 + prob_2 + prob_3 + prob_4 + prob_5

        correct_prob = 0
        corrupted_prob = 0
        for k in range(5):
            if k not in random_corrupted_index:
                correct_prob += prob_list[k]
            else:
                corrupted_prob += prob_list[k]

        correct_prob = correct_prob / 3
        corrupted_prob = corrupted_prob / 2

        scores.append(prob)
        correct_scores.append(correct_prob)
        corrupted_scores.append(corrupted_prob)
        i += 1

    print("gold labels : corrupted labels = 3：2", sum(scores) / len(scores))
    print(sum(correct_scores) / len(correct_scores))
    print(sum(corrupted_scores) / len(corrupted_scores))

    # gold labels : corrupted labels = 2：3
    scores = []
    correct_scores = []
    corrupted_scores = []

    i = 0
    while i < 50:
        corrupted_index = [2, 3, 4]
        # select one positive and one negative example from the demonstration examples
        correct_demonstration = random.sample(correct_examples, k=2)
        corrupted_demonstration = random.sample(corrupted_examples, k=3)
        examples = correct_demonstration + corrupted_demonstration
        demonstration = {}
        for j, example in enumerate(examples):
            demonstration[j] = example
        item = list(demonstration.items())
        random.shuffle(item)
        demonstration = dict(item)
        demonstraion_values = list(demonstration.values())
        all_demonstraion = "".join(demonstraion_values)
        random_corrupted_index = []
        for idx in corrupted_index:
            if idx in demonstration:
                random_corrupted_index.append(list(demonstration).index(idx))
        input_string = all_demonstraion + test_examples
        length_1 = len(tokenizer.encode(demonstraion_values[0], return_tensors='pt')[0])
        length_2 = len(tokenizer.encode(demonstraion_values[1], return_tensors='pt')[0])
        length_3 = len(tokenizer.encode(demonstraion_values[2], return_tensors='pt')[0])
        length_4 = len(tokenizer.encode(demonstraion_values[3], return_tensors='pt')[0])
        length_5 = len(tokenizer.encode(demonstraion_values[4], return_tensors='pt')[0])
        # print(correct_length_1, correct_length_2, correct_length_3, correct_length_4, correct_length_5)
        position_1 = length_1 - 1 - 3
        position_2 = length_1 + length_2 - 2 - 3
        position_3 = length_1 + length_2 + length_3 - 3 - 3
        position_4 = length_1 + length_2 + length_3 + length_4 - 4 - 3
        position_5 = length_1 + length_2 + length_3 + length_4 + length_5 - 5 - 3
        # print(correct_position_1, correct_position_2, correct_position_3, correct_position_4, correct_position_5)

        inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        outputs = model(inputs)
        attention = outputs[-1]

        prob_list = []
        sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
        prob_1 = sum_attention[:, position_1]
        prob_list.append(prob_1)
        prob_2 = sum_attention[:, position_2]
        prob_list.append(prob_2)
        prob_3 = sum_attention[:, position_3]
        prob_list.append(prob_3)
        prob_4 = sum_attention[:, position_4]
        prob_list.append(prob_4)
        prob_5 = sum_attention[:, position_5]
        prob_list.append(prob_5)
        prob = prob_1 + prob_2 + prob_3 + prob_4 + prob_5

        correct_prob = 0
        corrupted_prob = 0
        for k in range(5):
            if k not in random_corrupted_index:
                correct_prob += prob_list[k]
            else:
                corrupted_prob += prob_list[k]

        correct_prob = correct_prob / 2
        corrupted_prob = corrupted_prob / 3

        scores.append(prob)
        correct_scores.append(correct_prob)
        corrupted_scores.append(corrupted_prob)
        i += 1

    print("gold labels : corrupted labels = 2：3", sum(scores) / len(scores))
    print(sum(correct_scores) / len(correct_scores))
    print(sum(corrupted_scores) / len(corrupted_scores))

    # gold labels : corrupted labels = 1：4
    scores = []

    i = 0
    while i < 50:
        # select one positive and one negative example from the demonstration examples
        corrupted_index = [1, 2, 3, 4]
        # select one positive and one negative example from the demonstration examples
        correct_demonstration = random.sample(correct_examples, k=1)
        corrupted_demonstration = random.sample(corrupted_examples, k=4)
        examples = correct_demonstration + corrupted_demonstration
        demonstration = {}
        for j, example in enumerate(examples):
            demonstration[j] = example
        item = list(demonstration.items())
        random.shuffle(item)
        demonstration = dict(item)
        demonstraion_values = list(demonstration.values())
        all_demonstraion = "".join(demonstraion_values)
        random_corrupted_index = []
        for idx in corrupted_index:
            if idx in demonstration:
                random_corrupted_index.append(list(demonstration).index(idx))
        input_string = all_demonstraion + test_examples
        length_1 = len(tokenizer.encode(demonstraion_values[0], return_tensors='pt')[0])
        length_2 = len(tokenizer.encode(demonstraion_values[1], return_tensors='pt')[0])
        length_3 = len(tokenizer.encode(demonstraion_values[2], return_tensors='pt')[0])
        length_4 = len(tokenizer.encode(demonstraion_values[3], return_tensors='pt')[0])
        length_5 = len(tokenizer.encode(demonstraion_values[4], return_tensors='pt')[0])
        # print(correct_length_1, correct_length_2, correct_length_3, correct_length_4, correct_length_5)
        position_1 = length_1 - 1 - 3
        position_2 = length_1 + length_2 - 2 - 3
        position_3 = length_1 + length_2 + length_3 - 3 - 3
        position_4 = length_1 + length_2 + length_3 + length_4 - 4 - 3
        position_5 = length_1 + length_2 + length_3 + length_4 + length_5 - 5 - 3
        # print(correct_position_1, correct_position_2, correct_position_3, correct_position_4, correct_position_5)

        inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        outputs = model(inputs)
        attention = outputs[-1]

        prob_list = []
        sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
        prob_1 = sum_attention[:, position_1]
        prob_list.append(prob_1)
        prob_2 = sum_attention[:, position_2]
        prob_list.append(prob_2)
        prob_3 = sum_attention[:, position_3]
        prob_list.append(prob_3)
        prob_4 = sum_attention[:, position_4]
        prob_list.append(prob_4)
        prob_5 = sum_attention[:, position_5]
        prob_list.append(prob_5)
        prob = prob_1 + prob_2 + prob_3 + prob_4 + prob_5

        correct_prob = 0
        corrupted_prob = 0
        for k in range(5):
            if k not in random_corrupted_index:
                correct_prob += prob_list[k]
            else:
                corrupted_prob += prob_list[k]

        corrupted_prob = corrupted_prob / 4

        scores.append(prob)
        correct_scores.append(correct_prob)
        corrupted_scores.append(corrupted_prob)
        i += 1

    print("gold labels : corrupted labels = 1：4", sum(scores) / len(scores))
    print(sum(correct_scores) / len(correct_scores))
    print(sum(corrupted_scores) / len(corrupted_scores))

    # all corrupted labels
    scores = []

    i = 0
    while i < 50:
        # select one positive and one negative example from the demonstration examples
        corrupted_demonstration = random.sample(corrupted_examples, k=5)
        corrupted = "".join(corrupted_demonstration)
        input_string = corrupted + test_examples
        corrputed_length_1 = len(tokenizer.encode(corrupted_demonstration[0], return_tensors='pt')[0])
        corrputed_length_2 = len(tokenizer.encode(corrupted_demonstration[1], return_tensors='pt')[0])
        corrputed_length_3 = len(tokenizer.encode(corrupted_demonstration[2], return_tensors='pt')[0])
        corrputed_length_4 = len(tokenizer.encode(corrupted_demonstration[3], return_tensors='pt')[0])
        corrputed_length_5 = len(tokenizer.encode(corrupted_demonstration[4], return_tensors='pt')[0])
        # print(correct_length_1, correct_length_2, correct_length_3, correct_length_4, correct_length_5)
        corrputed_position_1 = corrputed_length_1 - 1 - 3
        corrputed_position_2 = corrputed_length_1 + corrputed_length_2 - 2 - 3
        corrputed_position_3 = corrputed_length_1 + corrputed_length_2 + corrputed_length_3 - 3 - 3
        corrputed_position_4 = corrputed_length_1 + corrputed_length_2 + corrputed_length_3 + corrputed_length_4 - 4 - 3
        corrputed_position_5 = corrputed_length_1 + corrputed_length_2 + corrputed_length_3 + corrputed_length_4 + corrputed_length_5 - 5 - 3
        # print(corrputed_position_1, corrputed_position_2, corrputed_position_3, corrputed_position_4, corrputed_position_5)

        inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        outputs = model(inputs)
        attention = outputs[-1]

        sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
        corrputed_prob_1 = sum_attention[:, corrputed_position_1]
        corrputed_prob_2 = sum_attention[:, corrputed_position_2]
        corrputed_prob_3 = sum_attention[:, corrputed_position_3]
        corrputed_prob_4 = sum_attention[:, corrputed_position_4]
        corrputed_prob_5 = sum_attention[:, corrputed_position_5]
        prob = corrputed_prob_1 + corrputed_prob_2 + corrputed_prob_3 + corrputed_prob_4 + corrputed_prob_5

        scores.append(prob)

        i += 1

    print("all corrupted labels", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
