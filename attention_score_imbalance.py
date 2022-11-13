from transformers import AutoModelForCausalLM, AutoModel
from transformers import GPT2Tokenizer, AutoTokenizer
from bertviz import head_view
import torch
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random select demonstration examples
demonstration_data_dir = "/data/v-xindiwang/ICL_LL/data_imbalance/rotten_tomatoes_low/rotten_tomatoes_16_100_test.jsonl"
positive_examples = []
negative_examples = []
with open(demonstration_data_dir, "r") as f:
    for line in f:
        dp = json.loads(line)
        if dp["output"] == "positive":
            positive_examples.append(dp["input"])
        else:
            negative_examples.append(dp["input"])
all_examples = positive_examples + negative_examples
test_examples = random.choice(all_examples)

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", output_attentions=True)
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", output_attentions=True)
# model = AutoModelForCausalLM.from_pretrained("gpt2-large", output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.to(device)

# 1:1
pos_scores = []
neg_scores = []
i = 0
while i < 50:
    # select one positive and one negative example from the demonstration examples
    pos_demonstration = random.choice(positive_examples)
    neg_demonstration = random.choice(negative_examples)
    # input_string = pos_demonstration + " \n positive \n\n " + neg_demonstration + " \n negative \n\n " + test_examples
    input_string = neg_demonstration + " \n negative \n\n " + pos_demonstration + " \n positive \n\n " + test_examples
    pos_length = len(tokenizer.encode(pos_demonstration + " \n positive \n\n ", return_tensors='pt')[0])
    neg_length = len(tokenizer.encode(neg_demonstration + " \n negative \n\n ", return_tensors='pt')[0])
    # pos_position = pos_length - 1 - 3
    # neg_position = pos_length + neg_length - 2 -3
    neg_position = neg_length - 1 - 3
    pos_position = neg_length + pos_length - 2 - 3

    inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs[-1]

    sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
    pos_prob = sum_attention[:, pos_position]
    neg_prob = sum_attention[:, neg_position]

    pos_scores.append(pos_prob)
    neg_scores.append(neg_prob)

    i += 1

print(sum(pos_scores) / len(pos_scores))
print(sum(neg_scores) / len(neg_scores))

# 2:1
pos_scores = []
neg_scores = []
i = 0
while i < 50:
    # select one positive and one negative example from the demonstration examples
    # pos_demonstration = random.sample(positive_examples, k=2)
    # neg_demonstration = random.choice(negative_examples)
    # pos_examples = " \n positive \n\n ".join(pos_demonstration)
    pos_demonstration = random.choice(positive_examples)
    neg_demonstration = random.sample(negative_examples, k=2)
    neg_examples = " \n negative \n\n ".join(neg_demonstration)
    # input_string = pos_examples + " \n positive \n\n " + neg_demonstration + " \n negative \n\n " + test_examples
    input_string = neg_examples + " \n negative \n\n " + pos_demonstration + " \n positive \n\n " + test_examples
    # pos_length_1 = len(tokenizer.encode(pos_demonstration[0] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_2 = len(tokenizer.encode(pos_examples + " \n positive \n\n ", return_tensors='pt')[0])
    # neg_length = len(tokenizer.encode(neg_demonstration + " \n negative \n\n ", return_tensors='pt')[0])
    # pos_position_1 = pos_length_1 - 1 - 3
    # pos_position_2 = pos_length_2 - 1 - 3
    # neg_position = pos_length_2 + neg_length - 2 - 3
    neg_length_1 = len(tokenizer.encode(neg_demonstration[0] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_2 = len(tokenizer.encode(neg_examples + " \n negative \n\n ", return_tensors='pt')[0])
    pos_length = len(tokenizer.encode(pos_demonstration + " \n positive \n\n ", return_tensors='pt')[0])
    neg_position_1 = neg_length_1 - 1 - 3
    neg_position_2 = neg_length_2 - 1 - 3
    pos_position = neg_length_2 + pos_length - 2 - 3

    inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs[-1]

    sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
    # pos_prob_1 = sum_attention[:, pos_position_1]
    # pos_prob_2 = sum_attention[:, pos_position_2]
    # pos_prob = pos_prob_1 + pos_prob_2
    # neg_prob = sum_attention[:, neg_position]
    neg_prob_1 = sum_attention[:, neg_position_1]
    neg_prob_2 = sum_attention[:, neg_position_2]
    neg_prob = neg_prob_1 + neg_prob_2
    pos_prob = sum_attention[:, pos_position]

    pos_scores.append(pos_prob)
    neg_scores.append(neg_prob)

    i += 1

print(sum(pos_scores) / len(pos_scores))
print(sum(neg_scores) / len(neg_scores))

# 3:1
pos_scores = []
neg_scores = []
i = 0
while i < 50:
    # select one positive and one negative example from the demonstration examples
    # pos_demonstration = random.sample(positive_examples, k=3)
    # neg_demonstration = random.choice(negative_examples)
    # pos_examples = " \n positive \n\n ".join(pos_demonstration)
    # input_string = pos_examples + " \n positive \n\n " + neg_demonstration + " \n negative \n\n " + test_examples
    # pos_length_1 = len(tokenizer.encode(pos_demonstration[0] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_2 = len(tokenizer.encode(pos_demonstration[1] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_3 = len(tokenizer.encode(pos_examples + " \n positive \n\n ", return_tensors='pt')[0])
    # neg_length = len(tokenizer.encode(neg_demonstration + " \n negative \n\n ", return_tensors='pt')[0])
    # pos_position_1 = pos_length_1 - 1 - 3
    # pos_position_2 = pos_length_1 + pos_length_2 - 2 - 3
    # pos_position_3 = pos_length_3 - 1 - 3
    # neg_position = pos_length_3 + neg_length - 2 - 3

    pos_demonstration = random.choice(positive_examples)
    neg_demonstration = random.sample(negative_examples, k=3)
    neg_examples = " \n negative \n\n ".join(neg_demonstration)
    input_string = neg_examples + " \n negative \n\n " + pos_demonstration + " \n positive \n\n " + test_examples
    neg_length_1 = len(tokenizer.encode(neg_demonstration[0] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_2 = len(tokenizer.encode(neg_demonstration[1] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_3 = len(tokenizer.encode(neg_examples + " \n negative \n\n ", return_tensors='pt')[0])
    pos_length = len(tokenizer.encode(pos_demonstration + " \n positive \n\n ", return_tensors='pt')[0])
    neg_position_1 = neg_length_1 - 1 - 3
    neg_position_2 = neg_length_1 + neg_length_2 - 2 - 3
    neg_position_3 = neg_length_3 - 1 - 3
    pos_position = neg_length_3 + pos_length - 2 - 3

    inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs[-1]

    sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
    # pos_prob_1 = sum_attention[:, pos_position_1]
    # pos_prob_2 = sum_attention[:, pos_position_2]
    # pos_prob_3 = sum_attention[:, pos_position_3]
    # pos_prob = pos_prob_1 + pos_prob_2 + pos_prob_3
    # neg_prob = sum_attention[:, neg_position]
    neg_prob_1 = sum_attention[:, neg_position_1]
    neg_prob_2 = sum_attention[:, neg_position_2]
    neg_prob_3 = sum_attention[:, neg_position_3]
    neg_prob = neg_prob_1 + neg_prob_2 + neg_prob_3
    pos_prob = sum_attention[:, pos_position]

    pos_scores.append(pos_prob)
    neg_scores.append(neg_prob)

    i += 1

print(sum(pos_scores) / len(pos_scores))
print(sum(neg_scores) / len(neg_scores))

# 4:1
pos_scores = []
neg_scores = []
i = 0
while i < 50:
    # select one positive and one negative example from the demonstration examples
    # pos_demonstration = random.sample(positive_examples, k=4)
    # neg_demonstration = random.choice(negative_examples)
    # pos_examples = " \n positive \n\n ".join(pos_demonstration)
    # input_string = pos_examples + " \n positive \n\n " + neg_demonstration + " \n negative \n\n " + test_examples
    # pos_length_1 = len(tokenizer.encode(pos_demonstration[0] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_2 = len(tokenizer.encode(pos_demonstration[1] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_3 = len(tokenizer.encode(pos_demonstration[2] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_4 = len(tokenizer.encode(pos_examples + " \n positive \n\n ", return_tensors='pt')[0])
    # neg_length = len(tokenizer.encode(neg_demonstration + " \n negative \n\n ", return_tensors='pt')[0])
    # pos_position_1 = pos_length_1 - 1 - 3
    # pos_position_2 = pos_length_1 + pos_length_2 - 2 - 3
    # pos_position_3 = pos_length_1 + pos_length_2 + pos_length_3 - 3 - 3
    # pos_position_4 = pos_length_4 - 1 - 3
    # neg_position = pos_length_4 + neg_length - 2 - 3
    # print(pos_position_1, pos_position_2, pos_position_3, pos_position_4, neg_position)

    pos_demonstration = random.choice(positive_examples)
    neg_demonstration = random.sample(negative_examples, k=4)
    neg_examples = " \n negative \n\n ".join(neg_demonstration)
    input_string = neg_examples + " \n negative \n\n " + pos_demonstration + " \n positive \n\n " + test_examples
    neg_length_1 = len(tokenizer.encode(neg_demonstration[0] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_2 = len(tokenizer.encode(neg_demonstration[1] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_3 = len(tokenizer.encode(neg_demonstration[2] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_4 = len(tokenizer.encode(neg_examples + " \n negative \n\n ", return_tensors='pt')[0])
    pos_length = len(tokenizer.encode(pos_demonstration + " \n positive \n\n ", return_tensors='pt')[0])
    neg_position_1 = neg_length_1 - 1 - 3
    neg_position_2 = neg_length_1 + neg_length_2 - 2 - 3
    neg_position_3 = neg_length_1 + neg_length_2 + neg_length_3 - 3 - 3
    neg_position_4 = neg_length_4 - 1 - 3
    pos_position = neg_length_4 + pos_length - 2 - 3

    inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs[-1]

    sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
    # pos_prob_1 = sum_attention[:, pos_position_1]
    # pos_prob_2 = sum_attention[:, pos_position_2]
    # pos_prob_3 = sum_attention[:, pos_position_3]
    # pos_prob_4 = sum_attention[:, pos_position_4]
    # pos_prob = pos_prob_1 + pos_prob_2 + pos_prob_3 + pos_prob_4
    # neg_prob = sum_attention[:, neg_position]
    neg_prob_1 = sum_attention[:, neg_position_1]
    neg_prob_2 = sum_attention[:, neg_position_2]
    neg_prob_3 = sum_attention[:, neg_position_3]
    neg_prob_4 = sum_attention[:, neg_position_4]
    neg_prob = neg_prob_1 + neg_prob_2 + neg_prob_3 + neg_prob_4
    pos_prob = sum_attention[:, pos_position]

    pos_scores.append(pos_prob)
    neg_scores.append(neg_prob)

    i += 1

print(sum(pos_scores) / len(pos_scores))
print(sum(neg_scores) / len(neg_scores))

# 5:1
pos_scores = []
neg_scores = []
i = 0
while i < 50:
    # select one positive and one negative example from the demonstration examples
    # pos_demonstration = random.sample(positive_examples, k=5)
    # neg_demonstration = random.choice(negative_examples)
    # pos_examples = " \n positive \n\n ".join(pos_demonstration)
    # input_string = pos_examples + " \n positive \n\n " + neg_demonstration + " \n negative \n\n " + test_examples
    # pos_length_1 = len(tokenizer.encode(pos_demonstration[0] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_2 = len(tokenizer.encode(pos_demonstration[1] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_3 = len(tokenizer.encode(pos_demonstration[2] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_4 = len(tokenizer.encode(pos_demonstration[3] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_5 = len(tokenizer.encode(pos_examples + " \n positive \n\n ", return_tensors='pt')[0])
    # neg_length = len(tokenizer.encode(neg_demonstration + " \n negative \n\n ", return_tensors='pt')[0])
    # pos_position_1 = pos_length_1 - 1 - 3
    # pos_position_2 = pos_length_1 + pos_length_2 - 2 - 3
    # pos_position_3 = pos_length_1 + pos_length_2 + pos_length_3 - 3 - 3
    # pos_position_4 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 - 4 - 3
    # pos_position_5 = pos_length_5 - 1 - 3
    # neg_position = pos_length_5 + neg_length - 2 - 3

    pos_demonstration = random.choice(positive_examples)
    neg_demonstration = random.sample(negative_examples, k=5)
    neg_examples = " \n negative \n\n ".join(neg_demonstration)
    input_string = neg_examples + " \n negative \n\n " + pos_demonstration + " \n positive \n\n " + test_examples
    neg_length_1 = len(tokenizer.encode(neg_demonstration[0] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_2 = len(tokenizer.encode(neg_demonstration[1] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_3 = len(tokenizer.encode(neg_demonstration[2] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_4 = len(tokenizer.encode(neg_demonstration[3] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_5 = len(tokenizer.encode(neg_examples + " \n negative \n\n ", return_tensors='pt')[0])
    pos_length = len(tokenizer.encode(pos_demonstration + " \n positive \n\n ", return_tensors='pt')[0])
    neg_position_1 = neg_length_1 - 1 - 3
    neg_position_2 = neg_length_1 + neg_length_2 - 2 - 3
    neg_position_3 = neg_length_1 + neg_length_2 + neg_length_3 - 3 - 3
    neg_position_4 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 - 4 - 3
    neg_position_5 = neg_length_5 - 1 - 3
    pos_position = neg_length_5 + pos_length - 2 - 3

    inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs[-1]

    sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
    # pos_prob_1 = sum_attention[:, pos_position_1]
    # pos_prob_2 = sum_attention[:, pos_position_2]
    # pos_prob_3 = sum_attention[:, pos_position_3]
    # pos_prob_4 = sum_attention[:, pos_position_4]
    # pos_prob_5 = sum_attention[:, pos_position_5]
    # pos_prob = pos_prob_1 + pos_prob_2 + pos_prob_3 + pos_prob_4 + pos_prob_5
    # neg_prob = sum_attention[:, neg_position]
    neg_prob_1 = sum_attention[:, neg_position_1]
    neg_prob_2 = sum_attention[:, neg_position_2]
    neg_prob_3 = sum_attention[:, neg_position_3]
    neg_prob_4 = sum_attention[:, neg_position_4]
    neg_prob_5 = sum_attention[:, neg_position_5]
    neg_prob = neg_prob_1 + neg_prob_2 + neg_prob_3 + neg_prob_4 + neg_prob_5
    pos_prob = sum_attention[:, pos_position]

    pos_scores.append(pos_prob)
    neg_scores.append(neg_prob)

    i += 1

print(sum(pos_scores) / len(pos_scores))
print(sum(neg_scores) / len(neg_scores))

# 10:1
pos_scores = []
neg_scores = []
i = 0
while i < 50:
    # select one positive and one negative example from the demonstration examples
    # pos_demonstration = random.sample(positive_examples, k=10)
    # neg_demonstration = random.choice(negative_examples)
    # pos_examples = " \n positive \n\n ".join(pos_demonstration)
    # input_string = pos_examples + " \n positive \n\n " + neg_demonstration + " \n negative \n\n " + test_examples
    # pos_length_1 = len(tokenizer.encode(pos_demonstration[0] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_2 = len(tokenizer.encode(pos_demonstration[1] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_3 = len(tokenizer.encode(pos_demonstration[2] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_4 = len(tokenizer.encode(pos_demonstration[3] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_5 = len(tokenizer.encode(pos_demonstration[4] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_6 = len(tokenizer.encode(pos_demonstration[5] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_7 = len(tokenizer.encode(pos_demonstration[6] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_8 = len(tokenizer.encode(pos_demonstration[7] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_9 = len(tokenizer.encode(pos_demonstration[8] + " \n positive \n\n ", return_tensors='pt')[0])
    # pos_length_10 = len(tokenizer.encode(pos_examples + " \n positive \n\n ", return_tensors='pt')[0])
    # neg_length = len(tokenizer.encode(neg_demonstration + " \n negative \n\n ", return_tensors='pt')[0])
    # pos_position_1 = pos_length_1 - 1 - 3
    # pos_position_2 = pos_length_1 + pos_length_2 - 2 - 3
    # pos_position_3 = pos_length_1 + pos_length_2 + pos_length_3 - 3 - 3
    # pos_position_4 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 - 4 - 3
    # pos_position_5 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 + pos_length_5 - 5 - 3
    # pos_position_6 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 + pos_length_5 + pos_length_6 - 6 - 3
    # pos_position_7 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 + pos_length_5 + pos_length_6 + pos_length_7 - 7 - 3
    # pos_position_8 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 + pos_length_5 + pos_length_6 + pos_length_7 + pos_length_8 - 8 - 3
    # pos_position_9 = pos_length_1 + pos_length_2 + pos_length_3 + pos_length_4 + pos_length_5 + pos_length_6 + pos_length_7 + pos_length_8 + pos_length_9 - 9 - 3
    # pos_position_10 = pos_length_10 - 1 - 3
    # neg_position = pos_length_10 + neg_length - 2 - 3

    pos_demonstration = random.choice(positive_examples)
    neg_demonstration = random.sample(negative_examples, k=10)
    neg_examples = " \n negative \n\n ".join(neg_demonstration)
    input_string = neg_examples + " \n negative \n\n " + pos_demonstration + " \n positive \n\n " + test_examples
    neg_length_1 = len(tokenizer.encode(neg_demonstration[0] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_2 = len(tokenizer.encode(neg_demonstration[1] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_3 = len(tokenizer.encode(neg_demonstration[2] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_4 = len(tokenizer.encode(neg_demonstration[3] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_5 = len(tokenizer.encode(neg_demonstration[4] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_6 = len(tokenizer.encode(neg_demonstration[5] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_7 = len(tokenizer.encode(neg_demonstration[6] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_8 = len(tokenizer.encode(neg_demonstration[7] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_9 = len(tokenizer.encode(neg_demonstration[8] + " \n negative \n\n ", return_tensors='pt')[0])
    neg_length_10 = len(tokenizer.encode(neg_examples + " \n negative \n\n ", return_tensors='pt')[0])
    pos_length = len(tokenizer.encode(pos_demonstration + " \n positive \n\n ", return_tensors='pt')[0])
    neg_position_1 = neg_length_1 - 1 - 3
    neg_position_2 = neg_length_1 + neg_length_2 - 2 - 3
    neg_position_3 = neg_length_1 + neg_length_2 + neg_length_3 - 3 - 3
    neg_position_4 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 - 4 - 3
    neg_position_5 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 + neg_length_5 - 5 - 3
    neg_position_6 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 + neg_length_5 + neg_length_6 - 6 - 3
    neg_position_7 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 + neg_length_5 + neg_length_6 + neg_length_7 - 7 - 3
    neg_position_8 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 + neg_length_5 + neg_length_6 + neg_length_7 + neg_length_8 - 8 - 3
    neg_position_9 = neg_length_1 + neg_length_2 + neg_length_3 + neg_length_4 + neg_length_5 + neg_length_6 + neg_length_7 + neg_length_8 + neg_length_9 - 9 - 3
    neg_position_10 = neg_length_10 - 1 - 3
    pos_position = neg_length_10 + pos_length - 2 - 3

    inputs = tokenizer.encode(input_string, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs[-1]

    sum_attention = torch.sum(attention[0][:, :, -1, :], dim=1)
    # pos_prob_1 = sum_attention[:, pos_position_1]
    # pos_prob_2 = sum_attention[:, pos_position_2]
    # pos_prob_3 = sum_attention[:, pos_position_3]
    # pos_prob_4 = sum_attention[:, pos_position_4]
    # pos_prob_5 = sum_attention[:, pos_position_5]
    # pos_prob_6 = sum_attention[:, pos_position_6]
    # pos_prob_7 = sum_attention[:, pos_position_7]
    # pos_prob_8 = sum_attention[:, pos_position_8]
    # pos_prob_9 = sum_attention[:, pos_position_9]
    # pos_prob_10 = sum_attention[:, pos_position_10]
    # pos_prob = pos_prob_1 + pos_prob_2 + pos_prob_3 + pos_prob_4 + pos_prob_5 + pos_prob_6 + pos_prob_7 + pos_prob_8 + pos_prob_9 + pos_prob_10
    # neg_prob = sum_attention[:, neg_position]
    neg_prob_1 = sum_attention[:, neg_position_1]
    neg_prob_2 = sum_attention[:, neg_position_2]
    neg_prob_3 = sum_attention[:, neg_position_3]
    neg_prob_4 = sum_attention[:, neg_position_4]
    neg_prob_5 = sum_attention[:, neg_position_5]
    neg_prob_6 = sum_attention[:, neg_position_6]
    neg_prob_7 = sum_attention[:, neg_position_7]
    neg_prob_8 = sum_attention[:, neg_position_8]
    neg_prob_9 = sum_attention[:, neg_position_9]
    neg_prob_10 = sum_attention[:, neg_position_10]
    neg_prob = neg_prob_1 + neg_prob_2 + neg_prob_3 + neg_prob_4 + neg_prob_5 + neg_prob_6 + neg_prob_7 + neg_prob_8 + neg_prob_9 + neg_prob_10
    pos_prob = sum_attention[:, pos_position]

    pos_scores.append(pos_prob)
    neg_scores.append(neg_prob)

    i += 1

print(sum(pos_scores) / len(pos_scores))
print(sum(neg_scores) / len(neg_scores))
