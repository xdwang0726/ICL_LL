
# Installing dependencies
```
conda create --name ICL-LL python=3.8
conda activate ICL-LL
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/huggingface/transformers
pip install datasets
pip install sklearn

```
# Preparation
datasets are: `` ag_news``, ``glue-rte``, `` glue-sst2``, ``rotten_tomatoes``, ``trec``, ``superglue-cb``
## prepare the dataset
```
cd preprocess
python _build_gym.py --build --n_proc=40 --do_test --test_k {4|8|16|32}
```

# Noisy Label

Create data with different label corruption rate
```
python create_data.py --variant {75|50|25|0}_correct --dataset {dataset}
```
## In-context Learning
To run the evaluation of all gold labels
```
python test.py --dataset {dataset} --gpt2 {gpt2-large|gpt-neo|gpt-neox|gpt-j} --method direct --out_dir out/{model} --do_zeroshot --use_demonstrations --k 16 --seed 100,13,21,42,87
```
To run the evaluation of label corruption 25-100%
```
python test.py --dataset {dataset}_{75|50|25|0}_correct --gpt2 {gpt2-large|gpt-neo|gpt-neox|gpt-j} --method direct --out_dir out/{model} --do_zeroshot --use_demonstrations --k 16 --seed 100,13,21,42,87 
```

## Supervised Learning
### Grid search: search hyper-parameter for each dataset 
gpt2:
```
CUDA_VISIBLE_DEVICES=0 python grid_search.py  --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl} --out_dir hyperparameter/noisy_label/{gpt2-large|gpt2-xl}
```
gpt-j (distributed):
```
python grid_search_distributed.py  --dataset {dataset} --gpt2 {gpt-j} --out_dir hyperparameter/noisy_label/gpt-j
```

### Fine-tuning and do supervised learning
```
CUDA_VISIBLE_DEVICES=0  python fine-tuning.py --dataset {dataset} --gpt2 {gpt2-large|gpt-neo|gpt-neox|gpt-j} --correct {100|75|50|25|0} --result_dir supervised_learning_results/noisy_label
```

# Label distribution 
Create data with different imbalance ratio
```
python _build_gym.py --build --n_proc=6 --do_test 
```
## In-context Learning
To run the evaluation of different imbalance ratio
```
python test.py --dataset {dataset}_{75|50|25|0}_correct --gpt2 {gpt2-large|gpt-neo|gpt-neox|gpt-j} --method direct --out_dir out/{model} --do_zeroshot --use_demonstrations --k 16 --seed 100,13,21,42,87 
```
## Supervised Learning 
### Grid search:
gpt2:
```
CUDA_VISIBLE_DEVICES=0 python grid_search.py --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl} --label_imbalance --imbalance_level low --out_dir hyperparameter/label_imbalance/{gpt2-large|gpt2-xl}
```
gpt-j:
```
python grid_search_distributed.py  --dataset {dataset} --gpt2 {gpt-j} --label_imbalance --out_dir hyperparameter/noisy_label/gpt-j
```
### Fine-tuning and do supervised learning
```
CUDA_VISIBLE_DEVICES=0  python fine-tuning.py --dataset {dataset} --gpt2 {gpt2-large|gpt-neo|gpt-neox|gpt-j} --label_imbalance --imbalance_level {low|medium|high} --result_dir supervised_learning_results/noisy_label
```
