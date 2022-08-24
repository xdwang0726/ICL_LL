# aieng-template
aieng template repo, the static code checker runs on python3.8

# Installing dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
# Preparation
## prepare the dataset
```
cd preprocess
python _build_gym.py --build --n_proc=40 --do_test --test_k {4|8|16|32}
```
# Run
## Step 1: Grid search
search hyper-parameter for each dataset 
```
CUDA_VISIBLE_DEVICES=0 python grid_search.py --dataset {dataset} --gpt2 gpt2-large 
```

## Step 2: Fine-tuning and Do Supervised Learning
```
CUDA_VISIBLE_DEVICES=0  python fine-tuning.py --dataset {dataset} --gpt2 gpt2-large --correct {} 
```
