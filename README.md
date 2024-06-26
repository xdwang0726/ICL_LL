# Investigating the Learning Behaviour of In-context Learning:  A Comparison with Supervised Learning
This repository contains the implementations of our paper at ECAI 2023: "Investigating the Learning Behaviour of In-context Learning:  A Comparison with Supervised Learning". If you have any questions, feel free to contact us at: xwang842@uwo.ca

# Installing dependencies
```
conda create --name ICL-LL python=3.8
conda activate ICL-LL
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install datasets
pip install -U scikit-learn
```
# Preparation
datasets are: `` ag_news``, ``glue-rte``, `` glue-sst2``, ``rotten_tomatoes``, ``trec``, ``superglue-cb``
## prepare the dataset
```
cd preprocess
python _build_gym.py --build --n_proc=6 --do_test --test_k 16
```

# Noisy Label

Create data with different label corruption rate
```
python create_data.py --variant {75|50|25|0}_correct --dataset {dataset}
```
## In-context Learning
To run the evaluation of all gold labels
```
python test.py --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl|gpt-j} --method direct --out_dir out/{model} --do_zeroshot --use_demonstrations --k 16 --seed 100,13,21,42,87 --test_batch_size 32 --task_name {dataset}
```
To run the evaluation of label corruption 0-75%
```
python test.py --dataset {dataset}_{75|50|25|0}_correct --gpt2 {gpt2-large|gpt2-xl|gpt-j} --method direct --out_dir out/{model} --do_zeroshot --use_demonstrations --k 16 --seed 100,13,21,42,87 --test_batch_size 32 --task_name {dataset}
```

## Supervised Learning
### Grid search: search hyper-parameter for each dataset 
gpt2:
```
CUDA_VISIBLE_DEVICES=0 python grid_search.py  --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl} --out_dir hyperparameter/noisy_label/ --task_name {dataset}
```
gpt-j (distributed):
```
python grid_search.py  --dataset {dataset} --gpt2 {gpt-j} --out_dir hyperparameter/noisy_label/gpt-j --distributed --task_name {dataset}
```

### Fine-tuning and do supervised learning
gpt2:
```
CUDA_VISIBLE_DEVICES=0  python fine-tuning.py --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl} --correct {100|75|50|25|0} --result_dir supervised_learning_results/noisy_label --task_name {dataset}
```
gpt-j (distributed):
```
python fine-tuning.py --dataset {dataset} --gpt2 {gpt-j} --correct {100|75|50|25|0} --result_dir supervised_learning_results/noisy_label --distributed --task_name {dataset}
```

# Label distribution 
Create data with different imbalance ratio
```
python _build_gym.py --build --n_proc=6 --do_test 
```
## In-context Learning
To run the evaluation of different imbalance ratio
```
python test.py --dataset {dataset}_{75|50|25|0}_correct --gpt2 {gpt2-large|gpt2-xl|gpt-j} --method direct --out_dir out/{model} --do_zeroshot --use_demonstrations --k 16 --seed 100,13,21,42,87 --test_batch_size 32
```
## Supervised Learning 
### Grid search:
gpt2:
```
CUDA_VISIBLE_DEVICES=0 python grid_search.py --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl} --label_imbalance --imbalance_level low --out_dir hyperparameter/label_imbalance/{gpt2-large|gpt2-xl}
```
gpt-j:
```
python grid_search.py  --dataset {dataset} --gpt2 {gpt-j} --label_imbalance --out_dir hyperparameter/label_imbalance/gpt-j --distributed
```
### Fine-tuning and do supervised learning
gpt2:
```
CUDA_VISIBLE_DEVICES=0  python fine-tuning.py --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl} --label_imbalance --imbalance_level {low|medium|high} --result_dir supervised_learning_results/label_imbalance
```
gptj (distributed):
```
python fine-tuning.py --dataset {dataset} --gpt2 {gpt-j} --label_imbalance --imbalance_level {low|medium|high} --result_dir supervised_learning_results/label_imbalance --distributed
```
# Analysis
## Attention Score
### noisy label
```
python attention_score_noisy.py --gpt2 {gpt2-large|gpt2-xl|gpt-j} --dataset {dataset} --dir {data_dir}
```
### label imbalance
```
python attention_score_imbalance.py --gpt2 {gpt2-large|gpt2-xl|gpt-j} --dataset {dataset} --dir {data_dir}
```
## Agreement
```
python agreement.py --dataset {dataset} --gpt2 {gpt2-large|gpt2-xl|gpt-j} --correct {100|75|50|25|0} --icl_dir {icl_dir} --sl_dir {sl_dir}
```

# Citation 
If you use our codes in your work, please consider citing our paper：
```
@article{wang2023investigating,
  title={Investigating the learning behaviour of in-context learning: a comparison with supervised learning},
  author={Wang, Xindi and Wang, Yufei and Xu, Can and Geng, Xiubo and Zhang, Bowen and Tao, Chongyang and Rudzicz, Frank and Mercer, Robert E and Jiang, Daxin},
  journal={arXiv preprint arXiv:2307.15411},
  year={2023}
}
```
