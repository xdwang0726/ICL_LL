# aieng-template
aieng template repo, the static code checker runs on python3.8

# Installing dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

# Run
## Fine-tuning 
```
CUDA_VISIBLE_DEVICES=0 python fine-tuning.py --dataset {dataset} --gpt2 gpt2-large --num_training_steps 200 --lr 3e-5 --batch_size 2
```

## Test 
```
CUDA_VISIBLE_DEVICES=0  python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --checkpoint checkpoints/gpt2-large/{dataset}/model_{dataset}_{seed}.pt
```
