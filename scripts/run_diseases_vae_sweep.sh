#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for diseases"

python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=2 --eval_set='validation' --wandb --namestr='euclidean-2'
python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=4 --eval_set='validation' --wandb --namestr='euclidean-4'
python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=5 --eval_set='validation' --wandb --namestr='euclidean-5'
python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=6 --eval_set='validation' --wandb --namestr='euclidean-6'
python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=8 --eval_set='validation' --wandb --namestr='euclidean-8'
python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=10 --eval_set='validation' --wandb --namestr='euclidean-10'

echo "Running hyperbolic VAE for diseases"

python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=2 --eval_set='validation' --wandb --namestr='hyperbolic-2'
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=4 --eval_set='validation' --wandb --namestr='hyperbolic-4'
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=5 --eval_set='validation' --wandb --namestr='hyperbolic-5'
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=6 --eval_set='validation' --wandb --namestr='hyperbolic-6'
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=8 --eval_set='validation' --wandb --namestr='hyperbolic-8'
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=10 --eval_set='validation' --wandb --namestr='hyperbolic-10'
