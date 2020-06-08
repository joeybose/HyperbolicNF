#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for phylo"

python main.py --dataset='phylo' --model='euclidean' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=2 --eval_set='validation' --wandb --namestr='euclidean-2' --eval_set='validation' &&
python main.py --dataset='phylo' --model='euclidean' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=4 --eval_set='validation' --wandb --namestr='euclidean-4' --eval_set='validation' &&
python main.py --dataset='phylo' --model='euclidean' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=6 --eval_set='validation' --wandb --namestr='euclidean-6' --eval_set='validation' &&
python main.py --dataset='phylo' --model='euclidean' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=8 --eval_set='validation' --wandb --namestr='euclidean-8' --eval_set='validation' &&
python main.py --dataset='phylo' --model='euclidean' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=10 --eval_set='validation' --wandb --namestr='euclidean-10' --eval_set='validation' &&

echo "Running hyperbolic VAE for phylo"

python main.py --dataset='phylo' --model='hyperbolic' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=2 --eval_set='validation' --wandb --namestr='hyperbolic-2' --eval_set='validation' &&
python main.py --dataset='phylo' --model='hyperbolic' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=4 --eval_set='validation' --wandb --namestr='hyperbolic-4' --eval_set='validation' &&
python main.py --dataset='phylo' --model='hyperbolic' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=6 --eval_set='validation' --wandb --namestr='hyperbolic-6' --eval_set='validation' &&
python main.py --dataset='phylo' --model='hyperbolic' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=8 --eval_set='validation' --wandb --namestr='hyperbolic-8' --eval_set='validation' &&
python main.py --dataset='phylo' --model='hyperbolic' --decoder='distance' --epochs=2000 --hidden_dim=128 --z_dim=10 --eval_set='validation' --wandb --namestr='hyperbolic-10' --eval_set='validation' &&
