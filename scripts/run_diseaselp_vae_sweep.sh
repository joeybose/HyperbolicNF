#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for Diseases"

python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=2 --wandb --namestr='euclidean-2'
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=4 --wandb --namestr='euclidean-4'
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=6 --wandb --namestr='euclidean-6'
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=8 --wandb --namestr='euclidean-8'
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=10 --wandb --namestr='euclidean-10'

python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=2 --wandb --namestr='euclidean-2' --ll_estimate=iwae
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=4 --wandb --namestr='euclidean-4' --ll_estimate=iwae
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=6 --wandb --namestr='euclidean-6' --ll_estimate=iwae
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=8 --wandb --namestr='euclidean-8' --ll_estimate=iwae
python main.py --dataset='disease_lp' --model='euclidean' --epochs=3000 --hidden_dim=16 --z_dim=10 --wandb --namestr='euclidean-10' --ll_estimate=iwae

echo "Running hyperbolic VAE for Diseases"

python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=3000 --hidden_dim=16 --z_dim=2 --wandb --namestr='hyperbolic-2'
python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=3000 --hidden_dim=16 --z_dim=4 --wandb --namestr='hyperbolic-4'
python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=3000 --hidden_dim=16 --z_dim=6 --wandb --namestr='hyperbolic-6'
python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=3000 --hidden_dim=16 --z_dim=8 --wandb --namestr='hyperbolic-8'
python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=3000 --hidden_dim=16 --z_dim=10 --wandb --namestr='hyperbolic-10'
