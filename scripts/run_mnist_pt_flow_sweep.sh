#!/usr/bin/env bash

set -x

echo "Getting into the script"

python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=2 --wandb --namestr='hyperbolic-d2-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=6 --wandb --namestr='hyperbolic-d6-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=10 --wandb --namestr='hyperbolic-d10-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=20 --wandb --namestr='hyperbolic-d20-PTRealNVP'

echo "Running hyperbolic VAE with PTRealNVP flow for MNIST: changing number of layers"

python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=2 --wandb --namestr='hyperbolic-l2-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=4 --wandb --namestr='hyperbolic-l4-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=6 --wandb --namestr='hyperbolic-l6-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=8 --wandb --namestr='hyperbolic-l8-PTRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=10 --wandb --namestr='hyperbolic-l10-PTRealNVP'
