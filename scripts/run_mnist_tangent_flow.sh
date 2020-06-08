#!/usr/bin/env bash

set -x

echo "Getting into the script"

python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=2 --wandb --namestr='hyperbolic-d2-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=6 --wandb --namestr='hyperbolic-d6-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=10 --wandb --namestr='hyperbolic-d10-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=20 --wandb --namestr='hyperbolic-d20-TangentRealNVP'

echo "Running hyperbolic VAE with TangentRealNVP flow for MNIST: changing number of layers"

python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=2 --wandb --namestr='hyperbolic-l2-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=4 --wandb --namestr='hyperbolic-l4-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=6 --wandb --namestr='hyperbolic-l6-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=8 --wandb --namestr='hyperbolic-l8-TangentRealNVP'
python main.py --dataset='mnist' --model='hyperbolic' --flow_model='TangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=10 --wandb --namestr='hyperbolic-l10-TangentRealNVP'
