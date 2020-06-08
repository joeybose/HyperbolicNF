#!/usr/bin/env bash

set -x

echo "Getting into the script"

echo "Running hyperbolic VAE with WrappedRealNVP flow for csphd: changing number of layers"

python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --z_dim=6 --wandb --namestr='hyperbolic-l2-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=4 --z_dim=6 --wandb --namestr='hyperbolic-l4-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=6 --z_dim=6 --wandb --namestr='hyperbolic-l6-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=8 --z_dim=6 --wandb --namestr='hyperbolic-l8-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --z_dim=6 --wandb --namestr='hyperbolic-l10-WrappedRealNVP' --eval_set='validation'

python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --z_dim=8 --wandb --namestr='hyperbolic-l2-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=4 --z_dim=8 --wandb --namestr='hyperbolic-l4-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=6 --z_dim=8 --wandb --namestr='hyperbolic-l6-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=8 --z_dim=8 --wandb --namestr='hyperbolic-l8-WrappedRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --z_dim=8 --wandb --namestr='hyperbolic-l10-WrappedRealNVP' --eval_set='validation'