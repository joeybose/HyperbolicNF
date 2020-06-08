#!/usr/bin/env bash

set -x

echo "Running hyperbolic VAE with WrappedRealNVP flow for Diseases: changing dimension"

python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=2 --wandb --namestr='Test-hyperbolic-d2-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=4 --wandb --namestr='Test-hyperbolic-d4-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=6 --wandb --namestr='Test-hyperbolic-d6-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=8 --wandb --namestr='Test-hyperbolic-d8-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=10 --wandb --namestr='Test-hyperbolic-d10-WrappedRealNVP'

echo "Running hyperbolic VAE with WrappedRealNVP flow for Diseases: changing number of layers"

python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=2 --wandb --namestr='Test-hyperbolic-l2-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=4 --wandb --namestr='Test-hyperbolic-l4-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=6 --wandb --namestr='Test-hyperbolic-l6-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=8 --wandb --namestr='Test-hyperbolic-l8-WrappedRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=10 --wandb --namestr='Test-hyperbolic-l10-WrappedRealNVP'
