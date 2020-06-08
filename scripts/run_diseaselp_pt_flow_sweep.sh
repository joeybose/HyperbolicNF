#!/usr/bin/env bash

set -x

echo "Running hyperbolic VAE with PTRealNVP flow for Diseases: changing dimension"

python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=2 --wandb --namestr='hyperbolic-d2-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=4 --wandb --namestr='hyperbolic-d4-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=6 --wandb --namestr='hyperbolic-d6-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=8 --wandb --namestr='hyperbolic-d8-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --z_dim=10 --wandb --namestr='hyperbolic-d10-PTRealNVP'

echo "Running hyperbolic VAE with PTRealNVP flow for Diseases: changing number of layers"

python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=2 --wandb --namestr='hyperbolic-l2-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=4 --wandb --namestr='hyperbolic-l4-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=6 --wandb --namestr='hyperbolic-l6-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=8 --wandb --namestr='hyperbolic-l8-PTRealNVP'
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=16 --flow_hidden_size=32 --n_blocks=10 --wandb --namestr='hyperbolic-l10-PTRealNVP'
