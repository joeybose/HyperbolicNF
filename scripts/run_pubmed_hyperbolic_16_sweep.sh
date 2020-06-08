#!/usr/bin/env bash

set -x

echo "Running hyperbolic VAE for Pubmed"
python main.py --dataset='pubmed' --model='hyperbolic' --epochs=3000 --hidden_dim=128 --z_dim=16 --wandb --namestr='hyperbolic-16'

echo "Running hyperbolic VAE with TangentRealNVP flow for Pubmed: changing number of layers"

python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=2 --wandb --namestr='16-hyperbolic-l2-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=4 --wandb --namestr='16-hyperbolic-l4-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=6 --wandb --namestr='16-hyperbolic-l6-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=8 --wandb --namestr='16-hyperbolic-l8-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=10 --wandb --namestr='16-hyperbolic-l10-AllTangentRealNVP'

echo "Running hyperbolic VAE with WrappedRealNVP flow for Pubmed: changing number of layers"

python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=2 --wandb --namestr='16-hyperbolic-l2-WrappedRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=4 --wandb --namestr='16-hyperbolic-l4-WrappedRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=6 --wandb --namestr='16-hyperbolic-l6-WrappedRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=8 --wandb --namestr='16-hyperbolic-l8-WrappedRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=10 --wandb --namestr='16-hyperbolic-l10-WrappedRealNVP'

echo "Running hyperbolic VAE with PTRealNVP flow for Pubmed: changing number of layers"

python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=2 --wandb --namestr='16-hyperbolic-l2-PTRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=4 --wandb --namestr='16-hyperbolic-l4-PTRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=6 --wandb --namestr='16-hyperbolic-l6-PTRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=8 --wandb --namestr='16-hyperbolic-l8-PTRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='PTRealNVP' --epochs=3000 --hidden_dim=128 --z_dim=16 --flow_hidden_size=64 --n_blocks=10 --wandb --namestr='16-hyperbolic-l10-PTRealNVP'
