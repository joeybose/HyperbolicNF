#!/usr/bin/env bash

set -x

echo "Running euclidean VAE for Pubmed"
python main.py --dataset='pubmed' --model='euclidean' --epochs=3000 --hidden_dim=128 --z_dim=16 --wandb --namestr='16-euclidean'

echo "Running euclidean VAE with TangentRealNVP flow for Pubmed: changing number of layers"
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=16 --n_blocks=2 --wandb --namestr='16-euclidean-l2-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=16 --n_blocks=4  --wandb --namestr='16-euclidean-l4-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=16 --n_blocks=6 --wandb --namestr='16-euclidean-l6-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=16 --n_blocks=8 --wandb --namestr='16-euclidean-l8-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=16 --n_blocks=10 --wandb --namestr='16euclidean-l10-RealNVP'

echo "Running hyperbolic VAE for Pubmed"
python main.py --dataset='pubmed' --model='hyperbolic' --epochs=3000 --hidden_dim=128 --z_dim=16 --wandb --namestr='hyperbolic-16'
