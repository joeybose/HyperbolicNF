#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE with RealNVP flow for Pubmed"

python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --wandb --namestr='euclidean-2-RealNVP'
python main.py --dataset='pubmed' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=4 --wandb --namestr='euclidean-4-RealNVP'
python main.py --dataset='pubmed' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=6 --wandb --namestr='euclidean-6-RealNVP'
python main.py --dataset='pubmed' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=8 --wandb --namestr='euclidean-8-RealNVP'
python main.py --dataset='pubmed' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=10 --wandb --namestr='euclidean-10-RealNVP'

echo "Running euclidean VAE with TangentRealNVP flow for Pubmed: changing number of layers"
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --n_blocks=2 --wandb --namestr='euclidean-l2-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --n_blocks=4  --wandb --namestr='euclidean-l4-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --n_blocks=6 --wandb --namestr='euclidean-l6-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --n_blocks=8 --wandb --namestr='euclidean-l8-RealNVP'
python main.py --dataset='pubmed' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --n_blocks=10 --wandb --namestr='euclidean-l10-RealNVP'
echo "Running hyperbolic VAE with TangentRealNVP flow for Pubmed: changing dimension"

python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=2 --wandb --namestr='hyperbolic-d2-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=4 --wandb --namestr='hyperbolic-d4-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=6 --wandb --namestr='hyperbolic-d6-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=8 --wandb --namestr='hyperbolic-d8-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --z_dim=10 --wandb --namestr='hyperbolic-d10-AllTangentRealNVP'

echo "Running hyperbolic VAE with TangentRealNVP flow for Pubmed: changing number of layers"

python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --n_blocks=2 --wandb --namestr='hyperbolic-l2-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --n_blocks=4 --wandb --namestr='hyperbolic-l4-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --n_blocks=6 --wandb --namestr='hyperbolic-l6-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --n_blocks=8 --wandb --namestr='hyperbolic-l8-AllTangentRealNVP'
python main.py --dataset='pubmed' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=3000 --hidden_dim=128 --flow_hidden_size=64 --n_blocks=10 --wandb --namestr='hyperbolic-l10-AllTangentRealNVP'
