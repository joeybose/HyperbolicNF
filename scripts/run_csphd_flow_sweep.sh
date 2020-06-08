#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE with RealNVP flow for csphd"

python main.py --dataset='csphd' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --z_dim=2 --wandb --namestr='euclidean-2-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=4 --wandb --namestr='euclidean-4-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --wandb --namestr='euclidean-6-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=8 --wandb --namestr='euclidean-8-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=10 --wandb --namestr='euclidean-10-RealNVP' --eval_set='validation'

echo "Running euclidean VAE with RealNVP flow for csphd: changing number of layers"
python main.py --dataset='csphd' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --namestr='euclidean-l2-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --n_blocks=4 --z_dim=2 --wandb --namestr='euclidean-l4-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --n_blocks=6 --z_dim=2 --wandb --namestr='euclidean-l6-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --n_blocks=8 --z_dim=2 --wandb --namestr='euclidean-l8-RealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --n_blocks=10 --z_dim=2 --wandb --namestr='euclidean-l10-RealNVP' --eval_set='validation'

echo "Running hyperbolic VAE with TangentRealNVP flow for csphd: changing dimension"

python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=2 --wandb --namestr='hyperbolic-d2-AllTangentRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=4 --wandb --namestr='hyperbolic-d4-AllTangentRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --wandb --namestr='hyperbolic-d6-AllTangentRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=8 --wandb --namestr='hyperbolic-d8-AllTangentRealNVP' --eval_set='validation'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=10 --wandb --namestr='hyperbolic-d10-AllTangentRealNVP' --eval_set='validation'

echo "Running hyperbolic VAE with TangentRealNVP flow for csphd: changing number of layers"

python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=2 --z_dim=16 --wandb --namestr='hyperbolic-l2-AllTangentRealNVP'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=4 --z_dim=16 --wandb --namestr='hyperbolic-l4-AllTangentRealNVP'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=6 --z_dim=16 --wandb --namestr='hyperbolic-l6-AllTangentRealNVP'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=8 --z_dim=16 --wandb --namestr='hyperbolic-l8-AllTangentRealNVP'
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=10 --z_dim=16 --wandb --namestr='hyperbolic-l10-AllTangentRealNVP'
