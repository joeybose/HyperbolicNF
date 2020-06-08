#!/usr/bin/env bash

set -x

echo "Getting into the script"

echo "Running euclidean VAE with RealNVP flow for phylo: changing hidden dimension"
python main.py --dataset='phylo' --decoder='distance' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=16 --wandb --namestr='euclidean-h16-RealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --wandb --namestr='euclidean-h32-RealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=64 --wandb --namestr='euclidean-h64-RealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=128 --wandb --namestr='euclidean-h128-RealNVP' --eval_set='validation' &&

echo "Running euclidean VAE with RealNVP flow for phylo: changing flow hidden size"
python main.py --dataset='phylo' --decoder='distance' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=8 --wandb --namestr='euclidean-fh8-RealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=16 --wandb --namestr='euclidean-fh16-RealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=32 --wandb --namestr='euclidean-fh32-RealNVP' --eval_set='validation' &&

echo "Running hyperbolic VAE with TangentRealNVP flow for phylo: changing hidden dimension"
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic'  --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=16 --wandb --namestr='hyperbolic-h16-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --wandb --namestr='hyperbolic-h32-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=64 --wandb --namestr='hyperbolic-h64-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=128 --wandb --namestr='hyperbolic-h128-AllTangentRealNVP' --eval_set='validation' &&

echo "Running hyperbolic VAE with TangentRealNVP flow for phylo: changing flow hidden size"
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic'  --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=8 --wandb --namestr='hyperbolic-fh8-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=16 --wandb --namestr='hyperbolic-fh16-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='phylo' --decoder='distance' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=32 --wandb --namestr='hyperbolic-fh32-AllTangentRealNVP' --eval_set='validation' &&