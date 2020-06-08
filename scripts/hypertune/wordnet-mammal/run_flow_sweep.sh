#!/usr/bin/env bash

set -x

echo "Getting into the script"

echo "Running euclidean VAE with RealNVP flow for wordnet-mammal: changing hidden dimension"
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=16 --wandb --namestr='D6-euclidean-h16-RealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --wandb --namestr='D6-euclidean-h32-RealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=64 --wandb --namestr='D6-euclidean-h64-RealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=128 --wandb --namestr='D6-euclidean-h128-RealNVP' --eval_set='validation' &&

echo "Running euclidean VAE with RealNVP flow for wordnet-mammal: changing flow hidden size"
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean'  --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=8 --wandb --namestr='D6-euclidean-fh8-RealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=16 --wandb --namestr='D6-euclidean-fh16-RealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=32 --wandb --namestr='D6-euclidean-fh32-RealNVP' --eval_set='validation' &&

echo "Running hyperbolic VAE with TangentRealNVP flow for wordnet-mammal: changing hidden dimension"
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic'  --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=16 --wandb --namestr='D6-hyperbolic-h16-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --wandb --namestr='D6-hyperbolic-h32-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=64 --wandb --namestr='D6-hyperbolic-h64-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=128 --wandb --namestr='D6-hyperbolic-h128-AllTangentRealNVP' --eval_set='validation' &&

echo "Running hyperbolic VAE with TangentRealNVP flow for wordnet-mammal: changing flow hidden size"
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic'  --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=8 --wandb --namestr='D6-hyperbolic-fh8-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=16 --wandb --namestr='D6-hyperbolic-fh16-AllTangentRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --z_dim=6 --hidden_dim=32 --flow_hidden_size=32 --wandb --namestr='D6-hyperbolic-fh32-AllTangentRealNVP' --eval_set='validation' &&
