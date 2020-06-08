#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running flows with z_dim=2, decoder=inner, fh=128, h=200"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='disease_lp' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --flow_hidden_size=128 --hidden_dim=200 --namestr='euclidean-l2-RealNVP-fh128' --decoder='distance' &&
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --flow_hidden_size=128 --hidden_dim=200 --namestr='hyperbolic-AllTangentRealNVP-fh128' --decoder='distance' &&
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --flow_hidden_size=128 --hidden_dim=200 --namestr='hyperbolic-l2-WrappedRealNVP-fh128' --decoder='distance' &&
((counter++))
done

echo "Running flows with z_dim=2, decoder=inner, fh=128, h=128"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='disease_lp' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --flow_hidden_size=128 --hidden_dim=128 --namestr='euclidean-RealNVP-h128' --decoder='distance' &&
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --flow_hidden_size=128 --hidden_dim=128 --namestr='hyperbolic-AllTangentRealNVP-h128' --decoder='distance' &&
python main.py --dataset='disease_lp' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --z_dim=2 --wandb --flow_hidden_size=128 --hidden_dim=128 --namestr='hyperbolic-l2-WrappedRealNVP-h128' --decoder='distance' &&
((counter++))
done