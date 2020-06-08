#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running hyperbolic VAE with WrappedRealNVP flow for phylo: changing dimension"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --z_dim=2 --wandb --namestr='hyperbolic-d2-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --z_dim=4 --wandb --namestr='hyperbolic-d4-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --z_dim=6 --wandb --namestr='hyperbolic-d6-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --z_dim=8 --wandb --namestr='hyperbolic-d8-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --z_dim=10 --wandb --namestr='hyperbolic-d10-WrappedRealNVP'
((counter++))
done

echo "Running hyperbolic VAE with WrappedRealNVP flow for phylo: changing number of layers"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --wandb --namestr='hyperbolic-l2-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=4 --wandb --namestr='hyperbolic-l4-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=6 --wandb --namestr='hyperbolic-l6-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=8 --wandb --namestr='hyperbolic-l8-WrappedRealNVP'
python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --wandb --namestr='hyperbolic-l10-WrappedRealNVP'
((counter++))
done
