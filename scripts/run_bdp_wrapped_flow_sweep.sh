#!/usr/bin/env bash

set -x

echo "Running hyperbolic VAE with WrappedRealNVP flow for BDP: changing dimension"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --z_dim=2 --wandb --namestr='Test-hyperbolic-d2-WrappedRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --z_dim=4 --wandb --namestr='Test-hyperbolic-d4-WrappedRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --z_dim=6 --wandb --namestr='Test-hyperbolic-d6-WrappedRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --z_dim=8 --wandb --namestr='Test-hyperbolic-d8-WrappedRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --z_dim=10 --wandb --namestr='Test-hyperbolic-d10-WrappedRealNVP'
#((counter++))
#done

echo "Running hyperbolic VAE with WrappedRealNVP flow for BDP: changing number of layers"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --n_blocks=4 --wandb --namestr='Test-hyperbolic-l4-WrappedRealNVP' &&
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --n_blocks=6 --wandb --namestr='Test-hyperbolic-l6-WrappedRealNVP' &&
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --n_blocks=8 --wandb --namestr='Test-hyperbolic-l8-WrappedRealNVP' &&
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=3000 --n_blocks=10 --wandb --namestr='Test-hyperbolic-l10-WrappedRealNVP' &&
((counter++))
done
