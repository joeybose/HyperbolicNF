#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE with RealNVP flow for BDP"

echo "Running hyperbolic VAE with TangentRealNVP flow for BDP: changing dimension"
counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=4 --wandb --namestr='hyperbolic-d4-AllTangentRealNVP' &&
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=6 --wandb --namestr='hyperbolic-d6-AllTangentRealNVP' &&
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=8 --wandb --namestr='hyperbolic-d8-AllTangentRealNVP' &&
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=10 --wandb --namestr='hyperbolic-d10-AllTangentRealNVP' &&
((counter++))
done
