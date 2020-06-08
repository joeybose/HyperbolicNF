#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running flows with tuned parameters"

counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='csphd' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=128 --flow_hidden_size=128 --z_dim=6 --wandb --namestr='euclidean-h128-RealNVP' &&
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=128 --flow_hidden_size=128 --z_dim=6 --wandb --namestr='hyperbolic-h128-AllTangentRealNVP' &&
python main.py --dataset='csphd' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --z_dim=6 --wandb --namestr='hyperbolic-l10-WrappedRealNVP' &&
((counter++))
done