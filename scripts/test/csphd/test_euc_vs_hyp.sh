#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running euclidean VAE vs. hyperbolic VAE with tuned parameters"

counter=1
while [ $counter -le 10 ]
do

python main.py --dataset='csphd' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=5 --wandb --namestr='euclidean-5'
python main.py --dataset='csphd' --model='euclidean' --epochs=2000 --hidden_dim=128 --z_dim=6 --wandb --namestr='euclidean-6'
python main.py --dataset='csphd' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=2 --wandb --namestr='hyperbolic-2'
python main.py --dataset='csphd' --model='hyperbolic' --epochs=2000 --hidden_dim=128 --z_dim=6 --wandb --namestr='hyperbolic-6'

((counter++))
done

