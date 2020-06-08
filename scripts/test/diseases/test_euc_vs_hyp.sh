#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running euclidean VAE vs. hyperbolic VAE with inner product decoder"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='diseases' --model='euclidean' --epochs=2000 --hidden_dim=200 --z_dim=6 --wandb --namestr='euclidean-2' &&
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=200 --z_dim=6 --wandb --namestr='hyperbolic-2' &&
((counter++))
done

counter=1
echo "Run hyperbolic VAE with learning curvature"
while [ $counter -le 5 ]
do
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=200 --z_dim=6 --wandb --namestr='hyperbolic-2' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --epochs=2000 --hidden_dim=200 --z_dim=6 --wandb --namestr='hyperbolic-2' --fixed_curvature=True &&
((counter++))
done


