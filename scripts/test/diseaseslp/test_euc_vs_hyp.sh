#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running euclidean VAE vs. hyperbolic VAE with distance decoder"

counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='disease_lp' --model='euclidean' --epochs=2000 --hidden_dim=16 --z_dim=2 --wandb --namestr='euclidean-6-h128' &&
python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=2000 --hidden_dim=16 --z_dim=2 --wandb --namestr='hyperbolic-6-h128' &&
((counter++))
done

echo "Run hyperbolic VAE with learning curvature"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='disease_lp' --model='hyperbolic' --epochs=2000 --hidden_dim=16 --z_dim=2 --wandb --namestr='hyperbolic-6-h128' --fixed_curvature=False &&
((counter++))
done


