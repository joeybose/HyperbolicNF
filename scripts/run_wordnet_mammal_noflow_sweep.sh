#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for wordnet-mammal"

counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='wordnet-mammal' --model='euclidean' --decoder='tanh' --epochs=2000 --z_dim=6 --wandb --namestr='euclidean-6' &&
((counter++))
done

echo "Running hyperbolic VAE for wordnet-mammal"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='wordnet-mammal' --model='hyperbolic' --decoder='tanh' --epochs=2000 --z_dim=6 --fixed_curvature=False --wandb --namestr='hyperbolic-6' &&
((counter++))
done
