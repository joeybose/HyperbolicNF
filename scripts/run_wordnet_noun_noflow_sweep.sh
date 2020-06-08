#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for wordnet-noun"

counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='wordnet-noun' --model='euclidean' --epochs=500 --use_rand_feats=True --z_dim=2 --wandb --namestr='euclidean-2' &&
((counter++))
done

echo "Running hyperbolic VAE for wordnet-noun"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='wordnet-noun' --model='hyperbolic' --epochs=500 --z_dim=2 --use_rand_feats=True --fixed_curvature=False --wandb --namestr='hyperbolic-2' &&
((counter++))
done
