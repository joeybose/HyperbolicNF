#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for MNIST"

counter=1
while [ $counter -le 3 ]
do
#python main.py --dataset='mnist' --model='euclidean' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=2 --wandb --namestr='euclidean-2' &&
python main.py --dataset='mnist' --model='euclidean' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=4 --wandb --namestr='euclidean-4' &&
#python main.py --dataset='mnist' --model='euclidean' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=10 --wandb --namestr='euclidean-10' &&
#python main.py --dataset='mnist' --model='euclidean' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=20 --wandb --namestr='euclidean-20' &&
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=2 --wandb --namestr='hyperbolic-2'
python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=4 --wandb --namestr='hyperbolic-4'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=10 --wandb --namestr='hyperbolic-10'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=20 --wandb --namestr='hyperbolic-20'
((counter++))
done

