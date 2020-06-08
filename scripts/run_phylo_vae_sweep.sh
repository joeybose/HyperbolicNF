#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE for phylo"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='phylo' --model='euclidean' --epochs=2000  --z_dim=2  --wandb --namestr='euclidean-2' &&
python main.py --dataset='phylo' --model='euclidean' --epochs=2000  --z_dim=4  --wandb --namestr='euclidean-4' &&
python main.py --dataset='phylo' --model='euclidean' --epochs=2000  --z_dim=6  --wandb --namestr='euclidean-6' &&
python main.py --dataset='phylo' --model='euclidean' --epochs=2000  --z_dim=8  --wandb --namestr='euclidean-8' &&
python main.py --dataset='phylo' --model='euclidean' --epochs=2000  --z_dim=10  --wandb --namestr='euclidean-10' &&
((counter++))
done

#echo "Running hyperbolic VAE for phylo"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --fixed_curvature=False --epochs=2000  --z_dim=2  --wandb --namestr='hyperbolic-2' &&
#python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --fixed_curvature=False --epochs=2000  --z_dim=4  --wandb --namestr='hyperbolic-4' &&
#python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --fixed_curvature=False --epochs=2000  --z_dim=6  --wandb --namestr='hyperbolic-6' &&
#python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --fixed_curvature=False --epochs=2000  --z_dim=8  --wandb --namestr='hyperbolic-8' &&
#python main.py --dataset='phylo' --model='hyperbolic' --decoder='tanh' --fixed_curvature=False --epochs=2000  --z_dim=10  --wandb --namestr='hyperbolic-10' &&
#((counter++))
#done
