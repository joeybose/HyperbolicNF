#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE with RealNVP flow for MNIST"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=4 --wandb --namestr='hyperbolic-d4-AllTangentRealNVP' &&
python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=6 --wandb --namestr='hyperbolic-d6-AllTangentRealNVP' &&
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=10 --wandb --namestr='hyperbolic-d10-AllTangentRealNVP' &&
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --z_dim=20 --wandb --namestr='hyperbolic-d20-AllTangentRealNVP'
((counter++))
done

echo "Running hyperbolic VAE with TangentRealNVP flow for MNIST: changing number of layers"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=4 --wandb --namestr='hyperbolic-l4-AllTangentRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=6 --wandb --namestr='hyperbolic-l6-AllTangentRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=8 --wandb --namestr='hyperbolic-l8-AllTangentRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=80 --batch_size=128 --hidden_dim=600 --n_blocks=10 --wandb --namestr='hyperbolic-l10-AllTangentRealNVP'
#((counter++))
#done
