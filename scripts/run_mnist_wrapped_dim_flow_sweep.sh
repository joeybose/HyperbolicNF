#!/usr/bin/env bash

set -x

echo "Running hyperbolic VAE with WrappedRealNVP flow for MNIST: changing number of layers"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=2 --hidden_dim=600 --n_blocks=2 --wandb --namestr='D2-hyperbolic-l2-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=2 --hidden_dim=600 --n_blocks=4 --wandb --namestr='D2-hyperbolic-l4-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=2 --hidden_dim=600 --n_blocks=6 --wandb --namestr='D2-hyperbolic-l6-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=2 --hidden_dim=600 --n_blocks=8 --wandb --namestr='D2-hyperbolic-l8-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=2 --hidden_dim=600 --n_blocks=10 --wandb --namestr='D2-hyperbolic-l10-WrappedRealNVP'
#((counter++))
#done

echo "Running hyperbolic VAE with WrappedRealNVP flow for MNIST: changing number of layers"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=4 --hidden_dim=600 --n_blocks=2 --wandb --namestr='D4-hyperbolic-l2-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=6 --hidden_dim=600 --n_blocks=4 --wandb --namestr='D6-hyperbolic-l4-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=6 --hidden_dim=600 --n_blocks=6 --wandb --namestr='D6-hyperbolic-l6-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=6 --hidden_dim=600 --n_blocks=8 --wandb --namestr='D6-hyperbolic-l8-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=6 --hidden_dim=600 --n_blocks=10 --wandb --namestr='D6-hyperbolic-l10-WrappedRealNVP'
((counter++))
done

echo "Running hyperbolic VAE with WrappedRealNVP flow for MNIST: changing number of layers"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=10 --hidden_dim=600 --n_blocks=2 --wandb --namestr='D10-hyperbolic-l2-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=10 --hidden_dim=600 --n_blocks=4 --wandb --namestr='D10-hyperbolic-l4-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=10 --hidden_dim=600 --n_blocks=6 --wandb --namestr='D10-hyperbolic-l6-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=10 --hidden_dim=600 --n_blocks=8 --wandb --namestr='D10-hyperbolic-l8-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=10 --hidden_dim=600 --n_blocks=10 --wandb --namestr='D10-hyperbolic-l10-WrappedRealNVP'
#((counter++))
#done

#echo "Running hyperbolic VAE with WrappedRealNVP flow for MNIST: changing number of layers"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=20 --hidden_dim=600 --n_blocks=2 --wandb --namestr='D20-hyperbolic-l2-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=20 --hidden_dim=600 --n_blocks=4 --wandb --namestr='D20-hyperbolic-l4-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=20 --hidden_dim=600 --n_blocks=6 --wandb --namestr='D20-hyperbolic-l6-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=20 --hidden_dim=600 --n_blocks=8 --wandb --namestr='D20-hyperbolic-l8-WrappedRealNVP'
#python main.py --dataset='mnist' --model='hyperbolic' --fixed_curvature=False --flow_model='WrappedRealNVP' --epochs=80 --batch_size=128 --z_dim=20 --hidden_dim=600 --n_blocks=10 --wandb --namestr='D20-hyperbolic-l10-WrappedRealNVP'
#((counter++))
#done
