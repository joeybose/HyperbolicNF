#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE with RealNVP flow for MNIST"

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='lobster' --model='euclidean' --z_dim=2 --wandb --lr=1e-3 --deterministic=True --decoder='distance' --epochs=100 --batch_size=20 --namestr="100-euclidean-2-det" --n_blocks=2 --conv_type='GAT' --load_gae=False --use_rand_feats=False --hidden_dim=32
python main.py --dataset='lobster' --model='hyperbolic' --fixed_curvature=True --z_dim=2 --wandb --lr=1e-3 --deterministic=True --decoder='distance' --epochs=100 --batch_size=20 --namestr="100-hyperbolic-2-det" --n_blocks=2 --conv_type='GAT' --load_gae=False --use_rand_feats=False --hidden_dim=32
python main.py --dataset='lobster' --model='euclidean' --z_dim=2 --wandb --lr=1e-3 --deterministic=True --decoder='distance' --epochs=100 --batch_size=20 --namestr="100-euclidean-2-det-RealNVP" --n_blocks=2 --conv_type='GAT' --flow_layer_type='GAT' --load_gae=False --use_rand_feats=False --flow_model='RealNVP' --hidden_dim=32 --flow_hidden_size=32 --flow_epochs=100
python main.py --dataset='lobster' --model='hyperbolic' --fixed_curvature=True --z_dim=2 --wandb --lr=1e-3 --deterministic=True --epochs=100 --batch_size=20 --namestr="100-hyperbolic-2-det-AllTangent" --n_blocks=2 --conv_type='GAT' --flow_layer_type='GAT' --load_gae=False --use_rand_feats=False --flow_model='AllTangentRealNVP' --hidden_dim=32 --flow_hidden_size=32 --flow_epochs=100 --decoder='distance'
python main.py --dataset='lobster' --model='hyperbolic' --fixed_curvature=True --z_dim=2 --wandb --lr=1e-3 --deterministic=True --epochs=100 --batch_size=20 --namestr="100-hyperbolic-2-det-Wrapped" --n_blocks=2 --conv_type='GAT' --flow_layer_type='GAT' --load_gae=False --use_rand_feats=False --flow_model='WrappedRealNVP' --hidden_dim=32 --flow_hidden_size=32 --flow_epochs=100 --decoder='distance'
((counter++))
done

counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='lobster' --model='euclidean' --z_dim=2 --wandb --lr=1e-3 --deterministic=True --decoder='distance' --epochs=200 --batch_size=20 --namestr="euclidean-2" --n_blocks=2 --conv_type='GAT' --load_gae=False --use_rand_feats=False --hidden_dim=32
python main.py --dataset='lobster' --model='hyperbolic' --fixed_curvature=True --z_dim=2 --wandb --lr=1e-3 --deterministic=True --decoder='distance' --epochs=200 --batch_size=20 --namestr="hyperbolic-2" --n_blocks=2 --conv_type='GAT' --load_gae=False --use_rand_feats=False --hidden_dim=32
python main.py --dataset='lobster' --model='euclidean' --z_dim=2 --wandb --lr=1e-3 --deterministic=True --decoder='distance' --epochs=200 --batch_size=20 --namestr="euclidean-2-RealNVP" --n_blocks=2 --conv_type='GAT' --flow_layer_type='GAT' --load_gae=False --use_rand_feats=False --flow_model='RealNVP' --hidden_dim=32 --flow_hidden_size=32 --flow_epochs=100
python main.py --dataset='lobster' --model='hyperbolic' --fixed_curvature=True --z_dim=2 --wandb --lr=1e-3 --deterministic=True --epochs=200 --batch_size=20 --namestr="hyperbolic-2-AllTangent" --n_blocks=2 --conv_type='GAT' --flow_layer_type='GAT' --load_gae=False --use_rand_feats=False --flow_model='AllTangentRealNVP' --hidden_dim=32 --flow_hidden_size=32 --flow_epochs=100 --decoder='distance'
python main.py --dataset='lobster' --model='hyperbolic' --fixed_curvature=True --z_dim=2 --wandb --lr=1e-3 --deterministic=True --epochs=200 --batch_size=20 --namestr="hyperbolic-2-Wrapped" --n_blocks=2 --conv_type='GAT' --flow_layer_type='GAT' --load_gae=False --use_rand_feats=False --flow_model='WrappedRealNVP' --hidden_dim=32 --flow_hidden_size=32 --flow_epochs=100 --decoder='distance'
((counter++))
done
