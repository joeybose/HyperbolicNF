#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running flows with z_dim=2, decoder=distance, fixed_curvature=False"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='diseases' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='euclidean-RealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=4 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-h128-AllTangentRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=6 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=8 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=12 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=14 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=16 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=False &&
((counter++))
done

echo "Running flows with z_dim=2, decoder=distance, fixed_curvature=True"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='diseases' --model='euclidean' --flow_model='RealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='euclidean-RealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='AllTangentRealNVP' --epochs=2000 --n_blocks=4 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-h128-AllTangentRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=6 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=8 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=12 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=14 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
python main.py --dataset='diseases' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=16 --hidden_dim=200 --flow_hidden_size=128 --z_dim=2 --wandb --namestr='hyperbolic-l10-WrappedRealNVP-dd' --decoder='distance' --fixed_curvature=True &&
((counter++))
done