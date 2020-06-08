#!/usr/bin/env bash

set -x

echo "Getting into the test script"

echo "Running wrapped for learning curvature"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='wordnet-noun' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=500 --use_rand_feats=True --fixed_curvature=False --n_blocks=2  --z_dim=2 --wandb --fixed_curvature=False --namestr='D2-hyperbolic-l2-WrappedRealNVP' &&
python main.py --dataset='wordnet-noun' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=500 --use_rand_feats=True --fixed_curvature=False --n_blocks=4  --z_dim=2 --wandb --fixed_curvature=False --namestr='D2-hyperbolic-l4-WrappedRealNVP' &&
python main.py --dataset='wordnet-noun' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=500 --use_rand_feats=True --fixed_curvature=False --n_blocks=6  --z_dim=2 --wandb --fixed_curvature=False --namestr='D2-hyperbolic-l6-WrappedRealNVP' &&
python main.py --dataset='wordnet-noun' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=500 --use_rand_feats=True --fixed_curvature=False --n_blocks=8  --z_dim=2 --wandb --fixed_curvature=False --namestr='D2-hyperbolic-l8-WrappedRealNVP' &&
python main.py --dataset='wordnet-noun' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=500 --use_rand_feats=True --fixed_curvature=False --n_blocks=10  --z_dim=2 --wandb --fixed_curvature=False --namestr='D2-hyperbolic-l10-WrappedRealNVP' &&
((counter++))
done

