#!/usr/bin/env bash

set -x

echo "Getting into the script"

counter=1
while [ $counter -le 10 ]
do
echo "Running euclidean VAE with RealNVP flow for wordnet-noun: changing hidden dimension"
python main.py --dataset='wordnet-noun' --model='euclidean' --use_rand_feats=True --flow_model='RealNVP' --epochs=500 --z_dim=2 --n_blocks=2 --wandb --namestr='D2-euclidean-l2-RealNVP'  &&
python main.py --dataset='wordnet-noun' --model='euclidean' --use_rand_feats=True --flow_model='RealNVP' --epochs=500 --z_dim=2 --n_blocks=4 --wandb --namestr='D2-euclidean-l4-RealNVP'  &&
python main.py --dataset='wordnet-noun' --model='euclidean' --use_rand_feats=True --flow_model='RealNVP' --epochs=500 --z_dim=2 --n_blocks=6 --wandb --namestr='D2-euclidean-l6-RealNVP'  &&
python main.py --dataset='wordnet-noun' --model='euclidean' --use_rand_feats=True --flow_model='RealNVP' --epochs=500 --z_dim=2 --n_blocks=8 --wandb --namestr='D2-euclidean-l8-RealNVP'  &&
python main.py --dataset='wordnet-noun' --model='euclidean' --use_rand_feats=True --flow_model='RealNVP' --epochs=500 --z_dim=2 --n_blocks=10 --wandb --namestr='D2-euclidean-l10-RealNVP'  &&
((counter++))
done

#counter=1
#while [ $counter -le 10 ]
#do
#echo "Running hyperbolic VAE with TangentRealNVP flow for wordnet-noun: changing hidden dimension"
#python main.py --dataset='wordnet-noun'  --model='hyperbolic' --use_rand_feats=True --flow_model='AllTangentRealNVP' --epochs=500 --fixed_curvature=False --z_dim=2 --n_blocks=2 --wandb --namestr='D2-hyperbolic-l2-AllTangentRealNVP'  &&
#python main.py --dataset='wordnet-noun'  --model='hyperbolic' --use_rand_feats=True --flow_model='AllTangentRealNVP' --epochs=500 --fixed_curvature=False --z_dim=2 --n_blocks=4 --wandb --namestr='D2-hyperbolic-l4-AllTangentRealNVP'  &&
#python main.py --dataset='wordnet-noun'  --model='hyperbolic' --use_rand_feats=True --flow_model='AllTangentRealNVP' --epochs=500 --fixed_curvature=False --z_dim=2 --n_blocks=6 --wandb --namestr='D2-hyperbolic-l6-AllTangentRealNVP'  &&
#python main.py --dataset='wordnet-noun'  --model='hyperbolic' --use_rand_feats=True --flow_model='AllTangentRealNVP' --epochs=500 --fixed_curvature=False --z_dim=2 --n_blocks=8 --wandb --namestr='D2-hyperbolic-l8-AllTangentRealNVP'  &&
#python main.py --dataset='wordnet-noun'  --model='hyperbolic' --use_rand_feats=True --flow_model='AllTangentRealNVP' --epochs=500 --fixed_curvature=False --z_dim=2 --n_blocks=10 --wandb --namestr='D2-hyperbolic-l10-AllTangentRealNVP'  &&
#((counter++))
#done
