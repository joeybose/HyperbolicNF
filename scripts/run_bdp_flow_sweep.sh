#!/usr/bin/env bash

set -x

echo "Getting into the script"
echo "Running euclidean VAE with RealNVP flow for BDP"

#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='bdp' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --z_dim=4 --wandb --namestr='euclidean-4-RealNVP'
#python main.py --dataset='bdp' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --z_dim=6 --wandb --namestr='euclidean-6-RealNVP'
#python main.py --dataset='bdp' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --z_dim=8 --wandb --namestr='euclidean-8-RealNVP'
#python main.py --dataset='bdp' --model='euclidean' --flow_model='RealNVP' --epochs=3000 --z_dim=10 --wandb --namestr='euclidean-10-RealNVP'
#((counter++))
#done

#counter=1
#while [ $counter -le 5 ]
#do
#echo "Running hyperbolic VAE with TangentRealNVP flow for BDP: changing number of layers"
#python main.py --dataset='bdp' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --n_blocks=2 --z_dim=2 --wandb --namestr='euclidean-l2-RealNVP' &&
#python main.py --dataset='bdp' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --n_blocks=4 --z_dim=2 --wandb --namestr='euclidean-l4-RealNVP' &&
#python main.py --dataset='bdp' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --n_blocks=6 --z_dim=2 --wandb --namestr='euclidean-l6-RealNVP' &&
#python main.py --dataset='bdp' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --n_blocks=8 --z_dim=2 --wandb --namestr='euclidean-l8-RealNVP' &&
#python main.py --dataset='bdp' --model='euclidean'  --flow_model='RealNVP' --epochs=3000 --n_blocks=10 --z_dim=2 --wandb --namestr='euclidean-l10-RealNVP' &&
#((counter++))
#done

#echo "Running hyperbolic VAE with TangentRealNVP flow for BDP: changing dimension"
#counter=1
#while [ $counter -le 5 ]
#do
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=4 --wandb --namestr='hyperbolic-d4-AllTangentRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=6 --wandb --namestr='hyperbolic-d6-AllTangentRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=8 --wandb --namestr='hyperbolic-d8-AllTangentRealNVP'
#python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --z_dim=10 --wandb --namestr='hyperbolic-d10-AllTangentRealNVP'
#((counter++))
#done

echo "Running hyperbolic VAE with TangentRealNVP flow for BDP: changing number of layers"
counter=1
while [ $counter -le 5 ]
do
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --n_blocks=2 --wandb --namestr='hyperbolic-l2-AllTangentRealNVP'
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --n_blocks=4 --wandb --namestr='hyperbolic-l4-AllTangentRealNVP'
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --n_blocks=6 --wandb --namestr='hyperbolic-l6-AllTangentRealNVP'
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --n_blocks=8 --wandb --namestr='hyperbolic-l8-AllTangentRealNVP'
python main.py --dataset='bdp' --model='hyperbolic' --fixed_curvature=False --flow_model='AllTangentRealNVP' --epochs=3000 --n_blocks=10 --wandb --namestr='hyperbolic-l10-AllTangentRealNVP'
((counter++))
done

