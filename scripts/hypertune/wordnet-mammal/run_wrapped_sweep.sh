#!/usr/bin/env bash

set -x

echo "Getting into the test script"

#echo "Running wrapped for final param tuning"
#counter=1
#while [ $counter -le 10 ]
#do
#python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --z_dim=6 --wandb --namestr='D6-hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
#python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --flow_hidden_size=128 --z_dim=8 --wandb --namestr='D8-hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
#python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --z_dim=6 --wandb --namestr='D6-hyperbolic-l10-WrappedRealNVP' --eval_set='validation' &&
#((counter++))
#done

echo "Running wrapped for learning curvature"
counter=1
while [ $counter -le 10 ]
do
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --z_dim=6 --wandb --fixed_curvature=False --namestr='D6-hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --flow_hidden_size=128 --z_dim=8 --wandb --fixed_curvature=False --namestr='D8-hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
python main.py --dataset='wordnet-mammal' --decoder='tanh' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --z_dim=6 --wandb --fixed_curvature=False --namestr='D6-hyperbolic-l10-WrappedRealNVP' --eval_set='validation' &&
((counter++))
done

# echo "Running wrapped with different decoders"
# counter=1
# while [ $counter -le 10 ]
# do
# python main.py --dataset='wordnet-mammal' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --z_dim=6 --wandb --decoder='fermi' --namestr='hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
# python main.py --dataset='wordnet-mammal' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --z_dim=6 --wandb --decoder='tanh' --namestr='hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
# python main.py --dataset='wordnet-mammal' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --flow_hidden_size=128 --z_dim=8 --wandb --decoder='fermi' --namestr='hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
# python main.py --dataset='wordnet-mammal' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=2 --hidden_dim=200 --flow_hidden_size=128 --z_dim=8 --wandb --decoder='tanh' --namestr='hyperbolic-l2-WrappedRealNVP' --eval_set='validation' &&
# python main.py --dataset='wordnet-mammal' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --z_dim=6 --wandb --decoder='fermi' --namestr='hyperbolic-l10-WrappedRealNVP' --eval_set='validation' &&
# python main.py --dataset='wordnet-mammal' --model='hyperbolic' --flow_model='WrappedRealNVP' --epochs=2000 --n_blocks=10 --hidden_dim=200 --z_dim=6 --wandb --decoder='tanh' --namestr='hyperbolic-l10-WrappedRealNVP' --eval_set='validation' &&
# ((counter++))
# done
