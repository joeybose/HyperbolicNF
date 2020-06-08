# Normalizing Flows for Hyperbolic Spaces and Beyond!

## Installation
Python Packages:
- Pytorch Geometric: https://github.com/rusty1s/pytorch_geometric
Follow the installation instructions carefully for this package! Make sure all
your environment Path variables are exactly as outlined otherwise you will get
weird symbol errors
- Pytorch
- WandB for logging

Download the datasets:

`python -m data.download`

## Running Hyperbolic VAE
`python main.py --dataset=mnist --batch_size=100 --epochs=100 --model=hyperbolic --wandb --namestr="MNIST 2-HyperbolicVAE"`

## Running Euclidean Flow
`python main.py --dataset=mnist --batch_size=100 --epochs=100 --model=euclidean --flow_model=RealNVP --wandb --namestr="MNIST 2-Hyperbolic 2-RealNVP"`

## Running Flow Hyperbolic VAE
`python main.py --dataset=mnist --batch_size=100 --epochs=100 --model=hyperbolic --flow_model=TangentRealNVP --n_blocks=4 --wandb --namestr="MNIST 2-Hyperbolic 4-TangentRealNVP"`

## Reference code repos
1. "A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based
   Learning": https://github.com/pfnet-research/hyperbolic_wrapped_distribution
2. "Mixed-Curvature Variational Autoencoder":
   https://www.dropbox.com/s/tzilf229js1gsqu/mvae.zip?dl=0
3. "Hyperbolic Neural Networks": https://github.com/dalab/hyperbolic_nn
4. "Hyperbolic Graph Convolutional Neural Networks": https://github.com/HazyResearch/hgcn
