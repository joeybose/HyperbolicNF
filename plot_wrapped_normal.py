"""29', zorder=10)29', zorder=10)29', zorder=10)29', zorder=10)29', zorder=10)
Plot a grid on H2
with Poincare Disk visualization.
"""

import argparse
import matplotlib.pyplot as plt
from utils.hyperbolics import expmap_logdet, _logdet, logmap_logdet
from matplotlib import cm
import numpy as np
import os
import ipdb

import torch
import seaborn as sns
from scipy.stats import kde
import numpy as np
import sys
import math

from torch import optim
from torch.utils import data

from flows.flows import TangentRealNVP, WrappedRealNVP, AllTangentRealNVP, MAFRealNVP, RealNVP

sys.path.append("..")  # Adds higher directory to python modules path.
from utils.math_ops import clamp, acosh, cosh, sinh, sqrt, logsinh, e_i, expand_proj_dims
from utils.hyperbolics import exp_map, exp_map_mu0, inverse_exp_map_mu0, logmap_logdet, lorentz_to_poincare
from distributions.wrapped_normal import HyperboloidWrappedNormal

max_clamp_norm = 40

kwargs_flows = {'AllTangentRealNVP': AllTangentRealNVP, 'WrappedRealNVP': WrappedRealNVP}


class FlowDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x):
        'Initialization'
        self.x = x

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.x[index].squeeze()

def sample_2d_data(dataset, n_samples):

    z = torch.randn(n_samples, 2)
    z = z*0.25

    if dataset == '8gaussians':
        # z = z*0.25
        scale = 1
        sq2 = 1/math.sqrt(2)
        # centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = [(1,0), (-1,0), (0,1), (0,-1)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 20
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        # x1 = torch.rand(n_samples) - 1
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) / 4

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4) * 0.5
        circ4_y = torch.sin(linspace4) * 0.5
        circ3_x = torch.cos(linspace4) * 0.325
        circ3_y = torch.sin(linspace3) * 0.325
        circ2_x = torch.cos(linspace2) * 0.25
        circ2_y = torch.sin(linspace2) * 0.25
        circ1_x = torch.cos(linspace1) * 0.125
        circ1_y = torch.sin(linspace1) * 0.125

        # x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         # torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0
        x = torch.stack([torch.cat([2*circ4_x, 2*circ3_x, 2*circ2_x, 2*circ1_x]),
                         torch.cat([2*circ4_y, 2*circ3_y, 2*circ2_y, 2*circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return (x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x)))/2

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')


def plot_density(xy_poincare, probs, radius, namestr, mu=None, flow=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = xy_poincare[:, 0].view(-1, 100).detach().cpu()
    y = xy_poincare[:, 1].view(-1, 100).detach().cpu()
    z = probs.view(-1, 100).detach().cpu()
    range_lim = 2
    # Define points within circle
    if mu is not None:
        mu = mu.cpu().numpy()
        plt.plot(mu[:, 0], mu[:, 1], 'b+')
    ax.contourf(x, y, z, 100, antialiased=False, cmap='magma')
    # ax.pcolormesh(x, y, z,)
    # ax.hist2d(x.numpy().flatten(), y.numpy().flatten(), range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=500, cmap='magma')
    # ax.contourf(x, y, z, 100, antialiased=False, cmap=plt.cm.jet)

    # boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
    # ax.add_artist(boundary)
    ax.axis('off')
    # Makes the circle look like a circle
    ax.axis('equal')
    ax.set_xlim(-args.axis_lim, args.axis_lim)
    ax.set_ylim(-args.axis_lim, args.axis_lim)
    if flow is not None:
        fig.savefig('install/{}_{}.png'.format(namestr, flow))
        print("saved to install/{}_{}.png".format(namestr, flow))
    else:
        # Save the full figure...
        fig.savefig('install/{}.png'.format(namestr))
        print("saved to install/{}.png".format(namestr))

def compute_kl_qp_loss(model, target_potential_fn, batch_size):
    """ Compute BNAF eq 3 & 20:
    KL(q_inv||p) where q_inv is the inverse flow transform (log_q_inv = log_q_base - logdet), p is the target distribution (energy potential)
    Returns the minimization objective for density matching. """
    z = model.base_dist.sample((batch_size,))
    q_log_prob = model.base_dist.log_prob(z)
    zk, logdet = model(z)
    p_log_prob = - target_potential_fn(zk)  # p = exp(-potential) => log_p = - potential
    return q_log_prob.sum(1) - logdet.sum(1) - p_log_prob  # BNAF eq 20

def train_potential_flow(flow_model, n_blocks, radius, target):
    flow_model = kwargs_flows[flow_model](n_blocks, 2, 128, 1, layer_type='Linear',
                                          radius=torch.tensor(radius)).cuda()
    flow_opt = optim.Adam(flow_model.parameters(), lr=1e-2)

    sample_shape = torch.Size([10000])
    num_samples = torch.Size([256])
    mu_0_shape = torch.Size([1, 3])
    std_0_shape = torch.Size([1, 2])
    prior = HyperboloidWrappedNormal(radius, torch.zeros(mu_0_shape).cuda(),
				     torch.ones(std_0_shape).cuda())
    train_loss_avg = []
    for epoch in range(0, 1000):
        flow_opt.zero_grad()
        z_0 = prior.rsample(num_samples).squeeze()
        z_0 = clamp(z_0, min=-max_clamp_norm, max=max_clamp_norm)
        q_log_prob = prior.log_prob(z_0)
        z_hyper, logdet = flow_model.inverse(z_0)
        z_hyper = clamp(z_hyper, min=-max_clamp_norm, max=max_clamp_norm)
        z_k = inverse_exp_map_mu0(z_hyper, radius)
        z_mu0 = z_k[..., 1:]
        logdet += logmap_logdet(z_k, radius)
        p_log_prob = -1*target(z_mu0)
        loss = (q_log_prob - p_log_prob - logdet).mean()
        loss.backward()
        flow_opt.step()
        print("Epoch:{} Loss:{}".format(epoch, loss.item()))

    return flow_model

def train_flow(args, flow_model, radius, target, clamped_threedim, on_mani):
    flow_model = kwargs_flows[flow_model](4, 2, 32, 1, layer_type='Linear',
                                          radius=torch.tensor(radius)).cuda()
    flow_opt = optim.Adam(flow_model.parameters())

    sample_shape = torch.Size([10000])
    z, posterior_parts = target.rsample_with_parts(sample_shape)
    z = clamp(z, min=-max_clamp_norm, max=max_clamp_norm)

    train_dataset = FlowDataset(z)
    train_loader = data.DataLoader(train_dataset, batch_size=512)

    train_loss_avg = []
    for epoch in range(0, args.flow_epochs):
        train_loss_avg.append(0)
        for batch_idx, data_batch in enumerate(train_loader):

            data_batch = data_batch.cuda()
            flow_model.base_dist_mean = torch.zeros_like(data_batch).cuda()
            flow_model.base_dist_var = torch.ones(data_batch.shape[0], 2).cuda()
            flow_opt.zero_grad()

            loss = -1 * flow_model.log_prob(data_batch).mean()
            loss.backward()
            flow_opt.step()
            train_loss_avg[-1] += loss.item()

        train_loss_avg[-1] /= len(train_loader.dataset)
        print("Loss:{}".format(train_loss_avg[-1]))
        print("Epoch:{}".format(epoch))

        # Calculate densities of x, y coords on Lorentz model.
        flow_model.base_dist_mean = torch.zeros_like(on_mani).cuda()
        flow_model.base_dist_var = torch.ones(on_mani.shape[0], 2).cuda()
        probs = flow_model.log_prob(on_mani)
        probs += logmap_logdet(clamped_threedim.cuda(), radius)
        probs = torch.exp(probs)

        on_mani_conv = on_mani.detach().cpu()

        # Calculate the poincare coordinates
        xy_poincare = lorentz_to_poincare(on_mani.squeeze(), radius)
        namestr = args.namestr + str(epoch)
        plot_density(xy_poincare, probs, flow_model.radius, namestr)

    return flow_model

def train_flow_density(args, flow_model, n_blocks, radius, samples,
        clamped_threedim, on_mani):
    flow_model = kwargs_flows[flow_model](n_blocks, 2, 256, 1, layer_type='Linear',
                                          radius=torch.tensor(radius)).cuda()
    flow_opt = optim.Adam(flow_model.parameters())
    samples = clamp(samples, min=-max_clamp_norm, max=max_clamp_norm)

    train_dataset = FlowDataset(samples)
    train_loader = data.DataLoader(train_dataset, batch_size=1024)

    train_loss_avg = []
    for epoch in range(0, args.flow_epochs):
        train_loss_avg.append(0)
        for batch_idx, data_batch in enumerate(train_loader):

            data_batch = data_batch.cuda()
            flow_model.base_dist_mean = torch.zeros_like(data_batch).cuda()
            flow_model.base_dist_var = torch.ones(data_batch.shape[0], 2).cuda()
            flow_opt.zero_grad()

            loss = -1 * flow_model.log_prob(data_batch).mean()
            loss.backward()
            flow_opt.step()
            train_loss_avg[-1] += loss.item()

        train_loss_avg[-1] /= len(train_loader.dataset)
        print("Loss:{}".format(train_loss_avg[-1]))
        print("Epoch:{}".format(epoch))

        # Calculate densities of x, y coords on Lorentz model.
        flow_model.base_dist_mean = torch.zeros_like(on_mani).cuda()
        flow_model.base_dist_var = torch.ones(on_mani.shape[0], 2).cuda()
        probs = flow_model.log_prob(on_mani)
        probs += logmap_logdet(clamped_threedim.cuda(), radius)
        probs = torch.exp(probs)

        on_mani_conv = on_mani.detach().cpu()

        # Calculate the poincare coordinates
        xy_poincare = lorentz_to_poincare(on_mani.squeeze(), radius)
        namestr = args.namestr + str(epoch)
        plot_density(xy_poincare, probs, flow_model.radius, namestr)

    return flow_model

def plot_flow(args, radius, flow, target, namestr, n_blocks=2, samples=None):
    fig = plt.figure()
    ax = fig.add_subplot(555)
    # Map x, y coordinates on tangent space at origin to manifold (Lorentz model).
    x = torch.linspace(-5, 5, 100)
    xx, yy = torch.meshgrid((x, x))
    # x = np.arange(-5, 5, 0.1)
    # y = np.arange(-5, 5, 0.1)
    # x, y = np.meshgrid(x, y)
    # x = torch.Tensor(x).view(-1, 1)
    # y = torch.Tensor(y).view(-1, 1)
    twodim = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    # twodim = torch.cat([x, y], dim=1)
    threedim = expand_proj_dims(twodim)
    clamped_threedim = clamp(threedim, min=-max_clamp_norm,
            max=max_clamp_norm).to(args.dev)
    on_mani = exp_map_mu0(clamped_threedim, radius).cuda()
    # flow_model = train_potential_flow(flow, radius, target)
    if samples is not None:
        flow_model = train_flow_density(args, flow, n_blocks, radius, samples,
                clamped_threedim, on_mani)
    else:
        flow_model = train_flow(args, flow, radius, target, clamped_threedim, on_mani)


    # Calculate densities of x, y coords on Lorentz model.
    flow_model.base_dist_mean = torch.zeros_like(on_mani).cuda()
    flow_model.base_dist_var = torch.ones(on_mani.shape[0], 2).cuda()
    probs = flow_model.log_prob(on_mani)
    probs += logmap_logdet(clamped_threedim.cuda(), radius)
    probs = torch.exp(probs)

    on_mani_conv = on_mani.detach().cpu()

    # Calculate the poincare coordinates
    xy_poincare = lorentz_to_poincare(on_mani.squeeze(), radius)
    plot_density(xy_poincare, probs, flow_model.radius, namestr, flow=flow)


def setup_grid(range_lim, n_pts):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.cuda()


def some_density(args):
    radius = torch.Tensor([args.radius]).cuda()
    n_pts = 100

    f1 = lambda z: torch.sin(6 * math.pi * z[:, 0] / 4)
    f2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
    f3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)
    xx, yy, zz = setup_grid(5, n_pts)
    base_prob_dist = -f1(zz)

    # Map x, y coordinates on tangent space at origin to manifold (Lorentz model).
    twodim = zz
    threedim = expand_proj_dims(twodim).cuda()
    clamped_threedim = clamp(threedim, min=-max_clamp_norm, max=max_clamp_norm).cuda()
    on_mani = exp_map_mu0(clamped_threedim, radius)

    # Calculate densities of x, y coords on Lorentz model.
    log_det = _logdet(clamped_threedim, radius)
    log_probs = base_prob_dist - log_det
    probs = torch.exp(log_probs)

    # Calculate the poincare coordinates
    xy_poincare = lorentz_to_poincare(on_mani.squeeze(), radius)

    plot_density(xy_poincare, probs, radius, args.namestr)
    if args.flow != 'none':
        plot_flow(args, radius, args.flow, f1, args.namestr)

def mixture(args):
    radius = torch.Tensor([args.radius]).to(args.dev)
    samples = sample_2d_data(args.dataset, 100000).to(args.dev)
    samples = clamp(samples, min=-max_clamp_norm, max=max_clamp_norm)
    xi = samples[:,0].detach().cpu().numpy()
    yi = samples[:,1].detach().cpu().numpy()
    samples_h = exp_map_mu0(expand_proj_dims(samples), radius)
    # Calculate the poincare coordinates
    xy_poincare = lorentz_to_poincare(samples_h.squeeze(), radius)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = xy_poincare[:, 0].view(-1, 100).detach().cpu()
    y = xy_poincare[:, 1].view(-1, 100).detach().cpu()
    p_z = None

    # Define points within circle
    range_lim = 5
    ax.hist2d(xy_poincare[:, 0].detach().cpu().numpy(), xy_poincare[:,1].detach().cpu().numpy(), range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=5000, cmap='magma')
    # ax.contourf(x, y, z, 100, antialiased=False, cmap='magma')

    ax.axis('off')
    # Makes the circle look like a circle
    ax.axis('equal')
    ax.set_xlim(-args.axis_lim, args.axis_lim)
    ax.set_ylim(-args.axis_lim, args.axis_lim)

    # Save the full figure...
    fig.savefig('install/{}.png'.format(args.namestr))
    print("saved to install/{}.png".format(args.namestr))

    if args.flow != 'none':
        plot_flow(args, radius, args.flow, p_z, args.namestr, n_blocks=args.n_blocks, samples=samples_h)

def gauss(args, mu, std):
    radius = torch.Tensor([args.radius]).to(args.dev)

    mu = clamp(mu, min=-max_clamp_norm, max=max_clamp_norm)
    mu_h = exp_map_mu0(expand_proj_dims(mu), radius)

    p_z = HyperboloidWrappedNormal(radius, mu_h, std)

    # Map x, y coordinates on tangent space at origin to manifold (Lorentz model).
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(x, y)
    x = torch.Tensor(x).view(-1, 1)
    y = torch.Tensor(y).view(-1, 1)
    twodim = torch.cat([x, y], dim=1)
    threedim = expand_proj_dims(twodim)
    clamped_threedim = clamp(threedim, min=-max_clamp_norm,
            max=max_clamp_norm).to(args.dev)
    on_mani = exp_map_mu0(clamped_threedim, radius)

    # Calculate densities of x, y coords on Lorentz model.
    probs = p_z.log_prob(on_mani)
    probs = torch.exp(probs)

    # Calculate the poincare coordinates
    xy_poincare = lorentz_to_poincare(on_mani.squeeze(), radius)
    mu_p = lorentz_to_poincare(mu_h, radius)

    plot_density(xy_poincare, probs, args.radius, args.namestr, mu=mu_p)
    if args.flow != 'none':
        plot_flow(args, radius, args.flow, p_z, args.namestr, args.n_blocks)


def main(args):
    # mean_1 = torch.Tensor([-1., 1.]).unsqueeze(0).to(args.dev)
    # std_1 = torch.Tensor([[0.5], [0.5]]).T.to(args.dev)
    mean_1 = torch.Tensor([-1., 1.]).unsqueeze(0).to(args.dev)
    std_1 = torch.Tensor([[1.0], [0.25]]).T.to(args.dev)

    # some_density(args)
    if args.dataset == 'gauss':
        gauss(args, mean_1, std_1)
    else:
        mixture(args)


if __name__ == "__main__":
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--radius', type=float, default='1')
    parser.add_argument('--axis_lim', type=float, default='1')
    parser.add_argument('--K', type=int, default=500, help='Number of samples.')
    parser.add_argument('--n_blocks', type=int, default=2, \
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--namestr', type=str, default='Floss',
                        help='output filename')
    parser.add_argument('--dataset', type=str, default='checkerboard',
                        help='output filename')
    parser.add_argument('--flow', type=str, default='none')
    parser.add_argument('--flow_epochs', default=100, type=int)
    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
