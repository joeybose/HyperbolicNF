import sys

# module_path = 'umap'
# if module_path not in sys.path:
#   sys.path.insert(0, module_path)

sys.path.append("..")
from hyperbolics import exp_map_mu0, clamp
from math_ops import expand_proj_dims
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal
from matplotlib import cm
import torch
import umap
import sklearn.datasets

sns.set(style='white', rc={'figure.figsize': (100, 100)})


def create_hyperboloid(curvature, offset):

    x = np.arange(-5, 5, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    z = np.sqrt(4. * (x ** 2 + y ** 2) / 1. + curvature) + offset
    return x, y, z


def create_gauss():
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    x = np.arange(-10, 3, 0.1)
    y = np.arange(-4, 4, 0.1)
    x, y = np.meshgrid(x, y)

    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    sigma = np.array([[1., -0.5], [-0.5, 1.5]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    F = multivariate_normal(mu, sigma)
    z = F.pdf(pos)

    return x, y, z, mu, sigma, pos


def project_gauss(gx, gy, gz, mu, sigma, curvature):
    gx = torch.Tensor(gx).view(-1,1)
    gy = torch.Tensor(gy).view(-1,1)
    twodim = torch.cat([gx,gy], dim=1)
    threedim = expand_proj_dims(twodim)
    clamped_threedim = clamp(threedim, min=-40, max=40)

    on_mani = exp_map_mu0(threedim, curvature)

    return on_mani


def draw_fig(args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    hx, hy, hz = create_hyperboloid(args.curvature, offset=0)
    gx, gy, gz, mu, sigma, pos = create_gauss()
    on_mani = project_gauss(gx, gy, gz, mu, sigma, args.curvature)
    mx = on_mani.numpy()[:,0]
    my = on_mani.numpy()[:,1]
    mz = on_mani.numpy()[:,2]

    # Create surfaces
    # hyp_surf = ax.plot_surface(hx, hy, hz, linewidth=0.0, color='DarkKhaki', antialiased=False, alpha=0.25)
    gauss_cont = ax.contourf(gx, gy, gz, 20, cmap=cm.magma)
    gauss_proj = ax.scatter(mx, my, mz, color='red')

    # Plot
    plt.show()
    plt.savefig('install/{}.pdf'.format(args.plot_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--curvature', type=int, default=1)
    parser.add_argument('--plot_name', type=str, default='test')
    args = parser.parse_args()

    draw_fig(args)


