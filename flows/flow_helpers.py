import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
import torch.nn.init as init
from torch_geometric.nn import GCNConv, GATConv

from utils.math_ops import expand_proj_dims, logsinh
from utils.hyperbolics import parallel_transport_mu0,inverse_sample_projection_mu0,exp_map, inverse_exp_map, lorentz_norm,inverse_exp_map_mu0, exp_map_mu0
from utils.utils import MultiInputSequential
import math
import os
import math
import argparse
import pprint
import numpy as np
import copy


# --------------------
# Model layers and helpers
# --------------------

kwargs_layer = {'Linear': nn.Linear, 'GCN': GCNConv, 'GAT': GATConv}

def create_real_nvp_blocks(input_size, hidden_size, n_blocks, n_hidden,
                          layer_type='Linear'):
    nets, nett = [], []
    # Build the Flow Block by Block
    if layer_type == 'GCN':
        GCNConv.cached = False
    elif layer_type == 'GAT':
        GATConv.heads = 8

    for i in range(n_blocks):
        block_nets = [kwargs_layer[layer_type](input_size, hidden_size)]
        block_nett = [kwargs_layer[layer_type](input_size, hidden_size)]
        for _ in range(n_hidden):
            block_nets += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, hidden_size)]
            block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, hidden_size)]
        block_nets += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, input_size)]
        block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, input_size)]
        nets +=[MultiInputSequential(*block_nets)]
        nett +=[MultiInputSequential(*block_nett)]

    s = nets = MultiInputSequential(*nets)
    t = nett = MultiInputSequential(*nett)
    return s,t

def create_wrapped_real_nvp_blocks(input_size, hidden_size, n_blocks, n_hidden,
                          layer_type='Linear'):
    nets, nett = [], []
    # Build the Flow Block by Block
    if layer_type == 'GCN':
        GCNConv.cached = False
    elif layer_type == 'GAT':
        GATConv.heads = 8

    t_outsize = int(np.ceil(input_size/2))
    for i in range(n_blocks):
        block_nets = [kwargs_layer[layer_type](input_size, hidden_size)]
        block_nett = [kwargs_layer[layer_type](input_size, hidden_size)]
        for _ in range(n_hidden):
            block_nets += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, hidden_size)]
            block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, hidden_size)]
        block_nets += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, input_size)]
        block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, t_outsize)]
        nets +=[MultiInputSequential(*block_nets)]
        nett +=[MultiInputSequential(*block_nett)]

    s = nets = MultiInputSequential(*nets)
    t = nett = MultiInputSequential(*nett)
    return s,t

def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian

class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
#        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
#            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = y.mean(0)
            var = y.var(0)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)

class HyperbolicLinear(nn.Module):
    """
    Hyperbolic neural networks layer:
    Source: https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py
    """
    def __init__(self, in_features, out_features, radius, use_bias=False):
        super(HyperbolicLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.radius = radius
        self.use_bias = use_bias
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def hyper_act(self, x):
        '''
        Op: \sigma(x)
        Input:
        x: A Hyperbolic Vector in H^n

        Output:
        sigma_x_mu0: A Hyperbolic Vector in H^n after \sigma(x)
        '''
        x_tangent_mu0 = inverse_exp_map_mu0(x, self.radius)
        sigma_x = F.relu(x_tangent_mu0)
        sigma_x_mu0 = exp_map_mu0(sigma_x, self.radius)
        return sigma_x_mu0

    def forward(self, x):
        x_tangent_mu0 = inverse_exp_map_mu0(x, self.radius)
        output = exp_map_mu0(x_tangent_mu0.matmul(self.weight.t()), self.radius)
        if self.use_bias:
            output = parallel_transport_mu0(self.bias, output, self.radius)
            output = exp_map(output, x)
        ret = output
        h = self.hyper_act(ret)
        return h

