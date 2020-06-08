import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from flows.flow_helpers import *
from utils.math_ops import clamp, expand_proj_dims, logsinh, e_i
from utils.hyperbolics import inverse_parallel_transport_mu0, parallel_transport_mu0,inverse_sample_projection_mu0,exp_map, inverse_exp_map
from utils.hyperbolics import proj_vec_to_tang, proj_vec, lorentz_norm,inverse_exp_map_mu0, exp_map_mu0, _logdet, logmap_logdet
from distributions.normal import EuclideanNormal
from distributions.wrapped_normal import HyperboloidWrappedNormal


#Reference: https://github.com/ritheshkumar95/pytorch-normalizing-flows/blob/master/modules.py
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
max_clamp_norm = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# All code below this line is taken from
# https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        i = len(self)
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
            i -= 1
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        i = 0
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
            i += 1
        return u, sum_log_abs_det_jacobians

# --------------------
# Models
# --------------------

class MAFRealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 radius=torch.Tensor([0]), cond_label_size=None, batch_norm=False):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.p_z = EuclideanNormal
        self.radius = radius

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            # modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

## Taken from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 layer_type='Linear', radius=torch.Tensor([0])):
        super(RealNVP, self).__init__()
        mask = torch.arange(input_size).float() % 2
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.radius = radius
        self.layer_type = layer_type
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(n_blocks/2),1)
        self.p_z = EuclideanNormal
        self.s, self.t = create_real_nvp_blocks(input_size, hidden_size,
                                                n_blocks, n_hidden, layer_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z, edge_index=None):
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(0,self.n_blocks):
            x_ = x*self.mask[i]
            if self.layer_type != 'Linear':
                s = self.s[i](x_, edge_index)
                t = self.t[i](x_, edge_index)
            else:
                s = self.s[i](x_)
                t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += ((1-self.mask[i])*s).sum(dim=1)  # log det dx/du
        return x, log_det_J

    def forward(self, x, edge_index=None):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(0,self.n_blocks)):
            z_ = self.mask[i] * z
            if self.layer_type != 'Linear':
                s = self.s[i](z_, edge_index)
                t = self.t[i](z_, edge_index)
            else:
                s = self.s[i](z_)
                t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=1)
        return z, log_det_J

    def log_prob(self, x, edge_index=None):
        z, logp = self.forward(x, edge_index)
        p_z = self.p_z(torch.zeros_like(x, device=self.dev),
                         torch.ones_like(x))
        return p_z.log_prob(z) + logp

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x

class WrappedRealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, radius,
                 layer_type='Linear'):
        super(WrappedRealNVP, self).__init__()
        self.radius = radius
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.preclamp_norm = torch.Tensor([0])
        self.input_size = input_size
        self.layer_type = layer_type
        mask = torch.arange(input_size).float() % 2
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p_z = HyperboloidWrappedNormal
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(n_blocks/2),1)
        self.s, self.t = create_wrapped_real_nvp_blocks(input_size, hidden_size,
                                                n_blocks, n_hidden, layer_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size+1))
        self.register_buffer('base_dist_var', torch.ones(input_size+1))
        self.mask = nn.Parameter(mask, requires_grad=False)

    def create_masked_t(self, mask, t1, t_rest):
        count = 0
        zero_vector = torch.zeros(len(t1), 1).to(t1.device)
        for i in range(0,len(mask)):
            if mask[i].item() == 0:
                t1 = torch.cat((t1, zero_vector), dim=1)
            else:
                column = t_rest[:,count].view(-1,1)
                t1 = torch.cat((t1, column), dim=1)
                count += 1
        return t1

    def inverse(self, z_hyper, edge_index=None):
        z = inverse_exp_map_mu0(z_hyper, self.radius)
        z_mu0 = z[..., 1:]
        log_det_J, x = z_mu0.new_zeros(z_mu0.shape[0]), z_mu0
        log_det_J = logmap_logdet(z, self.radius)
        preclamp_norm_list = []
        for i in range(0,self.n_blocks):
            x_ = x*self.mask[i]
            if self.layer_type != 'Linear':
                s = self.s[i](x_, edge_index)
                t_out = self.t[i](x_, edge_index)
            else:
                s = self.s[i](x_)
                t_out = self.t[i](x_)
            t_proj = proj_vec(t_out, self.radius)
            t1, t_rest = t_proj[:,0].unsqueeze(1), t_proj[:,1:]
            t = self.create_masked_t((1-self.mask[i]), t1, t_rest)
            # (1-b) \odot \tilde{x} \odot exp(s(b \odot \tilde{x}))
            x_pt_arg = expand_proj_dims((1 - self.mask[i]) * x * torch.exp(s))

            # (1-b) \odot \textnormal{PT}_{\textbf{o}\to t(b \odot \tilde{x})
            pt = parallel_transport_mu0(x_pt_arg, dst=t, radius=self.radius)
            preclamp_norm = pt.max()
            pt = clamp(pt, min=-max_clamp_norm, max=max_clamp_norm)
            if pt.max() == max_clamp_norm:
                preclamp_norm_list.append(preclamp_norm)
            x_t = exp_map(x=pt, at_point=t, radius=self.radius)
            log_det_J += _logdet(pt, self.radius, subdim=(self.mask[i]).sum())
            preclamp_norm = x_t.max()
            x_t = clamp(x_t, min=-max_clamp_norm, max=max_clamp_norm)
            if x_t.max() == max_clamp_norm:
                preclamp_norm_list.append(preclamp_norm)

            #\log_{\textbf{o}}(\textnormal{exp}_{t()}(\textnormal{PT}_{\textbf{o}\to t()))
            x_0_full = inverse_exp_map_mu0(x_t, self.radius)
            x_0 = x_0_full[...,1:]
            log_det_J += logmap_logdet(x_0_full, self.radius, subdim=(self.mask[i]).sum())
            x = x_ + (1 - self.mask[i]) * x_0
            log_det_J += ((1-self.mask[i])*s).sum(dim=1)  # log det dx/du

            preclamp_norm = x.max()
            x = clamp(x, min=-max_clamp_norm, max=max_clamp_norm)
            if x.max() == max_clamp_norm:
                preclamp_norm_list.append(preclamp_norm)

        x_mu0 = expand_proj_dims(x)
        # Project back to Manifold
        x = exp_map_mu0(x_mu0, self.radius)
        log_det_J += _logdet(x_mu0, self.radius)

        self.preclamp_norm = torch.Tensor([sum(preclamp_norm_list)
                                           /len(preclamp_norm_list)]) if preclamp_norm_list else self.preclamp_norm
        return x, log_det_J

    def forward(self, x_hyper, edge_index=None):
        x = inverse_exp_map_mu0(x_hyper, self.radius)
        x_mu0 = x[..., 1:]
        log_det_J, z = x.new_zeros(x_mu0.shape[0]), x_mu0
        log_det_J = -1*logmap_logdet(x, self.radius)
        for i in reversed(range(0,self.n_blocks)):
            z_ = self.mask[i] * z
            if self.layer_type != 'Linear':
                s = self.s[i](z_, edge_index)
                t_out = self.t[i](z_, edge_index)
            else:
                s = self.s[i](z_)
                t_out = self.t[i](z_)
            t_proj = proj_vec(t_out, self.radius)

            t1, t_rest = t_proj[:,0].unsqueeze(1), t_proj[:,1:]
            t = self.create_masked_t((1-self.mask[i]), t1, t_rest)

            z_2 = expand_proj_dims((1 - self.mask[i]) * z)
            z_2 = clamp(z_2, min=-max_clamp_norm, max=max_clamp_norm)
            z_exp_2 = exp_map_mu0(z_2, self.radius)
            log_det_J -= _logdet(z_2, self.radius, subdim=(self.mask[i]).sum())

            z_exp_2 = clamp(z_exp_2, min=-max_clamp_norm, max=max_clamp_norm)
            z_inv_pt_arg = inverse_exp_map(x=z_exp_2, at_point=t, radius=self.radius)
            log_det_J -= logmap_logdet(z_inv_pt_arg, self.radius, subdim=(self.mask[i]).sum())

            z_inv_pt_arg = clamp(z_inv_pt_arg, min=-max_clamp_norm, max=max_clamp_norm)
            pt = inverse_parallel_transport_mu0(z_inv_pt_arg, src=t, radius=self.radius)
            pt = pt[..., 1:]

            z = (1 - self.mask[i]) * pt * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=1)

        z_mu0 = expand_proj_dims(z)
        z = exp_map_mu0(z_mu0, self.radius)
        log_det_J -= _logdet(z_mu0, self.radius)
        return z, log_det_J

    def log_prob(self,x, edge_index=None):
        z, logp = self.forward(x, edge_index)
        mu_0 = e_i(i=0, shape=self.base_dist_mean.shape,
                   device=self.base_dist_mean.device) * self.radius
        p_z = self.p_z(self.radius, torch.zeros_like(mu_0, device=self.dev),
                         torch.ones_like(self.base_dist_var))
        return p_z.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x

class AllTangentRealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, radius,
                 layer_type='Linear'):
        super(AllTangentRealNVP, self).__init__()
        self.radius = radius
        mask = torch.arange(input_size).float() % 2
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.layer_type = layer_type
        self.preclamp_norm = torch.Tensor([0])
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p_z = HyperboloidWrappedNormal
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(n_blocks/2),1)
        self.s, self.t = create_real_nvp_blocks(input_size, hidden_size,
                                                n_blocks, n_hidden, layer_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size+1))
        self.register_buffer('base_dist_var', torch.ones(input_size+1))
        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z_hyper, edge_index=None):
        z = inverse_exp_map_mu0(z_hyper, self.radius)
        z_mu0 = z[..., 1:]
        log_det_J, x = z_mu0.new_zeros(z_mu0.shape[0]), z_mu0
        log_det_J = logmap_logdet(z, self.radius)
        for i in range(0,self.n_blocks):
            x_ = x*self.mask[i]
            if self.layer_type != 'Linear':
                s = self.s[i](x_, edge_index)
                t = self.t[i](x_, edge_index)
            else:
                s = self.s[i](x_)
                t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            self.preclamp_norm = x.max()
            x = clamp(x, min=-max_clamp_norm, max=max_clamp_norm)
            log_det_J += ((1-self.mask[i])*s).sum(dim=1)  # log det dx/du
        x_mu0 = expand_proj_dims(x)
        x = exp_map_mu0(x_mu0, self.radius)
        log_det_J += _logdet(x_mu0, self.radius)
        return x, log_det_J

    def forward(self, x_hyper, edge_index=None):
        x = inverse_exp_map_mu0(x_hyper, self.radius)
        x_mu0 = x[..., 1:]
        log_det_J, z = x.new_zeros(x_mu0.shape[0]), x_mu0
        log_det_J = -1*logmap_logdet(x, self.radius)
        for i in reversed(range(0,self.n_blocks)):
            z_ = self.mask[i] * z
            if self.layer_type != 'Linear':
                s = self.s[i](z_, edge_index)
                t = self.t[i](z_, edge_index)
            else:
                s = self.s[i](z_)
                t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=1)
        z_mu0 = expand_proj_dims(z)
        z = exp_map_mu0(z_mu0, self.radius)
        log_det_J -= _logdet(z_mu0, self.radius)
        return z, log_det_J

    def log_prob(self, x, edge_index=None):
        z, logp = self.forward(x, edge_index)
        mu_0 = e_i(i=0, shape=self.base_dist_mean.shape,
                   device=self.base_dist_mean.device) * self.radius
        p_z = self.p_z(self.radius, torch.zeros_like(mu_0, device=self.dev),
                         torch.ones_like(self.base_dist_var))
        return p_z.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x

class TangentRealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, radius,
                 layer_type='Linear'):
        super(TangentRealNVP, self).__init__()
        self.radius = radius
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        mask = torch.arange(input_size).float() % 2
        self.preclamp_norm = torch.Tensor([0])
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p_z = HyperboloidWrappedNormal
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(n_blocks/2),1)
        nets, nett = [], []
        self.s, self.t = create_real_nvp_blocks(input_size, hidden_size,
                                                n_blocks, n_hidden, layer_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size+1))
        self.register_buffer('base_dist_var', torch.ones(input_size+1))
        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z_hyper):
        z = inverse_exp_map_mu0(z_hyper, self.radius)
        z_mu0 = z[..., 1:]
        log_det_J, x = z_mu0.new_zeros(z_mu0.shape[0]), z_mu0
        log_det_J = logmap_logdet(z, self.radius)
        for i in range(0,self.n_blocks):
            if i > 0:
                # Project between Flow Layers
                x_proj_mu0 = inverse_exp_map_mu0(x, self.radius)
                x = x_proj_mu0[..., 1:]
                log_det_J += logmap_logdet(x_proj_mu0, self.radius)
            x_ = x*self.mask[i]
            if self.layer_type != 'Linear':
                s = self.s[i](x_, edge_index)
                t = self.t[i](x_, edge_index)
            else:
                s = self.s[i](x_)
                t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            self.preclamp_norm = x.max()
            x = clamp(x, min=-max_clamp_norm, max=max_clamp_norm)
            log_det_J += ((1-self.mask[i])*s).sum(dim=1)  # log det dx/du
            x_mu0 = expand_proj_dims(x)
            # Project back to Manifold
            x = exp_map_mu0(x_mu0, self.radius)
            log_det_J += _logdet(x_mu0, self.radius)
        return x, log_det_J

    def forward(self, x_hyper):
        x = inverse_exp_map_mu0(x_hyper, self.radius)
        x_mu0 = x[..., 1:]
        log_det_J, z = x.new_zeros(x_mu0.shape[0]), x_mu0
        log_det_J = -1*logmap_logdet(x, self.radius)
        for i in reversed(range(0,self.n_blocks)):
            if i > 0:
                # Project between Flow Layers
                z_proj_mu0 = inverse_exp_map_mu0(z, self.radius)
                z = z_proj_mu0[..., 1:]
                log_det_J -= logmap_logdet(z_proj_mu0, self.radius)
            z_ = self.mask[i] * z
            if self.layer_type != 'Linear':
                s = self.s[i](z_, edge_index)
                t = self.t[i](z_, edge_index)
            else:
                s = self.s[i](z_)
                t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=1)
            z_mu0 = expand_proj_dims(z)
            # Project back to Manifold
            z = exp_map_mu0(z_mu0, self.radius)
            log_det_J -= _logdet(z_mu0, self.radius)
        return z, log_det_J

    def log_prob(self, x, edge_index=None):
        z, logp = self.forward(x, edge_index)
        mu_0 = e_i(i=0, shape=self.base_dist_mean.shape,
                   device=self.base_dist_mean.device) * self.radius
        p_z = self.p_z(self.radius, torch.zeros_like(mu_0, device=self.dev),
                         torch.ones_like(self.base_dist_var))
        return p_z.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.p_z.sample((batchSize, 1))
        logp = self.p_z.log_prob(z)
        x = self.inverse(z)
        return x
