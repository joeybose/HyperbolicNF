from typing import Dict, Any, Tuple, TypeVar, Type, Optional
import torch
import math
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
from torch.distributions import Normal
from utils.math_ops import clamp, acosh, cosh, sinh, sqrt, logsinh, e_i, expand_proj_dims
from utils.hyperbolics import exp_map, exp_map_mu0, inverse_exp_map_mu0, logmap_logdet
from distributions.wrapped_normal import HyperboloidWrappedNormal
import sys

sys.path.append("..")  # Adds higher directory to python modules path.
from flows.flows import FlowSequential, AllTangentRealNVP, TangentRealNVP
from flows.flows import WrappedRealNVP

max_clamp_norm = 40
kwargs_flows = {'AllTangentRealNVP': AllTangentRealNVP, 'TangentRealNVP':
                TangentRealNVP, 'WrappedRealNVP': WrappedRealNVP}

class FeedForwardHyperboloidVAE(nn.Module):
    def __init__(self, beta, input_dim, hidden_dim, z_dim, recon_loss,
                 ll_estimate, K, flow_args, dev, fixed_curvature, radius=0):
        super(FeedForwardHyperboloidVAE, self).__init__()
        self.type = 'hyperbolic'
        self.dev = dev
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.beta = beta
        self.recon_loss = recon_loss
        self.ll_estimate = ll_estimate
        self.K = K
        self.analytic_kl = False
        n_blocks, flow_hidden_size, n_hidden = flow_args[0], flow_args[1], flow_args[2]
        flow_model, flow_layer_type = flow_args[3], flow_args[4]
        self.radius = torch.nn.Parameter(torch.tensor(radius), requires_grad= not fixed_curvature)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        if flow_model is not None:
            self.flow_model = kwargs_flows[flow_model](n_blocks, self.z_dim,
                                                       flow_hidden_size,
                                                       n_hidden, self.radius,
                                                       flow_layer_type).to(self.dev)
        else:
            self.flow_model = None

    def curvature(self):
        return 1. / self.radius.pow(2)

    def mu_0(self, shape: Tuple[int, ...], **kwargs: Any) -> Tensor:
        return e_i(i=0, shape=shape, **kwargs) * self.radius

    def kl_loss(self, q_z: HyperboloidWrappedNormal, p_z:
                HyperboloidWrappedNormal, z: Tensor, z_k: Tensor,
                data: Tuple[Tensor, ...]) -> Tensor:
        logqz, logpz = self._log_prob(q_z, p_z, z, z_k, data)
        KLD = logqz - logpz
        return KLD

    def rsample_log_probs(self, sample_shape: torch.Size, q_z: HyperboloidWrappedNormal,
                          p_z: HyperboloidWrappedNormal) -> Tuple[Tensor, Tensor, Tensor]:
        sum_log_det_jac = 0
        z, posterior_parts = q_z.rsample_with_parts(sample_shape)
        z = clamp(z, min=-max_clamp_norm, max=max_clamp_norm)
        if self.flow_model:
            z_k = z.view(-1,self.z_dim+1)
            z_k, sum_log_det_jac = self.flow_model.inverse(z_k)
            z_k = clamp(z_k, min=-max_clamp_norm, max=max_clamp_norm)
            z_k = z_k.view(sample_shape[0],-1,self.z_dim+1)
            sum_log_det_jac = sum_log_det_jac.view(sample_shape[0],-1)
        else:
            z_k = z

        z_mu0 = inverse_exp_map_mu0(z_k, self.radius)
        log_q_z_x, log_p_z_k = self._log_prob(q_z, p_z, z, z_k, posterior_parts)
        log_q_z_k_x = log_q_z_x - sum_log_det_jac - logmap_logdet(z_mu0,self.radius)
        log_p_z_k = log_p_z_k - logmap_logdet(z_mu0,self.radius)
        return z_mu0, log_q_z_k_x, log_p_z_k

    def _log_prob(self, q_z: HyperboloidWrappedNormal, p_z:
                  HyperboloidWrappedNormal, z: Tensor, z_k: Tensor,
                  posterior_parts: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        log_q_z_x = q_z.log_prob_from_parts(z, posterior_parts)
        log_p_z = p_z.log_prob(z_k)
        return log_q_z_x, log_p_z

    def encode(self, x):
        h = self.encoder(x)
        z, mu_h, std = self.bottleneck(h)
        return z, mu_h, std

    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        mu = clamp(mu, min=-max_clamp_norm, max=max_clamp_norm)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()
        mu_h = exp_map_mu0(expand_proj_dims(mu), self.radius)
        assert torch.isfinite(mu_h).all()

        # +eps prevents collapse
        std = F.softplus(logvar) + 1e-5
        assert torch.isfinite(std).all()
        q_z, p_z = self.reparametrize(mu_h, std)
        self.q_z = q_z
        self.p_z = p_z
        z, data = q_z.rsample_with_parts()
        self.data = data
        return z, mu_h, std

    def reparametrize(self, mu_h, std):
        q_z = HyperboloidWrappedNormal(self.radius, mu_h, std)
        mu_0 = self.mu_0(mu_h.shape, device=mu_h.device)
        std_0 = torch.ones_like(std, device=mu_h.device)
        p_z = HyperboloidWrappedNormal(self.radius, mu_0, std_0)
        return q_z, p_z

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        sum_log_det_jac = 0
        z, mu_h, std = self.encode(x)
        z = clamp(z, min=-max_clamp_norm, max=max_clamp_norm)
        ### Flow ###
        if self.flow_model:
            self.flow_model.base_dist_mean = mu_h
            self.flow_model.base_dist_var = std
            self.flow_model.radius = self.radius
            z_k, sum_log_det_jac = self.flow_model.inverse(z)
            z_k = clamp(z_k, min=-max_clamp_norm, max=max_clamp_norm)
        else:
            z_k = z

        kld = self.kl_loss(self.q_z, self.p_z, z, z_k, self.data)
        z_mu0 = inverse_exp_map_mu0(z_k, self.radius)
        # This is not really the same KL Divergence and can be negative
        kld = kld - sum_log_det_jac - logmap_logdet(z_mu0, self.radius)
        x_tilde = self.decode(z_mu0)
        return x_tilde, kld

    def MC_log_likelihood(self, x):
        """
        :param x: Mini-batch of inputs.
        :param n: Number of MC samples
        :return: Monte Carlo estimate of log-likelihood.
        """
        n = self.K
        sample_shape = torch.Size([n])
        batch_size = x.shape[0]
        prob_shape = torch.Size([n, batch_size])

        x_encoded = self.encoder(x)
        mu, logvar = self.fc_mean(x_encoded), self.fc_logvar(x_encoded)
        mu = clamp(mu, min=-max_clamp_norm, max=max_clamp_norm)
        mu_h = exp_map_mu0(expand_proj_dims(mu), self.radius)

        # +eps prevents collapse
        std = F.softplus(logvar) + 1e-5
        q_z, p_z = self.reparametrize(mu_h, std)
        log_p_z = torch.zeros(prob_shape, device=x.device)
        log_q_z_x = torch.zeros(prob_shape, device=x.device)

        # Numerically more stable.
        z, log_q_z_x, log_p_z = self.rsample_log_probs(sample_shape, q_z, p_z)
        z = inverse_exp_map_mu0(z, self.radius)
        log_q_z_x = log_q_z_x - logmap_logdet(z, self.radius)

        x_mb_ = self.decode(z)
        x_orig = x.repeat((n, 1, 1))
        log_p_x_z = -self.recon_loss(x_mb_, x_orig).sum(dim=-1)

        assert log_p_x_z.shape == log_p_z.shape
        assert log_q_z_x.shape == log_p_z.shape
        joint = (log_p_x_z + log_p_z - log_q_z_x)
        log_p_x = joint.logsumexp(dim=0) - np.log(n)

        assert log_q_z_x.shape == log_p_z.shape
        mi = (log_q_z_x - log_p_z).logsumexp(dim=0) - np.log(n)

        return log_p_x, mi

    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))


class CifarHyperboloidVAE(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=192, z_dim=32):
        super(CifarHyperboloidVAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(image_channels, 3, kernel_size=3, stride=1, padding=1),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 16, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
