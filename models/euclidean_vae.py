from typing import Tuple
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from distributions.normal import EuclideanNormal
from utils.utils import log_mean_exp


sys.path.append("..")  # Adds higher directory to python modules path.
from flows.flows import MAFRealNVP, RealNVP

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP}


class FeedForwardVAE(nn.Module):
    def __init__(self, beta, input_dim, hidden_dim, z_dim, recon_loss, ll_estimate, K,
                 flow_args, dev, fixed_curvature, radius):
        super(FeedForwardVAE, self).__init__()
        self.type = 'euclidean'
        # prior distribution
        self.p_z = EuclideanNormal
        # posterior distribution
        self.qz_x = EuclideanNormal
        # likelihood distribution
        self.px_z = EuclideanNormal
        self.dev = dev
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.beta = beta
        self.recon_loss = recon_loss
        self.ll_estimate = ll_estimate
        self.K = K
        self.radius = torch.tensor(radius)
        n_blocks, flow_hidden_size, n_hidden = flow_args[0], flow_args[1], flow_args[2]
        flow_model, flow_layer_type = flow_args[3], flow_args[4]
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.LeakyReLU(0.2))
        self.fc_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU(0.2),
                                     nn.Linear(self.hidden_dim, self.input_dim))
        self.partial_decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU(0.2))
        self.fc_dmean = nn.Linear(self.hidden_dim, self.input_dim)
        self.fc_dlogvar = nn.Linear(self.hidden_dim, self.input_dim)
        if flow_model is not None:
            self.flow_model = kwargs_flows[flow_model](n_blocks, self.z_dim,
                                                       flow_hidden_size,
                                                       n_hidden,
                                                       flow_layer_type).to(self.dev)
            self.analytic_kl = False
        else:
            self.flow_model = None
            self.analytic_kl = True

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = F.softplus(logvar) + 1e-5
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder(z)
        return z

    def kl_loss(self, base_mu, base_logvar, z_0, z_k):
        if self.analytic_kl:
            ### -D_KL https://arxiv.org/abs/1312.6114 Appendix
            kld = -0.5 * torch.sum(1 + base_logvar - base_mu.pow(2) -
                                   base_logvar.exp(), dim=1)
            return kld
        base_std = F.softplus(base_logvar) + 1e-5
        q_z_0_x = self.qz_x(base_mu, base_std)
        p_z_k = self.p_z(torch.zeros_like(base_mu, device=self.dev),
                         torch.ones_like(base_std))
        log_q_z_0_x = q_z_0_x.log_prob(z_0)
        log_p_z_k = p_z_k.log_prob(z_k)
        kld = log_q_z_0_x - log_p_z_k
        return kld

    def forward(self, x):
        sum_log_det_jac = 0
        z, mu, logvar = self.encode(x)
        ### Flow ###
        if self.flow_model:
            self.flow_model.base_dist_mean = mu
            self.flow_model.base_dist_var = torch.exp(logvar)
            z_k, sum_log_det_jac = self.flow_model.inverse(z)
        else:
            z_k = z

        kld = self.kl_loss(mu, logvar, z, z_k)
        kld = kld - sum_log_det_jac
        x_tilde = self.decode(z_k)
        return x_tilde, kld

    def _log_prob(self, q_z, p_z, z, z_k, posterior_parts):
        log_q_z_x_ = q_z.log_prob_from_parts(z, posterior_parts)
        log_p_z_k = p_z.log_prob(z_k)
        return log_q_z_x_, log_p_z_k

    def rsample_log_probs(self, sample_shape, qz_x, p_z):
        sum_log_det_jac = 0
        z, posterior_parts = qz_x.rsample_with_parts(sample_shape)
        # Numerically more stable.
        if self.flow_model:
            z_k = z.view(-1,self.z_dim)
            z_k, sum_log_det_jac = self.flow_model.inverse(z_k)
            z_k = z_k.view(sample_shape[0],-1,self.z_dim)
            sum_log_det_jac = sum_log_det_jac.view(sample_shape[0],-1)
        else:
            z_k = z

        log_q_z_x, log_p_z_k = self._log_prob(qz_x, p_z, z, z_k, posterior_parts)
        log_q_z_k_x = log_q_z_x - sum_log_det_jac
        return z_k, log_q_z_k_x, log_p_z_k

    def likelihood(self, z):
        d = self.partial_decoder(z)
        mu = self.fc_dmean(d).view(*z.size()[:-1], self.input_dim) # reshape data
        px_z = self.px_z(mu, torch.ones_like(mu))
        return px_z

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
        _, mu, logvar = self.encode(x)
        std = F.softplus(logvar) + 1e-5

        qz_x = self.qz_x(mu, std)
        p_z = self.p_z(torch.zeros_like(mu, device=self.dev), torch.ones_like(std))

        z, log_qz_x, log_p_z = self.rsample_log_probs(sample_shape, qz_x, p_z)

        x_mb_ = self.decode(z)
        x_orig = x.repeat((n, 1, 1))
        log_p_x_z = -self.recon_loss(x_mb_, x_orig).sum(dim=-1)

        assert log_p_x_z.shape == log_p_z.shape
        assert log_qz_x.shape == log_p_z.shape
        joint = (log_p_x_z + log_p_z - log_qz_x)
        log_p_x = joint.logsumexp(dim=0) - np.log(n)

        assert log_qz_x.shape == log_p_z.shape
        mi = (log_qz_x - log_p_z).logsumexp(dim=0) - np.log(n)

        return log_p_x, mi

    def iwae(self, x):
        n = self.K
        sample_shape = torch.Size([n])
        batch_size = x.shape[0]
        prob_shape = torch.Size([n, batch_size])

        _, mu, logvar = self.encode(x)
        std = F.softplus(logvar) + 1e-5

        qz_x = self.qz_x(mu, std)
        p_z = self.p_z(torch.zeros_like(mu, device=self.dev), torch.ones_like(std))

        # Numerically more stable.
        z, log_qz_x, log_p_z = self.rsample_log_probs(sample_shape, qz_x, p_z)
        # TODO: This part is confusing, the partial decoder is just random
        # weights
        px_z = self.likelihood(z)
        flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
        log_px_z = px_z.log_prob(x.expand(z.size(0), *x.size())).view(flat_rest).sum(-1)

        obj = log_p_z.squeeze(-1) + log_px_z.view(log_p_z.squeeze(-1).shape) - log_qz_x.squeeze(-1)
        return -log_mean_exp(obj).sum(), None

    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))


class CifarVAE(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=192, z_dim=32):
        super(CifarVAE, self).__init__()
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
        # return torch.normal(mu, std)
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
