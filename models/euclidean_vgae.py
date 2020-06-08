import math
import random

import torch
import numpy as np
import torch_geometric
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv, GATConv
from typing import Tuple
import sys
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from distributions.normal import EuclideanNormal, MultivariateEuclideanNormal
from utils.utils import log_mean_exp, filter_state_dict, MultiInputSequential


sys.path.append("..")  # Adds higher directory to python modules path.
from flows.flows import MAFRealNVP, RealNVP

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP}
kwargs_enc_conv = {'GCN': GCNConv, 'GAT': GATConv}

EPS = 1e-15
LOG_VAR_MAX = 10
LOG_VAR_MIN = EPS

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def negative_sampling(pos_edge_index, num_nodes):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).to(pos_edge_index.device)

class Encoder(torch.nn.Module):
    def __init__(self, deterministic, n_blocks, conv_type, beta, input_dim, hidden_dim, z_dim, recon_loss,
                 ll_estimate, K, flow_args, dev, fixed_curvature, radius):
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks
        self.conv1, self.conv_mu, self.conv_logvar = self.create_enc_blocks(input_dim, hidden_dim, z_dim, n_blocks,
                                                                            conv_type)
        self.type = 'euclidean'
        self.dev = dev
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.beta = beta
        self.recon_loss = recon_loss
        self.ll_estimate = ll_estimate
        self.deterministic = deterministic
        self.K = K
        self.radius = torch.tensor(radius)
        n_blocks, flow_hidden_size, n_hidden = flow_args[0], flow_args[1], flow_args[2]
        flow_model, flow_layer_type = flow_args[3], flow_args[4]
        if flow_model is not None and not deterministic:
            self.flow_model = kwargs_flows[flow_model](n_blocks, self.z_dim,
                                                       flow_hidden_size,
                                                       n_hidden,
                                                       flow_layer_type).to(self.dev)
            self.analytic_kl = False
        else:
            self.flow_model = None
            self.analytic_kl = True

    def create_enc_blocks(self, input_dim, hidden_dim, z_dim, n_blocks,
                          conv_type='GCN'):
        # Build the Flow Block by Block
        if conv_type == 'GCN':
            GCNConv.cached = False
        elif conv_type == 'GAT':
            GATConv.heads = 8

        block_net_enc = [kwargs_enc_conv[conv_type](input_dim, hidden_dim)]
        for i in range(n_blocks):
            block_net_enc += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim, hidden_dim)]

        enc = MultiInputSequential(*block_net_enc)
        conv_mu = kwargs_enc_conv[conv_type](hidden_dim, z_dim)
        conv_logvar = kwargs_enc_conv[conv_type](hidden_dim, z_dim)
        return enc, conv_mu, conv_logvar

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        logvar = self.conv_logvar(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper
    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})
    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class TanhDecoder(torch.nn.Module):
    """TanH to compute edge probabilities based on distances."""

    def __init__(self, radius):
        super(TanhDecoder, self).__init__()
        self.radius = radius

    def forward(self, z, edge_index):
        dist = -1*F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        scores = torch.tanh(dist)
        return scores.squeeze()

    def forward_all(self, z):
        dist = -1*F.pairwise_distance(z, z.t())
        scores = torch.tanh(dist)
        return scores.squeeze()

class SoftmaxDecoder(torch.nn.Module):
    """Distance to compute edge probabilities based on distances."""

    def __init__(self, radius, p):
        super(SoftmaxDecoder, self).__init__()
        self.radius = radius
        self.p = torch.nn.Parameter(torch.tensor(p))

    def forward(self, z, edge_index):
        dist = 1./ F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        scores = torch.sigmoid(self.p) * F.softmax(dist)
        probs = scores * 1. / max(scores)
        return probs.squeeze()

    def forward_all(self, z, edge_index):
        adj = torch.eye(z.shape[0])
        dist = 1./F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        scores = torch.sigmoid(self.p) * F.softmax(dist)
        probs = scores * 1. / max(scores)
        for i in range(0, len(edge_index[0])):
            adj[edge_index[0][i]][edge_index[1][i]] = probs[i]
        return adj

class DistanceDecoder(torch.nn.Module):
    """Distance to compute edge probabilities based on distances."""

    def __init__(self, radius, conv_type, r, t, hidden_dim, z_dim, n_blocks):
        super(DistanceDecoder, self).__init__()
        self.radius = radius
        block_net_r = [kwargs_enc_conv[conv_type](z_dim, hidden_dim)]
        block_net_t = [kwargs_enc_conv[conv_type](z_dim, hidden_dim)]
        for i in range(n_blocks):
            block_net_r += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim, hidden_dim)]
            block_net_t += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim, hidden_dim)]

        block_net_r += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim,
                                                              int(hidden_dim/2))]
        block_net_t += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim,
                                                              int(hidden_dim/2))]
        self.r = MultiInputSequential(*block_net_r)
        self.r_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_dim, 1))
        self.t_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_dim, 1))
        self.t = MultiInputSequential(*block_net_r)

    def forward(self, z, edge_index):
        dist = -1*F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        r_gnn = self.r(z, edge_index)
        t_gnn = self.t(z, edge_index)
        inp_r = torch.cat((r_gnn[edge_index[0]],r_gnn[edge_index[1]]), dim=1)
        inp_t = torch.cat((t_gnn[edge_index[0]],t_gnn[edge_index[1]]), dim=1)
        r = self.r_mlp(inp_r).squeeze()
        t = self.t_mlp(inp_t).squeeze()
        probs = torch.sigmoid((dist - r) / t)
        return probs.squeeze()

    def forward_all(self, z, edge_index):
        adj = torch.eye(z.shape[0])
        dist = -1*F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        r_gnn = self.r(z, edge_index)
        t_gnn = self.t(z, edge_index)
        inp_r = torch.cat((r_gnn[edge_index[0]],r_gnn[edge_index[1]]), dim=1)
        inp_t = torch.cat((t_gnn[edge_index[0]],t_gnn[edge_index[1]]), dim=1)
        r = self.r_mlp(inp_r).squeeze()
        t = self.t_mlp(inp_r).squeeze()
        probs = torch.sigmoid((dist - r) / t)
        for i in range(0, len(edge_index[0])):
            adj[edge_index[0][i]][edge_index[1][i]] = probs[i]
        return adj

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def kl_loss(self, z_0, z_k, mu=None, logvar=None):
        r"""There is no KL here as everything is deterministic"""
        return torch.Tensor([0]).to(self.dev)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.
        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def ranking_metrics(self, logits, y):
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10."""

        y = y.to(logits.device)
        adj = torch.mm(logits, logits.t())
        adj = adj[y[0]]
        _, perm = adj.sort(dim=1, descending=True)

        mask = (y[1].view(-1, 1) == perm)

        mrr = (1 / (mask.nonzero()[:, -1] + 1).to(torch.float)).mean().item()
        hits1 = mask[:, :1].sum().item() / y.size(1)
        hits3 = mask[:, :3].sum().item() / y.size(1)
        hits10 = mask[:, :10].sum().item() / y.size(1)

        return mrr, hits1, hits3, hits10

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def save(self, fn_enc):
        torch.save(self.encoder.state_dict(), fn_enc)

    def load(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc))

    def load_no_flow(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc),strict=False)
        pretrained_dict = self.encoder.state_dict()
        filtered_dict = filter_state_dict(pretrained_dict, "flow")
        pretrained_dict.update(filtered_dict)

class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder, r, temperature):
        self.sum_log_det_jac = 0
        # prior distribution
        self.p_z = EuclideanNormal
        # posterior distribution
        self.qz_x = EuclideanNormal
        # likelihood distribution
        self.px_z = EuclideanNormal
        self.r = r
        self.temperature = temperature
        self.deterministic = encoder.deterministic
        self.z_dim = encoder.z_dim
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_name = decoder
        if self.decoder_name == 'tanh':
            decoder = TanhDecoder(encoder.radius)
        elif self.decoder_name == 'distance':
            decoder = DistanceDecoder(encoder.radius, 'GAT',
                                      self.r, self.temperature,
                                      encoder.hidden_dim, self.z_dim, 1)
        elif self.decoder_name == 'softmax':
            decoder = SoftmaxDecoder(encoder.radius, self.r)

        else:
            decoder = InnerProductDecoder()
        super(VGAE, self).__init__(encoder, decoder=decoder)

    def reparametrize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.sum_log_det_jac = 0
        node_feats, edge_index = args[0], args[1]
        self.__mu__, self.__logvar__ = self.encoder(*args, **kwargs)
        self.__std__ = self.__logvar__.mul(0.5).exp_()
        self.mask = (self.__logvar__.sum(dim=-1) != 0).int().unsqueeze(1)
        z = self.mask * self.reparametrize(self.__mu__, self.__std__)
        if self.encoder.flow_model:
            self.encoder.flow_model.base_dist_mean = self.__mu__
            self.encoder.flow_model.base_dist_var = self.__logvar__
            z_k, sum_log_det_jac = self.encoder.flow_model.inverse(z, edge_index)
            self.sum_log_det_jac = sum_log_det_jac
            self.z_k = z_k
        else:
            self.z_k = z
            z_k = z

        if self.deterministic:
            return self.__mu__, z_k

        return z, z_k

    def kl_loss(self, z_0, z_k, mu=None, std=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`std`, or based on latent variables from last encoding.
        Args:
            z_0 (Tensor): The latent sample prior to going through the flow
            z_k (Tensor): The latent sample after the flow. If there is no flow
            then z_0 == z_k
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            std (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        base_mu = self.__mu__ if mu is None else mu
        base_std = self.__std__ if std is None else std
        base_logvar = self.__logvar__ if std is None else torch.log(std)
        if self.encoder.analytic_kl:
            masked_ones = self.mask * torch.ones_like(base_logvar)
            kld = -0.5 * torch.sum(masked_ones + self.mask*base_logvar -
                                   self.mask*base_mu.pow(2) -
                                   self.mask*base_logvar.exp(), dim=1)
            mean_kld = kld.sum() / self.mask.sum()
            return mean_kld
        q_z_0_x = self.qz_x(base_mu, base_std)
        p_z_k = self.p_z(torch.zeros_like(base_mu, device=self.dev),
                         torch.ones_like(base_std))
        log_q_z_0_x = self.mask * q_z_0_x.log_prob(z_0)
        log_p_z_k = self.mask * p_z_k.log_prob(z_k)
        kld = log_q_z_0_x - log_p_z_k
        kld = self.mask * (kld - self.sum_log_det_jac)
        mean_kld = kld.sum() / self.mask.sum()
        return mean_kld

    def save(self, fn_enc):
        torch.save(self.encoder.state_dict(), fn_enc)

    def load(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc))

    def load_no_flow(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc),strict=False)
        pretrained_dict = self.encoder.state_dict()
        filtered_dict = filter_state_dict(pretrained_dict, "flow")
        pretrained_dict.update(filtered_dict)




