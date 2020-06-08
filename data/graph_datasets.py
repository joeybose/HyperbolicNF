import os
import os.path as osp
import torch
import numpy as np

import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data,Dataset,DataLoader,DataListLoader
from torch_geometric.utils import remove_self_loops
from typing import Any, List, Tuple
import urllib
from random import shuffle
from torch_geometric.datasets import Planetoid,PPI,TUDataset
from .vae_dataset import VaeDataset
from .data_utils import *


class CoraDataset:
    def __init__(self, name):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        self.dataset = Planetoid(path, "Cora", T.NormalizeFeatures())
        data = self.dataset[0]
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        self.num_features = self.dataset.num_features
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.dataset, self.dataset

class PubmedDataset:
    def __init__(self, name):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        self.dataset = Planetoid(path, "PubMed", T.NormalizeFeatures())
        self.num_features = self.dataset.num_features
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.dataset, self.dataset

class NXDataset:
    def __init__(self, name, train_batch_size, num_fixed_features, node_order,
                 use_rand_feats, seed, train_ratio=0.8):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        nx_dataset, self.max_nodes, self.max_edges, self.node_dist = create_nx_graphs(name,seed)
        self.train_batch_size = train_batch_size
        self.feats = 0.3*torch.randn(self.max_nodes,num_fixed_features,requires_grad=False)
        self.reconstruction_loss = None
        self.dataset = []
        for nx_graph in nx_dataset:
            num_nodes = len(nx_graph.nodes)
            nodes_to_pad = self.max_nodes - num_nodes
            perm = torch.randperm(self.feats.size(0))
            perm_idx = perm[:num_nodes]
            feats = self.feats[perm_idx]
            adj_mat = torch.Tensor(nx.to_numpy_matrix(nx_graph))
            col_zeros = torch.zeros(num_nodes, nodes_to_pad)
            adj_mat = torch.cat((adj_mat, col_zeros),dim=1)
            row_zeros = torch.zeros(nodes_to_pad, self.max_nodes)
            adj_mat = torch.cat((adj_mat, row_zeros),dim=0)
            edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            if use_rand_feats:
                self.num_features = num_fixed_features
                self.dataset.append(Data(edge_index=edge_index, x=feats))
            else:
                self.dataset.append(Data(edge_index=edge_index, x=adj_mat))
                self.num_features = self.max_nodes

        train_cutoff = int(np.round(train_ratio*len(self.dataset)))
        self.train_dataset = self.dataset[:train_cutoff]
        self.test_dataset = self.dataset[train_cutoff:]

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataListLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=False)
        test_loader = DataListLoader(self.test_dataset, batch_size=1, shuffle=False)
        return train_loader, test_loader

class PPIDataset:
    def __init__(self, name):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        self.path = path
        self.train_dataset = PPI(self.path, split='train',transform=T.NormalizeFeatures())
        self.test_dataset = PPI(self.path, split='test',transform=T.NormalizeFeatures())
        self.num_features = self.train_dataset.num_features
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataListLoader(self.train_dataset, batch_size=args.train_batch_size, shuffle=False)
        test_loader = DataListLoader(self.test_dataset, batch_size=args.test_batch_size, shuffle=False)
        return train_loader,test_loader

class DiseasesLPDataset:
    def __init__(self, name, val_prop, test_prop, normalize_adj, normalize_feats):
        self.path = osp.join(
            osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        G, features = load_data(self.path, name, val_prop, test_prop, normalize_adj,
                         normalize_feats)
        self.num_features = features.shape[1]
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        edge_index, _ = remove_self_loops(edge_index)
        self.dataset = Data(edge_index=edge_index, x=features)
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return [self.dataset], [self.dataset]

class CsphdDataset:
    def __init__(self, name):
        # Write path
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        adj, features, edge_index = load_data_network(name)
        self.num_features = features.shape[0]
        self.dataset = Data(edge_index=edge_index, x=features)
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return [self.dataset], [self.dataset]

class PhyloDataset:
    def __init__(self, name):
        # Write path
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        adj, features, edge_index = load_data_network(name)
        self.num_features = features.shape[0]
        self.dataset = Data(edge_index=edge_index, x=features)
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return [self.dataset], [self.dataset]

class DiseasesDataset:
    def __init__(self, name):
        # Write path
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        adj, features, edge_index = load_data_network(name)
        self.num_features = features.shape[0]
        self.dataset = Data(edge_index=edge_index, x=features)
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return [self.dataset], [self.dataset]

class WordnetDataset:
    def __init__(self, name, num_fixed_features, use_rand_feats):
        # Write path
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        self.edges, self.objects, self.weights = load_edge_list(self.path, False)
        num_nodes = len(self.objects)
        self.feats = torch.randn(num_nodes,num_fixed_features,requires_grad=False)
        perm = torch.randperm(self.feats.size(0))
        perm_idx = perm[:num_nodes]
        feats = self.feats[perm_idx]
        G = nx.Graph()
        G.add_edges_from(self.edges)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        edge_index, _ = remove_self_loops(edge_index)
        if use_rand_feats:
            self.num_features = num_fixed_features
            self.dataset = Data(edge_index=edge_index, x=feats)
        else:
            adj_mat = torch.Tensor(nx.to_numpy_matrix(G))
            self.dataset = Data(edge_index=edge_index, x=adj_mat)
            self.num_features = num_nodes
        self.reconstruction_loss = None

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return [self.dataset], [self.dataset]


