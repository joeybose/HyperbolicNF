"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
from _csv import reader


import pandas
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import os.path as osp
sys.path.append("..")  # Adds higher directory to python modules path.
from visualization.utils import draw_nx_graph

### Data Loading taken from: https://github.com/HazyResearch/hgcn/blob/master/utils/data_utils.py
def load_data(datapath, dataset_name, val_prop, test_prop, normalize_adj,
              normalize_feats, use_feats=True):
    data = load_data_lp(dataset_name, use_feats, datapath)
    adj = data['adj_train']
    unnormed_feats = data['features']
    adj_mat, features = process(adj,unnormed_feats,normalize_adj,normalize_feats)
    G = nx.from_numpy_matrix(adj_mat)
    return G, features


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj.todense(), features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels

def get_graph(adj):
    """ get a graph from zero-padded adj """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G

def create_node_ordered_graph(G, node_order='DFS'):
    node_degree_list = [(n, d) for n, d in G.degree()]
    adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### BFS & DFS from largest-degree node
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
        node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)

        bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
        dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

        node_list_bfs += list(bfs_tree.nodes())
        node_list_dfs += list(dfs_tree.nodes())

        adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
        adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

        if node_order == 'degree_decent':
            adj_list = [adj_1]
        elif node_order == 'degree_accent':
            adj_list = [adj_2]
        elif node_order == 'BFS':
            adj_list = [adj_3]
        elif node_order == 'DFS':
            adj_list = [adj_4]
        else:
            adj_list = [adj_0]

    return adj_list



def create_nx_graphs(graph_type, node_order='DFS', seed=1234):
    # Taken from GRAN: https://github.com/lrjconan/GRAN/blob/master/utils/data_helper.py
    npr = np.random.RandomState(seed)
    ### load datasets
    graphs = []
    # synthetic graphs
    if graph_type == 'grid':
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
    elif graph_type == 'lobster':
        graphs = []
        p1 = 0.7
        p2 = 0.7
        count = 0
        min_node = 20
        max_node = 100
        max_edge = 0
        mean_node = 80
        num_graphs = 100

        seed_tmp = seed
        while count < num_graphs:
            G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
            if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
                adj_mat = create_node_ordered_graph(G, node_order)[0]
                G = nx.from_numpy_matrix(adj_mat)
                graphs.append(G)
                draw_nx_graph(G,"Lobster_" + str(count))
                if G.number_of_edges() > max_edge:
                  max_edge = G.number_of_edges()

                count += 1

            seed_tmp += 1
    elif graph_type == 'prufer':
        graphs = []
        count = 0
        min_node = 20
        max_node = 100
        max_edge = 0
        num_graphs = 100
        save_path = './visualization/plots/Prufer/'
        seed_tmp = seed
        while count < num_graphs:
            num_nodes = np.random.randint(min_node, max_node)
            G = nx.random_tree(num_nodes, seed=seed_tmp)
            if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
                adj_mat = create_node_ordered_graph(G, node_order)[0]
                G = nx.from_numpy_matrix(adj_mat)
                graphs.append(G)
                # draw_nx_graph(G,"Prufer_" + str(count), path=save_path)
                if G.number_of_edges() > max_edge:
                  max_edge = G.number_of_edges()

                count += 1

            seed_tmp += 1

    num_nodes = [gg.number_of_nodes() for gg in graphs]
    num_edges = [gg.number_of_edges() for gg in graphs]
    print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
    print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
    return graphs, max(num_nodes), max(num_edges), num_nodes


def load_data_network(dataset):
    file = open(osp.join(osp.dirname(osp.realpath(__file__)), '{}.csv'.format(dataset)), "r")
    lines = reader(file)
    edges = list(lines)
    edges = np.array(edges).astype(np.int)
    edges_unique = np.unique(edges.flatten())
    N = len(edges_unique)  # 5242
    sorted_idx = np.argsort(edges_unique)
    sorted_edges_unique = np.sort(edges_unique)
    map_idx = np.ones(np.max(edges_unique) + 1).astype(np.int) * -1
    map_idx[sorted_edges_unique] = sorted_idx

    row = map_idx[edges[:, 0]]
    col = map_idx[edges[:, 1]]
    data = np.ones(len(row))
    adj = sp.coo_matrix((data, (row, col)), shape=(N, N))
    edge_index = torch.tensor(np.array([row, col]))

    features = torch.eye(N)
    return adj, features, edge_index

def load_edge_list(path, symmetrize=True):
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights

def load_adjacency_matrix(path, symmetrize=False, objects=None):
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')

    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])

    idmap = {}
    idlist = []

    def convert(id):
        if id not in idmap:
            idmap[id] = len(idlist)
            idlist.append(id)
        return idmap[id]
    if objects is not None:
        objects = pandas.DataFrame.from_dict({'obj': objects, 'id': np.arange(len(objects))})
        df = df.merge(objects, left_on='id1', right_on='obj').merge(objects, left_on='id2', right_on='obj')
        df['id1'] = df['id_x']
        df['id2'] = df['id_y']
    else:
        df.loc[:, 'id1'] = df['id1'].apply(convert)
        df.loc[:, 'id2'] = df['id2'].apply(convert)
        objects = np.array(idlist)

    groups = df.groupby('id1').apply(lambda x: x.sort_values(by='id2'))
    counts = df.groupby('id1').id2.size()

    ids = groups.index.levels[0].values
    offsets = counts.loc[ids].values
    offsets[1:] = np.cumsum(offsets)[:-1]
    offsets[0] = 0
    neighbors = groups['id2'].values
    weights = groups['weight'].values
    return ids, objects, weights
