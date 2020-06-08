import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def draw_nx_graph(G, name='Lobster', path='./visualization/plots/'):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title(name, fontsize=10)
    nx.draw(G)
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path + name + '.png'
    plt.savefig(save_name, format="PNG")
    plt.close()

def draw_pyg_graph(G, name='Lobster', path='./visualization/plots/'):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title(name, fontsize=10)
    nx_graph = to_networkx(G)
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path + name + '.png'
    nx.draw(nx_graph)
    plt.savefig(save_name, format="PNG")
    plt.close()
