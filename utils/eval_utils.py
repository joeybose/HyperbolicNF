##############################################################################################
#
# Some code is adapted from https://github.com/lrjconan/GRAN/blob/master/utils/eval_helper.py
#
#############################################################################################
import os
import copy
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

from random import shuffle
from datetime import datetime
from scipy.linalg import eigvalsh
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv

PRINT_TIME = False
__all__ = [
    'clean_graphs', 'degree_stats', 'clustering_stats', 'orbit_stats_all', 'spectral_stats',
    'eval_acc_lobster_graph'
]

def evaluate_generated(graph_gt, graph_pred, dataset='prufer',
                       degree_only=False):

    mmd_degree = degree_stats(graph_gt, graph_pred)
    mmd_4orbits = -1.0
    mmd_clustering = -1.0
    mmd_spectral = -1.0
    accuracy = 0

    if degree_only:
        mmd_4orbits = 0.0
        mmd_clustering = 0.0
        mmd_spectral = 0.0
    else:
        try:
            mmd_spectral = spectral_stats(graph_gt, graph_pred)
            mmd_clustering = clustering_stats(graph_gt, graph_pred)
            mmd_4orbits = orbit_stats_all(graph_gt, graph_pred)
        except:
            pass

    if dataset == 'lobster':
        accuracy = eval_acc_lobster_graph(graph_pred)
    else:
        accuracy = sum([nx.is_tree(G) for G in graph_pred]) / float(len(graph_pred))
    return mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral, accuracy

def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(G_list))

def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not
    Removing leaf nodes twice:
    lobster -> caterpillar -> path
    """
    ### Check if G is a tree
    if nx.is_tree(G):
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False

def find_nearest_idx(array, value):
  idx = (np.abs(array - value)).argmin()
  return idx


def clean_graphs(graph_real, graph_pred, npr=None):
  ''' Selecting graphs generated that have the similar sizes.
    It is usually necessary for GraphRNN-S version, but not the full GraphRNN model.
    '''

  if npr is None:
    shuffle(graph_real)
    shuffle(graph_pred)
  else:
    npr.shuffle(graph_real)
    npr.shuffle(graph_pred)

  # get length
  real_graph_len = np.array(
      [len(graph_real[i]) for i in range(len(graph_real))])
  pred_graph_len = np.array(
      [len(graph_pred[i]) for i in range(len(graph_pred))])

  # select pred samples
  # The number of nodes are sampled from the similar distribution as the training set
  pred_graph_new = []
  pred_graph_len_new = []
  for value in real_graph_len:
    pred_idx = find_nearest_idx(pred_graph_len, value)
    pred_graph_new.append(graph_pred[pred_idx])
    pred_graph_len_new.append(pred_graph_len[pred_idx])

  return graph_real, pred_graph_new


def degree_worker(G):
  return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
  ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
  sample_ref = []
  sample_pred = []
  # in case an empty graph is generated
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for deg_hist in executor.map(degree_worker, graph_ref_list):
        sample_ref.append(deg_hist)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
        sample_pred.append(deg_hist)
  else:
    for i in range(len(graph_ref_list)):
      degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
      sample_ref.append(degree_temp)
    for i in range(len(graph_pred_list_remove_empty)):
      degree_temp = np.array(
          nx.degree_histogram(graph_pred_list_remove_empty[i]))
      sample_pred.append(degree_temp)

  mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing degree mmd: ', elapsed)
  return mmd_dist

###############################################################################

def spectral_worker(G):
  eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
  spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
  spectral_pmf = spectral_pmf / spectral_pmf.sum()
  return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
  ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
  sample_ref = []
  sample_pred = []
  # in case an empty graph is generated
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_ref_list):
        sample_ref.append(spectral_density)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        sample_pred.append(spectral_density)
  else:
    for i in range(len(graph_ref_list)):
      spectral_temp = spectral_worker(graph_ref_list[i])
      sample_ref.append(spectral_temp)
    for i in range(len(graph_pred_list_remove_empty)):
      spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
      sample_pred.append(spectral_temp)
  mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing degree mmd: ', elapsed)
  return mmd_dist

###############################################################################

def clustering_worker(param):
  G, bins = param
  clustering_coeffs_list = list(nx.clustering(G).values())
  hist, _ = np.histogram(
      clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
  return hist


def clustering_stats(graph_ref_list,
                     graph_pred_list,
                     bins=100,
                     is_parallel=True):
  sample_ref = []
  sample_pred = []
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for clustering_hist in executor.map(clustering_worker,
                                          [(G, bins) for G in graph_ref_list]):
        sample_ref.append(clustering_hist)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for clustering_hist in executor.map(
          clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
        sample_pred.append(clustering_hist)
  else:
    for i in range(len(graph_ref_list)):
      clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
      hist, _ = np.histogram(
          clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
      sample_ref.append(hist)

    for i in range(len(graph_pred_list_remove_empty)):
      clustering_coeffs_list = list(
          nx.clustering(graph_pred_list_remove_empty[i]).values())
      hist, _ = np.histogram(
          clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
      sample_pred.append(hist)

  mmd_dist = compute_mmd(
      sample_ref,
      sample_pred,
      kernel=gaussian_tv,
      sigma=1.0 / 10)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing clustering mmd: ', elapsed)
  return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
  idx = 0
  id2idx = dict()
  for u in G.nodes():
    id2idx[str(u)] = idx
    idx += 1

  edges = []
  for (u, v) in G.edges():
    edges.append((id2idx[str(u)], id2idx[str(v)]))
  return edges


def orca(graph):
  tmp_fname = 'utils/orca/tmp.txt'
  f = open(tmp_fname, 'w')
  f.write(
      str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
  for (u, v) in edge_list_reindexed(graph):
    f.write(str(u) + ' ' + str(v) + '\n')
  f.close()

  output = sp.check_output(
      ['./utils/orca/orca', 'node', '4', 'utils/orca/tmp.txt', 'std'])
  output = output.decode('utf8').strip()

  idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
  output = output[idx:]
  node_orbit_counts = np.array([
      list(map(int,
               node_cnts.strip().split(' ')))
      for node_cnts in output.strip('\n').split('\n')
  ])

  try:
    os.remove(tmp_fname)
  except OSError:
    pass

  return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
  total_counts_ref = []
  total_counts_pred = []

  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  for G in graph_ref_list:
    try:
      orbit_counts = orca(G)
    except:
      continue
    orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
    total_counts_ref.append(orbit_counts_graph)

  for G in graph_pred_list:
    try:
      orbit_counts = orca(G)
    except:
      continue
    orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
    total_counts_pred.append(orbit_counts_graph)

  total_counts_ref = np.array(total_counts_ref)
  total_counts_pred = np.array(total_counts_pred)

  mmd_dist = compute_mmd(
      total_counts_ref,
      total_counts_pred,
      kernel=gaussian_tv,
      is_hist=False,
      sigma=30.0)

  return mmd_dist


def eval_acc_lobster_graph(G_list):
  G_list = [copy.deepcopy(gg) for gg in G_list]

  count = 0
  for gg in G_list:
    if is_lobster_graph(gg):
      count += 1

  return count / float(len(G_list))


def is_lobster_graph(G):
  """
    Check a given graph is a lobster graph or not
    Removing leaf nodes twice:
    lobster -> caterpillar -> path
  """
  ### Check if G is a tree
  if nx.is_tree(G):
    # import pdb; pdb.set_trace()
    ### Check if G is a path after removing leaves twice
    leaves = [n for n, d in G.degree() if d == 1]
    G.remove_nodes_from(leaves)

    leaves = [n for n, d in G.degree() if d == 1]
    G.remove_nodes_from(leaves)

    num_nodes = len(G.nodes())
    num_degree_one = [d for n, d in G.degree() if d == 1]
    num_degree_two = [d for n, d in G.degree() if d == 2]

    if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
      return True
    elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
      return True
    else:
      return False
  else:
    return False

