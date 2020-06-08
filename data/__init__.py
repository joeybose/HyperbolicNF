from typing import Any
import argparse
from .image_reconstruction import CifarVaeDataset, MnistVaeDataset, OmniglotVaeDataset
from .synthetic import BdpVaeDataset
from .vae_dataset import VaeDataset
from .graph_datasets import *

__all__ = [
    "BdpVaeDataset",
    "CifarVaeDataset",
    "MnistVaeDataset",
    "OmniglotVaeDataset",
    "VaeDataset",
    "create_dataset",
    "CoraDataset",
    "PubmedDataset",
    "PPIDataset",
    "NXDataset",
    "CsphdDataset",
    "PhyloDataset",
    "DiseasesDataset",
    "WordnetDataset"
]


def create_dataset(arg_parse: argparse.Namespace, dataset_type: str, *args: Any, **kwargs: Any) -> VaeDataset:
    dataset_type = dataset_type.lower()
    if dataset_type == "bdp":
        arg_parse.input_dim = 50
        return BdpVaeDataset(*args, **kwargs)
    elif dataset_type == "mnist":
        arg_parse.input_dim = 784
        arg_parse.log_freq = 5
        return MnistVaeDataset(*args, **kwargs)
    elif dataset_type == "omniglot":
        return OmniglotVaeDataset(*args, **kwargs)
    elif dataset_type == "cifar":
        return CifarVaeDataset(*args, **kwargs)
    elif dataset_type == "pbt":
        arg_parse.input_dim = 255
        return PbtVaeDataset(*args, **kwargs)
    elif dataset_type == "cora":
        dataset = CoraDataset(dataset_type)
        arg_parse.input_dim = dataset.num_features
        return dataset
    elif dataset_type == "pubmed":
        dataset = PubmedDataset(dataset_type)
        arg_parse.input_dim = dataset.num_features
        return dataset
    elif dataset_type == "ppi":
        #TODO: This currently doesn't work
        dataset = PPIDataset(dataset_type)
        arg_parse.input_dim = dataset.num_features
        return dataset
    elif dataset_type == "disease_lp":
        dataset = DiseasesLPDataset(dataset_type, arg_parse.val_edge_ratio,
                                    arg_parse.test_edge_ratio,
                                    arg_parse.normalize_adj,
                                    arg_parse.normalize_feats)
        arg_parse.input_dim = dataset.num_features
        return dataset
    elif dataset_type in ["lobster", "grid", "prufer"]:
        dataset = NXDataset(dataset_type, arg_parse.batch_size,
                            arg_parse.num_fixed_features, arg_parse.node_order,
                            arg_parse.use_rand_feats, arg_parse.seed)
        arg_parse.input_dim = dataset.num_features
        arg_parse.num_features = dataset.num_features
        arg_parse.node_dist = dataset.node_dist
        return dataset
    elif dataset_type == "csphd":
        dataset = CsphdDataset(dataset_type)
        arg_parse.input_dim = dataset.num_features
        return dataset
    elif dataset_type == "phylo":
        dataset = PhyloDataset(dataset_type)
        arg_parse.input_dim = dataset.num_features
        return dataset
    elif dataset_type == "diseases":
        dataset = DiseasesDataset(dataset_type)
        arg_parse.input_dim = dataset.num_features
        return dataset
    # elif dataset_type == "wordnet":
    elif "wordnet" in dataset_type:
        wordnet, dataset_name = dataset_type.split('-')
        closure_file = "/" + dataset_name + "_closure.csv"
        dataset_type = wordnet + closure_file
        dataset = WordnetDataset(dataset_type, arg_parse.num_fixed_features,
                                 arg_parse.use_rand_feats)
        arg_parse.input_dim = dataset.num_features
        return dataset
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
