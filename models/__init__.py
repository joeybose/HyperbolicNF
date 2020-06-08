from typing import Any
from .euclidean_vae import FeedForwardVAE, CifarVAE
from .hyperbolic_vae import FeedForwardHyperboloidVAE, CifarHyperboloidVAE
from .euclidean_vgae import Encoder, GAE, VGAE
from .hyperbolic_vgae import HyperboloidEncoder, HyperboloidGAE, HyperboloidVGAE


__all__ = [
    "FeedForwardVAE",
    "FeedForwardHyperboloidVAE",
    "CifarVAE",
    "CifarHyperboloidVAE",
    "VGAE",
    "HyperboloidVGAE"
]

def create_model(model_type: str, deterministic: bool, enc_blocks: int,
                 conv_type: str, dataset_type: str, decoder: str, r: float,
                 temperature: float, *args: Any, **kwargs: Any):
    if dataset_type in ["mnist", "bdp", "pbt"]:
        if model_type == 'euclidean':
            return FeedForwardVAE(*args, **kwargs)
        else:
            return FeedForwardHyperboloidVAE(*args, **kwargs)
    elif dataset_type in ["cora", "pubmed", "ppi", "disease_lp", "lobster",
                          "grid", "prufer", "csphd", "phylo", "diseases",
                          "wordnet-noun", "wordnet-mammal"]:
        if model_type == 'euclidean':
            return VGAE(Encoder(deterministic, enc_blocks, conv_type, *args,
                                **kwargs), decoder, r, temperature)
        else:
            return HyperboloidVGAE(HyperboloidEncoder(deterministic,
                                                      enc_blocks, conv_type,
                                                      *args, **kwargs),
                                   decoder, r, temperature)
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
