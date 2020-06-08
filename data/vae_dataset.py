from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader


class VaeDataset:

    def __init__(self, batch_size: int, in_dim: int, img_dims: Optional[Tuple[int, ...]]) -> None:
        self.batch_size = batch_size
        self._in_dim = in_dim
        self._img_dims = img_dims

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    @property
    def img_dims(self) -> Optional[Tuple[int, ...]]:
        return self._img_dims

    @property
    def in_dim(self) -> int:
        return self._in_dim

    def metrics(self, x_mb_: torch.Tensor, mode: str = "train") -> Dict[str, float]:
        return {}
