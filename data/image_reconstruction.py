from typing import Tuple, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributions
from torch import Tensor
from .vae_dataset import VaeDataset

class ToDefaultTensor(transforms.Lambda):

    def __init__(self) -> None:
        super().__init__(lambda x: x.to(torch.get_default_dtype()))


def flatten_transform(img: torch.Tensor) -> torch.Tensor:
    return img.view(-1)


# TODO: Move this to Utils Later
class VaeDistribution:

    def rsample_with_parts(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        z = self.rsample(shape)
        return z, None

    def log_prob_from_parts(self, z: Tensor, data: Optional[Tuple[Tensor, ...]]) -> Tensor:
        log_prob = self.log_prob(z)
        assert torch.isfinite(log_prob).all()
        return log_prob

    def rsample_log_prob(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tensor]:
        z, data = self.rsample_with_parts(shape)
        return z, self.log_prob_from_parts(z, data)


class EuclideanNormal(torch.distributions.Normal, VaeDistribution):

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value).sum(dim=-1)


class EuclideanUniform(torch.distributions.Uniform, VaeDistribution):

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value).sum(dim=-1)


class ImageDynamicBinarization:

    def __init__(self, train: bool, invert: bool = False) -> None:
        self.uniform = EuclideanUniform(0, 1)
        self.train = train
        self.invert = invert

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1,))  # Reshape per element, not batched yet.
        if self.invert:
            x = 1 - x
        if self.train:
            x = x > self.uniform.sample(x.shape)  # dynamic binarization
        else:
            x = x > 0.5  # fixed binarization for eval
        x = x.to(torch.get_default_dtype())
        return x


class MnistVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 1, 28, 28), in_dim=784)
        self.data_folder = data_folder

    def _get_dataset(self, train: bool, download: bool, transform: Any) -> torch.utils.data.Dataset:
        return datasets.MNIST(self.data_folder, train=train, download=download, transform=transform)

    def _load_mnist(self, train: bool, download: bool) -> DataLoader:
        # transformation = transforms.Compose(
            # [transforms.ToTensor(),
             # ToDefaultTensor(),
             # transforms.Lambda(ImageDynamicBinarization(train=train))])
        transformation = transforms.Compose(
            [transforms.ToTensor(),
             ToDefaultTensor(),
             ImageDynamicBinarization(train=train)])
        return DataLoader(dataset=self._get_dataset(train, download, transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_mnist(train=True, download=False)
        test_loader = self._load_mnist(train=False, download=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class OmniglotVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 1, 28, 28), in_dim=784)
        self.data_folder = data_folder

    def _get_dataset(self, train: bool, transform: Any) -> torch.utils.data.Dataset:
        return datasets.Omniglot(self.data_folder, background=train, download=False, transform=transform)

    def _load_omniglot(self, train: bool) -> DataLoader:
        # transformation = transforms.Compose([
            # transforms.Resize((28, 28)),
            # transforms.ToTensor(),
            # ToDefaultTensor(),
            # transforms.Lambda(ImageDynamicBinarization(train=train, invert=True))
        # ])
        transformation = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            ToDefaultTensor(),
            ImageDynamicBinarization(train=train, invert=True)
        ])
        return DataLoader(dataset=self._get_dataset(train, transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_omniglot(train=True)
        test_loader = self._load_omniglot(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class CifarVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 3, 32, 32), in_dim=3072)
        self.data_folder = data_folder

    def _load_cifar(self, train: bool) -> DataLoader:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            ToDefaultTensor(),
            transforms.Lambda(flatten_transform),
        ])
        return DataLoader(dataset=datasets.CIFAR10(self.data_folder,
                                                   train=train,
                                                   download=False,
                                                   transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_cifar(train=True)
        test_loader = self._load_cifar(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")
