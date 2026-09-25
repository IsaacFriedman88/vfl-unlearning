from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import vfl_paths

_DEFAULT_DATA_ROOT = str(vfl_paths.resolve().data_root)


@dataclass(frozen=True)
class CFG:
    DATA_ROOT: str = _DEFAULT_DATA_ROOT
    BATCH_SIZE: int = 128
    TEST_BATCH_SIZE: int = 256
    NUM_EPOCHS: int = 3
    LR: float = 1e-3
    EMBED_DIM: int = 64
    NUM_CLASSES: int = 10
    IMAGE_SIZE: int = 28


class VerticalMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, download: bool = False):
        transform = transforms.Compose([transforms.ToTensor()])
        self.base = datasets.MNIST(root=root, train=train, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        left = image[:, :, :14]
        right = image[:, :, 14:]
        return left, right, label


def get_mnist_loaders(
    root: str = CFG.DATA_ROOT,
    batch_size: int = CFG.BATCH_SIZE,
    test_batch_size: int = CFG.TEST_BATCH_SIZE,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds = VerticalMNISTDataset(root=root, train=True, download=download)
    test_ds = VerticalMNISTDataset(root=root, train=False, download=download)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


class PartialImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = CFG.EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ServerClassifier(nn.Module):
    def __init__(self, embed_dim: int = CFG.EMBED_DIM, num_classes: int = CFG.NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, left_embed: torch.Tensor, right_embed: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([left_embed, right_embed], dim=1)
        joined = F.relu(self.fc1(joined))
        return self.fc2(joined)
