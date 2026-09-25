from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
    ATTR_DIM: int = 1
    IMAGE_SIZE: int = 64


CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def attr_index(name: str) -> int:
    return CELEBA_ATTRS.index(name)


def get_sensitive_attribute(target_vec: torch.Tensor) -> torch.Tensor:
    idx = attr_index("Eyeglasses")
    return (target_vec[:, idx] > 0).float().unsqueeze(1)


def get_task_label(target_vec: torch.Tensor) -> torch.Tensor:
    idx = attr_index("Smiling")
    return (target_vec[:, idx] > 0).float()


class VerticalCelebADataset(Dataset):
    def __init__(self, root: str, split: str):
        self.root = Path(root)
        self.rows = _load_celeba_rows(self.root, split=split)
        transform = transforms.Compose(
            [
                transforms.Resize(CFG.IMAGE_SIZE),
                transforms.CenterCrop(CFG.IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        attr_tensor = torch.tensor(row[CELEBA_ATTRS].to_numpy(dtype=np.float32))
        sensitive = get_sensitive_attribute(attr_tensor.unsqueeze(0)).squeeze(0)
        label = get_task_label(attr_tensor.unsqueeze(0)).squeeze(0)
        return image, sensitive, label


def get_celeba_loaders(
    root: str = CFG.DATA_ROOT,
    batch_size: int = CFG.BATCH_SIZE,
    test_batch_size: int = CFG.TEST_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    train_ds = VerticalCelebADataset(root=root, split="train")
    test_ds = VerticalCelebADataset(root=root, split="test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = CFG.EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class AttributeParty(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ServerHead(nn.Module):
    def __init__(self, embed_dim: int = CFG.EMBED_DIM, attr_dim: int = CFG.ATTR_DIM):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim + attr_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, left_embed: torch.Tensor, right_attr: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([left_embed, right_attr], dim=1)
        joined = F.relu(self.fc1(joined))
        return self.fc2(joined).squeeze(1)


def _load_celeba_rows(root: Path, split: str) -> pd.DataFrame:
    celeba_root = root / "celeba" if (root / "celeba").exists() else root
    attrs = _read_celeba_table(celeba_root, "list_attr_celeba")
    partitions = _read_celeba_table(celeba_root, "list_eval_partition")

    image_dir_candidates = [
        celeba_root / "img_align_celeba" / "img_align_celeba",
        celeba_root / "img_align_celeba",
    ]
    image_dir = next((path for path in image_dir_candidates if path.exists()), None)
    if image_dir is None:
        raise FileNotFoundError(f"Could not find CelebA image directory under {celeba_root}.")

    partition_map = {"train": 0, "valid": 1, "test": 2}
    merged = attrs.merge(partitions, on="image_id")
    merged = merged[merged["partition"] == partition_map[split]].copy()
    merged["image_path"] = merged["image_id"].map(lambda name: str(image_dir / name))
    merged = merged[merged["image_path"].map(_is_readable_path)].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"No readable CelebA images were found for split '{split}' under {image_dir}.")
    return merged


def _read_celeba_table(root: Path, stem: str) -> pd.DataFrame:
    csv_path = root / f"{stem}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    txt_path = root / f"{stem}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing {stem}.csv or {stem}.txt in {root}.")

    if stem == "list_eval_partition":
        return pd.read_csv(txt_path, sep=r"\s+", names=["image_id", "partition"], engine="python")

    return pd.read_csv(
        txt_path,
        sep=r"\s+",
        skiprows=2,
        names=["image_id", *CELEBA_ATTRS],
        engine="python",
    )


def _is_readable_path(path_str: str) -> bool:
    path = Path(path_str)
    if not (path.exists() and os.access(path, os.R_OK)):
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, PermissionError):
        return False
