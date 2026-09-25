from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import vfl_paths

_DEFAULT_DATA_ROOT = str(vfl_paths.resolve().nih_root)


@dataclass(frozen=True)
class CFG:
    DATA_ROOT: str = _DEFAULT_DATA_ROOT
    CSV_NAME: str = "Data_Entry_2017.csv"
    BATCH_SIZE: int = 32
    TEST_BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 3
    LR: float = 1e-4
    EMBED_DIM: int = 64
    IMAGE_SIZE: int = 224
    TASK_LABEL: str = "Infiltration"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42


def _build_image_index(root: Path) -> dict[str, Path]:
    image_paths = {}
    for path in root.rglob("*.png"):
        image_paths[path.name] = path
    if not image_paths:
        raise FileNotFoundError(
            f"No PNG images were found under {root}. Place the NIH Chest X-ray14 images under this folder."
        )
    return image_paths


def _load_metadata(root: Path, task_label: str) -> pd.DataFrame:
    csv_path = root / CFG.CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {CFG.CSV_NAME} in {root}. Put the official NIH metadata CSV in that folder."
        )

    image_index = _build_image_index(root)
    df = pd.read_csv(csv_path)
    df = df[df["Image Index"].isin(image_index)].copy()
    if df.empty:
        raise RuntimeError("Metadata loaded, but none of the listed images were found under the dataset folder.")

    def to_target(label_text: str) -> int:
        labels = {item.strip() for item in str(label_text).split("|")}
        if task_label == "No Finding":
            return int(labels == {"No Finding"})
        return int(task_label in labels)

    df["target"] = df["Finding Labels"].apply(to_target)
    df["image_path"] = df["Image Index"].map(lambda name: str(image_index[name]))
    return df[["image_path", "target"]]


class VerticalNIHDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_size: int = CFG.IMAGE_SIZE):
        self.rows = rows.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image = Image.open(row["image_path"]).convert("L")
        image = self.transform(image)
        left = image[:, :, : image.shape[2] // 2]
        right = image[:, :, image.shape[2] // 2 :]
        label = torch.tensor(row["target"], dtype=torch.float32)
        return left, right, label


def get_nih_loaders(
    root: str = CFG.DATA_ROOT,
    task_label: str = CFG.TASK_LABEL,
    batch_size: int = CFG.BATCH_SIZE,
    test_batch_size: int = CFG.TEST_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    root_path = Path(root)
    rows = _load_metadata(root_path, task_label=task_label)

    train_rows, test_rows = train_test_split(
        rows,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE,
        stratify=rows["target"],
    )

    train_ds = VerticalNIHDataset(train_rows)
    test_ds = VerticalNIHDataset(test_rows)
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
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ServerBinaryClassifier(nn.Module):
    def __init__(self, embed_dim: int = CFG.EMBED_DIM):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, left_embed: torch.Tensor, right_embed: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([left_embed, right_embed], dim=1)
        joined = F.relu(self.fc1(joined))
        return self.fc2(joined).squeeze(1)
