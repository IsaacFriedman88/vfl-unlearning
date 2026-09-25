from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


TaskType = Literal["multiclass", "binary"]


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    retain_train_loader: DataLoader
    forget_train_loader: DataLoader
    train_eval_loader: DataLoader
    retain_eval_loader: DataLoader
    forget_eval_loader: DataLoader
    forget_count: int
    retain_count: int


@dataclass(frozen=True)
class MetricSnapshot:
    train_loss: float
    train_acc: float
    retain_loss: float
    retain_acc: float
    forget_loss: float
    forget_acc: float
    test_loss: float
    test_acc: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}train_loss": self.train_loss,
            f"{prefix}train_acc": self.train_acc,
            f"{prefix}retain_loss": self.retain_loss,
            f"{prefix}retain_acc": self.retain_acc,
            f"{prefix}forget_loss": self.forget_loss,
            f"{prefix}forget_acc": self.forget_acc,
            f"{prefix}test_loss": self.test_loss,
            f"{prefix}test_acc": self.test_acc,
        }


@dataclass(frozen=True)
class AttackMetrics:
    attack_accuracy: float
    attack_precision: float
    attack_recall: float
    attack_f1: float
    attack_auc: float
    attack_threshold: float
    member_mean_loss: float
    nonmember_mean_loss: float
    member_mean_confidence: float
    nonmember_mean_confidence: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}attack_accuracy": self.attack_accuracy,
            f"{prefix}attack_precision": self.attack_precision,
            f"{prefix}attack_recall": self.attack_recall,
            f"{prefix}attack_f1": self.attack_f1,
            f"{prefix}attack_auc": self.attack_auc,
            f"{prefix}attack_threshold": self.attack_threshold,
            f"{prefix}member_mean_loss": self.member_mean_loss,
            f"{prefix}nonmember_mean_loss": self.nonmember_mean_loss,
            f"{prefix}member_mean_confidence": self.member_mean_confidence,
            f"{prefix}nonmember_mean_confidence": self.nonmember_mean_confidence,
        }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_unlearning_loaders(
    dataset: Dataset,
    *,
    batch_size: int,
    forget_fraction: float,
    seed: int,
) -> LoaderBundle:
    if not 0.0 < forget_fraction < 1.0:
        raise ValueError("forget_fraction must be between 0 and 1.")

    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Unlearning requires at least two training samples.")

    forget_count = min(max(1, int(round(dataset_size * forget_fraction))), dataset_size - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    forget_indices = permutation[:forget_count]
    retain_indices = permutation[forget_count:]

    retain_subset = Subset(dataset, retain_indices)
    forget_subset = Subset(dataset, forget_indices)

    return LoaderBundle(
        train_loader=_make_loader(dataset, batch_size=batch_size, shuffle=True, seed=seed),
        retain_train_loader=_make_loader(retain_subset, batch_size=batch_size, shuffle=True, seed=seed + 1),
        forget_train_loader=_make_loader(forget_subset, batch_size=batch_size, shuffle=True, seed=seed + 2),
        train_eval_loader=_make_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed),
        retain_eval_loader=_make_loader(retain_subset, batch_size=batch_size, shuffle=False, seed=seed),
        forget_eval_loader=_make_loader(forget_subset, batch_size=batch_size, shuffle=False, seed=seed),
        forget_count=forget_count,
        retain_count=len(retain_indices),
    )


def collect_metric_snapshot(
    *,
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    loaders: LoaderBundle,
    test_loader: DataLoader,
    device: torch.device,
    task_type: TaskType,
    max_eval_batches: int | None = None,
) -> MetricSnapshot:
    train_loss, train_acc = evaluate_model(
        left_party=left_party,
        right_party=right_party,
        server_head=server_head,
        loader=loaders.train_eval_loader,
        device=device,
        task_type=task_type,
        max_batches=max_eval_batches,
    )
    retain_loss, retain_acc = evaluate_model(
        left_party=left_party,
        right_party=right_party,
        server_head=server_head,
        loader=loaders.retain_eval_loader,
        device=device,
        task_type=task_type,
        max_batches=max_eval_batches,
    )
    forget_loss, forget_acc = evaluate_model(
        left_party=left_party,
        right_party=right_party,
        server_head=server_head,
        loader=loaders.forget_eval_loader,
        device=device,
        task_type=task_type,
        max_batches=max_eval_batches,
    )
    test_loss, test_acc = evaluate_model(
        left_party=left_party,
        right_party=right_party,
        server_head=server_head,
        loader=test_loader,
        device=device,
        task_type=task_type,
        max_batches=max_eval_batches,
    )
    return MetricSnapshot(
        train_loss=train_loss,
        train_acc=train_acc,
        retain_loss=retain_loss,
        retain_acc=retain_acc,
        forget_loss=forget_loss,
        forget_acc=forget_acc,
        test_loss=test_loss,
        test_acc=test_acc,
    )


def collect_attack_metrics(
    *,
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: torch.device,
    task_type: TaskType,
    max_eval_batches: int | None = None,
) -> AttackMetrics:
    member_losses, member_confidences = collect_per_sample_outputs(
        left_party=left_party,
        right_party=right_party,
        server_head=server_head,
        loader=member_loader,
        device=device,
        task_type=task_type,
        max_batches=max_eval_batches,
    )
    nonmember_losses, nonmember_confidences = collect_per_sample_outputs(
        left_party=left_party,
        right_party=right_party,
        server_head=server_head,
        loader=nonmember_loader,
        device=device,
        task_type=task_type,
        max_batches=max_eval_batches,
    )

    member_mean_loss = float(member_losses.mean().item())
    nonmember_mean_loss = float(nonmember_losses.mean().item())
    member_mean_confidence = float(member_confidences.mean().item())
    nonmember_mean_confidence = float(nonmember_confidences.mean().item())
    threshold = (member_mean_loss + nonmember_mean_loss) / 2.0

    member_preds = member_losses <= threshold
    nonmember_preds = nonmember_losses <= threshold
    tp = int(member_preds.sum().item())
    fn = int(member_preds.numel() - tp)
    fp = int(nonmember_preds.sum().item())
    tn = int(nonmember_preds.numel() - fp)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    auc = _roc_auc_from_scores(
        scores=torch.cat([-member_losses, -nonmember_losses]),
        labels=torch.cat(
            [
                torch.ones_like(member_losses, dtype=torch.int64),
                torch.zeros_like(nonmember_losses, dtype=torch.int64),
            ]
        ),
    )

    return AttackMetrics(
        attack_accuracy=accuracy,
        attack_precision=precision,
        attack_recall=recall,
        attack_f1=f1,
        attack_auc=auc,
        attack_threshold=threshold,
        member_mean_loss=member_mean_loss,
        nonmember_mean_loss=nonmember_mean_loss,
        member_mean_confidence=member_mean_confidence,
        nonmember_mean_confidence=nonmember_mean_confidence,
    )


def create_optimizer(
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    *,
    lr: float,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        list(left_party.parameters()) + list(right_party.parameters()) + list(server_head.parameters()),
        lr=lr,
    )


def train_epoch_with_loss_scale(
    *,
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: TaskType,
    loss_scale: float = 1.0,
    max_batches: int | None = None,
) -> tuple[float, float]:
    left_party.train()
    right_party.train()
    server_head.train()
    criterion = _criterion_for_task(task_type)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (left_x, right_x, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        left_x = left_x.to(device)
        right_x = right_x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        left_embed = left_party(left_x)
        right_embed = right_party(right_x)
        logits = server_head(left_embed, right_embed)
        loss = criterion(logits, labels)
        (loss * loss_scale).backward()
        optimizer.step()

        batch_total, batch_correct = _batch_metrics(logits=logits, labels=labels, task_type=task_type)
        total_loss += loss.item() * batch_total
        correct += batch_correct
        total += batch_total

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(
    *,
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: TaskType,
    max_batches: int | None = None,
) -> tuple[float, float]:
    left_party.eval()
    right_party.eval()
    server_head.eval()
    criterion = _criterion_for_task(task_type)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (left_x, right_x, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        left_x = left_x.to(device)
        right_x = right_x.to(device)
        labels = labels.to(device)

        left_embed = left_party(left_x)
        right_embed = right_party(right_x)
        logits = server_head(left_embed, right_embed)
        loss = criterion(logits, labels)

        batch_total, batch_correct = _batch_metrics(logits=logits, labels=labels, task_type=task_type)
        total_loss += loss.item() * batch_total
        correct += batch_correct
        total += batch_total

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def collect_per_sample_outputs(
    *,
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: TaskType,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_party.eval()
    right_party.eval()
    server_head.eval()

    losses: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []

    for batch_idx, (left_x, right_x, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        left_x = left_x.to(device)
        right_x = right_x.to(device)
        labels = labels.to(device)

        left_embed = left_party(left_x)
        right_embed = right_party(right_x)
        logits = server_head(left_embed, right_embed)
        batch_losses, batch_confidences = _per_sample_stats(logits=logits, labels=labels, task_type=task_type)
        losses.append(batch_losses.cpu())
        confidences.append(batch_confidences.cpu())

    if not losses:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    return torch.cat(losses), torch.cat(confidences)


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _criterion_for_task(task_type: TaskType) -> nn.Module:
    if task_type == "multiclass":
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()


def _per_sample_stats(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: TaskType,
) -> tuple[torch.Tensor, torch.Tensor]:
    if task_type == "multiclass":
        losses = nn.functional.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=1)
        confidences = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        return losses, confidences

    labels = labels.float()
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    confidences = torch.where(labels > 0.5, probs, 1.0 - probs)
    return losses, confidences


def _batch_metrics(logits: torch.Tensor, labels: torch.Tensor, task_type: TaskType) -> tuple[int, int]:
    if task_type == "multiclass":
        preds = logits.argmax(dim=1)
        total = labels.size(0)
        correct = (preds == labels).sum().item()
        return total, correct

    preds = (torch.sigmoid(logits) >= 0.5).float()
    total = labels.size(0)
    correct = (preds == labels).sum().item()
    return total, correct


def _roc_auc_from_scores(*, scores: torch.Tensor, labels: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.0

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    positive_mask = labels == 1
    positive_count = int(positive_mask.sum().item())
    negative_count = int((labels == 0).sum().item())
    if positive_count == 0 or negative_count == 0:
        return 0.0

    positive_rank_sum = float(ranks[positive_mask].sum().item())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return auc
