from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


TaskType = Literal["multiclass", "binary"]


def train_epoch(
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: TaskType,
    max_batches: int | None = None,
) -> tuple[float, float]:
    left_party.train()
    right_party.train()
    server_head.train()
    criterion: nn.Module
    if task_type == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

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
        loss.backward()
        optimizer.step()

        batch_total, batch_correct = _batch_metrics(logits=logits, labels=labels, task_type=task_type)
        total_loss += loss.item() * batch_total
        correct += batch_correct
        total += batch_total

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    left_party: nn.Module,
    right_party: nn.Module,
    server_head: nn.Module,
    loader,
    device: torch.device,
    task_type: TaskType,
    max_batches: int | None = None,
) -> tuple[float, float]:
    left_party.eval()
    right_party.eval()
    server_head.eval()
    criterion: nn.Module
    if task_type == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

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

    return total_loss / total, correct / total


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
