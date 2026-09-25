from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vfl_unlearning import (
    build_unlearning_loaders,
    collect_attack_metrics,
    collect_metric_snapshot,
    create_optimizer,
    seed_everything,
    train_epoch_with_loss_scale,
)


class SyntheticVFLDataset(Dataset):
    def __init__(self, sample_count: int = 96, seed: int = 123) -> None:
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(sample_count, 4, generator=generator)
        logits = 1.5 * features[:, 0] - 1.0 * features[:, 1] + 1.2 * features[:, 2] - 0.8 * features[:, 3]
        labels = (logits > 0).long()
        self.left = features[:, :2]
        self.right = features[:, 2:]
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int):
        return self.left[index], self.right[index], self.labels[index]


class TinyParty(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, left_embed: torch.Tensor, right_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left_embed, right_embed], dim=1))


class UnlearningTests(unittest.TestCase):
    def test_split_is_repeatable_for_same_seed(self) -> None:
        dataset = SyntheticVFLDataset()
        first = build_unlearning_loaders(dataset, batch_size=8, forget_fraction=0.25, seed=11)
        second = build_unlearning_loaders(dataset, batch_size=8, forget_fraction=0.25, seed=11)

        self.assertEqual(first.forget_count, second.forget_count)
        self.assertEqual(first.retain_count, second.retain_count)
        self.assertEqual(first.forget_train_loader.dataset.indices, second.forget_train_loader.dataset.indices)
        self.assertEqual(first.retain_train_loader.dataset.indices, second.retain_train_loader.dataset.indices)

    def test_unlearning_run_is_repeatable(self) -> None:
        first = self._run_cycle()
        second = self._run_cycle()

        self.assertEqual(first["forget_indices"], second["forget_indices"])
        self.assertAlmostEqual(first["pre"].forget_acc, second["pre"].forget_acc, places=6)
        self.assertAlmostEqual(first["post"].forget_acc, second["post"].forget_acc, places=6)
        self.assertAlmostEqual(first["post"].retain_acc, second["post"].retain_acc, places=6)
        self.assertAlmostEqual(first["pre_attack"].attack_auc, second["pre_attack"].attack_auc, places=6)
        self.assertAlmostEqual(first["post_attack"].attack_auc, second["post_attack"].attack_auc, places=6)
        self.assertGreater(first["post"].forget_loss, first["pre"].forget_loss)
        self.assertLess(first["post_attack"].attack_auc, first["pre_attack"].attack_auc)

    def _run_cycle(self) -> dict[str, object]:
        seed_everything(7)
        dataset = SyntheticVFLDataset(seed=21)
        test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        loaders = build_unlearning_loaders(dataset, batch_size=8, forget_fraction=0.25, seed=7)

        left_party = TinyParty()
        right_party = TinyParty()
        server_head = TinyHead()
        optimizer = create_optimizer(left_party, right_party, server_head, lr=0.05)
        device = torch.device("cpu")

        for _ in range(6):
            train_epoch_with_loss_scale(
                left_party=left_party,
                right_party=right_party,
                server_head=server_head,
                loader=loaders.train_loader,
                optimizer=optimizer,
                device=device,
                task_type="multiclass",
            )

        pre_snapshot = collect_metric_snapshot(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            loaders=loaders,
            test_loader=test_loader,
            device=device,
            task_type="multiclass",
        )
        pre_attack = collect_attack_metrics(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            member_loader=loaders.forget_eval_loader,
            nonmember_loader=test_loader,
            device=device,
            task_type="multiclass",
        )

        for _ in range(3):
            train_epoch_with_loss_scale(
                left_party=left_party,
                right_party=right_party,
                server_head=server_head,
                loader=loaders.forget_train_loader,
                optimizer=optimizer,
                device=device,
                task_type="multiclass",
                loss_scale=-2.0,
            )
            train_epoch_with_loss_scale(
                left_party=left_party,
                right_party=right_party,
                server_head=server_head,
                loader=loaders.retain_train_loader,
                optimizer=optimizer,
                device=device,
                task_type="multiclass",
                loss_scale=1.0,
            )

        post_snapshot = collect_metric_snapshot(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            loaders=loaders,
            test_loader=test_loader,
            device=device,
            task_type="multiclass",
        )
        post_attack = collect_attack_metrics(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            member_loader=loaders.forget_eval_loader,
            nonmember_loader=test_loader,
            device=device,
            task_type="multiclass",
        )

        return {
            "forget_indices": tuple(loaders.forget_train_loader.dataset.indices),
            "pre": pre_snapshot,
            "post": post_snapshot,
            "pre_attack": pre_attack,
            "post_attack": post_attack,
        }


if __name__ == "__main__":
    unittest.main()
