from __future__ import annotations

from typing import List, Tuple, Optional

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
from datetime import datetime

from common import CFG, seed_everything, ServerHead
from sklearn.metrics import roc_auc_score

print("SERVER STARTING...")


def match_by_id(
    ids_img: List[int],
    z: np.ndarray,
    labels: List[int],
    ids_attr: List[int],
    a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Align (z, a, y) by shared sample ids and return common_ids order."""
    img_map = {int(i): j for j, i in enumerate(ids_img)}
    attr_map = {int(i): j for j, i in enumerate(ids_attr)}

    common_ids = sorted(set(img_map.keys()).intersection(set(attr_map.keys())))
    if not common_ids:
        raise RuntimeError("No overlapping IDs between image and attribute payloads.")

    z_out = np.stack([z[img_map[i]] for i in common_ids], axis=0)
    a_out = np.stack([a[attr_map[i]] for i in common_ids], axis=0)
    y_out = np.array([labels[img_map[i]] for i in common_ids], dtype=np.int64)
    return z_out, a_out, y_out, common_ids


class Attacker(nn.Module):
    """Attacker: embedding -> attribute (scalar regression baseline)."""
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        return self.fc2(x).squeeze(1)  # (B,)


class VFLCollectStrategy(fl.server.strategy.Strategy):
    """
    Each round, request "collect" payloads from exactly 2 clients:
      - image client returns: [z, ids, labels]
      - attribute client returns: [a, ids, a_true]

    Server trains:
      - head model on [z||a] -> y
      - attacker on z -> a_true

    Prints head loss/acc and attacker ASR.
    """

    def __init__(self):
        super().__init__()
        seed_everything(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.head = ServerHead(embed_dim=CFG.EMBED_DIM, attr_dim=CFG.ATTR_DIM).to(self.device)
        self.opt = torch.optim.Adam(self.head.parameters(), lr=1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.attacker = Attacker(embed_dim=CFG.EMBED_DIM).to(self.device)
        self.att_opt = torch.optim.Adam(self.attacker.parameters(), lr=1e-3)
        self.att_loss = nn.BCEWithLogitsLoss()

        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"run_{ts}.csv")

        with open(self.log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "head_loss", "head_acc", "att_loss", "asr", "auc", "n"])

        print(f"[Logging] Writing metrics to {self.log_path}")

    def initialize_parameters(self, client_manager):
        # We don't broadcast model weights in this custom protocol.
        return fl.common.Parameters(tensors=[], tensor_type="")

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=2, min_num_clients=2)
        cfg = {"round": server_round - 1}  # 0-index for batching
        return [(c, fl.common.FitIns(parameters, cfg)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print(f"[Round {server_round}] Failures: {len(failures)}")

        img_payload = None
        attr_payload = None

        for _, fit_res in results:
            role = fit_res.metrics.get("role", "unknown")
            arrs = fl.common.parameters_to_ndarrays(fit_res.parameters)

            if role == "img":
                # [z, ids, labels]
                img_payload = {
                    "z": arrs[0],
                    "ids": arrs[1],
                    "labels": arrs[2],
                }
            elif role == "attr":
                # [a, ids, a_true]
                attr_payload = {
                    "a": arrs[0],
                    "ids": arrs[1],
                    "a_true": arrs[2],
                }

        if img_payload is None or attr_payload is None:
            print(f"[Round {server_round}] Missing one party payload (img or attr).")
            return fl.common.Parameters(tensors=[], tensor_type=""), {}

        # Align by ID
        z_np, a_np, y_np, common_ids = match_by_id(
            img_payload["ids"].tolist(),
            img_payload["z"],
            img_payload["labels"].tolist(),
            attr_payload["ids"].tolist(),
            attr_payload["a"],
        )

        z = torch.tensor(z_np, dtype=torch.float32, device=self.device)
        a = torch.tensor(a_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.long, device=self.device)

        # ---- Train head: [z||a] -> y_task (binary) ----
        self.head.train()
        self.opt.zero_grad()

        logits = self.head(z, a)
        y_float = y.float()

        loss = self.loss_fn(logits, y_float)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            acc = (preds == y).float().mean().item()

        # ---- Train attacker: z -> a_true ----
        # Align attacker ground truth by the same common_ids ordering
        ids_attr = attr_payload["ids"].tolist()
        a_true_list = attr_payload["a_true"].tolist()
        attr_map = {int(i): j for j, i in enumerate(ids_attr)}
        a_true_np = np.array([a_true_list[attr_map[i]] for i in common_ids], dtype=np.float32)
        a_true = torch.tensor(a_true_np, dtype=torch.float32, device=self.device)

        self.attacker.train()
        self.att_opt.zero_grad()
        a_pred = self.attacker(z)  # (B,)
        att_loss = self.att_loss(a_pred, a_true)
        att_loss.backward()
        self.att_opt.step()

        # ASR for binary attribute inference: accuracy after sigmoid thresholding
        with torch.no_grad():
            probs = torch.sigmoid(a_pred)
            preds = (probs >= 0.5).float()
            asr = (preds == a_true).float().mean().item()

        # AUC (ROC)
        try:
            auc = roc_auc_score(a_true.detach().cpu().numpy(), probs.detach().cpu().numpy())
        except ValueError:
            # Happens if a batch has only one class (all 0s or all 1s)
            auc = float("nan")

        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([server_round, float(loss.item()), float(acc), float(att_loss.item()), float(asr), float(auc), int(len(y_np))])

        print(
            f"[Round {server_round}] head_loss={loss.item():.4f} head_acc={acc:.4f} "
            f"att_loss={att_loss.item():.4f} ASR={asr:.4f} AUC={auc:.4f} (n={len(y_np)})"
        )

        metrics = {
            "head_loss": float(loss.item()),
            "head_acc": float(acc),
            "att_loss": float(att_loss.item()),
            "asr": float(asr),
            "auc": float(auc) if auc == auc else auc, # Keep NaN if NaN
        }
        return fl.common.Parameters(tensors=[], tensor_type=""), metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        # No evaluation phase in this custom protocol.
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


if __name__ == "__main__":
    strategy = VFLCollectStrategy()

    fl.server.start_server(
        server_address=CFG.SERVER_ADDR,
        config=fl.server.ServerConfig(num_rounds=CFG.NUM_ROUNDS),
        strategy=strategy,
    )