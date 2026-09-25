from __future__ import annotations

import os
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from common import (
    CFG,
    seed_everything,
    get_celeba_dataset,
    make_shared_pool_indices,
    round_batch_ids,
    get_sensitive_attribute,
    get_task_label,
    ImageEncoder,
)

print("CLIENT STARTING...")


def to_ndarrays(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class VFLClient(fl.client.NumPyClient):
    """
    Two roles:
      - ROLE=img  -> returns embeddings + ids + labels
      - ROLE=attr -> returns attributes + ids + attribute_ground_truth

    IMPORTANT:
      Flower client metrics must be scalar-only (no lists).
      We put all batch data into the payload arrays returned as "parameters".
    """

    def __init__(self, role: str, device: torch.device):
        self.role = role
        self.device = device

        seed_everything(42)
        self.trainset, _ = get_celeba_dataset(root="./data")
        self.shared_pool = make_shared_pool_indices(len(self.trainset), CFG.SHARED_POOL_SIZE, seed=42)

        self.encoder = None
        if self.role == "img":
            self.encoder = ImageEncoder(embed_dim=CFG.EMBED_DIM).to(self.device)
            self.encoder.eval()

    def get_parameters(self, config):
        # We are not doing FedAvg parameters here.
        return []

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        rnd = int(config.get("round", 0))
        batch_ids = round_batch_ids(self.shared_pool, rnd, CFG.BATCH_PER_ROUND)

        xs, tars = [], []
        for idx in batch_ids:
            x, t = self.trainset[int(idx)] # t is (40,) tensor of -1/1 attributes
            xs.append(x)
            tars.append(t)

        x_tensor = torch.stack(xs, dim=0).to(self.device)  # (B,3,64,64)
        t_tensor = torch.stack(tars, dim=0).to(self.device)      # (B,40)

        y_task = get_task_label(t_tensor)   # (B,) long 0/1
        a_sensitive = get_sensitive_attribute(t_tensor) # (B,) float 0/1

        if self.role == "img":
            assert self.encoder is not None
            with torch.no_grad():
                z = self.encoder(x_tensor)  # (B, EMBED_DIM)

            z_np = to_ndarrays(z).astype(np.float32)          # (B, EMBED_DIM)
            ids_np = batch_ids.astype(np.int64)               # (B,)
            y_np = y_task.detach().cpu().numpy().astype(np.int64)     # (B,)

            payload = [z_np, ids_np, y_np]
            metrics = {"role": "img"}  # scalar-only
            return payload, len(batch_ids), metrics

        if self.role == "attr":
            # "a" is what server will use as the attribute feature input
            # "a_true" is ground truth for attacker (same here)
            a_np = a_sensitive.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)         # (B,1)
            ids_np = batch_ids.astype(np.int64)              # (B,)
            a_true_np = a_sensitive.detach().cpu().numpy().astype(np.float32)  # (B,)

            payload = [a_np, ids_np, a_true_np]
            metrics = {"role": "attr"}  # scalar-only
            return payload, len(batch_ids), metrics

        raise ValueError(f"Unknown ROLE={self.role}")

    def evaluate(self, parameters, config):
        # Not used in this protocol.
        return 0.0, 0, {}


if __name__ == "__main__":
    role = os.environ.get("ROLE", "img").strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = VFLClient(role=role, device=device)

    # This is deprecated in newer Flower versions, but works.
    # We can modernize later once your pipeline is stable.
    fl.client.start_numpy_client(server_address=CFG.SERVER_ADDR, client=client)