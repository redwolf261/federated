"""Client partitioning strategies for federated simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class PartitionResult:
    client_indices: dict[int, np.ndarray]


class PartitionStrategies:
    """Collection of deterministic partitioning strategies."""

    @staticmethod
    def by_writer_ids(writer_ids: Iterable[str], num_clients: int) -> PartitionResult:
        writer_ids_np = np.array(list(writer_ids))
        unique_writers = np.unique(writer_ids_np)
        writer_to_client: dict[str, int] = {}

        for idx, writer in enumerate(unique_writers):
            writer_to_client[str(writer)] = idx % num_clients

        buckets: dict[int, list[int]] = {cid: [] for cid in range(num_clients)}
        for idx, writer in enumerate(writer_ids_np):
            buckets[writer_to_client[str(writer)]].append(idx)

        return PartitionResult(
            client_indices={cid: np.array(indices, dtype=np.int64) for cid, indices in buckets.items()}
        )

    @staticmethod
    def dirichlet_by_label(
        labels: torch.Tensor,
        num_clients: int,
        alpha: float,
        seed: int,
    ) -> PartitionResult:
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for Dirichlet partitioning")

        rng = np.random.default_rng(seed)
        labels_np = labels.cpu().numpy().astype(np.int64)
        classes = np.unique(labels_np)

        buckets: dict[int, list[int]] = {cid: [] for cid in range(num_clients)}

        for cls in classes:
            cls_indices = np.where(labels_np == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            split_chunks = np.split(cls_indices, split_points)
            for cid, chunk in enumerate(split_chunks):
                if len(chunk) > 0:
                    buckets[cid].extend(chunk.tolist())

        for cid in range(num_clients):
            rng.shuffle(buckets[cid])

        return PartitionResult(
            client_indices={cid: np.array(indices, dtype=np.int64) for cid, indices in buckets.items()}
        )
