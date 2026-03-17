"""Client-level DataLoader construction for federated simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config.experiment_config import ExperimentConfig
from .dataset_registry import DatasetRegistry
from .partition_strategies import PartitionStrategies


@dataclass
class ClientDatasetBundle:
    client_id: int
    train_loader: DataLoader
    eval_loader: DataLoader
    num_samples: int
    class_histogram: dict[int, int]


class ClientDataManager:
    """Builds per-client dataloaders for FEMNIST and CIFAR-100 experiments."""

    def __init__(self, workspace_root: str | Path, config: ExperimentConfig) -> None:
        self.workspace_root = Path(workspace_root)
        self.config = config
        self.registry = DatasetRegistry(self.workspace_root)

    def build_client_bundles(self) -> list[ClientDatasetBundle]:
        max_samples = self.config.training.max_samples_per_client
        max_rows = (
            int(self.config.num_clients * max_samples * 4)
            if max_samples is not None
            else None
        )

        artifact = self.registry.load(
            self.config.dataset_name,
            max_train_samples=(self.config.num_clients * max_samples * 2) if max_samples is not None else None,
            max_test_samples=(self.config.num_clients * max_samples) if max_samples is not None else None,
            max_rows=max_rows,
        )
        if artifact.name == "femnist":
            return self._build_femnist_bundles(artifact.payload)
        if artifact.name == "cifar100":
            return self._build_cifar100_bundles(artifact.payload)
        raise ValueError(f"Unsupported artifact: {artifact.name}")

    def _build_femnist_bundles(self, payload: dict[str, object]) -> list[ClientDatasetBundle]:
        images = payload["images"]
        labels = payload["labels"]
        writer_ids = payload["writer_ids"]

        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Expected tensor payloads for FEMNIST images and labels")

        partition = PartitionStrategies.by_writer_ids(writer_ids, self.config.num_clients)
        bundles: list[ClientDatasetBundle] = []

        for client_id, indices in partition.client_indices.items():
            if len(indices) == 0:
                continue
            idx_tensor = torch.from_numpy(indices)
            x_client = images[idx_tensor]
            y_client = labels[idx_tensor]
            bundles.append(self._make_bundle(client_id, x_client, y_client))

        return bundles

    def _build_cifar100_bundles(self, payload: dict[str, object]) -> list[ClientDatasetBundle]:
        train_images = payload["train_images"]
        train_labels = payload["train_labels"]

        if not isinstance(train_images, torch.Tensor) or not isinstance(train_labels, torch.Tensor):
            raise TypeError("Expected tensor payloads for CIFAR-100 train images and labels")

        partition = PartitionStrategies.dirichlet_by_label(
            labels=train_labels,
            num_clients=self.config.num_clients,
            alpha=0.5,
            seed=self.config.random_seed,
        )

        bundles: list[ClientDatasetBundle] = []
        for client_id, indices in partition.client_indices.items():
            if len(indices) == 0:
                continue
            idx_tensor = torch.from_numpy(indices)
            x_client = train_images[idx_tensor]
            y_client = train_labels[idx_tensor]
            bundles.append(self._make_bundle(client_id, x_client, y_client))

        return bundles

    def _make_bundle(
        self,
        client_id: int,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> ClientDatasetBundle:
        max_samples = self.config.training.max_samples_per_client
        if max_samples is not None and int(y_tensor.shape[0]) > max_samples:
            x_tensor = x_tensor[:max_samples]
            y_tensor = y_tensor[:max_samples]

        dataset = TensorDataset(x_tensor, y_tensor)
        batch_size = self.config.training.batch_size

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        eval_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        unique_labels, counts = torch.unique(y_tensor, return_counts=True)
        class_hist = {
            int(label.item()): int(count.item())
            for label, count in zip(unique_labels, counts, strict=True)
        }

        return ClientDatasetBundle(
            client_id=client_id,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_samples=int(y_tensor.shape[0]),
            class_histogram=class_hist,
        )
