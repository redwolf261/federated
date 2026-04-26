"""Client-level DataLoader construction for federated simulation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config.experiment_config import ExperimentConfig
from .dataset_registry import DatasetRegistry
from .partition_strategies import PartitionResult, PartitionStrategies


@dataclass
class ClientDatasetBundle:
    client_id: int
    train_loader: DataLoader
    eval_loader: DataLoader
    num_samples: int
    class_histogram: dict[int, int]
    source_indices: list[int]


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
        if artifact.name == "cifar10":
            return self._build_cifar_bundles(artifact.payload, dataset_name="cifar10")
        if artifact.name == "cifar100":
            return self._build_cifar_bundles(artifact.payload, dataset_name="cifar100")
        raise ValueError(f"Unsupported artifact: {artifact.name}")

    def _build_femnist_bundles(self, payload: dict[str, object]) -> list[ClientDatasetBundle]:
        images = payload["images"]
        labels = payload["labels"]
        writer_ids = payload["writer_ids"]

        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Expected tensor payloads for FEMNIST images and labels")

        partition = self._resolve_partition(
            labels=labels,
            writer_ids=writer_ids,
            dataset_name="femnist",
        )
        bundles: list[ClientDatasetBundle] = []

        for client_id, indices in partition.client_indices.items():
            if len(indices) == 0:
                continue
            idx_tensor = torch.from_numpy(indices)
            x_client = images[idx_tensor]
            y_client = labels[idx_tensor]
            bundles.append(self._make_bundle(client_id, x_client, y_client, source_indices=indices))

        return bundles

    def _build_cifar_bundles(
        self,
        payload: dict[str, object],
        dataset_name: str,
    ) -> list[ClientDatasetBundle]:
        train_images = payload["train_images"]
        train_labels = payload["train_labels"]

        if not isinstance(train_images, torch.Tensor) or not isinstance(train_labels, torch.Tensor):
            raise TypeError("Expected tensor payloads for CIFAR-100 train images and labels")

        partition = self._resolve_partition(
            labels=train_labels,
            writer_ids=None,
            dataset_name=dataset_name,
        )

        bundles: list[ClientDatasetBundle] = []
        for client_id, indices in partition.client_indices.items():
            if len(indices) == 0:
                continue
            idx_tensor = torch.from_numpy(indices)
            x_client = train_images[idx_tensor]
            y_client = train_labels[idx_tensor]
            bundles.append(self._make_bundle(client_id, x_client, y_client, source_indices=indices))

        return bundles

    def _make_bundle(
        self,
        client_id: int,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        source_indices: np.ndarray,
    ) -> ClientDatasetBundle:
        max_samples = self.config.training.max_samples_per_client
        if max_samples is not None and int(y_tensor.shape[0]) > max_samples:
            x_tensor = x_tensor[:max_samples]
            y_tensor = y_tensor[:max_samples]

        total_samples = int(y_tensor.shape[0])
        if total_samples <= 1:
            train_x = x_tensor
            train_y = y_tensor
            eval_x = x_tensor
            eval_y = y_tensor
        else:
            generator = torch.Generator()
            generator.manual_seed(int(self.config.random_seed + client_id))

            permutation = torch.randperm(total_samples, generator=generator)
            eval_size = max(1, int(total_samples * 0.2))
            eval_size = min(eval_size, total_samples - 1)

            eval_indices = permutation[:eval_size]
            train_indices = permutation[eval_size:]

            train_x = x_tensor[train_indices]
            train_y = y_tensor[train_indices]
            eval_x = x_tensor[eval_indices]
            eval_y = y_tensor[eval_indices]

        train_dataset = TensorDataset(train_x, train_y)
        eval_dataset = TensorDataset(eval_x, eval_y)
        batch_size = self.config.training.batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=self._make_loader_generator(client_id),
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            generator=self._make_loader_generator(client_id + 10_000),
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
            num_samples=total_samples,
            class_histogram=class_hist,
            source_indices=[int(i) for i in source_indices.tolist()],
        )

    def _make_loader_generator(self, client_id: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.random_seed) + int(client_id))
        return generator

    @staticmethod
    def partition_fingerprint(bundles: list[ClientDatasetBundle]) -> str:
        payload = {
            str(bundle.client_id): {
                "indices": bundle.source_indices,
                "class_histogram": bundle.class_histogram,
            }
            for bundle in sorted(bundles, key=lambda b: b.client_id)
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _resolve_partition(
        self,
        labels: torch.Tensor,
        writer_ids: np.ndarray | None,
        dataset_name: str,
    ) -> PartitionResult:
        mode = self.config.partition_mode

        if mode == "iid":
            return PartitionStrategies.iid_even(
                num_samples=int(labels.shape[0]),
                num_clients=self.config.num_clients,
                seed=self.config.random_seed,
            )

        if mode == "dirichlet":
            return PartitionStrategies.dirichlet_by_label(
                labels=labels,
                num_clients=self.config.num_clients,
                alpha=self.config.dirichlet_alpha,
                seed=self.config.random_seed,
            )

        if dataset_name == "femnist":
            if writer_ids is None:
                raise ValueError("writer_ids are required for FEMNIST natural partitioning")
            return PartitionStrategies.by_writer_ids(writer_ids, self.config.num_clients)

        return PartitionStrategies.dirichlet_by_label(
            labels=labels,
            num_clients=self.config.num_clients,
            alpha=self.config.dirichlet_alpha,
            seed=self.config.random_seed,
        )
