"""Dataset registry for known local dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cifar100_loader import Cifar100Loader
from .cifar10_loader import Cifar10Loader
from .femnist_loader import FemnistLoader


@dataclass(frozen=True)
class DatasetArtifact:
    name: str
    payload: dict[str, Any]


class DatasetRegistry:
    """Creates dataset payloads for supported dataset names."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def load(
        self,
        dataset_name: str,
        max_train_samples: int | None = None,
        max_test_samples: int | None = None,
        max_rows: int | None = None,
    ) -> DatasetArtifact:
        normalized = dataset_name.lower().strip()
        if normalized == "cifar10":
            loader = Cifar10Loader(self.workspace_root / "dataset" / "cifar-10-batches-py")
            return DatasetArtifact(
                name="cifar10",
                payload=loader.load(
                    max_train_samples=max_train_samples,
                    max_test_samples=max_test_samples,
                ),
            )
        if normalized == "cifar100":
            loader = Cifar100Loader(self.workspace_root / "dataset" / "cifar-100-python")
            return DatasetArtifact(
                name="cifar100",
                payload=loader.load(
                    max_train_samples=max_train_samples,
                    max_test_samples=max_test_samples,
                ),
            )
        if normalized == "femnist":
            loader = FemnistLoader(
                self.workspace_root / "dataset" / "femnist" / "train-00000-of-00001.parquet"
            )
            return DatasetArtifact(name="femnist", payload=loader.load(max_rows=max_rows))

        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
