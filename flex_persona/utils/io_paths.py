"""Path utilities for dataset and output locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def cifar100_dir(self) -> Path:
        return self.root / "dataset" / "cifar-100-python"

    @property
    def femnist_parquet(self) -> Path:
        return self.root / "dataset" / "femnist" / "train-00000-of-00001.parquet"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
