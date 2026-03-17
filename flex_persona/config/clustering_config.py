"""Spectral clustering configuration schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    """Hyperparameters for client clustering."""

    num_clusters: int = 2
    assign_labels: str = "kmeans"
    random_state: int = 42

    def validate(self) -> None:
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if self.assign_labels not in {"kmeans", "discretize"}:
            raise ValueError("assign_labels must be 'kmeans' or 'discretize'")
