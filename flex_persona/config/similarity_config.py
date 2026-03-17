"""Similarity and distance configuration schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimilarityConfig:
    """Controls distance and affinity computation hyperparameters."""

    sigma: float = 1.0
    use_euclidean_baseline: bool = True
    use_wasserstein: bool = True

    def validate(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if not (self.use_euclidean_baseline or self.use_wasserstein):
            raise ValueError("At least one similarity mode must be enabled")
