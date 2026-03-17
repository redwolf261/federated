"""Training configuration schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters for local and cluster-aware optimization."""

    rounds: int = 50
    local_epochs: int = 1
    cluster_aware_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lambda_cluster: float = 0.1
    max_samples_per_client: int | None = None

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.cluster_aware_epochs <= 0:
            raise ValueError("cluster_aware_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.lambda_cluster < 0.0:
            raise ValueError("lambda_cluster must be non-negative")
        if self.max_samples_per_client is not None and self.max_samples_per_client <= 0:
            raise ValueError("max_samples_per_client must be positive when set")
