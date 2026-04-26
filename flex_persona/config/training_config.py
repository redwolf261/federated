"""Training configuration schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters for local and cluster-aware optimization."""

    rounds: int = 100  # Increased for full convergence
    local_epochs: int = 20  # Increased for meaningful training
    cluster_aware_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lambda_cluster: float = 0.1
    lambda_cluster_center: float = 0.01
    cluster_center_warmup_rounds: int = 8
    max_samples_per_client: int | None = None
    aggregation_mode: str = "prototype"
    fedprox_mu: float = 0.0
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    max_unlimited_rounds: int = 10000

    # Ablation study toggles
    use_clustering: bool = True
    use_guidance: bool = True
    ablation_mode: str = "full"

    def validate(self) -> None:

        if self.rounds == 0 or self.rounds < -1:
            raise ValueError("rounds must be positive, or -1 for unlimited")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.cluster_aware_epochs < 0:
            raise ValueError("cluster_aware_epochs must be non-negative")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.lambda_cluster < 0.0:
            raise ValueError("lambda_cluster must be non-negative")
        if self.lambda_cluster_center < 0.0:
            raise ValueError("lambda_cluster_center must be non-negative")
        if self.cluster_center_warmup_rounds <= 0:
            raise ValueError("cluster_center_warmup_rounds must be positive")
        if self.max_samples_per_client is not None and self.max_samples_per_client <= 0:
            raise ValueError("max_samples_per_client must be positive when set")
        if self.aggregation_mode not in {"prototype", "fedavg", "fedprox"}:
            raise ValueError("aggregation_mode must be one of: 'prototype', 'fedavg', 'fedprox'")
        if self.fedprox_mu < 0.0:
            raise ValueError("fedprox_mu must be non-negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.max_unlimited_rounds <= 0:
            raise ValueError("max_unlimited_rounds must be positive")
        if self.rounds == -1 and not self.early_stopping_enabled:
            raise ValueError("unlimited rounds (-1) requires early_stopping_enabled=True")
        if self.ablation_mode not in {"full", "no_clustering", "random_clusters", "no_guidance", "no_prototype_sharing", "self_only", "shuffled_prototypes", "noise_prototypes"}:
            raise ValueError("ablation_mode must be one of: 'full', 'no_clustering', 'random_clusters', 'no_guidance', 'no_prototype_sharing', 'self_only', 'shuffled_prototypes', 'noise_prototypes'")
