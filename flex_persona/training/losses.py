"""Loss composition utilities for FLEX-Persona training."""

from __future__ import annotations

import torch
from torch import nn


class LossComposer:
    """Composes local and cluster-aware objective terms."""

    def __init__(self) -> None:
        self._criterion = nn.CrossEntropyLoss()

    def local_task_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._criterion(logits, labels)

    @staticmethod
    def total_loss(local_loss: torch.Tensor, cluster_loss: torch.Tensor, lambda_cluster: float) -> torch.Tensor:
        if lambda_cluster < 0.0:
            raise ValueError("lambda_cluster must be non-negative")
        return local_loss + (lambda_cluster * cluster_loss)
