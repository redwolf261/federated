"""Prototype distribution data structure for client-level semantic summaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrototypeDistribution:
    """Discrete distribution mu_k = sum_c w_k,c delta_{p_k,c}."""

    client_id: int
    support_points: torch.Tensor
    support_labels: torch.Tensor
    weights: torch.Tensor
    num_classes: int

    def validate(self) -> None:
        if self.support_points.ndim != 2:
            raise ValueError("support_points must be 2D with shape [n_support, d]")
        if self.support_labels.ndim != 1:
            raise ValueError("support_labels must be 1D")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if self.support_points.shape[0] != self.support_labels.shape[0]:
            raise ValueError("support_points and support_labels length mismatch")
        if self.support_points.shape[0] != self.weights.shape[0]:
            raise ValueError("support_points and weights length mismatch")
        if (self.weights < 0).any():
            raise ValueError("weights must be non-negative")

    def normalized(self) -> "PrototypeDistribution":
        total = self.weights.sum()
        if total.item() <= 0.0:
            raise ValueError("Sum of weights must be positive")
        return PrototypeDistribution(
            client_id=self.client_id,
            support_points=self.support_points,
            support_labels=self.support_labels,
            weights=self.weights / total,
            num_classes=self.num_classes,
        )

    @property
    def num_support(self) -> int:
        return int(self.support_points.shape[0])

    @property
    def shared_dim(self) -> int:
        return int(self.support_points.shape[1])
