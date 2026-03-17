"""Optimizer construction helpers."""

from __future__ import annotations

import torch
from torch import nn


class OptimizerFactory:
    """Factory for optimizers used in client training."""

    @staticmethod
    def adam(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
