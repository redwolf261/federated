"""Ground cost matrix utilities for prototype distributions."""

from __future__ import annotations

import torch


def squared_euclidean_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return pairwise squared Euclidean distance matrix between supports."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")

    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    cross = x @ y.transpose(0, 1)
    cost = x_norm + y_norm - 2.0 * cross
    return torch.clamp(cost, min=0.0)
