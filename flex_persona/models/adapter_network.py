"""Adapter mapping from client-specific features to shared latent space."""

from __future__ import annotations

import torch
from torch import nn


class AdapterNetwork(nn.Module):
    """Linear adapter A_k(z) = W_k z + b_k."""

    def __init__(self, input_dim: int, shared_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if shared_dim <= 0:
            raise ValueError("shared_dim must be positive")

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.proj = nn.Linear(input_dim, shared_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)
