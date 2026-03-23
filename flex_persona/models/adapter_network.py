"""Adapter network: Maps client-specific features to a shared latent space.

This is a core component of FLEX-Persona's representation-based collaboration approach.
Instead of sharing model parameters across heterogeneous clients, each client maintains
a private backbone but shares a learned adapter that projects client-specific features
into a shared latent space. This enables:

1. **Cross-architecture collaboration**: Different clients can use different backbones
   (CNN, ResNet, MLP) because they unify via the shared latent space.

2. **Efficient knowledge transfer**: The server computes cluster prototypes in the
   shared space and guides clients without requiring full model sharing.

3. **Privacy preservation**: Only low-dimensional prototype distributions are
   communicated, not raw features or model weights.

The adapter is trained jointly with the local task loss and cluster guidance loss,
allowing each client to learn a representation that is both discriminative for the
local task and aligned with the shared cluster structure.
"""

from __future__ import annotations

import torch
from torch import nn


class AdapterNetwork(nn.Module):
    """Maps client-specific features to a shared latent representation space.

    This is a lightweight linear projection layer A_k: R^d → R^s that transforms
    client k's local feature representation (dimension d) into a shared latent space
    (dimension s). All clients project into the same shared space, enabling the server
    to compute meaningful similarity and clustering even though clients use different
    model architectures.

    Args:
        input_dim: Dimension of client-specific features from the backbone (d).
        shared_dim: Dimension of the shared latent space (s). Typically smaller than
                   input_dim to encourage compression.

    Example:
        >>> backbone = SmallCNN()  # outputs 128-dim features
        >>> adapter = AdapterNetwork(input_dim=128, shared_dim=64)
        >>> x = torch.randn(32, 128)  # batch of 32 client features
        >>> shared_repr = adapter(x)  # (32, 64) in shared space
    """

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
        """Project client features into the shared latent space.

        Args:
            features: Tensor of shape (batch_size, input_dim) containing client-specific
                     feature representations from the backbone.

        Returns:
            Tensor of shape (batch_size, shared_dim) in the shared latent space.
        """
        return self.proj(features)
