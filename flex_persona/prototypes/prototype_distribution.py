"""Compact prototype distribution: Client summaries for server clustering.

A PrototypeDistribution is a discrete probability distribution in the shared latent
space. It represents a client's learned representation as a weighted mixture of
class-specific prototypes, enabling efficient communication and clustering without
sharing raw features or model weights.

In FLEX-Persona's representation-based collaboration:

    Client k learns: backbone + adapter + classifier
                     ↓
    Extracts features through shared adapter (latent space)
                     ↓
    Computes per-class prototypes: p_k,c = mean{h_i : y_i = c}
                     ↓
    Packages as: μ_k = Σ_c w_k,c δ_{p_k,c}
    (weighted mixture of class prototypes)
                     ↓
    Sends to server (compact summary)
                     ↓
    Server clusters clients based on Wasserstein distance between μ_k's
    Server aggregates: μ_cluster = averaged prototypes of similar clients
                     ↓
    Sends back to client as cluster guidance

This achieves:
- **Communication efficiency**: Prototypes are compact vs. full weights
- **Privacy**: Only aggregated prototypes shared, not individual features
- **Cross-architecture collaboration**: All clients unify in shared space
- **Personalization**: Cluster guidance helps client align with cohort
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrototypeDistribution:
    """Discrete probability distribution in shared latent space via class prototypes.

    Represents a client or cluster as a weighted mixture of class-specific prototypes
    extracted from learned representations in the shared latent space.

    Attributes:
        client_id: Identifier of the client (or cluster aggregate) represented.
        support_points: Prototype tensors, shape [n_support, shared_dim].
                       support_points[i] is a prototype for label support_labels[i].
        support_labels: Class labels for each support point, shape [n_support].
                       Can have multiple prototypes per class if aggregating.
        weights: Probability weight for each support point, shape [n_support].
                Non-negative, typically normalized to sum to 1.
        num_classes: Number of task classes (for validation).

    Mathematical representation:
        μ_k = Σ_{i=1}^n w_i δ_{p_i}
        where p_i = prototype (support point) in shared space
              w_i = weight/probability
              δ = Dirac delta (concentrated probability mass)

    The Wasserstein distance W(μ_i, μ_j) measures similarity between clients,
    enabling meaningful clustering in the representation space.
    """

    client_id: int
    support_points: torch.Tensor
    support_labels: torch.Tensor
    weights: torch.Tensor
    num_classes: int

    def validate(self) -> None:
        """Validate consistency of the distribution."""
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
        """Return a copy with weights normalized to sum to 1."""
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
        """Number of support points (prototypes) in this distribution."""
        return int(self.support_points.shape[0])

    @property
    def shared_dim(self) -> int:
        """Dimension of the shared latent space (prototype dimension)."""
        return int(self.support_points.shape[1])
