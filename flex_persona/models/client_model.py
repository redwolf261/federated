"""Unified client model with private backbone and shared adapter for federated learning.

This model is the centerpiece of FLEX-Persona's representation-based collaboration:

    Backbone z = B_k(x)  [client-specific features]
         ↓
    Adapter h = A_k(z)   [projection to shared latent space]
         ↓
    Classifier y = C(h)  [task-specific predictions and prototypes]

The backbone and adapter are jointly optimized:
- Local task loss optimizes the backbone + adapter + classifier for the local task
- Cluster guidance loss optimizes the adapter to align with other clients' representations

This enables heterogeneous clients with different backbones to collaborate through a
unified shared latent space while maintaining privacy (only prototypes are shared,
not features or weights).
"""

from __future__ import annotations

import torch
from torch import nn

from .adapter_network import AdapterNetwork
from .backbones import FeatureBackbone


class ClientModel(nn.Module):
    """Heterogeneous client model: (Backbone → Adapter → Classifier).

    Supports FLEX-Persona's representation-based collaboration approach where:
    1. Each client has a private backbone (can differ across clients)
    2. All clients share the same adapter architecture (maps to shared space)
    3. Classifier produces task-specific predictions
    4. The adapter enables similarity computation and clustering in shared space

    Args:
        backbone: Feature extraction network (SmallCNN, ResNet8, or MLP). Different
                 clients can use different backbones.
        adapter: Adapter network mapping backbone features (dim d) to shared space (dim s).
        num_classes: Number of classes for the task classifier.

    Attributes:
        backbone: Client-specific feature extractor.
        adapter: Maps features to shared latent space (same dim s for all clients).
        classifier: Linear classifier head (input_dim = backbone.output_dim).
        num_classes: Number of task classes.
    """

    def __init__(self, backbone: FeatureBackbone, adapter: AdapterNetwork, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        self.backbone = backbone
        self.adapter = adapter
        self.num_classes = num_classes
        self.classifier = nn.Linear(adapter.shared_dim, num_classes)

    def forward_task(self, x: torch.Tensor) -> torch.Tensor:
        """Compute task predictions (labels).

        Used for: local supervised learning, classification accuracy.

        Args:
            x: Input tensor (images, shape ∝ [batch_size, channels, height, width]).

        Returns:
            Logits of shape [batch_size, num_classes].
        """
        features = self.extract_features(x)
        shared = self.project_shared(features)
        return self.classifier(shared)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract client-specific features from the backbone.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor of shape [batch_size, backbone.output_dim].
        """
        return self.backbone(x)

    def project_shared(self, features: torch.Tensor) -> torch.Tensor:
        """Project features into the shared latent space via the adapter.

        This is the key operation for representation-based collaboration:
        Maps client-specific features into a unified space shared by all clients,
        enabling meaningful similarity/clustering computation.

        Args:
            features: Client-specific features from extract_features().
                     Shape: [batch_size, backbone.output_dim].

        Returns:
            Shared latent representations, shape [batch_size, adapter.shared_dim].
            These are used to compute prototypes, similarity, and clustering.
        """
        return self.adapter(features)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end: input → backbone → adapter → shared representation.

        Convenience method combining extract_features() and project_shared().
        Used in: prototype extraction, cluster guidance loss computation.

        Args:
            x: Input tensor.

        Returns:
            Shared latent representation, shape [batch_size, adapter.shared_dim].
        """
        features = self.extract_features(x)
        return self.project_shared(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass: compute task predictions.

        Equivalent to forward_task(x).
        """
        return self.forward_task(x)
