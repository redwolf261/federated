"""Improved Client Model with proper adapter-classifier alignment.

Addresses critical architectural issues identified in the original implementation:

1. **Alignment Issue**: Original classifier worked directly on backbone features (6272-dim)
   while adapter projected to shared space (64-dim). These were separate, unaligned paths.

2. **Architecture Fix**: New design ensures classifier and adapter share representations,
   enabling proper alignment and avoiding the "two separate networks" problem.

3. **Improved Adapter**: Uses the new improved adapter with reduced compression and non-linearity.

Key architectural improvements:
- Classifier now works on adapter output (aligned representations)
- Added alignment loss computation between adapter and classifier paths
- Support for both improved and alignment-aware adapters
- Proper gradient flow between all components

This resolves the fundamental design flaw where adapter and classifier were learning
completely separate representations with no alignment mechanism.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .adapter_network import AdapterNetwork  # Keep for backward compatibility
from .improved_adapter_network import ImprovedAdapterNetwork, AlignmentAwareAdapter
from .backbones import FeatureBackbone


class ImprovedClientModel(nn.Module):
    """Client model with proper adapter-classifier alignment.

    Architecture: Backbone → Adapter → Classifier (aligned pathway)

    Key improvements over original ClientModel:
    1. Classifier works on adapter output, not backbone output (proper alignment)
    2. Uses improved adapter with less compression and non-linearity
    3. Supports alignment loss computation
    4. Maintains backward compatibility with original adapter

    Args:
        backbone: Feature extraction network (output_dim used for adapter input)
        adapter: Adapter network (improved or alignment-aware recommended)
        num_classes: Number of classes for task classifier
        use_alignment_loss: Whether to compute alignment losses during training
    """

    def __init__(
        self,
        backbone: FeatureBackbone,
        adapter: AdapterNetwork | ImprovedAdapterNetwork | AlignmentAwareAdapter,
        num_classes: int,
        use_alignment_loss: bool = False
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        self.backbone = backbone
        self.adapter = adapter
        self.num_classes = num_classes
        self.use_alignment_loss = use_alignment_loss

        # CRITICAL FIX: Classifier now works on adapter output, not backbone output
        # This ensures adapter and classifier representations are aligned
        self.classifier = nn.Linear(adapter.shared_dim, num_classes)

        # For alignment loss computation, we need a projection from backbone to alignment space
        if use_alignment_loss and isinstance(adapter, AlignmentAwareAdapter):
            self.backbone_alignment_proj = nn.Sequential(
                nn.Linear(backbone.output_dim, adapter.alignment_dim),
                nn.BatchNorm1d(adapter.alignment_dim),
                nn.ReLU(inplace=True),
                nn.Linear(adapter.alignment_dim, adapter.alignment_dim)
            )
        else:
            self.backbone_alignment_proj = None

    def forward_task(self, x: torch.Tensor) -> torch.Tensor:
        """Compute task predictions through aligned backbone→adapter→classifier path.

        CRITICAL ARCHITECTURAL CHANGE: Now classifier works on adapter output,
        ensuring adapter and classifier representations are properly aligned.

        Args:
            x: Input tensor (images)

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # Extract backbone features
        backbone_features = self.backbone(x)

        # Project to shared space via adapter
        shared_repr = self.adapter(backbone_features)

        # Classify using shared representation (ALIGNED PATH)
        logits = self.classifier(shared_repr)

        return logits

    def forward_task_with_alignment(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass with alignment loss computation.

        Returns both task predictions and alignment information for loss computation.

        Args:
            x: Input tensor

        Returns:
            Tuple of (logits, alignment_info) where alignment_info contains
            features needed for computing alignment losses
        """
        backbone_features = self.backbone(x)

        alignment_info = {}

        if isinstance(self.adapter, AlignmentAwareAdapter):
            # Use alignment-aware adapter
            shared_repr, adapter_alignment_features = self.adapter.forward_with_alignment(backbone_features)
            alignment_info['adapter_alignment_features'] = adapter_alignment_features

            # Project backbone features to alignment space
            if self.backbone_alignment_proj is not None:
                backbone_alignment_features = self.backbone_alignment_proj(backbone_features)
                alignment_info['backbone_alignment_features'] = backbone_alignment_features
        else:
            # Regular adapter
            shared_repr = self.adapter(backbone_features)

        # Store intermediate representations for analysis
        alignment_info.update({
            'backbone_features': backbone_features,
            'shared_repr': shared_repr,
        })

        # Classify using shared representation
        logits = self.classifier(shared_repr)

        return logits, alignment_info

    def compute_alignment_loss(self, alignment_info: dict, weight: float = 0.1) -> torch.Tensor:
        """Compute alignment loss between adapter and backbone representations.

        This addresses the critique about missing alignment by providing an explicit
        mechanism to align the adapter's shared representation with the original
        backbone representation space.

        Args:
            alignment_info: Dict from forward_task_with_alignment
            weight: Weight for alignment loss

        Returns:
            Alignment loss scalar
        """
        if not isinstance(self.adapter, AlignmentAwareAdapter):
            # For regular adapters, use simple cosine similarity between backbone and shared repr
            backbone_features = alignment_info['backbone_features']
            shared_repr = alignment_info['shared_repr']

            # Normalize features
            backbone_norm = F.normalize(backbone_features, p=2, dim=1)
            shared_norm = F.normalize(shared_repr, p=2, dim=1)

            # Project to same dimension for comparison (use first N dims)
            min_dim = min(backbone_features.shape[1], shared_repr.shape[1])
            backbone_proj = backbone_norm[:, :min_dim]
            shared_proj = shared_norm[:, :min_dim]

            # Cosine similarity loss
            cosine_sim = torch.sum(backbone_proj * shared_proj, dim=1)
            alignment_loss = -cosine_sim.mean() * weight

            return alignment_loss

        else:
            # Use alignment-aware adapter's built-in alignment loss
            adapter_features = alignment_info['adapter_alignment_features']
            backbone_features = alignment_info['backbone_alignment_features']

            alignment_loss = self.adapter.compute_alignment_loss(
                adapter_features, backbone_features
            ) * weight

            return alignment_loss

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract client-specific features from backbone (unchanged for compatibility)."""
        return self.backbone(x)

    def project_shared(self, features: torch.Tensor) -> torch.Tensor:
        """Project backbone features to shared space via adapter (unchanged for compatibility)."""
        return self.adapter(features)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end shared representation (unchanged for compatibility)."""
        features = self.extract_features(x)
        return self.project_shared(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: use new aligned task prediction."""
        return self.forward_task(x)

    def get_model_info(self) -> dict:
        """Get detailed model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())

        adapter_info = {}
        if hasattr(self.adapter, 'get_info_summary'):
            adapter_info = self.adapter.get_info_summary()

        return {
            'backbone_type': type(self.backbone).__name__,
            'backbone_output_dim': self.backbone.output_dim,
            'adapter_type': type(self.adapter).__name__,
            'adapter_info': adapter_info,
            'classifier_input_dim': self.classifier.in_features,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'use_alignment_loss': self.use_alignment_loss,
            'alignment_aware': isinstance(self.adapter, AlignmentAwareAdapter)
        }


# For backward compatibility, alias the original model
class ClientModel(nn.Module):
    """Original ClientModel - kept for backward compatibility.

    WARNING: This has the architectural alignment issue. Use ImprovedClientModel instead.
    """

    def __init__(self, backbone: FeatureBackbone, adapter: AdapterNetwork, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        self.backbone = backbone
        self.adapter = adapter
        self.num_classes = num_classes

        # ISSUE: Classifier works on backbone output, not adapter output (misaligned)
        self.classifier = nn.Linear(backbone.output_dim, num_classes)

    def forward_task(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project_shared(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter(features)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.project_shared(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_task(x)