"""Alignment-aware training for FLEX-Persona with representation alignment losses.

Addresses the critique about missing alignment between adapter and classifier spaces
by implementing explicit alignment mechanisms during training.

Key components:
1. **Alignment Loss**: Explicit loss to align adapter and classifier representations
2. **Multi-objective Training**: Balances task loss with alignment constraints
3. **Progressive Alignment**: Gradually increase alignment weight during training
4. **Alignment Metrics**: Track alignment quality throughout training

This enables the adapter and classifier to learn aligned representations instead
of completely separate, uncoordinated feature spaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class AlignmentConfig:
    """Configuration for alignment-aware training."""
    alignment_weight: float = 0.1          # Weight for alignment loss
    alignment_warmup_epochs: int = 5       # Epochs to warm up alignment loss
    alignment_schedule: str = "linear"     # "linear", "cosine", or "constant"
    alignment_target: str = "cosine"       # "cosine", "mse", or "kl"
    temperature: float = 0.1               # Temperature for cosine similarity
    min_alignment_weight: float = 0.01     # Minimum alignment weight


class AlignmentAwareTrainer:
    """Trainer with explicit representation alignment between adapter and classifier.

    This addresses the fundamental issue where adapter and classifier learn
    separate, unaligned representations by enforcing alignment constraints
    during training.

    The trainer implements multi-objective optimization:
    - Task Loss: Standard cross-entropy for classification
    - Alignment Loss: Ensures adapter and classifier representations are aligned
    - Total Loss: Weighted combination with scheduling

    Args:
        model: Model with alignment capabilities (ImprovedClientModel)
        optimizer: Optimizer for model parameters
        alignment_config: Configuration for alignment training
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        alignment_config: AlignmentConfig = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.alignment_config = alignment_config or AlignmentConfig()

        # Check if model supports alignment
        self.supports_alignment = hasattr(model, 'compute_alignment_loss')
        if not self.supports_alignment:
            print("WARNING: Model does not support alignment - will use task loss only")

        # Training state
        self.current_epoch = 0
        self.alignment_weight_history = []
        self.alignment_loss_history = []

    def train_epoch(
        self,
        train_loader,
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        """Train for one epoch with alignment-aware loss.

        Args:
            train_loader: DataLoader for training data
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Dict with training metrics including alignment information
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.train()
        self.current_epoch += 1

        # Calculate current alignment weight with scheduling
        alignment_weight = self._get_alignment_weight()
        self.alignment_weight_history.append(alignment_weight)

        # Training metrics
        total_loss = 0.0
        total_task_loss = 0.0
        total_alignment_loss = 0.0
        correct = 0
        total = 0
        alignment_scores = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Forward pass with alignment if supported
            if self.supports_alignment:
                logits, alignment_info = self.model.forward_task_with_alignment(data)
                task_loss = criterion(logits, targets)

                # Compute alignment loss
                alignment_loss = self.model.compute_alignment_loss(alignment_info)

                # Combined loss
                batch_loss = task_loss + alignment_weight * alignment_loss

                # Track alignment quality
                alignment_score = self._compute_alignment_score(alignment_info)
                alignment_scores.append(alignment_score)

                total_alignment_loss += alignment_loss.item()
            else:
                # Fallback to standard training
                logits = self.model.forward_task(data)
                batch_loss = criterion(logits, targets)
                alignment_loss = torch.tensor(0.0)

            # Backward pass
            batch_loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += batch_loss.item()
            total_task_loss += task_loss.item() if self.supports_alignment else batch_loss.item()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)
        avg_alignment_loss = total_alignment_loss / len(train_loader)
        accuracy = correct / total
        avg_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0

        # Store alignment loss history
        self.alignment_loss_history.append(avg_alignment_loss)

        return {
            'total_loss': avg_loss,
            'task_loss': avg_task_loss,
            'alignment_loss': avg_alignment_loss,
            'accuracy': accuracy,
            'alignment_weight': alignment_weight,
            'alignment_score': avg_alignment_score,
            'epoch': self.current_epoch
        }

    def validate(self, val_loader, criterion: nn.Module = None) -> Dict[str, float]:
        """Validate model with alignment metrics.

        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Dict with validation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()

        total_loss = 0.0
        total_task_loss = 0.0
        total_alignment_loss = 0.0
        correct = 0
        total = 0
        alignment_scores = []

        with torch.no_grad():
            for data, targets in val_loader:
                if self.supports_alignment:
                    logits, alignment_info = self.model.forward_task_with_alignment(data)
                    task_loss = criterion(logits, targets)
                    alignment_loss = self.model.compute_alignment_loss(alignment_info)

                    alignment_weight = self._get_alignment_weight()
                    batch_loss = task_loss + alignment_weight * alignment_loss

                    alignment_score = self._compute_alignment_score(alignment_info)
                    alignment_scores.append(alignment_score)

                    total_alignment_loss += alignment_loss.item()
                else:
                    logits = self.model.forward_task(data)
                    batch_loss = criterion(logits, targets)
                    task_loss = batch_loss

                total_loss += batch_loss.item()
                total_task_loss += task_loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        accuracy = correct / total
        avg_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0

        return {
            'total_loss': total_loss / len(val_loader),
            'task_loss': total_task_loss / len(val_loader),
            'alignment_loss': total_alignment_loss / len(val_loader),
            'accuracy': accuracy,
            'alignment_score': avg_alignment_score
        }

    def _get_alignment_weight(self) -> float:
        """Calculate current alignment weight based on schedule."""
        config = self.alignment_config

        if self.current_epoch <= config.alignment_warmup_epochs:
            # Warmup phase
            if config.alignment_schedule == "linear":
                weight = (self.current_epoch / config.alignment_warmup_epochs) * config.alignment_weight
            elif config.alignment_schedule == "cosine":
                progress = self.current_epoch / config.alignment_warmup_epochs
                weight = config.alignment_weight * (1 - np.cos(progress * np.pi / 2))
            else:  # constant
                weight = config.alignment_weight
        else:
            # Post-warmup
            weight = config.alignment_weight

        return max(weight, config.min_alignment_weight)

    def _compute_alignment_score(self, alignment_info: Dict[str, torch.Tensor]) -> float:
        """Compute alignment quality score for monitoring.

        Args:
            alignment_info: Dict containing alignment features

        Returns:
            Alignment score (higher = better aligned)
        """
        if not alignment_info:
            return 0.0

        # Try to compute alignment based on available features
        if 'adapter_alignment_features' in alignment_info and 'backbone_alignment_features' in alignment_info:
            # Alignment-aware model
            adapter_features = alignment_info['adapter_alignment_features']
            backbone_features = alignment_info['backbone_alignment_features']

            # Compute mean cosine similarity
            adapter_norm = F.normalize(adapter_features, p=2, dim=1)
            backbone_norm = F.normalize(backbone_features, p=2, dim=1)
            cosine_sim = torch.sum(adapter_norm * backbone_norm, dim=1)

            return cosine_sim.mean().item()

        elif 'backbone_features' in alignment_info and 'shared_repr' in alignment_info:
            # Regular model alignment approximation
            backbone_features = alignment_info['backbone_features']
            shared_repr = alignment_info['shared_repr']

            # Project to same dimensionality for comparison
            min_dim = min(backbone_features.shape[1], shared_repr.shape[1])
            backbone_proj = F.normalize(backbone_features[:, :min_dim], p=2, dim=1)
            shared_proj = F.normalize(shared_repr[:, :min_dim], p=2, dim=1)

            cosine_sim = torch.sum(backbone_proj * shared_proj, dim=1)
            return cosine_sim.mean().item()

        return 0.0

    def get_alignment_summary(self) -> Dict[str, Any]:
        """Get summary of alignment training progress."""
        if not self.alignment_weight_history:
            return {"status": "no_training"}

        return {
            "epochs_trained": len(self.alignment_weight_history),
            "current_alignment_weight": self.alignment_weight_history[-1],
            "final_alignment_loss": self.alignment_loss_history[-1] if self.alignment_loss_history else 0.0,
            "alignment_loss_trend": "decreasing" if len(self.alignment_loss_history) >= 2 and
                                   self.alignment_loss_history[-1] < self.alignment_loss_history[-2] else "stable",
            "supports_alignment": self.supports_alignment,
            "config": self.alignment_config
        }


class ContrastiveAlignmentLoss(nn.Module):
    """Advanced contrastive alignment loss for representation alignment.

    Implements contrastive learning between adapter and classifier representations
    to encourage alignment while maintaining discriminative power.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        adapter_features: torch.Tensor,
        classifier_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive alignment loss.

        Args:
            adapter_features: Features from adapter (batch_size, dim)
            classifier_features: Features from classifier space (batch_size, dim)
            labels: Class labels for contrastive grouping (batch_size,)

        Returns:
            Contrastive alignment loss
        """
        batch_size = adapter_features.size(0)

        # Normalize features
        adapter_norm = F.normalize(adapter_features, p=2, dim=1)
        classifier_norm = F.normalize(classifier_features, p=2, dim=1)

        # Compute similarity matrices
        adapter_sim = torch.matmul(adapter_norm, adapter_norm.t()) / self.temperature
        classifier_sim = torch.matmul(classifier_norm, classifier_norm.t()) / self.temperature

        # Create positive mask (same class)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.t()).float()

        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(batch_size, device=positive_mask.device)

        # Compute contrastive loss between similarity matrices
        sim_diff = torch.abs(adapter_sim - classifier_sim)

        # Positive pairs should have similar similarities
        positive_loss = (positive_mask * sim_diff).sum() / (positive_mask.sum() + 1e-8)

        # Negative pairs should maintain discriminative power
        negative_mask = 1.0 - positive_mask - torch.eye(batch_size, device=positive_mask.device)
        negative_loss = torch.clamp(self.margin - sim_diff, min=0.0)
        negative_loss = (negative_mask * negative_loss).sum() / (negative_mask.sum() + 1e-8)

        return positive_loss + negative_loss