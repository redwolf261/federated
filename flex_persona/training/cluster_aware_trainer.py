"""Cluster-aware training: Local optimization with cluster prototype guidance.

This module implements Phase 3 of FLEX-Persona federated learning:

    Server computes cluster prototypes C = aggregate {p_k : k ∈ cluster}
                            ↓
    Client receives cluster prototypes
                            ↓
    Client trains with: Loss = local_loss + λ × alignment_loss

Where:
    local_loss = classification loss on local task (y_pred vs y_true)
    alignment_loss = MSE between client prototypes and cluster prototypes
                   = mean over classes c: ||p_k^c - C^c||₂²

The alignment loss encourages the client's learned representations (via adapter)
to be similar to the cluster's aggregated prototypes. This achieves personalization
while preserving group coherence: each client learns its own model but is guided by
its cluster's collective knowledge.

Key insight: Only the adapter parameters are significantly affected by the alignment
loss, allowing fine-grained personalization while maintaining shared latent space
structure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.client_model import ClientModel
from ..prototypes.distribution_builder import PrototypeDistributionBuilder
from ..prototypes.prototype_distribution import PrototypeDistribution
from ..prototypes.prototype_extractor import PrototypeExtractor
from ..similarity.wasserstein_distance import WassersteinDistanceCalculator
from .losses import LossComposer
from .optim_factory import OptimizerFactory


class ClusterAwareTrainer:
    """Trains a client model with cluster prototype guidance.

    Each client trains on two objectives:
    1. **Task objective**: Minimize classification loss on local labels
    2. **Alignment objective**: Minimize distance between client prototypes
       and cluster prototypes in the shared latent space

    The combined objective is:
        minimize: L_task(model) + λ_cluster × L_align(model)

    where L_align encourages the adapter to project features toward the cluster's
    aggregated representation structure, enabling personalized yet coherent learning.

    Args:
        None (stateless trainer)

    Example:
        >>> trainer = ClusterAwareTrainer()
        >>> metrics = trainer.train(
        ...     model=client_model,
        ...     train_loader=train_loader,
        ...     device="cuda",
        ...     num_classes=10,
        ...     cluster_distribution=cluster_proto_dist,
        ...     lambda_cluster=0.1,
        ...     cluster_aware_epochs=1,
        ...     learning_rate=0.01,
        ...     weight_decay=1e-5
        ... )
    """

    def __init__(self) -> None:
        self.loss_composer = LossComposer()
        self.distance_calculator = WassersteinDistanceCalculator(prefer_pot=True)
        self._optimizer: torch.optim.Optimizer | None = None
        self._optimizer_signature: tuple[float, float] | None = None

    def _get_optimizer(
        self,
        model: ClientModel,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        signature = (float(learning_rate), float(weight_decay))
        if self._optimizer is None or self._optimizer_signature != signature:
            self._optimizer = OptimizerFactory.adam(
                model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            self._optimizer_signature = signature
        return self._optimizer

    def train(
        self,
        model: ClientModel,
        train_loader: DataLoader,
        device: str,
        num_classes: int,
        cluster_distribution: PrototypeDistribution,
        lambda_cluster: float,
        cluster_aware_epochs: int,
        learning_rate: float,
        weight_decay: float,
    ) -> dict[str, float]:
        """Train a client model with cluster guidance.

        During each epoch and batch:
        1. Compute local task loss (standard classification)
        2. Compute alignment loss (compare client prototypes to cluster prototypes)
        3. Backprop combined loss and update parameters
        4. Track both local and alignment loss components

        Args:
            model: ClientModel with backbone, adapter, and classifier.
            train_loader: DataLoader with (x, y) batches for local data.
            device: Compute device ("cpu" or "cuda").
            num_classes: Number of task classes.
            cluster_distribution: Server-provided cluster prototype distribution
                                 (aggregate of similar clients' representations).
            lambda_cluster: Weight for alignment loss (typically 0.05-0.5).
                           Higher values enforce stronger cluster alignment.
            cluster_aware_epochs: Number of training epochs with cluster guidance.
            learning_rate: Learning rate for optimizer (default Adam).
            weight_decay: L2 regularization coefficient.

        Returns:
            Dictionary of metrics:
            - cluster_local_loss: Mean task loss over all batches
            - cluster_alignment_loss: Mean per-batch alignment loss
            - cluster_loss: Wasserstein distance between final and cluster distributions
            - total_objective: local_loss + λ × cluster_loss
        """
        model.train()
        optimizer = self._get_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
        cluster_class_prototypes = self._cluster_class_prototypes(cluster_distribution, device)

        total_local_loss = 0.0
        total_alignment_loss = 0.0
        total_samples = 0

        for _ in range(cluster_aware_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model.forward_task(x_batch)
                local_loss = self.loss_composer.local_task_loss(logits, y_batch)
                shared_features = model.forward_shared(x_batch)
                cluster_alignment_loss = self._batch_cluster_alignment_loss(
                    shared_features=shared_features,
                    labels=y_batch,
                    cluster_class_prototypes=cluster_class_prototypes,
                )

                total_loss = self.loss_composer.total_loss(
                    local_loss=local_loss,
                    cluster_loss=cluster_alignment_loss,
                    lambda_cluster=lambda_cluster,
                )
                total_loss.backward()
                optimizer.step()

                batch_size = int(y_batch.shape[0])
                total_local_loss += float(local_loss.item()) * batch_size
                total_alignment_loss += float(cluster_alignment_loss.item()) * batch_size
                total_samples += batch_size

        current_distribution = self._compute_current_distribution(
            model=model,
            train_loader=train_loader,
            device=device,
            num_classes=num_classes,
            client_id=cluster_distribution.client_id,
        )

        cluster_loss_value = self.distance_calculator.wasserstein_distance(
            current_distribution,
            cluster_distribution,
        )
        avg_local_loss = total_local_loss / max(total_samples, 1)
        avg_alignment_loss = total_alignment_loss / max(total_samples, 1)
        total_objective = avg_local_loss + (lambda_cluster * cluster_loss_value)

        return {
            "cluster_local_loss": avg_local_loss,
            "cluster_alignment_loss": avg_alignment_loss,
            "cluster_loss": float(cluster_loss_value),
            "total_objective": float(total_objective),
        }

    @staticmethod
    def _cluster_class_prototypes(
        cluster_distribution: PrototypeDistribution,
        device: str,
    ) -> dict[int, torch.Tensor]:
        """Extract per-class prototype tensors from cluster distribution.

        The cluster distribution is a weighted mixture of client prototypes.
        This extracts the mean prototype for each class by aggregating support
        points weighted by the class-wise probability weights.

        Args:
            cluster_distribution: PrototypeDistribution from server aggregation.
            device: Target device for tensors.

        Returns:
            Dictionary mapping class_id → prototype_tensor (shape: [shared_dim]).
            Each prototype is the weighted mean of support points for that class.
        """
        support_points = cluster_distribution.support_points.to(device=device)
        support_labels = cluster_distribution.support_labels.to(device=device)
        weights = cluster_distribution.weights.to(device=device, dtype=support_points.dtype)

        class_prototypes: dict[int, torch.Tensor] = {}
        for label in torch.unique(support_labels):
            mask = support_labels == label
            class_points = support_points[mask]
            class_weights = weights[mask]
            denom = class_weights.sum()
            if bool(denom <= 0):
                continue
            normalized = class_weights / denom
            class_prototypes[int(label.item())] = torch.sum(class_points * normalized.unsqueeze(1), dim=0)

        return class_prototypes

    @staticmethod
    def _batch_cluster_alignment_loss(
        shared_features: torch.Tensor,
        labels: torch.Tensor,
        cluster_class_prototypes: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-batch alignment loss between client and cluster prototypes.

        For each class in the batch:
        1. Extract batch samples belonging to that class
        2. Compute client's mean prototype for that class from batch features
        3. Compute MSE to cluster's mean prototype for that class
        4. Average over all classes

        This loss is differentiable w.r.t. the model parameters (especially the adapter),
        allowing gradient-based optimization to bring the client's learned representations
        into alignment with the cluster structure.

        Args:
            shared_features: Batch features in shared space, shape [batch_size, shared_dim].
            labels: Batch class labels, shape [batch_size].
            cluster_class_prototypes: Dict mapping class_id → cluster prototype tensor.

        Returns:
            Scalar tensor (0 if no valid classes in batch, otherwise MSE loss).
        """
        losses: list[torch.Tensor] = []

        for class_id, cluster_proto in cluster_class_prototypes.items():
            mask = labels == class_id
            if not bool(mask.any()):
                continue
            client_proto = shared_features[mask].mean(dim=0)
            losses.append(F.mse_loss(client_proto, cluster_proto))

        if not losses:
            return shared_features.new_zeros(())
        return torch.stack(losses).mean()

    @staticmethod
    def _compute_current_distribution(
        model: ClientModel,
        train_loader: DataLoader,
        device: str,
        num_classes: int,
        client_id: int,
    ) -> PrototypeDistribution:
        """Compute the client's current prototype distribution after training.

        After cluster-aware training, re-extract all training samples through
        the updated model to compute the new prototype distribution. This is
        compared with the cluster distribution to measure alignment progress.

        Args:
            model: Updated ClientModel with new backbone/adapter parameters.
            train_loader: DataLoader for re-evaluation on training data.
            device: Compute device.
            num_classes: Number of classes.
            client_id: Client identifier for the distribution.

        Returns:
            PrototypeDistribution with updated prototypes reflecting the trained model.
        """
        model.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                z = model.extract_features(x_batch)
                h = model.project_shared(z)
                features.append(h.detach().cpu())
                labels.append(y_batch.detach().cpu())

        if not features:
            raise RuntimeError("Cannot compute cluster-aware loss on empty train_loader")

        all_features = torch.cat(features, dim=0)
        all_labels = torch.cat(labels, dim=0)
        prototype_dict, class_counts = PrototypeExtractor.compute_class_prototypes(
            shared_features=all_features,
            labels=all_labels,
            num_classes=num_classes,
        )
        return PrototypeDistributionBuilder.build_distribution(
            client_id=client_id,
            prototype_dict=prototype_dict,
            class_counts=class_counts,
            num_classes=num_classes,
        )
