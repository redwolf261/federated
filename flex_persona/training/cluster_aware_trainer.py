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
from typing import Optional
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

    def _get_optimizer(
        self,
        model: ClientModel,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        # Always create a new optimizer per model for safety
        return OptimizerFactory.adam(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

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
        cluster_feature_mean: Optional[torch.Tensor] = None,
        lambda_cluster_center: float = 0.0,
        cluster_center_warmup_scale: float = 1.0,
        alignment_mode: str = "cluster_prototype",
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
            alignment_mode: Type of alignment signal to use.
                           - cluster_prototype: Use server cluster prototypes (default)
                           - class_centroid: Use per-class centroids from client data
                           - global_centroid: Use single global centroid
                           - random_centroid: Use random fixed centroids
                           - feature_norm: L2 normalize features
                           - variance_min: Minimize intra-batch feature variance

        Returns:
            Dictionary of metrics:
            - cluster_local_loss: Mean task loss over all batches
            - cluster_alignment_loss: Mean per-batch alignment loss
            - cluster_loss: Wasserstein distance between final and cluster distributions
            - total_objective: local_loss + λ × cluster_loss
        """

        import torch
        scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
        model.train()
        optimizer = self._get_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)

        # Pre-compute centroids based on alignment mode
        class_centroids: Optional[dict[int, torch.Tensor]] = None
        global_centroid: Optional[torch.Tensor] = None
        random_centroids: Optional[dict[int, torch.Tensor]] = None

        if alignment_mode == "cluster_prototype":
            cluster_class_prototypes = self._cluster_class_prototypes(cluster_distribution, device)
        elif alignment_mode == "class_centroid":
            class_centroids = self._compute_class_centroids(model, train_loader, device, num_classes)
            cluster_class_prototypes = class_centroids
        elif alignment_mode == "global_centroid":
            global_centroid = self._compute_global_centroid(model, train_loader, device)
            cluster_class_prototypes = None
        elif alignment_mode == "random_centroid":
            random_centroids = self._compute_random_centroids(num_classes, cluster_distribution, device)
            cluster_class_prototypes = random_centroids
        else:
            # feature_norm and variance_min don't use cluster_class_prototypes
            cluster_class_prototypes = None

        total_local_loss = 0.0
        total_alignment_loss = 0.0
        total_samples = 0

        for _ in range(cluster_aware_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
                    logits = model.forward_task(x_batch)
                    local_loss = self.loss_composer.local_task_loss(logits, y_batch)
                    shared_features = model.forward_shared(x_batch)

                    # Compute alignment loss based on mode
                    if alignment_mode == "cluster_prototype":
                        cluster_alignment_loss = self._batch_cluster_alignment_loss(
                            shared_features=shared_features,
                            labels=y_batch,
                            cluster_class_prototypes=cluster_class_prototypes,
                        )
                    elif alignment_mode in ("class_centroid", "random_centroid"):
                        cluster_alignment_loss = self._batch_cluster_alignment_loss(
                            shared_features=shared_features,
                            labels=y_batch,
                            cluster_class_prototypes=cluster_class_prototypes,
                        )
                    elif alignment_mode == "global_centroid":
                        cluster_alignment_loss = self._batch_global_centroid_alignment_loss(
                            shared_features=shared_features,
                            global_centroid=global_centroid,
                        )
                    elif alignment_mode == "feature_norm":
                        cluster_alignment_loss = self._batch_feature_norm_loss(
                            shared_features=shared_features,
                        )
                    elif alignment_mode == "variance_min":
                        cluster_alignment_loss = self._batch_variance_min_loss(
                            shared_features=shared_features,
                        )
                    else:
                        cluster_alignment_loss = shared_features.new_zeros(())

                    # Structured alignment: client batch mean -> assigned cluster center.
                    cluster_center_reg = 0.0
                    if cluster_feature_mean is not None and lambda_cluster_center > 0.0:
                        batch_mean = shared_features.mean(dim=0)
                        cluster_feature_mean = cluster_feature_mean.to(batch_mean.device)
                        cluster_center_reg = torch.norm(batch_mean - cluster_feature_mean, p=2) ** 2

                    total_loss = self.loss_composer.total_loss(
                        local_loss=local_loss,
                        cluster_loss=cluster_alignment_loss,
                        lambda_cluster=lambda_cluster,
                    )
                    if cluster_feature_mean is not None and lambda_cluster_center > 0.0:
                        total_loss = total_loss + (
                            lambda_cluster_center * float(cluster_center_warmup_scale) * cluster_center_reg
                        )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
    def _compute_class_centroids(
        model: ClientModel,
        train_loader: DataLoader,
        device: str,
        num_classes: int,
    ) -> dict[int, torch.Tensor]:
        """Compute per-class centroids from client's own training data.

        Args:
            model: ClientModel to extract features.
            train_loader: DataLoader with training data.
            device: Compute device.
            num_classes: Number of classes.

        Returns:
            Dictionary mapping class_id → centroid_tensor (shape: [shared_dim]).
        """
        model.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                z = model.extract_features(x_batch)
                h = model.project_shared(z)
                features.append(h.detach())
                labels.append(y_batch.detach())

        if not features:
            return {}

        all_features = torch.cat(features, dim=0)
        all_labels = torch.cat(labels, dim=0)

        centroids: dict[int, torch.Tensor] = {}
        for class_id in range(num_classes):
            mask = all_labels == class_id
            if bool(mask.any()):
                centroids[class_id] = all_features[mask].mean(dim=0)
            else:
                centroids[class_id] = torch.zeros(all_features.shape[1], device=device)

        return centroids

    @staticmethod
    def _compute_global_centroid(
        model: ClientModel,
        train_loader: DataLoader,
        device: str,
    ) -> torch.Tensor:
        """Compute a single global centroid from all training data.


        Args:
            model: ClientModel to extract features.
            train_loader: DataLoader with training data.
            device: Compute device.

        Returns:
            Global centroid tensor (shape: [shared_dim]).
        """
        model.eval()
        features: list[torch.Tensor] = []

        with torch.no_grad():
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                z = model.extract_features(x_batch)
                h = model.project_shared(z)
                features.append(h.detach())

        if not features:
            return torch.zeros(1, device=device)

        all_features = torch.cat(features, dim=0)
        return all_features.mean(dim=0)

    @staticmethod
    def _compute_random_centroids(
        num_classes: int,
        cluster_distribution: PrototypeDistribution,
        device: str,
    ) -> dict[int, torch.Tensor]:
        """Compute random fixed centroids per class.

        Args:
            num_classes: Number of classes.
            cluster_distribution: PrototypeDistribution to get dimension.
            device: Compute device.

        Returns:
            Dictionary mapping class_id → random_centroid_tensor (shape: [shared_dim]).
        """
        shared_dim = cluster_distribution.support_points.shape[1]
        generator = torch.Generator(device=device)
        generator.manual_seed(cluster_distribution.client_id + 42)

        centroids: dict[int, torch.Tensor] = {}
        for class_id in range(num_classes):
            centroids[class_id] = torch.randn(shared_dim, generator=generator, device=device)

        return centroids

    @staticmethod
    def _batch_cluster_alignment_loss(
        shared_features: torch.Tensor,
        labels: torch.Tensor,
        cluster_class_prototypes: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-batch alignment loss using cosine similarity between client and cluster prototypes.

        For each class in the batch:
        1. Extract batch samples belonging to that class
        2. Compute client's mean prototype for that class from batch features
        3. Compute 1 - cosine similarity to cluster's mean prototype for that class
        4. Average over all classes

        This loss is differentiable w.r.t. the model parameters (especially the adapter),
        allowing gradient-based optimization to bring the client's learned representations
        into alignment with the cluster structure.

        Args:
            shared_features: Batch features in shared space, shape [batch_size, shared_dim].
            labels: Batch class labels, shape [batch_size].
            cluster_class_prototypes: Dict mapping class_id → cluster prototype tensor.

        Returns:
            Scalar tensor (0 if no valid classes in batch, otherwise mean cosine loss).
        """
        losses: list[torch.Tensor] = []

        for class_id, cluster_proto in cluster_class_prototypes.items():
            mask = labels == class_id
            if not bool(mask.any()):
                continue
            client_proto = shared_features[mask].mean(dim=0)
            cosine_loss = 1.0 - F.cosine_similarity(client_proto, cluster_proto, dim=0)
            losses.append(cosine_loss)

        if not losses:
            return shared_features.new_zeros(())
        return torch.stack(losses).mean()

    @staticmethod
    def _batch_global_centroid_alignment_loss(
        shared_features: torch.Tensor,
        global_centroid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss to a single global centroid.

        All samples are pulled toward the same global centroid, regardless of class.

        Args:
            shared_features: Batch features in shared space, shape [batch_size, shared_dim].
            global_centroid: Global centroid tensor (shape: [shared_dim]).

        Returns:
            Scalar tensor (MSE between features and global centroid).
        """
        global_centroid = global_centroid.to(shared_features.device)
        return F.mse_loss(shared_features, global_centroid.unsqueeze(0).expand_as(shared_features))

    @staticmethod
    def _batch_feature_norm_loss(
        shared_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2 normalization loss on features.

        Encourages features to have unit norm, testing if geometry/scale alone matters.

        Args:
            shared_features: Batch features in shared space, shape [batch_size, shared_dim].

        Returns:
            Scalar tensor (deviation from unit norm).
        """
        norms = torch.norm(shared_features, p=2, dim=1)
        return torch.mean((norms - 1.0) ** 2)

    @staticmethod
    def _batch_variance_min_loss(
        shared_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intra-batch feature variance minimization loss.

        Minimizes the variance of features within the batch, testing generic regularization.

        Args:
            shared_features: Batch features in shared space, shape [batch_size, shared_dim].

        Returns:
            Scalar tensor (intra-batch variance).
        """
        return torch.var(shared_features, dim=0).mean()

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
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                z = model.extract_features(x_batch)
                h = model.project_shared(z)
                features.append(h.detach())
                labels.append(y_batch.detach())

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
