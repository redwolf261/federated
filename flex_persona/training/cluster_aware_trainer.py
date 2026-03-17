"""Cluster-aware optimization routine using cluster prototype guidance."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..models.client_model import ClientModel
from ..prototypes.distribution_builder import PrototypeDistributionBuilder
from ..prototypes.prototype_distribution import PrototypeDistribution
from ..prototypes.prototype_extractor import PrototypeExtractor
from ..similarity.wasserstein_distance import WassersteinDistanceCalculator
from .losses import LossComposer
from .optim_factory import OptimizerFactory


class ClusterAwareTrainer:
    """Runs local updates and reports cluster-aware objective components."""

    def __init__(self) -> None:
        self.loss_composer = LossComposer()
        self.distance_calculator = WassersteinDistanceCalculator(prefer_pot=True)

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
        model.train()
        optimizer = OptimizerFactory.adam(model, learning_rate=learning_rate, weight_decay=weight_decay)

        total_local_loss = 0.0
        total_samples = 0

        for _ in range(cluster_aware_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model.forward_task(x_batch)
                local_loss = self.loss_composer.local_task_loss(logits, y_batch)

                # Cluster regularization value is computed at epoch scope below.
                total_loss = self.loss_composer.total_loss(
                    local_loss=local_loss,
                    cluster_loss=torch.zeros((), device=local_loss.device),
                    lambda_cluster=lambda_cluster,
                )
                total_loss.backward()
                optimizer.step()

                batch_size = int(y_batch.shape[0])
                total_local_loss += float(local_loss.item()) * batch_size
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
        total_objective = avg_local_loss + (lambda_cluster * cluster_loss_value)

        return {
            "cluster_local_loss": avg_local_loss,
            "cluster_loss": float(cluster_loss_value),
            "total_objective": float(total_objective),
        }

    @staticmethod
    def _compute_current_distribution(
        model: ClientModel,
        train_loader: DataLoader,
        device: str,
        num_classes: int,
        client_id: int,
    ) -> PrototypeDistribution:
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
