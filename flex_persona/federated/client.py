"""Federated client entity for local training and prototype-based collaboration.

FLEX-Persona client workflow:

    Phase 1: Local Training
    ├─ train_local(): Standard task-specific training
    │  Loss: task_loss(forward_task(x), y)
    │  Optimizes: backbone, adapter, classifier
    │
    └─ Output: Local representations in shared space

    Phase 2: Prototype Extraction (Client → Server)
    ├─ extract_shared_representations(): Compute shared latent features
    │  Features h = adapter(backbone(x)) for all training samples
    │
    ├─ compute_prototype_distribution(): Create per-class prototypes in shared space
    │  Prototypes p_c = mean{h_i : y_i = c} for each class c
    │  Bundled as PrototypeDistribution with support points + weights
    │
    └─ Output: Compact PrototypeDistribution (sent to server)

    Phase 3: Cluster Guidance (Server → Client)
    ├─ apply_cluster_guidance(): Train with cluster prototype guidance
    │  Loss: task_loss + λ_cluster × alignment_loss(client_proto, cluster_proto)
    │  Optimizes: backbone, adapter
    │  Encourages: Client prototypes to align with cluster in shared space
    │
    └─ Output: Updated model tuned to cluster

This design enables:
- **Heterogeneous architectures**: Different backbones mapped to shared space
- **Privacy preservation**: Only prototypes shared, not features/weights
- **Personalization**: Each client balances local task with cluster alignment
- **Efficient communication**: Prototypes are compact vs. full weights
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..models.client_model import ClientModel
from ..prototypes.distribution_builder import PrototypeDistributionBuilder
from ..prototypes.prototype_distribution import PrototypeDistribution
from ..prototypes.prototype_extractor import PrototypeExtractor
from ..training.cluster_aware_trainer import ClusterAwareTrainer
from ..training.local_trainer import LocalTrainer
from .messages import ClientToServerMessage, ServerToClientMessage


@dataclass
class Client:
    """A federated learning client in FLEX-Persona's representation-based paradigm.

    Each client maintains:
    - model: Backbone + adapter + classifier (heterogeneous per client)
    - data: train/eval splits (non-IID local data)
    - prototypes: Per-class summary in shared latent space

    The client participates in federated rounds by:
    1. Training locally to optimize local task + cluster guidance
    2. Extracting shared representations and building prototype distributions
    3. Receiving cluster guidance from the server
    4. Evaluating accuracy on local eval split

    Attributes:
        client_id: Unique client identifier.
        model: ClientModel (backbone + adapter + classifier).
        train_loader: DataLoader for local training.
        eval_loader: DataLoader for local evaluation.
        num_classes: Number of task classes.
        device: Device for computation ("cpu" or "cuda").
    """

    client_id: int
    model: ClientModel
    train_loader: DataLoader
    eval_loader: DataLoader
    num_classes: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.local_trainer = LocalTrainer()
        self.cluster_trainer = ClusterAwareTrainer()
        self._last_cluster_id: int | None = None
        self._last_cluster_distribution: PrototypeDistribution | None = None

    def train_local(
        self,
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        fedprox_mu: float = 0.0,
        reference_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Phase 1: Local task-specific training.

        Trains the backbone, adapter, and classifier to minimize the task loss
        on local data, without any cluster guidance. This is the standard federated
        learning phase where clients learn from their own non-IID data.

        Args:
            local_epochs: Number of local training epochs.
            learning_rate: Learning rate for SGD/Adam.
            weight_decay: L2 regularization coefficient.

        Returns:
            Dictionary with metrics: {"local_loss": float, ...}
        """
        return self.local_trainer.train(
            model=self.model,
            train_loader=self.train_loader,
            device=self.device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fedprox_mu=fedprox_mu,
            reference_state=reference_state,
        )

    def extract_shared_representations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract learned representations in the shared latent space.

        Passes all training samples through backbone → adapter to get shared
        representations h_i. These are later used to compute class prototypes.

        Used in:
        - compute_prototype_distribution() to build prototypes from learned features
        - Typically called after train_local() when the model is well-trained

        Returns:
            Tuple of:
            - features: Tensor of shape (num_samples, adapter.shared_dim) with
                       shared latent representations.
            - labels: Tensor of shape (num_samples,) with class labels.

        Raises:
            RuntimeError: If the client has no training data.
        """
        self.model.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                z = self.model.extract_features(x_batch)
                h = self.model.project_shared(z)
                features.append(h.detach().cpu())
                labels.append(y_batch.detach().cpu())

        if not features:
            raise RuntimeError(f"Client {self.client_id} has no samples")

        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def compute_prototype_distribution(
        self,
    ) -> tuple[PrototypeDistribution, dict[int, torch.Tensor], dict[int, int]]:
        """Phase 2: Build prototype distribution for server communication.

        Extracts representations from all training samples and computes per-class
        prototypes in the shared latent space. These prototypes are sent to the server
        for clustering and generating cluster guidance.

        The prototype distribution is a compact summary of the client's learned
        representations, avoiding the need to send raw features or model weights.

        Returns:
            Tuple of:
            - distribution: PrototypeDistribution object with support_points, weights
                           representing the client's prototype distribution.
            - prototype_dict: Dict mapping class_id → prototype_tensor.
            - class_counts: Dict mapping class_id → count of samples in that class.
        """
        shared_features, labels = self.extract_shared_representations()
        prototype_dict, class_counts = PrototypeExtractor.compute_class_prototypes(
            shared_features=shared_features,
            labels=labels,
            num_classes=self.num_classes,
        )
        distribution = PrototypeDistributionBuilder.build_distribution(
            client_id=self.client_id,
            prototype_dict=prototype_dict,
            class_counts=class_counts,
            num_classes=self.num_classes,
        )
        return distribution, prototype_dict, class_counts

    def build_upload_message(self, round_idx: int) -> ClientToServerMessage:
        distribution, prototype_dict, class_counts = self.compute_prototype_distribution()
        return ClientToServerMessage(
            client_id=self.client_id,
            round_idx=round_idx,
            prototype_distribution=distribution,
            prototype_dict=prototype_dict,
            class_counts=class_counts,
            metadata={"num_samples": int(sum(class_counts.values()))},
        )

    def apply_cluster_guidance(
        self,
        message: ServerToClientMessage,
        cluster_aware_epochs: int,
        learning_rate: float,
        weight_decay: float,
        lambda_cluster: float,
        cluster_feature_mean: torch.Tensor | None = None,
        lambda_cluster_center: float = 0.0,
        cluster_center_warmup_scale: float = 1.0,
    ) -> dict[str, float]:
        self._last_cluster_id = int(message.cluster_id)
        self._last_cluster_distribution = message.cluster_prototype_distribution

        cluster_metrics = self.cluster_trainer.train(
            model=self.model,
            train_loader=self.train_loader,
            device=self.device,
            num_classes=self.num_classes,
            cluster_distribution=message.cluster_prototype_distribution,
            lambda_cluster=lambda_cluster,
            cluster_aware_epochs=cluster_aware_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cluster_feature_mean=cluster_feature_mean,
            lambda_cluster_center=lambda_cluster_center,
            cluster_center_warmup_scale=cluster_center_warmup_scale,
        )

        output = {
            "cluster_id": float(self._last_cluster_id),
            "cluster_support_size": float(message.cluster_prototype_distribution.num_support),
        }
        output.update(cluster_metrics)
        return output

    def evaluate_accuracy(self) -> float:
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in self.eval_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model.forward_task(x_batch)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == y_batch).sum().item())
                total_samples += int(y_batch.shape[0])

        if total_samples == 0:
            return 0.0
        return float(total_correct / total_samples)
