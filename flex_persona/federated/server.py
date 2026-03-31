"""Central server for prototype clustering and personalized guidance generation.

The server is the orchestrator of FLEX-Persona's representation-based clustering:

    Round t:
    [Receive Phase]
    ├─ Client k sends: μ_k (prototype distribution in shared space)
    │
    [Clustering Phase]
    ├─ compute_wasserstein_matrix(): W_ij = Wasserstein distance(μ_i, μ_j)
    │  → Measures similarity between clients' learned representations
    │
    ├─ build_similarity_and_adjacency(): Affinity = exp(-W² / σ²)
    │  → Converts distances to affinities for spectral clustering
    │
    ├─ cluster_clients(): Spectral clustering on affinity matrix
    │  → Groups similar clients (similar learned representations)
    │
    [Aggregation Phase]
    ├─ compute_cluster_distributions(): C_c = aggregate {μ_k : k ∈ cluster c}
    │  → Combines prototypes of similar clients into cluster knowledge
    │
    [Broadcast Phase]
    └─ build_broadcast_messages(): Send C_c back to client k
       → Client k receives cluster prototype guidance

This enables personalized cluster-aware training where each client:
1. Maintains its private model and adapter
2. Learns from its local data
3. Is guided by its cluster's aggregated knowledge
4. Improves through both local task loss and cluster alignment
"""

from __future__ import annotations

import torch

from ..clustering.cluster_aggregator import ClusterPrototypeAggregator
from ..clustering.spectral_clusterer import SpectralClusterer
from ..prototypes.prototype_distribution import PrototypeDistribution
from .messages import ClientToServerMessage, ServerToClientMessage


class Server:
    """Central server orchestrating representation clustering and guidance generation.

    Receives prototype distributions from clients, clusters them based on Wasserstein
    distance in the shared latent space, aggregates cluster knowledge, and sends
    personalized cluster guidance back to each client.

    Args:
        num_clusters: Number of client clusters to discover (K).
        sigma: Bandwidth parameter for exponential kernel in similarity affinity.
               Controls smoothness of affinity = exp(-distance²/σ²).
        random_state: Random seed for spectral clustering.

    The server workflow for each round:
    1. receive_client_messages(): Collect client prototype distributions
    2. compute_wasserstein_matrix(): Pairwise similarity in shared space
    3. build_similarity_and_adjacency(): RBF affinity + adjacency graph
    4. cluster_clients(): Spectral clustering on affinity
    5. compute_cluster_distributions(): Aggregate prototypes per cluster
    6. build_broadcast_messages(): Create personalized guidance messages

    This enables cross-architecture collaboration: clients with different model
    architectures collaborate through their unified prototype distributions.
    """

    def __init__(self, num_clusters: int, sigma: float, random_state: int = 42) -> None:
        self.num_clusters = num_clusters
        self.sigma = sigma
        self.clusterer = SpectralClusterer(num_clusters=num_clusters, random_state=random_state)
        self.aggregator = ClusterPrototypeAggregator()
        self._client_feature_means: dict[int, torch.Tensor] = {}
        self._client_messages: dict[int, ClientToServerMessage] = {}

    def receive_client_feature_means(self, feature_means: dict[int, torch.Tensor]) -> None:
        """Store client feature means for this round (FRepAlign)."""
        self._client_feature_means = feature_means.copy()

    def receive_client_messages(self, messages: list[ClientToServerMessage]) -> None:
        self._client_messages = {msg.client_id: msg for msg in messages}

    @property
    def client_ids(self) -> list[int]:
        """Sorted list of client IDs that sent messages this round."""
        return sorted(self._client_messages.keys())

    def get_client_distributions(self) -> dict[int, PrototypeDistribution]:
        """Extract all client prototype distributions from received messages.

        Returns:
            Dict mapping client_id → PrototypeDistribution in shared latent space.
        """
        return {
            cid: self._client_messages[cid].prototype_distribution
            for cid in self.client_ids
        }

    def compute_feature_mean_similarity_matrix(self) -> torch.Tensor:
        """Compute pairwise cosine similarity between client feature means."""
        if not self._client_feature_means:
            raise ValueError("No client feature means received.")
        client_ids = sorted(self._client_feature_means.keys())
        means = torch.stack([self._client_feature_means[cid] for cid in client_ids], dim=0)  # [N, D]
        # Normalize for cosine similarity
        means_norm = means / (means.norm(dim=1, keepdim=True) + 1e-8)
        cosine = means_norm @ means_norm.t()  # [-1, 1]
        similarity = (cosine + 1.0) * 0.5  # [0, 1]
        similarity.fill_diagonal_(1.0)
        return similarity

    def build_similarity_and_adjacency(self, similarity_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build adjacency matrix for clustering from similarity matrix."""
        # For FRepAlign, similarity_matrix is already cosine similarity [N, N]
        # Optionally, threshold or kNN for adjacency
        N = similarity_matrix.shape[0]
        k = min(self.num_clusters + 1, N)  # include self in kNN neighborhood
        adjacency = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        for i in range(N):
            topk = torch.topk(similarity_matrix[i], k=k).indices
            adjacency[i, topk] = True
        return similarity_matrix, adjacency

    def cluster_clients(self, affinity_matrix: torch.Tensor) -> torch.Tensor:
        """Perform spectral clustering on similarity affinity matrix.

        Uses spectral clustering algorithm to partition clients into num_clusters
        groups based on their representation similarity.

        Args:
            affinity_matrix: Similarity affinity matrix from build_similarity_and_adjacency().

        Returns:
            Tensor of shape [num_clients] where cluster_assignments[i] = cluster_id
            for client i. cluster_id ∈ [0, num_clusters-1].
        """
        return self.clusterer.fit_predict(affinity_matrix)

    def compute_cluster_distributions(
        self,
        cluster_assignments: torch.Tensor,
    ) -> dict[int, PrototypeDistribution]:
        """Aggregate client prototypes to compute cluster-level distributions.

        For each cluster c, aggregates the prototype distributions of all clients
        assigned to cluster c. This creates a unified cluster prototype distribution
        that represents the collective learned representation of the cluster.

        These cluster prototypes are sent back to clients for personalized guidance:
        clients in cluster c train to align with C_c's prototypes.

        Args:
            cluster_assignments: Cluster assignment tensor from cluster_clients().

        Returns:
            Dict mapping cluster_id → PrototypeDistribution (aggregated cluster prototypes).
        """
        return self.aggregator.aggregate_cluster_distributions(
            cluster_assignments=cluster_assignments,
            client_ids=self.client_ids,
            client_distributions=self.get_client_distributions(),
        )

    def build_broadcast_messages(
        self,
        round_idx: int,
        cluster_assignments: torch.Tensor,
        cluster_distributions: dict[int, PrototypeDistribution],
        affinity_matrix: torch.Tensor,
    ) -> list[ServerToClientMessage]:
        """Create personalized guidance messages to send back to clients.

        For each client k in cluster c:
        1. Include the cluster's aggregated prototype distribution C_c
        2. Include per-class prototypes for efficient batch loss computation
        3. Include similarity weights to other clients (affinity_matrix[k])
        4. Include cluster metadata (size, etc.)

        Args:
            round_idx: Current federated round number.
            cluster_assignments: Tensor mapping client_idx → cluster_id.
            cluster_distributions: Dict mapping cluster_id → aggregated prototypes.
            affinity_matrix: Similarity weights between all client pairs.

        Returns:
            List of ServerToClientMessage, one per client, containing cluster guidance.
        """
        messages: list[ServerToClientMessage] = []

        for row_idx, client_id in enumerate(self.client_ids):
            cluster_id = int(cluster_assignments[row_idx].item())
            cluster_dist = cluster_distributions[cluster_id]

            support_dict = {
                int(label.item()): cluster_dist.support_points[idx]
                for idx, label in enumerate(cluster_dist.support_labels)
            }

            similarity_row = affinity_matrix[row_idx].tolist()
            similarity_weights = {
                other_client_id: float(similarity_row[col_idx])
                for col_idx, other_client_id in enumerate(self.client_ids)
            }

            messages.append(
                ServerToClientMessage(
                    client_id=client_id,
                    round_idx=round_idx,
                    cluster_id=cluster_id,
                    cluster_prototype_distribution=cluster_dist,
                    cluster_prototype_dict=support_dict,
                    similarity_weights=similarity_weights,
                    metadata={"cluster_size": int((cluster_assignments == cluster_id).sum().item())},
                )
            )

        return messages
