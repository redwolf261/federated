"""Central server for prototype aggregation, similarity, and clustering."""

from __future__ import annotations

import torch

from ..clustering.cluster_aggregator import ClusterPrototypeAggregator
from ..clustering.spectral_clusterer import SpectralClusterer
from ..prototypes.prototype_distribution import PrototypeDistribution
from ..similarity.similarity_graph_builder import SimilarityGraphBuilder
from ..similarity.wasserstein_distance import WassersteinDistanceCalculator
from .messages import ClientToServerMessage, ServerToClientMessage


class Server:
    """Server-side orchestration for FLEX-Persona clustering rounds."""

    def __init__(self, num_clusters: int, sigma: float, random_state: int = 42) -> None:
        self.num_clusters = num_clusters
        self.sigma = sigma
        self.distance_calculator = WassersteinDistanceCalculator(prefer_pot=True)
        self.graph_builder = SimilarityGraphBuilder()
        self.clusterer = SpectralClusterer(num_clusters=num_clusters, random_state=random_state)
        self.aggregator = ClusterPrototypeAggregator()

        self._client_messages: dict[int, ClientToServerMessage] = {}

    def receive_client_messages(self, messages: list[ClientToServerMessage]) -> None:
        self._client_messages = {msg.client_id: msg for msg in messages}

    @property
    def client_ids(self) -> list[int]:
        return sorted(self._client_messages.keys())

    def get_client_distributions(self) -> dict[int, PrototypeDistribution]:
        return {
            cid: self._client_messages[cid].prototype_distribution
            for cid in self.client_ids
        }

    def compute_wasserstein_matrix(self) -> torch.Tensor:
        return self.distance_calculator.pairwise_wasserstein_matrix(self.get_client_distributions())

    def build_similarity_and_adjacency(
        self,
        distance_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        affinity = self.graph_builder.build_affinity_matrix(distance_matrix, sigma=self.sigma)
        adjacency = self.graph_builder.build_adjacency_matrix(affinity)
        return affinity, adjacency

    def cluster_clients(self, affinity_matrix: torch.Tensor) -> torch.Tensor:
        return self.clusterer.fit_predict(affinity_matrix)

    def compute_cluster_distributions(
        self,
        cluster_assignments: torch.Tensor,
    ) -> dict[int, PrototypeDistribution]:
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
