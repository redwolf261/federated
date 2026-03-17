"""Build client similarity graph tensors from pairwise distances."""

from __future__ import annotations

import torch


class SimilarityGraphBuilder:
    """Converts distance matrices to affinity matrices for graph clustering."""

    @staticmethod
    def distance_to_similarity(distance_matrix: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square")

        similarity = torch.exp(-(distance_matrix / (sigma**2)))
        similarity.fill_diagonal_(1.0)
        return similarity

    @staticmethod
    def build_affinity_matrix(distance_matrix: torch.Tensor, sigma: float) -> torch.Tensor:
        return SimilarityGraphBuilder.distance_to_similarity(distance_matrix, sigma)

    @staticmethod
    def build_adjacency_matrix(affinity_matrix: torch.Tensor) -> torch.Tensor:
        if affinity_matrix.ndim != 2 or affinity_matrix.shape[0] != affinity_matrix.shape[1]:
            raise ValueError("affinity_matrix must be square")
        return affinity_matrix.clone()
