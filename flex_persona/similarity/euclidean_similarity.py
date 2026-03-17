"""Euclidean prototype similarity baseline (Section 3.6)."""

from __future__ import annotations

import math

import torch

from ..prototypes.prototype_distribution import PrototypeDistribution


class EuclideanSimilarityCalculator:
    """Computes baseline prototype distances and Gaussian-kernel similarities."""

    @staticmethod
    def pairwise_distance(mu_i: PrototypeDistribution, mu_j: PrototypeDistribution) -> float:
        """Compute class-index matched Euclidean prototype distance baseline."""
        labels_i = set(mu_i.support_labels.tolist())
        labels_j = set(mu_j.support_labels.tolist())
        common = sorted(labels_i.intersection(labels_j))

        if not common:
            return float("inf")

        dist = 0.0
        for class_id in common:
            idx_i = int((mu_i.support_labels == class_id).nonzero(as_tuple=True)[0].item())
            idx_j = int((mu_j.support_labels == class_id).nonzero(as_tuple=True)[0].item())
            diff = mu_i.support_points[idx_i] - mu_j.support_points[idx_j]
            dist += float(torch.norm(diff, p=2).item())
        return dist

    @classmethod
    def build_similarity_matrix(
        cls,
        client_distributions: dict[int, PrototypeDistribution],
        sigma: float,
    ) -> torch.Tensor:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")

        client_ids = sorted(client_distributions.keys())
        num_clients = len(client_ids)
        matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)

        for i, cid_i in enumerate(client_ids):
            for j, cid_j in enumerate(client_ids):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                dist = cls.pairwise_distance(client_distributions[cid_i], client_distributions[cid_j])
                if math.isinf(dist):
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = float(math.exp(-(dist / (sigma**2))))

        return matrix
