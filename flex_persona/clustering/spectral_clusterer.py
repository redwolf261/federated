"""Spectral clustering wrapper for client grouping."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from .graph_laplacian import GraphLaplacianBuilder


class SpectralClusterer:
    """Runs spectral embedding on graph Laplacian and clusters with k-means."""

    def __init__(self, num_clusters: int, random_state: int = 42) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        self.num_clusters = num_clusters
        self.random_state = random_state

    def fit_predict(self, affinity_matrix: torch.Tensor) -> torch.Tensor:
        if affinity_matrix.ndim != 2 or affinity_matrix.shape[0] != affinity_matrix.shape[1]:
            raise ValueError("affinity_matrix must be square")

        n = affinity_matrix.shape[0]
        if self.num_clusters > n:
            raise ValueError("num_clusters cannot exceed number of clients")

        laplacian = GraphLaplacianBuilder.build_unnormalized_laplacian(affinity_matrix)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        _ = eigenvalues
        embedding = eigenvectors[:, : self.num_clusters].detach().cpu().numpy()
        embedding = self._row_normalize(embedding)

        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(embedding)
        return torch.from_numpy(labels.astype(np.int64))

    @staticmethod
    def _row_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return matrix / norms
