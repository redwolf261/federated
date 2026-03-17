"""Graph Laplacian construction for spectral clustering."""

from __future__ import annotations

import torch


class GraphLaplacianBuilder:
    """Builds degree and Laplacian matrices from affinity graph."""

    @staticmethod
    def build_degree_matrix(affinity_matrix: torch.Tensor) -> torch.Tensor:
        if affinity_matrix.ndim != 2 or affinity_matrix.shape[0] != affinity_matrix.shape[1]:
            raise ValueError("affinity_matrix must be square")
        degrees = affinity_matrix.sum(dim=1)
        return torch.diag(degrees)

    @classmethod
    def build_unnormalized_laplacian(cls, affinity_matrix: torch.Tensor) -> torch.Tensor:
        degree = cls.build_degree_matrix(affinity_matrix)
        return degree - affinity_matrix

    @classmethod
    def build_normalized_laplacian(cls, affinity_matrix: torch.Tensor) -> torch.Tensor:
        degree = cls.build_degree_matrix(affinity_matrix)
        d = torch.diag(degree)
        inv_sqrt_d = torch.zeros_like(d)
        mask = d > 0
        inv_sqrt_d[mask] = torch.rsqrt(d[mask])
        D_inv_sqrt = torch.diag(inv_sqrt_d)
        identity = torch.eye(affinity_matrix.shape[0], device=affinity_matrix.device)
        return identity - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
