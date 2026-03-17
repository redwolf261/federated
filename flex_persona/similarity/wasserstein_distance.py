"""Wasserstein distance calculator for prototype distributions (Section 3.9)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.optimize import linprog

from ..prototypes.prototype_distribution import PrototypeDistribution
from .cost_matrix import squared_euclidean_cost_matrix


class WassersteinDistanceCalculator:
    """Computes pairwise W_2^2 distances between client prototype distributions."""

    def __init__(self, prefer_pot: bool = True) -> None:
        self.prefer_pot = prefer_pot

    def compute_cost_matrix(
        self,
        mu_i: PrototypeDistribution,
        mu_j: PrototypeDistribution,
    ) -> torch.Tensor:
        return squared_euclidean_cost_matrix(mu_i.support_points, mu_j.support_points)

    def wasserstein_distance(
        self,
        mu_i: PrototypeDistribution,
        mu_j: PrototypeDistribution,
    ) -> float:
        """Compute W_2^2(mu_i, mu_j) using OT with exact LP fallback."""
        mu_i = mu_i.normalized()
        mu_j = mu_j.normalized()

        cost = self.compute_cost_matrix(mu_i, mu_j)
        a = mu_i.weights.detach().cpu().numpy().astype(np.float64)
        b = mu_j.weights.detach().cpu().numpy().astype(np.float64)
        M = cost.detach().cpu().numpy().astype(np.float64)

        if self.prefer_pot:
            pot_value = self._wasserstein_with_pot(a, b, M)
            if pot_value is not None:
                return pot_value

        return self._wasserstein_with_linprog(a, b, M)

    @staticmethod
    def _wasserstein_with_pot(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> Optional[float]:
        try:
            import ot  # type: ignore

            value = float(ot.emd2(a, b, M))
            return value
        except Exception:
            return None

    @staticmethod
    def _wasserstein_with_linprog(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
        n, m = M.shape
        c = M.flatten()

        # Equality constraints for transport marginals.
        A_eq = np.zeros((n + m, n * m), dtype=np.float64)
        b_eq = np.concatenate([a, b], axis=0)

        for i in range(n):
            A_eq[i, i * m : (i + 1) * m] = 1.0
        for j in range(m):
            A_eq[n + j, j::m] = 1.0

        bounds = [(0.0, None)] * (n * m)
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(f"Wasserstein LP failed: {result.message}")
        return float(result.fun)

    def pairwise_wasserstein_matrix(
        self,
        client_distributions: dict[int, PrototypeDistribution],
    ) -> torch.Tensor:
        client_ids = sorted(client_distributions.keys())
        num_clients = len(client_ids)
        matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)

        for i, cid_i in enumerate(client_ids):
            for j, cid_j in enumerate(client_ids):
                if i == j:
                    matrix[i, j] = 0.0
                    continue
                if j < i:
                    matrix[i, j] = matrix[j, i]
                    continue
                dist = self.wasserstein_distance(
                    client_distributions[cid_i],
                    client_distributions[cid_j],
                )
                matrix[i, j] = float(dist)
                matrix[j, i] = float(dist)

        return matrix
