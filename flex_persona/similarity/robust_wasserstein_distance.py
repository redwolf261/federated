"""Robust Wasserstein distance calculator with fallback handling for edge cases."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.optimize import linprog

from ..prototypes.prototype_distribution import PrototypeDistribution
from .cost_matrix import squared_euclidean_cost_matrix


class RobustWassersteinDistanceCalculator:
    """Computes pairwise W_2^2 distances with robust handling of edge cases."""

    def __init__(self, prefer_pot: bool = True, fallback_distance: float = 1e6) -> None:
        self.prefer_pot = prefer_pot
        self.fallback_distance = fallback_distance  # Large penalty for invalid distributions

    def compute_cost_matrix(
        self,
        mu_i: PrototypeDistribution,
        mu_j: PrototypeDistribution,
    ) -> torch.Tensor:
        return squared_euclidean_cost_matrix(mu_i.support_points, mu_j.support_points)

    def _validate_distribution(self, mu: PrototypeDistribution) -> bool:
        """Check if distribution is valid for Wasserstein distance computation."""
        try:
            # Check basic validity
            mu.validate()

            # Check for empty support
            if mu.num_support <= 0:
                return False

            # Check for zero total weight
            if mu.weights.sum().item() <= 1e-10:
                return False

            # Check for NaN or infinite values
            if torch.isnan(mu.weights).any() or torch.isinf(mu.weights).any():
                return False

            if torch.isnan(mu.support_points).any() or torch.isinf(mu.support_points).any():
                return False

            return True

        except Exception:
            return False

    def wasserstein_distance(
        self,
        mu_i: PrototypeDistribution,
        mu_j: PrototypeDistribution,
    ) -> float:
        """Compute W_2^2(mu_i, mu_j) with robust error handling."""

        # Validate both distributions first
        if not self._validate_distribution(mu_i):
            print(f"Warning: Invalid distribution for client {mu_i.client_id}, using fallback distance")
            return self.fallback_distance

        if not self._validate_distribution(mu_j):
            print(f"Warning: Invalid distribution for client {mu_j.client_id}, using fallback distance")
            return self.fallback_distance

        # Handle identical distributions (should be zero distance)
        if mu_i.client_id == mu_j.client_id:
            return 0.0

        try:
            mu_i_norm = mu_i.normalized()
            mu_j_norm = mu_j.normalized()
        except Exception as e:
            print(f"Warning: Failed to normalize distributions {mu_i.client_id} <-> {mu_j.client_id}: {e}")
            return self.fallback_distance

        # Additional checks after normalization
        if mu_i_norm.weights.sum().item() <= 1e-10 or mu_j_norm.weights.sum().item() <= 1e-10:
            print(f"Warning: Zero weight distributions {mu_i.client_id} <-> {mu_j.client_id}, using fallback")
            return self.fallback_distance

        cost = self.compute_cost_matrix(mu_i_norm, mu_j_norm)
        a = mu_i_norm.weights.detach().cpu().numpy().astype(np.float64)
        b = mu_j_norm.weights.detach().cpu().numpy().astype(np.float64)
        M = cost.detach().cpu().numpy().astype(np.float64)

        # Additional validation on numpy arrays
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(M)):
            print(f"Warning: NaN values in transport problem {mu_i.client_id} <-> {mu_j.client_id}")
            return self.fallback_distance

        if np.sum(a) <= 1e-10 or np.sum(b) <= 1e-10:
            print(f"Warning: Zero marginal sums {mu_i.client_id} <-> {mu_j.client_id}")
            return self.fallback_distance

        # Try POT first if preferred
        if self.prefer_pot:
            pot_value = self._wasserstein_with_pot(a, b, M)
            if pot_value is not None and pot_value >= 0 and not np.isnan(pot_value):
                return pot_value
            print(f"Warning: POT failed for {mu_i.client_id} <-> {mu_j.client_id}, trying LP fallback")

        # Fallback to linear programming
        try:
            return self._wasserstein_with_linprog(a, b, M)
        except Exception as e:
            print(f"Warning: LP solver failed for {mu_i.client_id} <-> {mu_j.client_id}: {e}")
            return self.fallback_distance

    @staticmethod
    def _wasserstein_with_pot(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> Optional[float]:
        try:
            import ot  # type: ignore

            # Additional validation before POT
            if a.shape[0] == 0 or b.shape[0] == 0:
                return None

            value = float(ot.emd2(a, b, M))

            # Validate result
            if np.isnan(value) or np.isinf(value) or value < 0:
                return None

            return value
        except Exception:
            return None

    @staticmethod
    def _wasserstein_with_linprog(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
        n, m = M.shape

        # Validate inputs
        if n == 0 or m == 0:
            raise RuntimeError("Empty cost matrix")

        c = M.flatten()

        # Equality constraints for transport marginals.
        A_eq = np.zeros((n + m, n * m), dtype=np.float64)
        b_eq = np.concatenate([a, b], axis=0)

        for i in range(n):
            A_eq[i, i * m : (i + 1) * m] = 1.0
        for j in range(m):
            A_eq[n + j, j::m] = 1.0

        # Validate constraint matrix
        if np.any(np.isnan(A_eq)) or np.any(np.isnan(b_eq)):
            raise RuntimeError("NaN values in constraint matrix")

        # Check marginal consistency (necessary condition for feasibility)
        if abs(np.sum(a) - np.sum(b)) > 1e-10:
            raise RuntimeError(f"Marginal constraint violation: sum(a)={np.sum(a):.6f}, sum(b)={np.sum(b):.6f}")

        bounds = [(0.0, None)] * (n * m)

        # Use more robust solver options
        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 30}  # Add time limit
        )

        if not result.success:
            # Provide more detailed error information
            raise RuntimeError(
                f"Wasserstein LP failed: {result.message}. "
                f"Status: {result.get('status', 'unknown')}, "
                f"n={n}, m={m}, sum_a={np.sum(a):.6f}, sum_b={np.sum(b):.6f}"
            )

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


# Alias for backward compatibility
WassersteinDistanceCalculator = RobustWassersteinDistanceCalculator