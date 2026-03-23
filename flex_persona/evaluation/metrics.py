"""Core evaluation metrics for federated client performance."""

from __future__ import annotations


class Evaluator:
    """Computes aggregate and robustness-oriented metrics."""

    @staticmethod
    def mean_client_accuracy(client_accuracy: dict[int, float]) -> float:
        if not client_accuracy:
            return 0.0
        return float(sum(client_accuracy.values()) / len(client_accuracy))

    @staticmethod
    def worst_client_accuracy(client_accuracy: dict[int, float]) -> float:
        if not client_accuracy:
            return 0.0
        return float(min(client_accuracy.values()))

    @staticmethod
    def p10_client_accuracy(client_accuracy: dict[int, float]) -> float:
        """Return the 10th-percentile client accuracy using nearest-rank."""
        if not client_accuracy:
            return 0.0
        values = sorted(float(v) for v in client_accuracy.values())
        n = len(values)
        rank = max(1, int((0.10 * n) + 0.999999))  # ceil(0.10 * n)
        idx = min(rank - 1, n - 1)
        return float(values[idx])

    @staticmethod
    def bottom_k_client_accuracy(client_accuracy: dict[int, float], k: int = 3) -> float:
        """Return mean accuracy among the bottom-k clients."""
        if not client_accuracy:
            return 0.0
        values = sorted(float(v) for v in client_accuracy.values())
        k_eff = max(1, min(int(k), len(values)))
        return float(sum(values[:k_eff]) / k_eff)
