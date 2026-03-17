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
