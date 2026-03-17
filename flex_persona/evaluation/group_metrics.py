"""Group-based robustness metrics."""

from __future__ import annotations


class GroupMetrics:
    """Computes worst-group style metrics from group-level scores."""

    @staticmethod
    def worst_group_accuracy(group_accuracy: dict[str, float]) -> float:
        if not group_accuracy:
            return 0.0
        return float(min(group_accuracy.values()))
