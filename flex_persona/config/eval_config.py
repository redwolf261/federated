"""Evaluation configuration schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Metric and reporting toggles for experiments."""

    track_mean_accuracy: bool = True
    track_worst_client_accuracy: bool = True
    track_worst_group_accuracy: bool = True
    track_communication_overhead: bool = True
    track_convergence: bool = True

    def validate(self) -> None:
        enabled = [
            self.track_mean_accuracy,
            self.track_worst_client_accuracy,
            self.track_worst_group_accuracy,
            self.track_communication_overhead,
            self.track_convergence,
        ]
        if not any(enabled):
            raise ValueError("At least one evaluation metric must be enabled")
