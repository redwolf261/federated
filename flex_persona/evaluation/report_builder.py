"""Builds experiment summary reports from round-level logs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReportBuilder:
    """Assembles final run reports from tracked metrics."""

    def build(
        self,
        final_round_metrics: dict[str, float],
        communication_summary: dict[str, int],
        convergence_traces: dict[str, list[float]],
    ) -> dict[str, object]:
        return {
            "final_metrics": final_round_metrics,
            "communication": communication_summary,
            "convergence": convergence_traces,
        }
