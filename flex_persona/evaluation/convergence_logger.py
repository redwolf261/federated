"""Per-round convergence metric logging utilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceLogger:
    """Stores round-by-round metric traces for analysis."""

    traces: dict[str, list[float]] = field(default_factory=dict)

    def log(self, metric_name: str, value: float) -> None:
        self.traces.setdefault(metric_name, []).append(float(value))

    def as_dict(self) -> dict[str, list[float]]:
        return {name: values[:] for name, values in self.traces.items()}
