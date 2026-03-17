"""Round-level state container used by the federated simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..prototypes.prototype_distribution import PrototypeDistribution


@dataclass
class RoundState:
    round_idx: int
    client_ids: list[int]
    client_distributions: dict[int, PrototypeDistribution] = field(default_factory=dict)
    distance_matrix: torch.Tensor | None = None
    similarity_matrix: torch.Tensor | None = None
    adjacency_matrix: torch.Tensor | None = None
    cluster_assignments: torch.Tensor | None = None
    cluster_distributions: dict[int, PrototypeDistribution] = field(default_factory=dict)
    local_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
