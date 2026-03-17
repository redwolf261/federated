"""Client-server message contracts for each federated round."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..prototypes.prototype_distribution import PrototypeDistribution


@dataclass
class ClientToServerMessage:
    client_id: int
    round_idx: int
    prototype_distribution: PrototypeDistribution
    prototype_dict: dict[int, torch.Tensor]
    class_counts: dict[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerToClientMessage:
    client_id: int
    round_idx: int
    cluster_id: int
    cluster_prototype_distribution: PrototypeDistribution
    cluster_prototype_dict: dict[int, torch.Tensor]
    similarity_weights: dict[int, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
