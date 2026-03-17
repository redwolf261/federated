"""Shared typing aliases used across modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

Tensor = Any
ClassId = int
ClientId = int
PrototypeDict = Dict[ClassId, Tensor]
ClassCountDict = Dict[ClassId, int]


@dataclass(frozen=True)
class MatrixShapes:
    """Shape metadata for client-level pairwise matrices."""

    num_clients: int

    @property
    def square_shape(self) -> tuple[int, int]:
        return (self.num_clients, self.num_clients)
