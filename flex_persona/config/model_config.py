"""Model configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Defines model architecture options for heterogeneous clients."""

    num_classes: int = 100
    shared_dim: int = 64
    client_backbones: list[str] = field(
        default_factory=lambda: ["small_cnn", "resnet8", "mlp", "small_cnn"]
    )

    def validate(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if self.shared_dim <= 0:
            raise ValueError("shared_dim must be positive")
        if not self.client_backbones:
            raise ValueError("client_backbones must contain at least one architecture")
