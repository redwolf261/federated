"""Initialization utilities for model components."""

from __future__ import annotations

from torch import nn


def initialize_module_weights(module: nn.Module) -> None:
    """Apply a simple weight init policy for linear and convolutional layers."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.BatchNorm2d):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)
