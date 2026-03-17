"""Unified client model wrapper (backbone + adapter + classifier)."""

from __future__ import annotations

import torch
from torch import nn

from .adapter_network import AdapterNetwork
from .backbones import FeatureBackbone


class ClientModel(nn.Module):
    """Client network that exposes task and representation interfaces."""

    def __init__(self, backbone: FeatureBackbone, adapter: AdapterNetwork, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        self.backbone = backbone
        self.adapter = adapter
        self.num_classes = num_classes
        self.classifier = nn.Linear(backbone.output_dim, num_classes)

    def forward_task(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project_shared(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter(features)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.project_shared(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_task(x)
