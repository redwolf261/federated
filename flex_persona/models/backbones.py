"""Backbone networks used for heterogeneous clients."""

from __future__ import annotations

import torch
from torch import nn


class FeatureBackbone(nn.Module):
    """Base class with unified feature dimension contract."""

    output_dim: int


class SmallCNNBackbone(FeatureBackbone):
    def __init__(self, in_channels: int = 3, input_height: int = 28, input_width: int = 28) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # CRITICAL FIX: Remove AdaptiveAvgPool2d to preserve spatial information
            # AdaptiveAvgPool2d((1, 1)) was destroying spatial structure needed for FEMNIST
        )
        # After two 2x2 pooling layers, spatial resolution is quartered.
        pooled_h = max(int(input_height) // 4, 1)
        pooled_w = max(int(input_width) // 4, 1)
        self.output_dim = 128 * pooled_h * pooled_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(start_dim=1)


class MLPBackbone(FeatureBackbone):
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        in_channels, height, width = input_shape
        input_dim = in_channels * height * width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.output_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = out + x
        return self.relu(out)


class ResNet8Backbone(FeatureBackbone):
    """Compact residual backbone suitable for small federated clients."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)
