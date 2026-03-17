"""Tensor conversion and normalization transforms for local datasets."""

from __future__ import annotations

import numpy as np
import torch


def normalize_uint8_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize uint8-like image tensors to [0, 1] float range."""
    tensor = tensor.float()
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    return tensor


def numpy_images_to_nchw_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert NHWC uint8 arrays to normalized NCHW tensors."""
    tensor = torch.from_numpy(images)
    tensor = normalize_uint8_image_tensor(tensor)
    return tensor.permute(0, 3, 1, 2).contiguous()


def numpy_grayscale_to_nchw_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert N x H x W arrays to normalized N x 1 x H x W tensors."""
    tensor = torch.from_numpy(images)
    tensor = normalize_uint8_image_tensor(tensor)
    return tensor.unsqueeze(1).contiguous()
