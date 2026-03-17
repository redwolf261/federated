"""Helper functions for prototype tensor processing."""

from __future__ import annotations

import torch


def class_histogram(labels: torch.Tensor, num_classes: int) -> dict[int, int]:
    """Build a class-count map from label tensor."""
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")

    hist: dict[int, int] = {cls: 0 for cls in range(num_classes)}
    unique_labels, counts = torch.unique(labels, return_counts=True)
    for label, count in zip(unique_labels.tolist(), counts.tolist(), strict=True):
        hist[int(label)] = int(count)
    return hist


def stack_class_prototypes(
    prototype_dict: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert prototype dictionary to aligned support tensors."""
    if not prototype_dict:
        raise ValueError("prototype_dict cannot be empty")

    class_ids = sorted(prototype_dict.keys())
    points = [prototype_dict[class_id] for class_id in class_ids]
    support_points = torch.stack(points, dim=0)
    support_labels = torch.tensor(class_ids, dtype=torch.long, device=support_points.device)
    return support_points, support_labels
