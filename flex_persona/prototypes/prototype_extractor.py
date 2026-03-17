"""Class prototype extraction from shared representations."""

from __future__ import annotations

import torch

from .prototype_utils import class_histogram


class PrototypeExtractor:
    """Computes class-wise mean prototypes p_{k,c}."""

    @staticmethod
    def compute_class_prototypes(
        shared_features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
        if shared_features.ndim != 2:
            raise ValueError("shared_features must be 2D: [N, d]")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        if shared_features.shape[0] != labels.shape[0]:
            raise ValueError("Feature and label batch size mismatch")

        prototypes: dict[int, torch.Tensor] = {}
        counts = class_histogram(labels, num_classes)

        for class_id in range(num_classes):
            mask = labels == class_id
            if not bool(mask.any()):
                continue
            class_feats = shared_features[mask]
            prototypes[class_id] = class_feats.mean(dim=0)

        return prototypes, counts
