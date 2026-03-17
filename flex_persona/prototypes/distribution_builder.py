"""Build weighted prototype distributions from class prototypes and counts."""

from __future__ import annotations

import torch

from .prototype_distribution import PrototypeDistribution
from .prototype_utils import stack_class_prototypes


class PrototypeDistributionBuilder:
    """Creates mu_k from class prototypes and empirical class frequencies."""

    @staticmethod
    def build_distribution(
        client_id: int,
        prototype_dict: dict[int, torch.Tensor],
        class_counts: dict[int, int],
        num_classes: int,
    ) -> PrototypeDistribution:
        if not prototype_dict:
            raise ValueError("prototype_dict cannot be empty")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        total_samples = float(sum(class_counts.values()))
        if total_samples <= 0.0:
            raise ValueError("Total class_counts must be positive")

        support_points, support_labels = stack_class_prototypes(prototype_dict)

        weight_values = []
        for class_id in support_labels.tolist():
            count = class_counts.get(int(class_id), 0)
            weight_values.append(float(count) / total_samples)

        weights = torch.tensor(
            weight_values,
            dtype=support_points.dtype,
            device=support_points.device,
        )

        distribution = PrototypeDistribution(
            client_id=client_id,
            support_points=support_points,
            support_labels=support_labels,
            weights=weights,
            num_classes=num_classes,
        )
        distribution.validate()
        return distribution.normalized()
