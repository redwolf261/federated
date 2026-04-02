"""Cluster-level prototype distribution aggregation."""

from __future__ import annotations

from collections import defaultdict

import torch

from ..prototypes.prototype_distribution import PrototypeDistribution


class ClusterPrototypeAggregator:
    """Computes representative distributions for client clusters."""

    def aggregate_cluster_distributions(
        self,
        cluster_assignments: torch.Tensor,
        client_ids: list[int],
        client_distributions: dict[int, PrototypeDistribution],
        client_sample_counts: dict[int, int] = None,
    ) -> dict[int, PrototypeDistribution]:
        if cluster_assignments.ndim != 1:
            raise ValueError("cluster_assignments must be 1D")
        if len(client_ids) != int(cluster_assignments.shape[0]):
            raise ValueError("client_ids length must match cluster_assignments length")

        grouped: dict[int, list[PrototypeDistribution]] = defaultdict(list)
        grouped_sample_counts: dict[int, list[int]] = defaultdict(list)
        for index, client_id in enumerate(client_ids):
            cluster_id = int(cluster_assignments[index].item())
            grouped[cluster_id].append(client_distributions[client_id])
            if client_sample_counts is not None:
                grouped_sample_counts[cluster_id].append(client_sample_counts.get(client_id, 1))

        outputs: dict[int, PrototypeDistribution] = {}
        for cluster_id, members in grouped.items():
            sample_counts = grouped_sample_counts[cluster_id] if client_sample_counts is not None else None
            outputs[cluster_id] = self.empirical_mixture_barycenter(
                cluster_id=cluster_id,
                member_distributions=members,
                member_sample_counts=sample_counts,
            )

        return outputs

    def empirical_mixture_barycenter(
        self,
        cluster_id: int,
        member_distributions: list[PrototypeDistribution],
        member_sample_counts: list[int] = None,
    ) -> PrototypeDistribution:
        """Weighted mean of member prototype distributions, weighted by sample counts."""
        if not member_distributions:
            raise ValueError("member_distributions cannot be empty")

        first = member_distributions[0]
        dtype = first.support_points.dtype
        device = first.support_points.device
        num_classes = first.num_classes

        collected_points: list[torch.Tensor] = []
        collected_labels: list[torch.Tensor] = []
        collected_weights: list[torch.Tensor] = []

        if member_sample_counts is not None:
            total_samples = float(sum(member_sample_counts))
            scales = [float(n) / total_samples for n in member_sample_counts]
        else:
            scales = [1.0 / float(len(member_distributions))] * len(member_distributions)

        for dist, scale in zip(member_distributions, scales):
            normalized = dist.normalized()
            collected_points.append(normalized.support_points)
            collected_labels.append(normalized.support_labels)
            collected_weights.append(normalized.weights * scale)

        support_points = torch.cat(collected_points, dim=0)
        support_labels = torch.cat(collected_labels, dim=0)
        weights = torch.cat(collected_weights, dim=0)

        barycenter = PrototypeDistribution(
            client_id=cluster_id,
            support_points=support_points.to(dtype=dtype, device=device),
            support_labels=support_labels.to(device=device),
            weights=weights.to(dtype=dtype, device=device),
            num_classes=num_classes,
        )
        barycenter.validate()
        return barycenter.normalized()

    def wasserstein_barycenter(
        self,
        cluster_id: int,
        member_distributions: list[PrototypeDistribution],
    ) -> PrototypeDistribution:
        """Interface placeholder aligned with Section 3.13 objective contract."""
        return self.empirical_mixture_barycenter(cluster_id, member_distributions)
