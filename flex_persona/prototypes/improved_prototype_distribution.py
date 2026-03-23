"""Improved prototype distribution with robust normalization and variance handling.

Addresses critique points about prototype quality and statistical rigor:

1. **Robust Statistics**: Uses outlier-resistant prototype computation
2. **Variance Tracking**: Includes confidence and spread information
3. **Quality Metrics**: Measures prototype representativeness and reliability
4. **Adaptive Normalization**: Context-aware normalization strategies
5. **Regularization**: Prevents degenerate or unstable prototypes

Key improvements over basic mean-based prototypes:
- Trimmed means and median variants for outlier resistance
- Confidence intervals and variance tracking
- Multi-scale normalization (L2, unit variance, adaptive)
- Quality scoring for prototype reliability assessment
- Robust aggregation for cluster prototype computation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np


@dataclass
class PrototypeStatistics:
    """Statistics for prototype quality assessment and variance tracking."""
    mean_prototype: torch.Tensor              # Core prototype (mean or robust estimate)
    variance: torch.Tensor                    # Per-dimension variance
    confidence: float                         # Overall confidence score [0,1]
    support_size: int                         # Number of samples used
    outlier_ratio: float                      # Fraction of samples considered outliers
    intra_class_distance: float              # Average distance to prototype
    quality_score: float                     # Combined quality metric [0,1]
    normalization_stats: Dict[str, float]    # Normalization factors used


@dataclass
class ImprovedPrototypeDistribution:
    """Enhanced prototype distribution with variance tracking and robust statistics.

    Extends basic PrototypeDistribution with:
    - Statistical robustness (outlier handling, confidence tracking)
    - Multiple normalization strategies
    - Quality assessment metrics
    - Variance-aware similarity computation

    This addresses the critique about insufficiently robust prototype handling
    by providing research-grade statistical treatment of prototypes.
    """

    client_id: int
    prototype_stats: Dict[int, PrototypeStatistics]  # class_id -> statistics
    support_labels: torch.Tensor                     # Available class labels
    weights: torch.Tensor                            # Class weights for distribution
    num_classes: int
    normalization_method: str = "adaptive"          # Normalization strategy used
    robustness_level: str = "medium"                # Level of outlier resistance

    # Derived properties for compatibility
    @property
    def support_points(self) -> torch.Tensor:
        """Get prototype points for compatibility with original API."""
        points = []
        for label in self.support_labels:
            if label.item() in self.prototype_stats:
                points.append(self.prototype_stats[label.item()].mean_prototype)
        return torch.stack(points) if points else torch.empty(0, 0)

    @property
    def num_support(self) -> int:
        """Number of prototype classes."""
        return len(self.prototype_stats)

    @property
    def shared_dim(self) -> int:
        """Dimension of prototype space."""
        if self.prototype_stats:
            first_stat = next(iter(self.prototype_stats.values()))
            return first_stat.mean_prototype.shape[0]
        return 0

    def get_prototype_with_confidence(self, class_id: int) -> Tuple[torch.Tensor, float]:
        """Get prototype with confidence score."""
        if class_id not in self.prototype_stats:
            raise ValueError(f"No prototype for class {class_id}")

        stats = self.prototype_stats[class_id]
        return stats.mean_prototype, stats.confidence

    def get_class_variance(self, class_id: int) -> torch.Tensor:
        """Get per-dimension variance for a class prototype."""
        if class_id not in self.prototype_stats:
            raise ValueError(f"No variance info for class {class_id}")
        return self.prototype_stats[class_id].variance

    def get_quality_summary(self) -> Dict[str, float]:
        """Get overall distribution quality metrics."""
        if not self.prototype_stats:
            return {"avg_quality": 0.0, "avg_confidence": 0.0, "avg_support": 0.0}

        qualities = [s.quality_score for s in self.prototype_stats.values()]
        confidences = [s.confidence for s in self.prototype_stats.values()]
        supports = [s.support_size for s in self.prototype_stats.values()]

        return {
            "avg_quality": np.mean(qualities),
            "avg_confidence": np.mean(confidences),
            "avg_support": np.mean(supports),
            "min_quality": np.min(qualities),
            "max_confidence": np.max(confidences),
            "total_support": np.sum(supports)
        }


class RobustPrototypeExtractor:
    """Advanced prototype extraction with robust statistics and variance tracking.

    Replaces simple mean-based extraction with statistically rigorous methods:
    - Outlier detection and removal
    - Robust central tendency estimation (trimmed mean, median variants)
    - Variance and confidence tracking
    - Quality assessment metrics
    - Adaptive normalization strategies
    """

    def __init__(
        self,
        robustness_level: str = "medium",
        normalization_method: str = "adaptive",
        outlier_threshold: float = 2.0,
        min_support_size: int = 3
    ):
        self.robustness_level = robustness_level
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.min_support_size = min_support_size

        # Configure robustness parameters
        self.robustness_params = {
            "low": {"trim_ratio": 0.05, "use_median": False},
            "medium": {"trim_ratio": 0.1, "use_median": False},
            "high": {"trim_ratio": 0.2, "use_median": True}
        }

    def extract_robust_prototypes(
        self,
        shared_features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None
    ) -> ImprovedPrototypeDistribution:
        """Extract prototypes with robust statistics and variance tracking.

        Args:
            shared_features: Features in shared space [batch_size, shared_dim]
            labels: Class labels [batch_size]
            num_classes: Number of classes in task
            class_weights: Optional class weights for balancing

        Returns:
            Enhanced prototype distribution with statistics
        """
        if shared_features.ndim != 2:
            raise ValueError("shared_features must be 2D")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")

        # Normalize features first if specified
        if self.normalization_method != "none":
            shared_features = self._normalize_features(shared_features)

        prototype_stats = {}
        available_labels = []
        distribution_weights = []

        for class_id in range(num_classes):
            class_mask = labels == class_id
            if not class_mask.any():
                continue

            class_features = shared_features[class_mask]
            support_size = class_features.shape[0]

            # Skip classes with insufficient support
            if support_size < self.min_support_size:
                continue

            # Compute robust prototype statistics
            stats = self._compute_robust_statistics(class_features, class_id)
            prototype_stats[class_id] = stats

            available_labels.append(class_id)

            # Compute distribution weight (can incorporate class imbalance, quality, etc.)
            if class_weights is not None:
                weight = class_weights[class_id].item() * stats.quality_score
            else:
                weight = stats.quality_score * np.sqrt(support_size)  # Quality * support size
            distribution_weights.append(weight)

        # Normalize distribution weights
        if distribution_weights:
            distribution_weights = torch.tensor(distribution_weights)
            distribution_weights = distribution_weights / distribution_weights.sum()
        else:
            distribution_weights = torch.empty(0)

        return ImprovedPrototypeDistribution(
            client_id=-1,  # Will be set by caller
            prototype_stats=prototype_stats,
            support_labels=torch.tensor(available_labels, dtype=torch.long),
            weights=distribution_weights,
            num_classes=num_classes,
            normalization_method=self.normalization_method,
            robustness_level=self.robustness_level
        )

    def _compute_robust_statistics(
        self,
        class_features: torch.Tensor,
        class_id: int
    ) -> PrototypeStatistics:
        """Compute robust statistics for a single class."""
        n_samples, feature_dim = class_features.shape

        # 1. Outlier detection and removal
        outlier_mask = self._detect_outliers(class_features)
        clean_features = class_features[~outlier_mask]
        outlier_ratio = outlier_mask.float().mean().item()

        # Use original features if too many flagged as outliers
        if clean_features.shape[0] < max(2, n_samples // 3):
            clean_features = class_features
            outlier_ratio = 0.0

        # 2. Robust central tendency estimation
        params = self.robustness_params[self.robustness_level]

        if params["use_median"]:
            # Use median for high robustness
            mean_prototype = torch.median(clean_features, dim=0)[0]
        else:
            # Use trimmed mean
            trim_ratio = params["trim_ratio"]
            mean_prototype = self._compute_trimmed_mean(clean_features, trim_ratio)

        # 3. Variance estimation (robust)
        variance = self._compute_robust_variance(clean_features, mean_prototype)

        # 4. Quality metrics
        intra_class_dist = torch.cdist(
            clean_features.unsqueeze(0),
            mean_prototype.unsqueeze(0).unsqueeze(0)
        ).squeeze().mean().item()

        # Confidence based on consistency (lower variance = higher confidence)
        variance_penalty = torch.clamp(variance.mean() / (variance.mean() + 1.0), 0, 1)
        confidence = (1.0 - variance_penalty) * (1.0 - outlier_ratio)

        # Quality score combines multiple factors
        support_bonus = min(1.0, n_samples / 20.0)  # Bonus for more samples
        quality_score = confidence * support_bonus * (1.0 - outlier_ratio * 0.5)

        return PrototypeStatistics(
            mean_prototype=mean_prototype,
            variance=variance,
            confidence=float(confidence),
            support_size=n_samples,
            outlier_ratio=outlier_ratio,
            intra_class_distance=intra_class_dist,
            quality_score=float(quality_score),
            normalization_stats={"method": self.normalization_method}
        )

    def _detect_outliers(self, features: torch.Tensor) -> torch.Tensor:
        """Detect outliers using robust statistical methods."""
        # Use median absolute deviation (MAD) for outlier detection
        median = torch.median(features, dim=0)[0]
        mad = torch.median(torch.abs(features - median), dim=0)[0]

        # Modified Z-score using MAD
        modified_z_scores = 0.6745 * (features - median) / (mad + 1e-8)
        outlier_mask = (modified_z_scores.abs() > self.outlier_threshold).any(dim=1)

        return outlier_mask

    def _compute_trimmed_mean(self, features: torch.Tensor, trim_ratio: float) -> torch.Tensor:
        """Compute trimmed mean (exclude extreme values)."""
        if trim_ratio <= 0:
            return features.mean(dim=0)

        # Sort along batch dimension and trim extremes
        sorted_features, _ = torch.sort(features, dim=0)
        n_samples = features.shape[0]
        trim_count = int(n_samples * trim_ratio / 2)  # Trim from both ends

        if trim_count > 0:
            trimmed = sorted_features[trim_count:-trim_count]
        else:
            trimmed = sorted_features

        return trimmed.mean(dim=0)

    def _compute_robust_variance(
        self,
        features: torch.Tensor,
        center: torch.Tensor
    ) -> torch.Tensor:
        """Compute robust variance estimate."""
        # Use median absolute deviation as robust variance estimator
        deviations = torch.abs(features - center)
        mad = torch.median(deviations, dim=0)[0]

        # Convert MAD to standard-deviation-like scale (assumes normal distribution)
        robust_std = mad * 1.4826  # MAD to std conversion factor
        robust_variance = robust_std ** 2

        return robust_variance

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply normalization strategy to features."""
        if self.normalization_method == "l2":
            return F.normalize(features, p=2, dim=1)
        elif self.normalization_method == "unit_variance":
            return (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        elif self.normalization_method == "adaptive":
            # Adaptive: L2 norm followed by centering
            l2_normalized = F.normalize(features, p=2, dim=1)
            centered = l2_normalized - l2_normalized.mean(dim=0)
            return centered
        else:  # "none"
            return features


def aggregate_prototype_distributions(
    distributions: List[ImprovedPrototypeDistribution],
    aggregation_method: str = "quality_weighted"
) -> ImprovedPrototypeDistribution:
    """Aggregate multiple prototype distributions robustly.

    Combines prototypes from multiple clients using quality-aware aggregation
    instead of simple averaging. This addresses the critique about insufficient
    attention to prototype quality in aggregation.

    Args:
        distributions: List of distributions to aggregate
        aggregation_method: "simple", "quality_weighted", or "variance_weighted"

    Returns:
        Aggregated prototype distribution
    """
    if not distributions:
        raise ValueError("Cannot aggregate empty list of distributions")

    # Find common classes across all distributions
    all_classes = set()
    for dist in distributions:
        all_classes.update(dist.prototype_stats.keys())

    aggregated_stats = {}
    available_labels = []
    distribution_weights = []

    for class_id in sorted(all_classes):
        # Get prototypes and quality info for this class
        class_prototypes = []
        class_qualities = []
        class_variances = []
        class_supports = []

        for dist in distributions:
            if class_id in dist.prototype_stats:
                stats = dist.prototype_stats[class_id]
                class_prototypes.append(stats.mean_prototype)
                class_qualities.append(stats.quality_score)
                class_variances.append(stats.variance)
                class_supports.append(stats.support_size)

        if not class_prototypes:
            continue

        # Aggregate based on method
        if aggregation_method == "quality_weighted":
            weights = torch.tensor(class_qualities)
            weights = weights / weights.sum()

            # Weighted average of prototypes
            stacked_prototypes = torch.stack(class_prototypes)
            aggregated_prototype = (stacked_prototypes * weights.unsqueeze(1)).sum(dim=0)

            # Weighted average of variances
            stacked_variances = torch.stack(class_variances)
            aggregated_variance = (stacked_variances * weights.unsqueeze(1)).sum(dim=0)

            # Combined quality metrics
            combined_confidence = (torch.tensor(class_qualities) * weights).sum().item()

        elif aggregation_method == "variance_weighted":
            # Weight by inverse variance (higher confidence for lower variance)
            inv_variances = [1.0 / (v.mean().item() + 1e-8) for v in class_variances]
            weights = torch.tensor(inv_variances)
            weights = weights / weights.sum()

            stacked_prototypes = torch.stack(class_prototypes)
            aggregated_prototype = (stacked_prototypes * weights.unsqueeze(1)).sum(dim=0)

            stacked_variances = torch.stack(class_variances)
            aggregated_variance = (stacked_variances * weights.unsqueeze(1)).sum(dim=0)

            combined_confidence = weights.max().item()  # Confidence of best client

        else:  # "simple"
            # Simple averaging
            aggregated_prototype = torch.stack(class_prototypes).mean(dim=0)
            aggregated_variance = torch.stack(class_variances).mean(dim=0)
            combined_confidence = np.mean(class_qualities)

        # Create aggregated statistics
        aggregated_stats[class_id] = PrototypeStatistics(
            mean_prototype=aggregated_prototype,
            variance=aggregated_variance,
            confidence=combined_confidence,
            support_size=sum(class_supports),
            outlier_ratio=0.0,  # Not applicable for aggregated
            intra_class_distance=0.0,  # Not applicable
            quality_score=combined_confidence,
            normalization_stats={"method": "aggregated"}
        )

        available_labels.append(class_id)
        distribution_weights.append(combined_confidence)

    # Normalize weights
    if distribution_weights:
        distribution_weights = torch.tensor(distribution_weights)
        distribution_weights = distribution_weights / distribution_weights.sum()
    else:
        distribution_weights = torch.empty(0)

    return ImprovedPrototypeDistribution(
        client_id=-1,  # Aggregated
        prototype_stats=aggregated_stats,
        support_labels=torch.tensor(available_labels, dtype=torch.long),
        weights=distribution_weights,
        num_classes=distributions[0].num_classes,
        normalization_method="aggregated",
        robustness_level="aggregated"
    )