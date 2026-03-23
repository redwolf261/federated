"""Wasserstein distance calculator for prototype distributions (Section 3.9)."""

from __future__ import annotations

# Import the robust implementation
from .robust_wasserstein_distance import RobustWassersteinDistanceCalculator

# Alias for backward compatibility
WassersteinDistanceCalculator = RobustWassersteinDistanceCalculator
