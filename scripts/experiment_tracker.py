"""Comprehensive experiment tracking, logging, and validation framework."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


@dataclass
class SeedResult:
    """Per-seed experimental outcome."""
    seed: int
    method: str
    regime: str
    mean_accuracy: float
    worst_accuracy: float
    rounds_data: list[dict[str, float]] = field(default_factory=list)
    
    def collapsed(self, threshold: float = 0.10) -> bool:
        """Check if run collapsed (final accuracy below threshold)."""
        if not self.rounds_data:
            return False
        final = self.rounds_data[-1].get("mean_client_accuracy", 0)
        return final < threshold
    
    @property
    def stability_variance(self) -> float:
        """Intra-seed variance (lower = more stable)."""
        if not self.rounds_data:
            return 0
        accs = [r.get("mean_client_accuracy", 0) for r in self.rounds_data]
        return float(stdev(accs)) if len(accs) > 1 else 0


@dataclass
class MethodRegimeResults:
    """Aggregated results for method×regime combination."""
    method: str
    regime: str
    seeds: list[SeedResult] = field(default_factory=list)
    
    @property
    def mean_accuracy_avg(self) -> float:
        """Mean accuracy across seeds."""
        means = [s.mean_accuracy for s in self.seeds]
        return mean(means) if means else 0
    
    @property
    def mean_accuracy_std(self) -> float:
        """Standard deviation of mean accuracy."""
        means = [s.mean_accuracy for s in self.seeds]
        return stdev(means) if len(means) > 1 else 0
    
    @property
    def worst_accuracy_avg(self) -> float:
        """Worst-client accuracy across seeds."""
        worsts = [s.worst_accuracy for s in self.seeds]
        return mean(worsts) if worsts else 0
    
    @property
    def worst_accuracy_std(self) -> float:
        """Std dev of worst-client accuracy."""
        worsts = [s.worst_accuracy for s in self.seeds]
        return stdev(worsts) if len(worsts) > 1 else 0
    
    def collapse_rate(self, threshold: float = 0.10) -> float:
        """Percentage of seeds that collapsed."""
        if not self.seeds:
            return 0
        collapses = sum(1 for s in self.seeds if s.collapsed(threshold))
        return collapses / len(self.seeds)
    
    @property
    def stability_variance_avg(self) -> float:
        """Average intra-seed variance (lower = more stable)."""
        variances = [s.stability_variance for s in self.seeds]
        return mean(variances) if variances else 0
    
    def summary_table_row(self) -> str:
        """Format as markdown table row."""
        collapse_pct = int(self.collapse_rate() * 100)
        return (
            f"| {self.method:12} | {self.regime:20} | "
            f"{self.mean_accuracy_avg:.4f}±{self.mean_accuracy_std:.4f} | "
            f"{self.worst_accuracy_avg:.4f}±{self.worst_accuracy_std:.4f} | "
            f"{collapse_pct}% ({int(self.collapse_rate() * len(self.seeds))}/{len(self.seeds)}) |"
        )


class ExperimentTracker:
    """Manages experiment state, logging, and analysis."""
    
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.experiments_dir = self.workspace / "experiments"
        self.experiments_dir.mkdir(exist_ok=True, parents=True)
        self.results: dict[str, MethodRegimeResults] = {}
    
    def add_seed_result(
        self,
        seed: int,
        method: str,
        regime: str,
        mean_accuracy: float,
        worst_accuracy: float,
        rounds_data: list[dict[str, float]] | None = None,
    ) -> None:
        """Register a single-seed result."""
        key = f"{method}_{regime}"
        if key not in self.results:
            self.results[key] = MethodRegimeResults(method=method, regime=regime)
        
        seed_result = SeedResult(
            seed=seed,
            method=method,
            regime=regime,
            mean_accuracy=mean_accuracy,
            worst_accuracy=worst_accuracy,
            rounds_data=rounds_data or [],
        )
        self.results[key].seeds.append(seed_result)
    
    def save_json(self, filename: str = "experiment_results.json") -> Path:
        """Save all results to JSON."""
        data = {
            key: {
                "method": results.method,
                "regime": results.regime,
                "seeds": [asdict(s) for s in results.seeds],
                "summary": {
                    "mean_accuracy_avg": results.mean_accuracy_avg,
                    "mean_accuracy_std": results.mean_accuracy_std,
                    "worst_accuracy_avg": results.worst_accuracy_avg,
                    "worst_accuracy_std": results.worst_accuracy_std,
                    "collapse_rate": results.collapse_rate(),
                    "num_seeds": len(results.seeds),
                }
            }
            for key, results in self.results.items()
        }
        
        filepath = self.experiments_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath
    
    def markdown_summary(self) -> str:
        """Generate markdown summary table."""
        lines = [
            "# Comprehensive Experimental Results\n",
            "## Summary Statistics\n",
            "| Method | Regime | Mean Accuracy | Worst-Client | Collapse Rate |",
            "|--------|--------|---------------|--------------|---------------|",
        ]
        
        for key in sorted(self.results.keys()):
            lines.append(self.results[key].summary_table_row())
        
        return "\n".join(lines)


class DriftAnalyzer:
    """Analyzes client model divergence (drift) over rounds."""
    
    @staticmethod
    def compute_model_distance(model1_dict: dict, model2_dict: dict) -> float:
        """Compute L2 distance between two model state_dicts."""
        total_distance = 0.0
        count = 0
        
        for key in model1_dict:
            if key in model2_dict:
                t1 = model1_dict[key]
                t2 = model2_dict[key]
                # Flatten and compute distance
                distance = np.linalg.norm(t1.cpu().numpy().flatten() - t2.cpu().numpy().flatten())
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    @staticmethod
    def compute_prototype_distance(proto1: dict, proto2: dict) -> float:
        """Compute Wasserstein distance between two prototype distributions."""
        # Simplified: use L2 distance on prototype means
        mean1 = np.array([v for k, v in proto1.items() if "mean" in k])
        mean2 = np.array([v for k, v in proto2.items() if "mean" in k])
        
        if len(mean1) == 0 or len(mean2) == 0:
            return 0
        
        return float(np.linalg.norm(mean1 - mean2))


class ClusteringValidator:
    """Validates clustering quality and coherence."""
    
    @staticmethod
    def compute_cluster_purity(
        cluster_assignments: dict[int, int],  # client_id -> cluster_id
        true_groups: dict[int, int],  # client_id -> true_group_id
    ) -> float:
        """Compute cluster purity (higher = better clustering)."""
        from collections import defaultdict
        
        cluster_to_clients = defaultdict(list)
        for client, cluster in cluster_assignments.items():
            cluster_to_clients[cluster].append(client)
        
        correct = 0
        total = len(true_groups)
        
        for cluster, clients in cluster_to_clients.items():
            group_counts = defaultdict(int)
            for client in clients:
                if client in true_groups:
                    group_counts[true_groups[client]] += 1
            
            # Majority vote
            if group_counts:
                correct += max(group_counts.values())
        
        return correct / total if total > 0 else 0
    
    @staticmethod
    def compute_intra_cluster_distance(
        cluster_assignments: dict[int, int],
        client_representations: dict[int, np.ndarray],
    ) -> float:
        """Compute average intra-cluster distance (lower = better)."""
        from collections import defaultdict
        
        cluster_to_clients = defaultdict(list)
        for client, cluster in cluster_assignments.items():
            cluster_to_clients[cluster].append(client)
        
        distances = []
        for cluster, clients in cluster_to_clients.items():
            if len(clients) > 1:
                reps = [client_representations[c] for c in clients if c in client_representations]
                if len(reps) > 1:
                    for i, r1 in enumerate(reps):
                        for r2 in reps[i+1:]:
                            d = float(np.linalg.norm(r1 - r2))
                            distances.append(d)
        
        return mean(distances) if distances else 0
