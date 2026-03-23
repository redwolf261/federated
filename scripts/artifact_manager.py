"""Standardized experiment artifact manager for reproducible research."""

import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

@dataclass
class ExperimentConfig:
    """Complete experiment configuration for reproducibility."""

    # Experiment metadata (required)
    experiment_id: str
    name: str
    description: str
    git_commit_hash: str
    start_timestamp: str

    # Training parameters (required)
    method: str
    regime: str
    dataset_name: str
    num_clients: int
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float

    # Randomization (required)
    seed_list: List[int]

    # Optional parameters with defaults
    end_timestamp: Optional[str] = None
    max_samples_per_client: Optional[int] = None
    collapse_threshold: float = 0.10
    collapse_threshold_sensitive: float = 0.15
    use_clustering: bool = True
    use_guidance: bool = True
    additional_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class SeedResult:
    """Results from a single seed run."""

    # Required fields
    seed: int
    method: str
    regime: str
    mean_accuracy: float
    worst_accuracy: float
    p10_accuracy: float
    bottom3_accuracy: float
    collapsed: bool
    collapsed_sensitive: bool
    stability_variance: float
    rounds_data: List[Dict[str, float]]

    # Optional fields with defaults
    total_bytes_sent: int = 0
    execution_time_seconds: float = 0.0
    bytes_per_round: Optional[List[int]] = None
    errors: Optional[List[str]] = None

@dataclass
class AggregateMetrics:
    """Aggregate statistics across all seeds."""
    method: str
    regime: str
    num_seeds: int

    # Mean accuracy statistics
    mean_accuracy_avg: float
    mean_accuracy_std: float
    mean_accuracy_min: float
    mean_accuracy_max: float
    mean_accuracy_ci_lower: float
    mean_accuracy_ci_upper: float

    # Worst client statistics
    worst_accuracy_avg: float
    worst_accuracy_std: float
    worst_accuracy_min: float
    worst_accuracy_max: float

    # Collapse statistics
    collapse_rate: float
    collapse_count: int
    collapse_rate_sensitive: float
    collapse_count_sensitive: int

    # Stability metrics
    stability_variance_avg: float
    stability_variance_std: float

    # Communication metrics
    avg_bytes_per_round: float
    total_communication_cost: float


class ExperimentArtifactManager:
    """Manages standardized experiment artifacts and reproducibility."""

    def __init__(self, workspace_root: Path, experiment_name: str):
        self.workspace_root = Path(workspace_root)

        # Create experiment directory structure
        self.experiments_dir = self.workspace_root / "experiments"
        self.experiments_dir.mkdir(exist_ok=True, parents=True)

        # Generate unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_hash = hashlib.md5(f"{experiment_name}_{timestamp}".encode()).hexdigest()[:8]
        self.experiment_id = f"{experiment_name}_{timestamp}_{exp_hash}"

        # Create experiment-specific directory
        self.experiment_dir = self.experiments_dir / self.experiment_id
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

        # Define standard artifact paths
        self.config_path = self.experiment_dir / "config.json"
        self.per_seed_results_path = self.experiment_dir / "per_seed_results.json"
        self.aggregate_metrics_path = self.experiment_dir / "aggregate_metrics.json"
        self.summary_path = self.experiment_dir / "summary.md"
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)

        # Initialize collections
        self.seed_results: List[SeedResult] = []
        self.config: Optional[ExperimentConfig] = None

        print(f"Created experiment: {self.experiment_id}")
        print(f"Artifacts directory: {self.experiment_dir}")

    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash for reproducibility."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def initialize_config(
        self,
        experiment_name: str,
        description: str,
        method: str,
        regime: str,
        dataset_name: str,
        seed_list: List[int],
        **kwargs
    ) -> ExperimentConfig:
        """Initialize and save experiment configuration."""

        self.config = ExperimentConfig(
            experiment_id=self.experiment_id,
            name=experiment_name,
            description=description,
            git_commit_hash=self._get_git_commit_hash(),
            start_timestamp=datetime.now().isoformat(),
            method=method,
            regime=regime,
            dataset_name=dataset_name,
            seed_list=seed_list,
            **kwargs
        )

        # Save config immediately
        with open(self.config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"Experiment config saved: {self.config_path}")
        return self.config

    def add_seed_result(self, result: SeedResult) -> None:
        """Add a single seed result."""
        self.seed_results.append(result)

        # Save incrementally for crash safety
        self._save_per_seed_results()

    def _save_per_seed_results(self) -> None:
        """Save all per-seed results."""
        data = [asdict(result) for result in self.seed_results]
        with open(self.per_seed_results_path, 'w') as f:
            json.dump(data, f, indent=2)

    def compute_aggregates(self) -> AggregateMetrics:
        """Compute aggregate statistics from all seed results."""
        if not self.seed_results:
            raise ValueError("No seed results to aggregate")

        # Filter results by method and regime for consistency
        method = self.seed_results[0].method
        regime = self.seed_results[0].regime

        # Extract metrics
        mean_accs = [r.mean_accuracy for r in self.seed_results]
        worst_accs = [r.worst_accuracy for r in self.seed_results]
        collapses = [r.collapsed for r in self.seed_results]
        collapses_sens = [r.collapsed_sensitive for r in self.seed_results]
        stab_vars = [r.stability_variance for r in self.seed_results]

        from statistics import mean, stdev
        import numpy as np

        # Compute confidence intervals (95%)
        mean_acc_ci = np.percentile(mean_accs, [2.5, 97.5]) if len(mean_accs) > 1 else [mean(mean_accs), mean(mean_accs)]

        aggregate = AggregateMetrics(
            method=method,
            regime=regime,
            num_seeds=len(self.seed_results),

            # Mean accuracy stats
            mean_accuracy_avg=mean(mean_accs),
            mean_accuracy_std=stdev(mean_accs) if len(mean_accs) > 1 else 0,
            mean_accuracy_min=min(mean_accs),
            mean_accuracy_max=max(mean_accs),
            mean_accuracy_ci_lower=mean_acc_ci[0],
            mean_accuracy_ci_upper=mean_acc_ci[1],

            # Worst client stats
            worst_accuracy_avg=mean(worst_accs),
            worst_accuracy_std=stdev(worst_accs) if len(worst_accs) > 1 else 0,
            worst_accuracy_min=min(worst_accs),
            worst_accuracy_max=max(worst_accs),

            # Collapse stats
            collapse_rate=sum(collapses) / len(collapses),
            collapse_count=sum(collapses),
            collapse_rate_sensitive=sum(collapses_sens) / len(collapses_sens),
            collapse_count_sensitive=sum(collapses_sens),

            # Stability stats
            stability_variance_avg=mean(stab_vars),
            stability_variance_std=stdev(stab_vars) if len(stab_vars) > 1 else 0,

            # Communication placeholder
            avg_bytes_per_round=0.0,
            total_communication_cost=0.0
        )

        # Save aggregates
        with open(self.aggregate_metrics_path, 'w') as f:
            json.dump(asdict(aggregate), f, indent=2)

        return aggregate

    def generate_summary_report(self, aggregates: AggregateMetrics) -> None:
        """Generate markdown summary report."""

        summary_md = f"""# Experiment Summary: {self.config.name}

## Configuration
- **Experiment ID**: {self.experiment_id}
- **Method**: {self.config.method}
- **Regime**: {self.config.regime}
- **Dataset**: {self.config.dataset_name}
- **Git Commit**: {self.config.git_commit_hash}
- **Start Time**: {self.config.start_timestamp}
- **End Time**: {self.config.end_timestamp}

## Hyperparameters
- **Seeds**: {self.config.seed_list} ({len(self.config.seed_list)} total)
- **Clients**: {self.config.num_clients}
- **Rounds**: {self.config.rounds}
- **Local Epochs**: {self.config.local_epochs}
- **Batch Size**: {self.config.batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Max Samples per Client**: {self.config.max_samples_per_client}

## Results Summary

### Primary Metrics
- **Mean Accuracy**: {aggregates.mean_accuracy_avg:.4f} ± {aggregates.mean_accuracy_std:.4f}
  - Range: [{aggregates.mean_accuracy_min:.4f}, {aggregates.mean_accuracy_max:.4f}]
  - 95% CI: [{aggregates.mean_accuracy_ci_lower:.4f}, {aggregates.mean_accuracy_ci_upper:.4f}]

- **Worst-Client Accuracy**: {aggregates.worst_accuracy_avg:.4f} ± {aggregates.worst_accuracy_std:.4f}
  - Range: [{aggregates.worst_accuracy_min:.4f}, {aggregates.worst_accuracy_max:.4f}]

### Stability Metrics
- **Collapse Rate (< {self.config.collapse_threshold})**: {aggregates.collapse_rate:.1%} ({aggregates.collapse_count}/{aggregates.num_seeds})
- **Collapse Rate Sensitive (< {self.config.collapse_threshold_sensitive})**: {aggregates.collapse_rate_sensitive:.1%} ({aggregates.collapse_count_sensitive}/{aggregates.num_seeds})
- **Stability Variance**: {aggregates.stability_variance_avg:.4f} ± {aggregates.stability_variance_std:.4f}

## Reproducibility
All results can be reproduced using:
- Config file: `{self.config_path.name}`
- Git commit: `{self.config.git_commit_hash}`
- Seed list: {self.config.seed_list}

## Artifacts
- **Configuration**: {self.config_path.name}
- **Per-seed results**: {self.per_seed_results_path.name}
- **Aggregate metrics**: {self.aggregate_metrics_path.name}
- **Plots**: {self.plots_dir.name}/

---
*Generated on {datetime.now().isoformat()}*
"""

        with open(self.summary_path, 'w') as f:
            f.write(summary_md)

        print(f"Summary report saved: {self.summary_path}")

    def finalize_experiment(self) -> AggregateMetrics:
        """Finalize experiment by computing aggregates and generating reports."""

        if self.config:
            self.config.end_timestamp = datetime.now().isoformat()
            with open(self.config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)

        aggregates = self.compute_aggregates()
        self.generate_summary_report(aggregates)

        # Update global experiment registry
        self._update_experiment_registry()

        print(f"[SUCCESS] Experiment {self.experiment_id} finalized")
        print(f"   - {len(self.seed_results)} seeds completed")
        print(f"   - Collapse rate: {aggregates.collapse_rate:.1%}")
        print(f"   - Mean accuracy: {aggregates.mean_accuracy_avg:.4f} ± {aggregates.mean_accuracy_std:.4f}")

        return aggregates

    def _update_experiment_registry(self) -> None:
        """Update master experiment registry."""
        registry_path = self.experiments_dir / "experiment_registry.json"

        # Load existing registry
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"experiments": []}

        # Add this experiment
        experiment_entry = {
            "experiment_id": self.experiment_id,
            "name": self.config.name if self.config else "unknown",
            "method": self.config.method if self.config else "unknown",
            "regime": self.config.regime if self.config else "unknown",
            "num_seeds": len(self.seed_results),
            "start_timestamp": self.config.start_timestamp if self.config else None,
            "end_timestamp": self.config.end_timestamp if self.config else None,
            "git_commit": self.config.git_commit_hash if self.config else "unknown",
            "directory": str(self.experiment_dir.relative_to(self.workspace_root)),
            "status": "completed" if self.config and self.config.end_timestamp else "running"
        }

        registry["experiments"].append(experiment_entry)

        # Save registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        print(f"Updated experiment registry: {registry_path}")


def load_experiment(workspace_root: Path, experiment_id: str) -> Dict[str, Any]:
    """Load a completed experiment by ID."""
    experiments_dir = Path(workspace_root) / "experiments"
    experiment_dir = experiments_dir / experiment_id

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} not found")

    # Load all artifacts
    config_path = experiment_dir / "config.json"
    per_seed_path = experiment_dir / "per_seed_results.json"
    aggregate_path = experiment_dir / "aggregate_metrics.json"

    result = {}

    if config_path.exists():
        with open(config_path, 'r') as f:
            result['config'] = json.load(f)

    if per_seed_path.exists():
        with open(per_seed_path, 'r') as f:
            result['per_seed_results'] = json.load(f)

    if aggregate_path.exists():
        with open(aggregate_path, 'r') as f:
            result['aggregate_metrics'] = json.load(f)

    return result