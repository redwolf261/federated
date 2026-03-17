"""Run lightweight ablation experiments for FLEX-Persona."""

from __future__ import annotations

import json
from pathlib import Path

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed


def run_variant(workspace_root: Path, name: str, lambda_cluster: float, num_clusters: int) -> dict[str, object]:
    config = ExperimentConfig(experiment_name=name)
    config.training.lambda_cluster = lambda_cluster
    config.clustering.num_clusters = num_clusters
    config.validate()

    set_global_seed(config.random_seed)

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    history = simulator.run_experiment()
    report = simulator.build_report(history)
    return report


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    variants = [
        ("baseline_lambda_0", 0.0, 2),
        ("cluster_strong_lambda", 0.3, 2),
        ("more_clusters", 0.1, 3),
    ]

    outputs: dict[str, object] = {}
    for name, lambda_cluster, num_clusters in variants:
        outputs[name] = run_variant(
            workspace_root=workspace_root,
            name=name,
            lambda_cluster=lambda_cluster,
            num_clusters=num_clusters,
        )

    output_dir = workspace_root / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_reports.json"
    output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    print(f"Ablation runs complete. Report saved to: {output_path}")


if __name__ == "__main__":
    main()
