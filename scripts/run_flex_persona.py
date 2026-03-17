"""Run a FLEX-Persona experiment from default configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLEX-Persona experiment")
    parser.add_argument("--dataset", choices=["femnist", "cifar100"], default="femnist")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--cluster-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples-per-client", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-cluster", type=float, default=0.1)
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--experiment-name", type=str, default="flex_persona_smoke")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(experiment_name=args.experiment_name, dataset_name=args.dataset)
    config.num_clients = args.num_clients
    config.training.rounds = args.rounds
    config.training.local_epochs = args.local_epochs
    config.training.cluster_aware_epochs = args.cluster_epochs
    config.training.batch_size = args.batch_size
    config.training.max_samples_per_client = args.max_samples_per_client
    config.training.learning_rate = args.learning_rate
    config.training.lambda_cluster = args.lambda_cluster
    config.clustering.num_clusters = args.num_clusters

    # Keep class count aligned with dataset label spaces.
    if args.dataset == "femnist":
        config.model.num_classes = 62
    elif args.dataset == "cifar100":
        config.model.num_classes = 100

    return config


def main() -> None:
    args = parse_args()
    workspace_root = WORKSPACE_ROOT
    config = build_config(args)
    config.validate()

    set_global_seed(config.random_seed)

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    history = simulator.run_experiment()
    report = simulator.build_report(history)

    output_dir = workspace_root / config.output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.experiment_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Run complete. Report saved to: {report_path}")
    if report.get("final_metrics"):
        print("Final metrics:")
        for key, value in report["final_metrics"].items():
            print(f"- {key}: {value:.6f}")


if __name__ == "__main__":
    main()
