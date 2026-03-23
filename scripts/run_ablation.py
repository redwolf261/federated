"""Run multi-seed ablation studies with statistical summaries.

This script is designed for publication-grade comparisons where each variant is run
across multiple seeds and summarized with confidence intervals and significance tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scipy.stats import ttest_ind  # type: ignore

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed


@dataclass(frozen=True)
class AblationVariant:
    name: str
    aggregation_mode: str
    lambda_cluster: float
    num_clusters: int
    shared_dim: int
    sigma: float
    fedavg_backbone: str = "small_cnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed ablation experiments")
    parser.add_argument("--dataset", choices=["femnist", "cifar100"], default="femnist")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--cluster-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples-per-client", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--output-name", type=str, default="ablation_multiseed")
    parser.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    return parser.parse_args()


def _default_variants(preset: str) -> list[AblationVariant]:
    if preset == "smoke":
        return [
            AblationVariant(
                name="fedavg_baseline",
                aggregation_mode="fedavg",
                lambda_cluster=0.0,
                num_clusters=1,
                shared_dim=64,
                sigma=1.0,
            ),
            AblationVariant(
                name="prototype_lambda_0_1",
                aggregation_mode="prototype",
                lambda_cluster=0.1,
                num_clusters=2,
                shared_dim=64,
                sigma=1.0,
            ),
        ]

    return [
        AblationVariant(
            name="fedavg_baseline",
            aggregation_mode="fedavg",
            lambda_cluster=0.0,
            num_clusters=1,
            shared_dim=64,
            sigma=1.0,
        ),
        AblationVariant(
            name="prototype_lambda_0_0",
            aggregation_mode="prototype",
            lambda_cluster=0.0,
            num_clusters=2,
            shared_dim=64,
            sigma=1.0,
        ),
        AblationVariant(
            name="prototype_lambda_0_1",
            aggregation_mode="prototype",
            lambda_cluster=0.1,
            num_clusters=2,
            shared_dim=64,
            sigma=1.0,
        ),
        AblationVariant(
            name="prototype_lambda_0_3",
            aggregation_mode="prototype",
            lambda_cluster=0.3,
            num_clusters=2,
            shared_dim=64,
            sigma=1.0,
        ),
        AblationVariant(
            name="prototype_clusters_3",
            aggregation_mode="prototype",
            lambda_cluster=0.1,
            num_clusters=3,
            shared_dim=64,
            sigma=1.0,
        ),
        AblationVariant(
            name="prototype_shared_dim_32",
            aggregation_mode="prototype",
            lambda_cluster=0.1,
            num_clusters=2,
            shared_dim=32,
            sigma=1.0,
        ),
    ]


def _build_config(args: argparse.Namespace, variant: AblationVariant, seed: int) -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_name=f"{variant.name}_seed_{seed}",
        dataset_name=args.dataset,
        random_seed=int(seed),
    )
    config.num_clients = int(args.num_clients)

    config.training.aggregation_mode = variant.aggregation_mode
    config.training.rounds = int(args.rounds)
    config.training.local_epochs = int(args.local_epochs)
    config.training.cluster_aware_epochs = int(args.cluster_epochs)
    config.training.batch_size = int(args.batch_size)
    config.training.max_samples_per_client = int(args.max_samples_per_client)
    config.training.learning_rate = float(args.learning_rate)
    config.training.weight_decay = float(args.weight_decay)
    config.training.lambda_cluster = float(variant.lambda_cluster)

    config.clustering.num_clusters = int(variant.num_clusters)
    config.similarity.sigma = float(variant.sigma)
    config.model.shared_dim = int(variant.shared_dim)

    if args.dataset == "femnist":
        config.model.num_classes = 62
    else:
        config.model.num_classes = 100

    if variant.aggregation_mode == "fedavg":
        config.model.client_backbones = [variant.fedavg_backbone] * config.num_clients

    config.validate()
    return config


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * (stdev(values) / math.sqrt(len(values)))


def _aggregate_variant_stats(per_seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    mean_values = [float(r["final_metrics"].get("mean_client_accuracy", 0.0)) for r in per_seed_reports]
    worst_values = [float(r["final_metrics"].get("worst_client_accuracy", 0.0)) for r in per_seed_reports]
    comm_values = [float(r["communication"].get("total_bytes", 0)) for r in per_seed_reports]

    return {
        "runs": len(per_seed_reports),
        "mean_client_accuracy": {
            "values": mean_values,
            "mean": mean(mean_values) if mean_values else 0.0,
            "std": stdev(mean_values) if len(mean_values) > 1 else 0.0,
            "ci95": _ci95(mean_values),
        },
        "worst_client_accuracy": {
            "values": worst_values,
            "mean": mean(worst_values) if worst_values else 0.0,
            "std": stdev(worst_values) if len(worst_values) > 1 else 0.0,
            "ci95": _ci95(worst_values),
        },
        "total_communication_bytes": {
            "values": comm_values,
            "mean": mean(comm_values) if comm_values else 0.0,
            "std": stdev(comm_values) if len(comm_values) > 1 else 0.0,
            "ci95": _ci95(comm_values),
        },
    }


def _safe_ttest(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    stat = ttest_ind(a, b, equal_var=False)
    if stat.pvalue is None:
        return None
    return float(stat.pvalue)


def _build_markdown_summary(
    variants: list[AblationVariant],
    aggregate: dict[str, Any],
    baseline_name: str,
) -> str:
    lines: list[str] = []
    lines.append("# Multi-Seed Ablation Summary")
    lines.append("")
    lines.append(f"Baseline for significance: {baseline_name}")
    lines.append("")
    lines.append("| Variant | Mode | Mean Acc (mean±CI95) | Worst Acc (mean±CI95) | Comm Bytes (mean±CI95) | p-value vs baseline (mean acc) |")
    lines.append("|---|---|---:|---:|---:|---:|")

    baseline_values = aggregate[baseline_name]["mean_client_accuracy"]["values"]
    for variant in variants:
        name = variant.name
        agg = aggregate[name]
        mean_acc = agg["mean_client_accuracy"]
        worst_acc = agg["worst_client_accuracy"]
        comm = agg["total_communication_bytes"]

        p_value = _safe_ttest(mean_acc["values"], baseline_values) if name != baseline_name else None
        p_text = "-" if p_value is None else f"{p_value:.4g}"

        lines.append(
            "| "
            + f"{name}"
            + " | "
            + f"{variant.aggregation_mode}"
            + " | "
            + f"{mean_acc['mean']:.4f} ± {mean_acc['ci95']:.4f}"
            + " | "
            + f"{worst_acc['mean']:.4f} ± {worst_acc['ci95']:.4f}"
            + " | "
            + f"{comm['mean']:.0f} ± {comm['ci95']:.0f}"
            + " | "
            + p_text
            + " |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- CI95 uses normal approximation: 1.96 * std/sqrt(n).")
    lines.append("- p-values use Welch's t-test on per-seed mean client accuracy.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    variants = _default_variants(args.preset)

    output_dir = WORKSPACE_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{args.output_name}_{timestamp}"

    raw_results: dict[str, Any] = {
        "meta": {
            "run_name": run_name,
            "dataset": args.dataset,
            "rounds": args.rounds,
            "num_clients": args.num_clients,
            "local_epochs": args.local_epochs,
            "cluster_epochs": args.cluster_epochs,
            "batch_size": args.batch_size,
            "max_samples_per_client": args.max_samples_per_client,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seeds": [int(s) for s in args.seeds],
            "preset": args.preset,
        },
        "variants": {},
    }

    aggregate: dict[str, Any] = {}
    baseline_name = "fedavg_baseline"

    for variant in variants:
        print(f"\n[variant] {variant.name} ({variant.aggregation_mode})")
        per_seed_reports: list[dict[str, Any]] = []

        for seed in args.seeds:
            print(f"  - seed {seed}: running", flush=True)
            config = _build_config(args=args, variant=variant, seed=int(seed))
            set_global_seed(config.random_seed)

            simulator = FederatedSimulator(workspace_root=WORKSPACE_ROOT, config=config)
            history = simulator.run_experiment()
            report = simulator.build_report(history)

            per_seed_reports.append(report)
            print(
                "    done"
                + f" mean_acc={float(report['final_metrics'].get('mean_client_accuracy', 0.0)):.4f}"
                + f" worst_acc={float(report['final_metrics'].get('worst_client_accuracy', 0.0)):.4f}"
                + f" comm={int(report['communication'].get('total_bytes', 0))}",
                flush=True,
            )

        variant_payload = {
            "config": {
                "aggregation_mode": variant.aggregation_mode,
                "lambda_cluster": variant.lambda_cluster,
                "num_clusters": variant.num_clusters,
                "shared_dim": variant.shared_dim,
                "sigma": variant.sigma,
                "fedavg_backbone": variant.fedavg_backbone,
            },
            "runs": per_seed_reports,
        }
        raw_results["variants"][variant.name] = variant_payload
        aggregate[variant.name] = _aggregate_variant_stats(per_seed_reports)

        mean_acc = aggregate[variant.name]["mean_client_accuracy"]["mean"]
        ci95 = aggregate[variant.name]["mean_client_accuracy"]["ci95"]
        print(f"  summary: mean_acc={mean_acc:.4f} ± {ci95:.4f}", flush=True)

    raw_results["aggregate"] = aggregate

    json_path = output_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(raw_results, indent=2), encoding="utf-8")

    md_summary = _build_markdown_summary(variants=variants, aggregate=aggregate, baseline_name=baseline_name)
    md_path = output_dir / f"{run_name}.md"
    md_path.write_text(md_summary, encoding="utf-8")

    print("\nAblation complete.")
    print(f"- JSON: {json_path}")
    print(f"- Summary: {md_path}")


if __name__ == "__main__":
    main()
