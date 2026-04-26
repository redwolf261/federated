"""Phase-wise hyperparameter tuning for FLEX-Persona under severe Non-IID.

This runner implements a reproducible and resumable tuning workflow:
- Fixed protocol (dataset, clients, alpha, seeds)
- Phase 1 random coarse sweep over high-impact knobs
- Phase 2 focused neighbor sweep around top Phase 1 configs
- Deterministic objective score prioritizing worst-client accuracy
- JSON logging with per-run and per-config aggregates
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator


DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_PHASE1_SPACE = {
    "lambda_cluster": [0.02, 0.05, 0.1, 0.2, 0.3],
    "lambda_cluster_center": [0.0, 0.005, 0.01, 0.02],
    "cluster_center_warmup_rounds": [3, 5, 8, 12],
    "local_epochs": [1, 2, 3, 5],
    "cluster_aware_epochs": [1, 2, 3, 5],
    "learning_rate": [1e-3, 2e-3, 3e-3, 5e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [32, 64],
}


@dataclass(frozen=True)
class TuneParams:
    lambda_cluster: float
    lambda_cluster_center: float
    cluster_center_warmup_rounds: int
    local_epochs: int
    cluster_aware_epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int

    def key(self) -> str:
        return (
            f"lc={self.lambda_cluster}|lcc={self.lambda_cluster_center}|"
            f"wu={self.cluster_center_warmup_rounds}|le={self.local_epochs}|"
            f"ce={self.cluster_aware_epochs}|lr={self.learning_rate}|"
            f"wd={self.weight_decay}|bs={self.batch_size}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune FLEX-Persona hyperparameters")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="1")
    parser.add_argument("--dataset", choices=["femnist", "cifar100"], default="femnist")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-samples-per-client", type=int, default=2000)
    parser.add_argument("--shared-dim", type=int, default=64)
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--rng-seed", type=int, default=20260403)
    parser.add_argument("--max-configs-phase1", type=int, default=24)
    parser.add_argument("--max-configs-phase2", type=int, default=12)
    parser.add_argument("--topk-phase1", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(WORKSPACE_ROOT / "outputs" / "tuning" / "flex_tuning_results.json"),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(WORKSPACE_ROOT / "outputs" / "tuning" / "flex_tuning_summary.md"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny quick validation setup")
    return parser.parse_args()


def objective_score(mean_acc: float, worst_acc: float, p10: float, p90: float) -> float:
    gap = p90 - p10
    return float(worst_acc + (0.25 * mean_acc) - (0.1 * gap))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def build_phase1_candidates(rng: random.Random, max_count: int) -> list[TuneParams]:
    keys = list(DEFAULT_PHASE1_SPACE.keys())
    value_lists = [DEFAULT_PHASE1_SPACE[k] for k in keys]

    all_candidates: list[TuneParams] = []
    for combo in itertools.product(*value_lists):
        kwargs = {k: v for k, v in zip(keys, combo, strict=True)}
        all_candidates.append(TuneParams(**kwargs))

    rng.shuffle(all_candidates)
    return all_candidates[: max(1, max_count)]


def _neighbor(values: list[float] | list[int], current: float | int) -> list[float | int]:
    idx = values.index(current)
    picks = [values[idx]]
    if idx - 1 >= 0:
        picks.append(values[idx - 1])
    if idx + 1 < len(values):
        picks.append(values[idx + 1])
    return picks


def build_phase2_candidates(
    top_configs: list[TuneParams],
    rng: random.Random,
    max_count: int,
) -> list[TuneParams]:
    if not top_configs:
        return []

    candidates: list[TuneParams] = []
    for base in top_configs:
        lc_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["lambda_cluster"], base.lambda_cluster)
        lcc_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["lambda_cluster_center"], base.lambda_cluster_center)
        warm_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["cluster_center_warmup_rounds"], base.cluster_center_warmup_rounds)
        le_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["local_epochs"], base.local_epochs)
        ce_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["cluster_aware_epochs"], base.cluster_aware_epochs)
        lr_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["learning_rate"], base.learning_rate)
        wd_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["weight_decay"], base.weight_decay)
        bs_neighbors = _neighbor(DEFAULT_PHASE1_SPACE["batch_size"], base.batch_size)

        combos = itertools.product(
            lc_neighbors,
            lcc_neighbors,
            warm_neighbors,
            le_neighbors,
            ce_neighbors,
            lr_neighbors,
            wd_neighbors,
            bs_neighbors,
        )
        for combo in combos:
            candidates.append(
                TuneParams(
                    lambda_cluster=float(combo[0]),
                    lambda_cluster_center=float(combo[1]),
                    cluster_center_warmup_rounds=int(combo[2]),
                    local_epochs=int(combo[3]),
                    cluster_aware_epochs=int(combo[4]),
                    learning_rate=float(combo[5]),
                    weight_decay=float(combo[6]),
                    batch_size=int(combo[7]),
                )
            )

    dedup: dict[str, TuneParams] = {c.key(): c for c in candidates}
    dedup_values = list(dedup.values())
    rng.shuffle(dedup_values)
    return dedup_values[: max(1, max_count)]


def make_config(args: argparse.Namespace, params: TuneParams, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=f"tune_flex_s{seed}",
        dataset_name=args.dataset,
        num_clients=int(args.num_clients),
        random_seed=int(seed),
        output_dir=str(WORKSPACE_ROOT / "outputs" / "tuning"),
        partition_mode="dirichlet",
        dirichlet_alpha=float(args.alpha),
    )

    cfg.model.num_classes = 62 if args.dataset == "femnist" else 100
    cfg.model.shared_dim = int(args.shared_dim)
    cfg.model.client_backbones = ["small_cnn"] * cfg.num_clients

    cfg.clustering.num_clusters = int(args.num_clusters)
    cfg.similarity.sigma = float(args.sigma)

    cfg.training.aggregation_mode = "prototype"
    cfg.training.rounds = int(args.rounds)
    cfg.training.local_epochs = int(params.local_epochs)
    cfg.training.cluster_aware_epochs = int(params.cluster_aware_epochs)
    cfg.training.batch_size = int(params.batch_size)
    cfg.training.learning_rate = float(params.learning_rate)
    cfg.training.weight_decay = float(params.weight_decay)
    cfg.training.lambda_cluster = float(params.lambda_cluster)
    cfg.training.lambda_cluster_center = float(params.lambda_cluster_center)
    cfg.training.cluster_center_warmup_rounds = int(params.cluster_center_warmup_rounds)
    cfg.training.max_samples_per_client = int(args.max_samples_per_client)

    cfg.validate()
    return cfg


def run_once(args: argparse.Namespace, params: TuneParams, seed: int) -> dict[str, Any]:
    cfg = make_config(args, params, seed)
    sim = FederatedSimulator(workspace_root=WORKSPACE_ROOT, config=cfg)
    history = sim.run_experiment()

    final_eval = history[-1].metadata.get("evaluation", {}) if history else {}
    client_acc = {client.client_id: float(client.evaluate_accuracy()) for client in sim.clients}
    values = list(client_acc.values())

    p10 = percentile(values, 10)
    p90 = percentile(values, 90)
    mean_acc = float(final_eval.get("mean_client_accuracy", 0.0))
    worst_acc = float(final_eval.get("worst_client_accuracy", 0.0))

    score = objective_score(mean_acc=mean_acc, worst_acc=worst_acc, p10=p10, p90=p90)

    report = sim.build_report(history)
    comm = report.get("communication", {})
    run_schema = sim.build_run_schema(history)

    return {
        "seed": int(seed),
        "metrics": {
            "mean_accuracy": mean_acc,
            "worst_accuracy": worst_acc,
            "p10": p10,
            "p90": p90,
            "gap": float(p90 - p10),
            "score": score,
            "client_std": float(np.std(np.array(values, dtype=np.float64))) if values else 0.0,
        },
        "client_accuracies": {str(k): float(v) for k, v in client_acc.items()},
        "communication": {
            "total_bytes": int(comm.get("total_bytes", 0)),
            "rounds": int(comm.get("rounds", 0)),
        },
        "run_summary": report.get("run_summary", {}),
        "run_schema": run_schema,
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}

    def collect(path: str) -> list[float]:
        key1, key2 = path.split(".")
        return [float(r[key1][key2]) for r in runs]

    mean_values = collect("metrics.mean_accuracy")
    worst_values = collect("metrics.worst_accuracy")
    p10_values = collect("metrics.p10")
    p90_values = collect("metrics.p90")
    gap_values = collect("metrics.gap")
    score_values = collect("metrics.score")

    return {
        "n_runs": len(runs),
        "mean_accuracy": {
            "mean": float(mean(mean_values)),
            "std": float(stdev(mean_values)) if len(mean_values) > 1 else 0.0,
        },
        "worst_accuracy": {
            "mean": float(mean(worst_values)),
            "std": float(stdev(worst_values)) if len(worst_values) > 1 else 0.0,
        },
        "p10": {
            "mean": float(mean(p10_values)),
            "std": float(stdev(p10_values)) if len(p10_values) > 1 else 0.0,
        },
        "p90": {
            "mean": float(mean(p90_values)),
            "std": float(stdev(p90_values)) if len(p90_values) > 1 else 0.0,
        },
        "gap": {
            "mean": float(mean(gap_values)),
            "std": float(stdev(gap_values)) if len(gap_values) > 1 else 0.0,
        },
        "score": {
            "mean": float(mean(score_values)),
            "std": float(stdev(score_values)) if len(score_values) > 1 else 0.0,
        },
    }


def load_state(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(WORKSPACE_ROOT),
            "protocol": {
                "dataset": args.dataset,
                "num_clients": args.num_clients,
                "alpha": args.alpha,
                "rounds": args.rounds,
                "max_samples_per_client": args.max_samples_per_client,
                "shared_dim": args.shared_dim,
                "num_clusters": args.num_clusters,
                "sigma": args.sigma,
                "seeds": [int(s) for s in args.seeds],
            },
            "objective": "J = worst + 0.25*mean - 0.1*(p90-p10)",
            "phase1_space": DEFAULT_PHASE1_SPACE,
        },
        "runs": [],
        "aggregates": {},
        "rankings": {},
    }


def run_phase(
    phase_name: str,
    phase_candidates: list[TuneParams],
    args: argparse.Namespace,
    state: dict[str, Any],
) -> None:
    seen_run_ids = {r["run_id"] for r in state.get("runs", [])}

    for idx, params in enumerate(phase_candidates, start=1):
        config_id = f"{phase_name}_cfg_{idx:03d}"
        for seed in args.seeds:
            run_id = f"{phase_name}|{params.key()}|seed={seed}"
            if run_id in seen_run_ids:
                continue

            print(
                f"[{phase_name}] config {idx}/{len(phase_candidates)} | seed={seed} | "
                f"params={params.key()}"
            )
            result = run_once(args=args, params=params, seed=int(seed))
            state["runs"].append(
                {
                    "run_id": run_id,
                    "phase": phase_name,
                    "config_id": config_id,
                    "params": asdict(params),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                }
            )
            seen_run_ids.add(run_id)


def recompute_aggregates(state: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    params_by_group: dict[str, dict[str, Any]] = {}
    phase_by_group: dict[str, str] = {}

    for entry in state.get("runs", []):
        group_key = f"{entry['phase']}|{json.dumps(entry['params'], sort_keys=True)}"
        groups.setdefault(group_key, []).append(entry["result"])
        params_by_group[group_key] = entry["params"]
        phase_by_group[group_key] = entry["phase"]

    aggregates: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []

    for key, results in groups.items():
        agg = _aggregate_runs(results)
        aggregates[key] = {
            "phase": phase_by_group[key],
            "params": params_by_group[key],
            "aggregate": agg,
        }
        score_mean = float(agg.get("score", {}).get("mean", float("-inf")))
        ranking_rows.append(
            {
                "key": key,
                "phase": phase_by_group[key],
                "params": params_by_group[key],
                "score_mean": score_mean,
                "worst_mean": float(agg.get("worst_accuracy", {}).get("mean", 0.0)),
                "mean_acc": float(agg.get("mean_accuracy", {}).get("mean", 0.0)),
                "gap_mean": float(agg.get("gap", {}).get("mean", 0.0)),
            }
        )

    ranking_rows.sort(key=lambda r: (r["score_mean"], r["worst_mean"], r["mean_acc"]), reverse=True)

    state["aggregates"] = aggregates
    state["rankings"] = {
        "overall": ranking_rows,
        "top10": ranking_rows[:10],
    }
    state["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()


def persist_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def persist_markdown_summary(path: Path, state: dict[str, Any]) -> None:
    top = state.get("rankings", {}).get("top10", [])
    protocol = state.get("metadata", {}).get("protocol", {})
    objective = state.get("metadata", {}).get("objective", "")

    lines: list[str] = []
    lines.append("# FLEX Hyperparameter Tuning Summary")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- dataset: {protocol.get('dataset')}")
    lines.append(f"- num_clients: {protocol.get('num_clients')}")
    lines.append(f"- alpha: {protocol.get('alpha')}")
    lines.append(f"- rounds: {protocol.get('rounds')}")
    lines.append(f"- max_samples_per_client: {protocol.get('max_samples_per_client')}")
    lines.append(f"- seeds: {protocol.get('seeds')}")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(f"`{objective}`")
    lines.append("")
    lines.append("## Top Configurations")
    lines.append("")
    lines.append("| Rank | Phase | Score | Worst | Mean | Gap |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    for idx, row in enumerate(top, start=1):
        lines.append(
            f"| {idx} | {row.get('phase')} | {float(row.get('score_mean', 0.0)):.4f} | "
            f"{float(row.get('worst_mean', 0.0)):.4f} | {float(row.get('mean_acc', 0.0)):.4f} | "
            f"{float(row.get('gap_mean', 0.0)):.4f} |"
        )

    if top:
        best = top[0]
        lines.append("")
        lines.append("## Best Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(best.get("params", {}), indent=2))
        lines.append("```")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_top_phase1_configs(state: dict[str, Any], topk: int) -> list[TuneParams]:
    top_rows = [r for r in state.get("rankings", {}).get("overall", []) if r.get("phase") == "phase1"]
    selected = top_rows[: max(1, topk)]

    output: list[TuneParams] = []
    for row in selected:
        p = row["params"]
        output.append(
            TuneParams(
                lambda_cluster=float(p["lambda_cluster"]),
                lambda_cluster_center=float(p["lambda_cluster_center"]),
                cluster_center_warmup_rounds=int(p["cluster_center_warmup_rounds"]),
                local_epochs=int(p["local_epochs"]),
                cluster_aware_epochs=int(p["cluster_aware_epochs"]),
                learning_rate=float(p["learning_rate"]),
                weight_decay=float(p["weight_decay"]),
                batch_size=int(p["batch_size"]),
            )
        )
    return output


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.rounds = 1
        args.seeds = [int(args.seeds[0])]
        args.max_configs_phase1 = 1
        args.max_configs_phase2 = 1
        args.max_samples_per_client = min(int(args.max_samples_per_client), 64)

    rng = random.Random(int(args.rng_seed))
    out_path = Path(args.output_json)
    state = load_state(out_path, args)

    run_phase1 = args.phase in {"1", "all"}
    run_phase2 = args.phase in {"2", "all"}

    if run_phase1:
        phase1_candidates = build_phase1_candidates(rng=rng, max_count=int(args.max_configs_phase1))
        run_phase(phase_name="phase1", phase_candidates=phase1_candidates, args=args, state=state)
        recompute_aggregates(state)
        persist_state(out_path, state)

    if run_phase2:
        recompute_aggregates(state)
        top_phase1 = select_top_phase1_configs(state=state, topk=int(args.topk_phase1))
        phase2_candidates = build_phase2_candidates(
            top_configs=top_phase1,
            rng=rng,
            max_count=int(args.max_configs_phase2),
        )
        run_phase(phase_name="phase2", phase_candidates=phase2_candidates, args=args, state=state)
        recompute_aggregates(state)
        persist_state(out_path, state)

    recompute_aggregates(state)
    persist_state(out_path, state)
    persist_markdown_summary(Path(args.output_md), state)

    top = state.get("rankings", {}).get("top10", [])
    print("\nTop configs by objective:")
    for i, row in enumerate(top, start=1):
        print(
            f"#{i:02d} {row['phase']} | score={row['score_mean']:.4f} | "
            f"worst={row['worst_mean']:.4f} | mean={row['mean_acc']:.4f} | gap={row['gap_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
