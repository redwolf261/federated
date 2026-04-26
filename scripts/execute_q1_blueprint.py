"""Execute Q1-grade experiment blueprint phases with reproducible artifacts.

Implemented phases:
- Phase 0: system lockdown (determinism checks, environment snapshot, dependency freeze, dataset hashes)
- Phase 1: instrumentation correction validation via schema-compliant run artifact
- Phase 2: scaling failure diagnosis (controlled IID grid)
- Phase 3: local neighbor search (random sampled neighborhood)
- Phase 4: non-IID method comparison (FedAvg vs FLEX)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, cast

import numpy as np
import torch

try:
    from scipy.stats import ttest_rel, wilcoxon  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ttest_rel = None
    wilcoxon = None

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit_hash() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _experiment_contract(cfg: "BlueprintConfig") -> dict[str, Any]:
    return {
        "dataset": cfg.dataset,
        "model_architecture": "small_cnn",
        "shared_dim": 64,
        "preprocessing": "dataset_loader_default",
        "evaluation_metrics": [
            "mean_client_accuracy",
            "worst_client_accuracy",
            "p10_client_accuracy",
            "p90_client_accuracy",
            "p90_minus_p10_gap",
        ],
        "seed_protocol": list(cfg.seeds),
    }


def _prepare_run_directory(phase: str, contract: dict[str, Any], cfg: "BlueprintConfig") -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    commit = _git_commit_hash()
    config_for_hash: dict[str, Any] = {
        "phase": phase,
        "contract": contract,
        "config": asdict(cfg),
        "git_commit": commit,
    }
    config_hash = _stable_hash(config_for_hash)
    run_id = f"run_{timestamp}_{config_hash[:8]}"
    run_dir = WORKSPACE_ROOT / "experiments" / f"phase_{phase}" / run_id
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "git_commit": commit,
        "contract": contract,
    }
    return run_dir, run_meta


def _write_run_artifacts(run_dir: Path, run_meta: dict[str, Any], cfg: "BlueprintConfig", payload: dict[str, Any]) -> None:
    config_doc = {
        "run": run_meta,
        "config": asdict(cfg),
    }
    metrics_doc: dict[str, Any] = {
        "run_id": run_meta["run_id"],
        "phase": run_meta["phase"],
        "timestamp": _now(),
        "summary": {
            k: v
            for k, v in payload.items()
            if k in {"phase", "timestamp", "comparison", "grid_results", "ablation", "statistics", "communication_analysis"}
        },
    }
    raw_doc = {
        "run": run_meta,
        "payload": payload,
    }

    (run_dir / "config.json").write_text(json.dumps(config_doc, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")
    (run_dir / "raw_logs.json").write_text(json.dumps(raw_doc, indent=2), encoding="utf-8")


def _determinism_verification(cfg: "BlueprintConfig") -> dict[str, Any]:
    seed = int(cfg.seeds[0])
    exp1 = _base_config(seed=seed, cfg=cfg, mode="fedavg", partition_mode="iid", alpha=1.0)
    exp2 = _base_config(seed=seed, cfg=cfg, mode="fedavg", partition_mode="iid", alpha=1.0)
    exp1.training.rounds = 1
    exp2.training.rounds = 1
    exp1.training.max_samples_per_client = min(int(cfg.max_samples_per_client), 128)
    exp2.training.max_samples_per_client = min(int(cfg.max_samples_per_client), 128)

    run1 = _global_metrics(_run_and_collect(exp1))
    run2 = _global_metrics(_run_and_collect(exp2))

    tolerance = 1e-12
    mean_diff = abs(run1["mean"] - run2["mean"])
    worst_diff = abs(run1["worst"] - run2["worst"])
    if not (mean_diff < tolerance and worst_diff < tolerance):
        raise AssertionError(
            f"Determinism check failed: mean_diff={mean_diff}, worst_diff={worst_diff}, tolerance={tolerance}"
        )

    return {
        "seed": seed,
        "run1": run1,
        "run2": run2,
        "tolerance": tolerance,
        "mean_diff": mean_diff,
        "worst_diff": worst_diff,
        "passed": True,
    }


def _sample_signature(x: torch.Tensor, y: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(x.detach().cpu().numpy().tobytes())
    h.update(int(y.item()).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def _class_entropy(class_distribution: dict[str, int]) -> float:
    total = float(sum(int(v) for v in class_distribution.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in class_distribution.values():
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log(p + 1e-12)
    return float(entropy)


def _data_integrity_audit(cfg: "BlueprintConfig") -> dict[str, Any]:
    exp = _base_config(seed=int(cfg.seeds[0]), cfg=cfg, mode="prototype", partition_mode="dirichlet", alpha=cfg.alpha)
    exp.training.max_samples_per_client = min(int(cfg.max_samples_per_client), 256)
    manager = ClientDataManager(WORKSPACE_ROOT, exp)
    bundles = cast(list[Any], manager.build_client_bundles())

    source_sets = {int(bundle.client_id): set(cast(list[int], bundle.source_indices)) for bundle in bundles}
    total_source = sum(len(s) for s in source_sets.values())
    source_union: set[int] = set()
    for s in source_sets.values():
        source_union.update(s)
    union_source = len(source_union)
    client_overlap_ok = total_source == union_source

    per_client_train_eval_overlap: dict[str, int] = {}
    cross_client_train_overlap = 0
    all_train_signatures: dict[int, set[str]] = {}

    for bundle_any in bundles:
        train_ds = bundle_any.train_loader.dataset
        eval_ds = bundle_any.eval_loader.dataset
        train_sigs: set[str] = set()
        eval_sigs: set[str] = set()

        if hasattr(train_ds, "tensors"):
            tx, ty = train_ds.tensors
            tx = tx  # type: ignore[assignment]
            ty = ty  # type: ignore[assignment]
            for i in range(int(ty.shape[0])):
                train_sigs.add(_sample_signature(tx[i], ty[i]))

        if hasattr(eval_ds, "tensors"):
            ex, ey = eval_ds.tensors
            ex = ex  # type: ignore[assignment]
            ey = ey  # type: ignore[assignment]
            for i in range(int(ey.shape[0])):
                eval_sigs.add(_sample_signature(ex[i], ey[i]))

        overlap = len(train_sigs.intersection(eval_sigs))
        per_client_train_eval_overlap[str(bundle_any.client_id)] = overlap
        all_train_signatures[int(bundle_any.client_id)] = train_sigs

    client_ids = sorted(all_train_signatures.keys())
    for i, c1 in enumerate(client_ids):
        for c2 in client_ids[i + 1 :]:
            cross_client_train_overlap += len(all_train_signatures[c1].intersection(all_train_signatures[c2]))

    return {
        "partition_fingerprint": manager.partition_fingerprint(bundles),
        "client_overlap_ok": client_overlap_ok,
        "total_source_indices": total_source,
        "unique_source_indices": union_source,
        "per_client_train_eval_overlap": per_client_train_eval_overlap,
        "cross_client_train_overlap": int(cross_client_train_overlap),
        "class_distribution_per_client": {
            str(bundle_any.client_id): {str(k): int(v) for k, v in bundle_any.class_histogram.items()}
            for bundle_any in bundles
        },
    }


def _dataset_snapshot() -> dict[str, Any]:
    targets = [
        WORKSPACE_ROOT / "dataset" / "femnist" / "train-00000-of-00001.parquet",
        WORKSPACE_ROOT / "dataset" / "cifar-100-python" / "train",
        WORKSPACE_ROOT / "dataset" / "cifar-100-python" / "test",
        WORKSPACE_ROOT / "dataset" / "cifar-100-python" / "meta",
    ]

    snapshot: dict[str, Any] = {}
    for path in targets:
        if not path.exists():
            snapshot[str(path.relative_to(WORKSPACE_ROOT))] = {"exists": False}
            continue
        snapshot[str(path.relative_to(WORKSPACE_ROOT))] = {
            "exists": True,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    return snapshot


def _environment_snapshot(python_exe: Path, output_dir: Path) -> dict[str, Any]:
    req_path = output_dir / "requirements_frozen.txt"
    pip_freeze = subprocess.run(
        [str(python_exe), "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    req_path.write_text(pip_freeze.stdout, encoding="utf-8")

    return {
        "captured_at": _now(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "python_executable": str(python_exe),
        "pip_freeze_exit_code": int(pip_freeze.returncode),
        "requirements_file": str(req_path.relative_to(WORKSPACE_ROOT)),
    }


@dataclass(frozen=True)
class BlueprintConfig:
    dataset: str = "femnist"
    num_clients: int = 10
    rounds: int = 10
    max_samples_per_client: int = 2000
    alpha: float = 0.1
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


def _base_config(seed: int, cfg: BlueprintConfig, mode: str, partition_mode: str, alpha: float) -> ExperimentConfig:
    exp = ExperimentConfig(
        experiment_name=f"q1_blueprint_{mode}_s{seed}",
        dataset_name=cfg.dataset,
        num_clients=cfg.num_clients,
        random_seed=seed,
        output_dir=str(WORKSPACE_ROOT / "outputs" / "q1_blueprint"),
        partition_mode=partition_mode,
        dirichlet_alpha=alpha,
    )
    exp.model.num_classes = 62 if cfg.dataset == "femnist" else 100
    exp.model.client_backbones = ["small_cnn"] * cfg.num_clients
    exp.model.shared_dim = 64
    exp.training.rounds = cfg.rounds
    exp.training.max_samples_per_client = cfg.max_samples_per_client
    exp.training.aggregation_mode = mode
    exp.training.batch_size = 32
    exp.training.learning_rate = 0.003
    exp.training.local_epochs = 3
    exp.training.cluster_aware_epochs = 3
    exp.training.lambda_cluster = 0.1
    exp.training.lambda_cluster_center = 0.01
    exp.training.cluster_center_warmup_rounds = 8
    exp.clustering.num_clusters = 2
    exp.similarity.sigma = 1.0
    exp.validate()
    return exp


def _run_and_collect(exp: ExperimentConfig) -> dict[str, Any]:
    sim = FederatedSimulator(workspace_root=WORKSPACE_ROOT, config=exp)
    history = sim.run_experiment()
    return sim.build_run_schema(history)


def _global_metrics(schema: dict[str, Any]) -> dict[str, float]:
    rounds = schema.get("rounds", [])
    if not rounds:
        return {"mean": 0.0, "worst": 0.0, "p10": 0.0, "p90": 0.0, "gap": 0.0}
    gm = rounds[-1].get("global_metrics", {})
    mean_acc = float(gm.get("mean_client_accuracy", 0.0))
    worst_acc = float(gm.get("worst_client_accuracy", 0.0))
    p10 = float(gm.get("p10_client_accuracy", 0.0))
    client_acc = gm.get("client_accuracies", {})
    values = [float(v) for v in client_acc.values()]
    p90 = float(np.percentile(np.array(values, dtype=np.float64), 90)) if values else 0.0
    return {
        "mean": mean_acc,
        "worst": worst_acc,
        "p10": p10,
        "p90": p90,
        "gap": float(p90 - p10),
    }


def phase0_lockdown(cfg: BlueprintConfig, python_exe: Path, out_dir: Path) -> dict[str, Any]:
    set_global_seed(cfg.seeds[0], deterministic=True)
    determinism_check = _determinism_verification(cfg)
    data_audit = _data_integrity_audit(cfg)
    payload: dict[str, Any] = {
        "phase": "phase0",
        "timestamp": _now(),
        "environment": _environment_snapshot(python_exe=python_exe, output_dir=out_dir),
        "dataset_snapshot": _dataset_snapshot(),
        "determinism_verification": determinism_check,
        "data_integrity_audit": data_audit,
    }
    return payload


def phase1_instrumentation(cfg: BlueprintConfig) -> dict[str, Any]:
    exp = _base_config(seed=cfg.seeds[0], cfg=cfg, mode="prototype", partition_mode="dirichlet", alpha=cfg.alpha)
    schema = _run_and_collect(exp)
    return {
        "phase": "phase1",
        "timestamp": _now(),
        "schema_path_hint": "outputs/q1_blueprint/phase1_run_schema.json",
        "run_schema": schema,
    }


def phase2_scaling_diagnosis(cfg: BlueprintConfig) -> dict[str, Any]:
    local_epochs_grid = [1, 3, 5]
    lr_grid = [0.001, 0.003, 0.005]
    batch_grid = [32, 64]

    rows: list[dict[str, Any]] = []

    for local_epochs in local_epochs_grid:
        for lr in lr_grid:
            for batch_size in batch_grid:
                run_metrics: list[dict[str, float]] = []
                for seed in cfg.seeds:
                    exp = _base_config(seed=seed, cfg=cfg, mode="fedavg", partition_mode="iid", alpha=1.0)
                    exp.training.local_epochs = local_epochs
                    exp.training.learning_rate = lr
                    exp.training.batch_size = batch_size
                    schema = _run_and_collect(exp)
                    run_metrics.append(_global_metrics(schema))

                mean_values: list[float] = [float(r["mean"]) for r in run_metrics]
                worst_values: list[float] = [float(r["worst"]) for r in run_metrics]
                gap_values: list[float] = [float(r["gap"]) for r in run_metrics]
                rows.append(
                    {
                        "local_epochs": local_epochs,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "mean": float(mean(mean_values)),
                        "mean_std": float(stdev(mean_values)) if len(mean_values) > 1 else 0.0,
                        "worst": float(mean(worst_values)),
                        "worst_std": float(stdev(worst_values)) if len(worst_values) > 1 else 0.0,
                        "gap": float(mean(gap_values)),
                    }
                )

    rows.sort(key=lambda r: (r["worst"], r["mean"]), reverse=True)
    return {"phase": "phase2", "timestamp": _now(), "grid_results": rows}


def phase3_neighbor_search(cfg: BlueprintConfig, n_samples: int = 12) -> dict[str, Any]:
    rng = random.Random(20260406)
    space: dict[str, list[float] | list[int]] = {
        "lambda_cluster": [0.01, 0.02, 0.03],
        "local_epochs": [3, 5, 7],
        "cluster_aware_epochs": [2, 3, 4],
        "learning_rate": [0.003, 0.005, 0.007],
    }

    candidates: list[dict[str, float | int]] = []
    for _ in range(max(1, n_samples)):
        candidates.append(
            {
                "lambda_cluster": float(rng.choice(list(space["lambda_cluster"]))),
                "local_epochs": int(rng.choice(list(space["local_epochs"]))),
                "cluster_aware_epochs": int(rng.choice(list(space["cluster_aware_epochs"]))),
                "learning_rate": float(rng.choice(list(space["learning_rate"]))),
            }
        )

    dedup: dict[str, dict[str, float | int]] = {json.dumps(c, sort_keys=True): c for c in candidates}
    sampled: list[dict[str, float | int]] = list(dedup.values())

    results: list[dict[str, Any]] = []
    for params in sampled:
        seed_runs: list[dict[str, float]] = []
        for seed in cfg.seeds:
            exp = _base_config(seed=seed, cfg=cfg, mode="prototype", partition_mode="dirichlet", alpha=cfg.alpha)
            exp.training.lambda_cluster = float(params["lambda_cluster"])
            exp.training.local_epochs = int(params["local_epochs"])
            exp.training.cluster_aware_epochs = int(params["cluster_aware_epochs"])
            exp.training.learning_rate = float(params["learning_rate"])
            schema = _run_and_collect(exp)
            seed_runs.append(_global_metrics(schema))

        mean_acc = float(mean([float(r["mean"]) for r in seed_runs]))
        worst_acc = float(mean([float(r["worst"]) for r in seed_runs]))
        spread = float(mean([float(r["gap"]) for r in seed_runs]))
        score = float(worst_acc + (0.25 * mean_acc) - (0.1 * spread))
        results.append({
            "params": params,
            "mean": mean_acc,
            "worst": worst_acc,
            "gap": spread,
            "score": score,
        })

    results.sort(key=lambda r: (r["score"], r["worst"], r["mean"]), reverse=True)
    return {"phase": "phase3", "timestamp": _now(), "neighbor_results": results}


def phase4_noniid_compare(cfg: BlueprintConfig) -> dict[str, Any]:
    alphas = [1.0, 0.5, 0.1]
    methods = ["fedavg", "prototype"]

    table: list[dict[str, Any]] = []
    detailed_runs: list[dict[str, Any]] = []
    split_consistency: list[dict[str, Any]] = []
    for alpha in alphas:
        reference_split_by_seed: dict[int, str] = {}
        for method in methods:
            seed_rows: list[dict[str, float]] = []
            for seed in cfg.seeds:
                exp = _base_config(
                    seed=seed,
                    cfg=cfg,
                    mode=method,
                    partition_mode="dirichlet",
                    alpha=float(alpha),
                )
                schema = _run_and_collect(exp)
                metrics = _global_metrics(schema)
                seed_rows.append(metrics)
                rounds = schema.get("rounds", [])
                total_comm = int(sum(int(r.get("communication", {}).get("round_total_bytes", 0)) for r in rounds))
                partition_fingerprint = str(schema.get("partition_fingerprint", ""))
                if seed not in reference_split_by_seed:
                    reference_split_by_seed[seed] = partition_fingerprint
                elif reference_split_by_seed[seed] != partition_fingerprint:
                    raise AssertionError(
                        f"Split inconsistency for alpha={alpha}, seed={seed}: "
                        f"{reference_split_by_seed[seed]} != {partition_fingerprint}"
                    )

                partition_manifest = schema.get("partition_manifest", {})
                entropy_by_client = {
                    str(cid): _class_entropy(client_info.get("class_distribution", {}))
                    for cid, client_info in partition_manifest.items()
                }
                detailed_runs.append(
                    {
                        "alpha": float(alpha),
                        "method": method,
                        "seed": int(seed),
                        "metrics": metrics,
                        "total_communication_bytes": total_comm,
                        "partition_fingerprint": partition_fingerprint,
                        "entropy_by_client": entropy_by_client,
                        "mean_client_entropy": float(mean(entropy_by_client.values())) if entropy_by_client else 0.0,
                        "rounds": rounds,
                    }
                )

            split_consistency.append(
                {
                    "alpha": float(alpha),
                    "method": method,
                    "seeds_checked": [int(s) for s in cfg.seeds],
                    "consistent_with_reference": True,
                }
            )

            mean_acc = float(mean([float(r["mean"]) for r in seed_rows]))
            worst_acc = float(mean([float(r["worst"]) for r in seed_rows]))
            gap = float(mean([float(r["gap"]) for r in seed_rows]))
            std_acc = float(stdev([float(r["mean"]) for r in seed_rows])) if len(seed_rows) > 1 else 0.0
            table.append(
                {
                    "alpha": float(alpha),
                    "method": method,
                    "mean": mean_acc,
                    "worst": worst_acc,
                    "std": std_acc,
                    "gap": gap,
                }
            )

    return {
        "phase": "phase4",
        "timestamp": _now(),
        "comparison": table,
        "detailed_runs": detailed_runs,
        "split_consistency": split_consistency,
    }


def phase5_ablation(cfg: BlueprintConfig) -> dict[str, Any]:
    variants: list[dict[str, float | str]] = [
        {"name": "base", "mode": "fedavg", "lambda_cluster": 0.0, "lambda_center": 0.0},
        {"name": "+cluster_loss", "mode": "prototype", "lambda_cluster": 0.1, "lambda_center": 0.0},
        {"name": "+center_loss", "mode": "prototype", "lambda_cluster": 0.0, "lambda_center": 0.01},
        {"name": "full_model", "mode": "prototype", "lambda_cluster": 0.1, "lambda_center": 0.01},
    ]

    results: list[dict[str, Any]] = []
    for variant in variants:
        per_seed: list[dict[str, float]] = []
        for seed in cfg.seeds:
            exp = _base_config(
                seed=seed,
                cfg=cfg,
                mode=str(variant["mode"]),
                partition_mode="dirichlet",
                alpha=cfg.alpha,
            )
            exp.training.lambda_cluster = float(variant["lambda_cluster"])
            exp.training.lambda_cluster_center = float(variant["lambda_center"])
            schema = _run_and_collect(exp)
            per_seed.append(_global_metrics(schema))

        mean_values: list[float] = [float(r["mean"]) for r in per_seed]
        worst_values: list[float] = [float(r["worst"]) for r in per_seed]
        gap_values: list[float] = [float(r["gap"]) for r in per_seed]

        results.append(
            {
                "variant": variant["name"],
                "mode": variant["mode"],
                "mean": float(mean(mean_values)),
                "worst": float(mean(worst_values)),
                "gap": float(mean(gap_values)),
                "std": float(stdev(mean_values)) if len(mean_values) > 1 else 0.0,
                "per_seed": per_seed,
            }
        )

    return {"phase": "phase5", "timestamp": _now(), "ablation": results}


def _cohens_d_paired(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    diffs = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if std <= 0.0:
        return 0.0
    return float(np.mean(diffs) / std)


def _paired_t_statistic(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    diffs = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    if len(diffs) < 2:
        return 0.0
    std = float(np.std(diffs, ddof=1))
    if std <= 0.0:
        return 0.0
    return float(np.mean(diffs) / (std / np.sqrt(len(diffs))))


def _exact_sign_test_pvalue(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) == 0:
        return None
    diffs = [x - y for x, y in zip(a, b, strict=True) if (x - y) != 0.0]
    n = len(diffs)
    if n == 0:
        return 1.0
    positives = sum(1 for d in diffs if d > 0.0)
    k = min(positives, n - positives)

    # Two-sided exact binomial sign test under p=0.5.
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    p_two_sided = min(1.0, 2.0 * prob)
    return float(p_two_sided)


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def phase6_statistics(cfg: BlueprintConfig, phase4_payload: dict[str, Any] | None) -> dict[str, Any]:
    if phase4_payload is None:
        return {
            "phase": "phase6",
            "timestamp": _now(),
            "error": "phase4 results required for statistical validation",
        }

    detailed = phase4_payload.get("detailed_runs", [])
    stats_rows: list[dict[str, Any]] = []
    for alpha in [1.0, 0.5, 0.1]:
        fedavg = [
            r for r in detailed if float(r.get("alpha", -1.0)) == alpha and r.get("method") == "fedavg"
        ]
        ours = [
            r for r in detailed if float(r.get("alpha", -1.0)) == alpha and r.get("method") == "prototype"
        ]

        seed_to_fedavg = {int(r["seed"]): float(r["metrics"]["worst"]) for r in fedavg}
        seed_to_ours = {int(r["seed"]): float(r["metrics"]["worst"]) for r in ours}
        common = sorted(set(seed_to_fedavg.keys()).intersection(seed_to_ours.keys()))
        if not common:
            continue
        a = [seed_to_ours[s] for s in common]
        b = [seed_to_fedavg[s] for s in common]

        t_stat = _paired_t_statistic(a, b)
        sign_p = _exact_sign_test_pvalue(a, b)
        paired_t_pvalue: float | None = None
        wilcoxon_pvalue: float | None = None

        if ttest_rel is not None:
            try:
                paired_t_result: Any = ttest_rel(a, b)
                paired_t_pvalue = _as_float_or_none(getattr(paired_t_result, "pvalue", None))
            except Exception:
                paired_t_pvalue = None
        if wilcoxon is not None:
            try:
                wilcoxon_result: Any = wilcoxon(a, b)
                wilcoxon_pvalue = _as_float_or_none(getattr(wilcoxon_result, "pvalue", None))
            except Exception:
                wilcoxon_pvalue = None

        stats_rows.append(
            {
                "alpha": alpha,
                "n": len(common),
                "ours_worst_mean": float(mean(a)),
                "fedavg_worst_mean": float(mean(b)),
                "delta_worst": float(mean(a) - mean(b)),
                "paired_t_statistic": t_stat,
                "paired_t_pvalue": paired_t_pvalue,
                "wilcoxon_pvalue": wilcoxon_pvalue,
                "sign_test_pvalue": sign_p,
                "cohens_d_paired": _cohens_d_paired(a, b),
            }
        )

    return {"phase": "phase6", "timestamp": _now(), "statistics": stats_rows}


def _round_to_reach_accuracy(rounds: list[dict[str, Any]], target: float) -> int | None:
    for row in rounds:
        gm = row.get("global_metrics", {})
        if float(gm.get("mean_client_accuracy", 0.0)) >= target:
            return int(row.get("round", 0))
    return None


def phase7_communication(cfg: BlueprintConfig, phase4_payload: dict[str, Any] | None) -> dict[str, Any]:
    if phase4_payload is None:
        return {
            "phase": "phase7",
            "timestamp": _now(),
            "error": "phase4 results required for communication analysis",
        }

    detailed = phase4_payload.get("detailed_runs", [])
    output_rows: list[dict[str, Any]] = []
    target_acc = 0.80

    for alpha in [1.0, 0.5, 0.1]:
        for method in ["fedavg", "prototype"]:
            runs = [
                r
                for r in detailed
                if float(r.get("alpha", -1.0)) == alpha and r.get("method") == method
            ]
            if not runs:
                continue
            total_comm = [float(r.get("total_communication_bytes", 0.0)) for r in runs]
            final_mean = [float(r.get("metrics", {}).get("mean", 0.0)) for r in runs]
            rounds_to_target = [
                _round_to_reach_accuracy(r.get("rounds", []), target=target_acc) for r in runs
            ]
            reached = [r for r in rounds_to_target if r is not None]

            mean_comm = float(mean(total_comm)) if total_comm else 0.0
            mean_acc = float(mean(final_mean)) if final_mean else 0.0
            acc_gain_pct = max(mean_acc * 100.0, 1e-8)
            comm_per_pct = float(mean_comm / acc_gain_pct)

            output_rows.append(
                {
                    "alpha": alpha,
                    "method": method,
                    "mean_final_accuracy": mean_acc,
                    "mean_total_communication_bytes": mean_comm,
                    "communication_per_1pct_accuracy": comm_per_pct,
                    "mean_round_to_0.80": float(mean(reached)) if reached else None,
                    "target_reached_ratio": float(len(reached) / len(runs)),
                }
            )

    return {"phase": "phase7", "timestamp": _now(), "communication_analysis": output_rows}


def phase8_robustness(cfg: BlueprintConfig) -> dict[str, Any]:
    client_counts = [2, 5, 10]
    alphas = [1.0, 0.5, 0.1]
    methods = ["fedavg", "prototype"]

    rows: list[dict[str, Any]] = []
    for num_clients in client_counts:
        for alpha in alphas:
            for method in methods:
                metrics_by_seed: list[dict[str, float]] = []
                for seed in cfg.seeds:
                    local_cfg = BlueprintConfig(
                        dataset=cfg.dataset,
                        num_clients=num_clients,
                        rounds=cfg.rounds,
                        max_samples_per_client=cfg.max_samples_per_client,
                        alpha=alpha,
                        seeds=cfg.seeds,
                    )
                    exp = _base_config(
                        seed=seed,
                        cfg=local_cfg,
                        mode=method,
                        partition_mode="dirichlet",
                        alpha=alpha,
                    )
                    schema = _run_and_collect(exp)
                    metrics_by_seed.append(_global_metrics(schema))

                mean_vals = [float(m["mean"]) for m in metrics_by_seed]
                worst_vals = [float(m["worst"]) for m in metrics_by_seed]
                rows.append(
                    {
                        "num_clients": num_clients,
                        "alpha": alpha,
                        "method": method,
                        "mean": float(mean(mean_vals)),
                        "mean_std": float(stdev(mean_vals)) if len(mean_vals) > 1 else 0.0,
                        "worst": float(mean(worst_vals)),
                        "worst_std": float(stdev(worst_vals)) if len(worst_vals) > 1 else 0.0,
                    }
                )

    return {"phase": "phase8", "timestamp": _now(), "robustness": rows}


def phase9_assumptions() -> dict[str, Any]:
    return {
        "phase": "phase9",
        "timestamp": _now(),
        "assumptions": [
            "Synchronous training rounds",
            "No stragglers or dropped clients",
            "Lossless communication",
            "Equal aggregation weighting by sample count unless specified otherwise",
            "Homogeneous architecture for FedAvg-compatible runs",
        ],
    }


def phase10_claims(
    phase4_payload: dict[str, Any] | None,
    phase7_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    claim_template = {
        "problem": "FL fairness degradation under heterogeneity",
        "method": "cluster-aware regularization in shared latent space",
        "result": "TBD: fill with quantitative deltas from phase_4",
        "cost": "TBD: fill with communication/compute deltas from phase_7",
        "mechanism": "reduces representation drift via prototype and center alignment",
        "limitation": "TBD: identify failure regimes from robustness/ablation",
    }

    if phase4_payload is not None:
        comparison = phase4_payload.get("comparison", [])
        severe = [r for r in comparison if float(r.get("alpha", -1.0)) == 0.1]
        fedavg = next((r for r in severe if r.get("method") == "fedavg"), None)
        ours = next((r for r in severe if r.get("method") == "prototype"), None)
        if fedavg and ours:
            delta_worst = float(ours.get("worst", 0.0)) - float(fedavg.get("worst", 0.0)
            )
            delta_mean = float(ours.get("mean", 0.0)) - float(fedavg.get("mean", 0.0)
            )
            claim_template["result"] = (
                f"At alpha=0.1, worst-client delta vs FedAvg = {delta_worst:+.4f}, "
                f"mean delta = {delta_mean:+.4f}"
            )

    if phase7_payload is not None:
        rows = phase7_payload.get("communication_analysis", [])
        severe = [r for r in rows if float(r.get("alpha", -1.0)) == 0.1]
        fedavg = next((r for r in severe if r.get("method") == "fedavg"), None)
        ours = next((r for r in severe if r.get("method") == "prototype"), None)
        if fedavg and ours:
            comm_delta = float(ours.get("mean_total_communication_bytes", 0.0)) - float(
                fedavg.get("mean_total_communication_bytes", 0.0)
            )
            claim_template["cost"] = f"Communication delta vs FedAvg at alpha=0.1: {comm_delta:+.0f} bytes"

    return {"phase": "phase10", "timestamp": _now(), "claim_template": claim_template}


def phase11_baseline_extension() -> dict[str, Any]:
    baseline_path = WORKSPACE_ROOT / "outputs" / "phase2_q1" / "stage6_baselines.json"
    if not baseline_path.exists():
        return {
            "phase": "phase11",
            "timestamp": _now(),
            "status": "missing",
            "message": "stage6_baselines.json not found; run stage6 baselines first",
        }

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    return {
        "phase": "phase11",
        "timestamp": _now(),
        "status": "available",
        "source": str(baseline_path.relative_to(WORKSPACE_ROOT)),
        "baselines": data,
    }


def _build_traceability_table(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases = outputs.get("phases", {})
    for phase_name, phase_payload in phases.items():
        run_id = str(phase_payload.get("run_id", "unknown"))
        if phase_name == "phase_4":
            for row in phase_payload.get("detailed_runs", []):
                metrics = row.get("metrics", {})
                rows.append(
                    {
                        "table": "Table_D_NonIID",
                        "phase": phase_name,
                        "run_id": run_id,
                        "seed": int(row.get("seed", -1)),
                        "method": row.get("method"),
                        "alpha": float(row.get("alpha", -1.0)),
                        "mean": float(metrics.get("mean", 0.0)),
                        "worst": float(metrics.get("worst", 0.0)),
                        "std_proxy_gap": float(metrics.get("gap", 0.0)),
                    }
                )
        elif phase_name == "phase_5":
            for row in phase_payload.get("ablation", []):
                rows.append(
                    {
                        "table": "Table_E_Ablation",
                        "phase": phase_name,
                        "run_id": run_id,
                        "seed": "aggregate",
                        "variant": row.get("variant"),
                        "mean": float(row.get("mean", 0.0)),
                        "worst": float(row.get("worst", 0.0)),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Q1 blueprint phases")
    parser.add_argument(
        "--phase",
        choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "all"],
        default="0",
    )
    parser.add_argument("--dataset", choices=["femnist", "cifar100"], default="femnist")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-samples-per-client", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "outputs" / "q1_blueprint"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BlueprintConfig(
        dataset=args.dataset,
        num_clients=int(args.num_clients),
        rounds=int(args.rounds),
        max_samples_per_client=int(args.max_samples_per_client),
        alpha=float(args.alpha),
        seeds=tuple(int(s) for s in args.seeds),
    )
    if cfg.seeds != (0, 1, 2, 3, 4):
        raise ValueError("Seed protocol violation: seeds must be exactly [0, 1, 2, 3, 4]")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    python_exe = Path(sys.executable)

    phase_map = {
        "0": ["0"],
        "1": ["1"],
        "2": ["2"],
        "3": ["3"],
        "4": ["4"],
        "5": ["5"],
        "6": ["6"],
        "7": ["7"],
        "8": ["8"],
        "9": ["9"],
        "10": ["10"],
        "11": ["11"],
        "all": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
    }

    outputs: dict[str, Any] = {
        "created_at": _now(),
        "config": asdict(cfg),
        "contract": _experiment_contract(cfg),
        "phases": {},
    }

    phase4_payload: dict[str, Any] | None = None
    phase7_payload: dict[str, Any] | None = None
    for phase in phase_map[args.phase]:
        contract = _experiment_contract(cfg)
        run_dir, run_meta = _prepare_run_directory(phase=phase, contract=contract, cfg=cfg)
        if phase == "0":
            payload = phase0_lockdown(cfg, python_exe=python_exe, out_dir=out_dir)
        elif phase == "1":
            payload = phase1_instrumentation(cfg)
        elif phase == "2":
            payload = phase2_scaling_diagnosis(cfg)
        elif phase == "3":
            payload = phase3_neighbor_search(cfg)
        elif phase == "4":
            payload = phase4_noniid_compare(cfg)
            phase4_payload = payload
        elif phase == "5":
            payload = phase5_ablation(cfg)
        elif phase == "6":
            payload = phase6_statistics(cfg, phase4_payload=phase4_payload)
        elif phase == "7":
            payload = phase7_communication(cfg, phase4_payload=phase4_payload)
            phase7_payload = payload
        elif phase == "8":
            payload = phase8_robustness(cfg)
        elif phase == "9":
            payload = phase9_assumptions()
        elif phase == "10":
            payload = phase10_claims(phase4_payload=phase4_payload, phase7_payload=phase7_payload)
        elif phase == "11":
            payload = phase11_baseline_extension()
        else:
            continue

        payload["run_id"] = run_meta["run_id"]
        payload["config_hash"] = run_meta["config_hash"]
        payload["git_commit"] = run_meta["git_commit"]
        payload["contract"] = contract

        _write_run_artifacts(run_dir=run_dir, run_meta=run_meta, cfg=cfg, payload=payload)

        outputs["phases"][f"phase_{phase}"] = payload
        outputs["phases"][f"phase_{phase}"]["artifact_dir"] = str(run_dir.relative_to(WORKSPACE_ROOT))
        (out_dir / f"phase_{phase}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    outputs["updated_at"] = _now()
    (out_dir / "execution_summary.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    traceability = _build_traceability_table(outputs)
    (out_dir / "traceability_table.json").write_text(json.dumps(traceability, indent=2), encoding="utf-8")

    print("Execution complete. Generated artifacts:")
    for phase in outputs["phases"].keys():
        print(f"- outputs/q1_blueprint/{phase.replace('phase_', 'phase_')}.json")
    print("- outputs/q1_blueprint/execution_summary.json")
    print("- outputs/q1_blueprint/traceability_table.json")


if __name__ == "__main__":
    main()
