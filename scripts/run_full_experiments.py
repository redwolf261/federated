from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import math
import multiprocessing as mp
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from scipy.stats import ttest_rel, wilcoxon  # type: ignore
except Exception:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

MODE = "smoke"
MODE_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "methods": ["flexfl", "moon", "scaffold"],
        "datasets": ["cifar10"],
        "alphas": [0.3],
        "seeds": [42],
        "rounds": 5,
        "local_epochs": 1,
        "max_samples_per_client": 64,
    },
    "validate": {
        "methods": ["flexfl", "moon", "scaffold"],
        "datasets": ["cifar10", "cifar100"],
        "alphas": [0.1, 1.0],
        "seeds": [42, 43, 44],
        "rounds": 20,
        "local_epochs": 2,
        "max_samples_per_client": 512,
    },
    "full": {
        "methods": ["flexfl", "moon", "scaffold"],
        "datasets": ["cifar10", "cifar100"],
        "alphas": [0.1, 0.3, 0.5, 1.0],
        "seeds": [42, 43, 44],
        "rounds": 50,
        "local_epochs": 3,
        "max_samples_per_client": 2000,
    },
    "firm2h": {
        "methods": ["flexfl", "fedavg", "moon", "scaffold"],
        "datasets": ["cifar10", "cifar100"],
        "alphas": [0.1, 0.5],
        "seeds": [42, 43, 44],
        "rounds": 6,
        "local_epochs": 2,
        "max_samples_per_client": 256,
    },
}

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.models.model_factory import ModelFactory
from flex_persona.utils.seed import set_global_seed
from generate_dataset_manifest import main as generate_manifest_main
from validators.validate_run import validate_run_schema

run_moon = None
run_scaffold = None
_baseline_module_path = WORKSPACE / "scripts" / "phase2_q1_validation.py"
if _baseline_module_path.exists():
    try:
        from scripts.phase2_q1_validation import run_moon, run_scaffold
    except Exception:
        try:
            spec = importlib.util.spec_from_file_location("phase2_q1_validation", _baseline_module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                run_moon = getattr(module, "run_moon", None)
                run_scaffold = getattr(module, "run_scaffold", None)
        except Exception:
            run_moon = None
            run_scaffold = None

RUNS_DIR = WORKSPACE / "artifacts" / "runs"
STATS_DIR = WORKSPACE / "artifacts" / "stats"
TRACE_DIR = WORKSPACE / "artifacts" / "traceability"
BASELINE_DIR = WORKSPACE / "artifacts" / "baselines"
SUCCESS_REGISTRY = WORKSPACE / "artifacts" / "success.json"
FAILED_REGISTRY = WORKSPACE / "artifacts" / "failed.json"
COMPLETED_REGISTRY = WORKSPACE / "artifacts" / "completed.json"
MAX_RETRIES = 2
_WORKER_DATASET_CACHE: dict[tuple[Any, ...], Any] = {}
_WORKER_CACHE_PATCHED = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def auto_detect_workers() -> int:
    try:
        if torch.cuda.is_available():
            total_mem = int(torch.cuda.get_device_properties(0).total_memory)
            return max(1, int(total_mem // (2 * 1024**3)))
    except Exception:
        pass
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu // 2))


MAX_WORKERS = auto_detect_workers()


def get_run_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_registry(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = _load_json(path)
    if not isinstance(data, list):
        return []
    return [str(x) for x in data]


def update_registry(run_id: str, status: str) -> None:
    target = SUCCESS_REGISTRY if status == "success" else FAILED_REGISTRY
    other = FAILED_REGISTRY if status == "success" else SUCCESS_REGISTRY
    target_rows = _load_registry(target)
    if run_id not in target_rows:
        target_rows.append(run_id)
    _write_json(target, target_rows)

    other_rows = [x for x in _load_registry(other) if x != run_id]
    _write_json(other, other_rows)

    completed_rows = sorted(set(_load_registry(SUCCESS_REGISTRY) + _load_registry(FAILED_REGISTRY)))
    _write_json(COMPLETED_REGISTRY, completed_rows)


def load_completed() -> set[str]:
    if COMPLETED_REGISTRY.exists():
        return set(_load_registry(COMPLETED_REGISTRY))
    return set(_load_registry(SUCCESS_REGISTRY))


def has_nan(obj: Any) -> bool:
    if isinstance(obj, dict):
        return any(has_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_nan(v) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return math.isnan(float(obj))
    return False


def validate_result(result: dict[str, Any]) -> bool:
    run_row = result["result"]
    if run_row is None:
        raise AssertionError("Empty result")
    rounds = run_row.get("metrics", {}).get("per_round", [])
    if not isinstance(rounds, list) or len(rounds) <= 0:
        raise AssertionError("No round data")
    if has_nan(run_row):
        raise AssertionError("NaN detected")

    total_bytes = float(run_row.get("communication", {}).get("total_bytes", 0.0))
    if total_bytes <= 0.0:
        raise AssertionError("Zero communication")

    expected = int(run_row.get("rounds_expected", 0))
    completed = int(run_row.get("rounds_completed", -1))
    if expected <= 0 or completed != expected:
        raise AssertionError(f"Missing rounds: completed={completed} expected={expected}")

    return True


def _phase_label(raw: str) -> str:
    v = raw.strip().lower()
    if v in {"a", "sanity", "phase-a", "phase_a"}:
        return "sanity"
    if v in {"b", "baselines", "phase-b", "phase_b"}:
        return "baselines"
    if v in {"c", "full", "phase-c", "phase_c"}:
        return "full"
    raise ValueError(f"Unsupported phase: {raw}")


def _build_run_configs(
    methods: list[str],
    datasets: list[str],
    alphas: list[float],
    seeds: list[int],
    rounds: int,
    local_epochs: int,
    max_samples_per_client: int,
    require_gpu: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "method": str(method),
            "dataset": str(dataset_name),
            "alpha": float(alpha),
            "seed": int(seed),
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "max_samples_per_client": int(max_samples_per_client),
            "require_gpu": bool(require_gpu),
        }
        for method in methods
        for dataset_name in datasets
        for alpha in alphas
        for seed in seeds
    ]


def _write_run_plan(
    phase: str,
    mode: str,
    run_configs: list[dict[str, Any]],
    pending_configs: list[dict[str, Any]],
    max_workers: int,
) -> None:
    plan_payload = {
        "created_at": _now(),
        "phase": phase,
        "mode": mode,
        "requested_runs": len(run_configs),
        "pending_runs": len(pending_configs),
        "max_workers": int(max_workers),
        "run_ids": [get_run_id(cfg) for cfg in run_configs],
        "pending_run_ids": [get_run_id(cfg) for cfg in pending_configs],
    }
    _write_json(STATS_DIR / "current_plan.json", plan_payload)


def _write_live_progress(
    *,
    phase: str,
    mode: str,
    requested_runs: int,
    pending_runs: int,
    done_runs: int,
    success_runs: int,
    failed_runs: int,
    max_workers: int,
    active: bool,
) -> None:
    payload = {
        "updated_at": _now(),
        "phase": phase,
        "mode": mode,
        "requested_runs": int(requested_runs),
        "pending_runs": int(pending_runs),
        "done_runs": int(done_runs),
        "success_runs": int(success_runs),
        "failed_runs": int(failed_runs),
        "max_workers": int(max_workers),
        "active": bool(active),
    }
    _write_json(STATS_DIR / "live_progress.json", payload)


def _read_run_artifacts(run_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in sorted(run_ids):
        run_path = RUNS_DIR / f"{run_id}.json"
        if run_path.exists():
            rows.append(_load_json(run_path))
    return rows


def _minimal_failure_row(config: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "run_id": get_run_id(config),
        "config": config,
        "method": str(config["method"]),
        "alpha": float(config["alpha"]),
        "dataset": str(config["dataset"]),
        "seed": int(config["seed"]),
        "status": "FAIL",
        "rounds_expected": int(config["rounds"]),
        "rounds_completed": 0,
        "error": error,
        "dataset_hash": None,
        "preprocessing": "dataset_loader_default",
        "metrics": {"per_round": [], "final": {}},
        "communication": {"total_bytes": 0.0},
    }


def save_result(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    run_id = get_run_id(config)

    run_row = result.get("result")
    if not isinstance(run_row, dict):
        run_row = _minimal_failure_row(config, str(result.get("error", "Unknown failure")))

    run_row["run_id"] = run_id
    run_row["config"] = config
    run_row["seed"] = int(config["seed"])
    run_row["execution"] = {
        "attempts": int(result.get("attempts", 1)),
        "status": str(result.get("status", "failed")),
        "last_error": result.get("error"),
        "saved_at": _now(),
    }

    status = "success" if result.get("status") == "success" else "failed"
    if status != "success":
        run_row["status"] = "FAIL"
        if not run_row.get("error"):
            run_row["error"] = str(result.get("error", "run failed"))

    run_path = RUNS_DIR / f"{run_id}.json"
    _write_json(run_path, run_row)
    update_registry(run_id, status)
    return run_row


def _load_global_config() -> dict[str, Any]:
    cfg = _load_json(WORKSPACE / "configs" / "global_config.json")
    required = {"learning_rate", "batch_size", "local_epochs", "rounds", "optimizer", "weight_decay"}
    missing = required.difference(cfg.keys())
    if missing:
        raise ValueError(f"global_config.json missing required keys: {sorted(missing)}")
    return cfg


def _install_worker_dataset_cache() -> None:
    global _WORKER_CACHE_PATCHED
    if _WORKER_CACHE_PATCHED:
        return

    from flex_persona.data.dataset_registry import DatasetRegistry

    original_load = DatasetRegistry.load

    def cached_load(
        self: Any,
        dataset_name: str,
        max_train_samples: int | None = None,
        max_test_samples: int | None = None,
        max_rows: int | None = None,
    ) -> Any:
        key = (
            str(self.workspace_root),
            dataset_name.lower().strip(),
            max_train_samples,
            max_test_samples,
            max_rows,
        )
        if key not in _WORKER_DATASET_CACHE:
            _WORKER_DATASET_CACHE[key] = original_load(
                self,
                dataset_name,
                max_train_samples=max_train_samples,
                max_test_samples=max_test_samples,
                max_rows=max_rows,
            )
        # Return deep copy so each run gets isolated tensors/arrays.
        return copy.deepcopy(_WORKER_DATASET_CACHE[key])

    DatasetRegistry.load = cached_load  # type: ignore[assignment]
    _WORKER_CACHE_PATCHED = True


def _load_seed_list() -> list[int]:
    seeds = [int(s) for s in _load_json(WORKSPACE / "configs" / "seeds.json")]
    if not seeds:
        raise ValueError("Seed protocol violation: configs/seeds.json is empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed protocol violation: duplicate seeds in configs/seeds.json")
    return seeds


def _ensure_gpu_ready(require_gpu: bool) -> None:
    if not require_gpu:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount\(\).*",
            category=UserWarning,
        )
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
    if not available or count <= 0:
        raise RuntimeError(
            "GPU preflight failed: CUDA device is not accessible by PyTorch. "
            "Run `nvidia-smi` and check driver/session permissions before launching experiments."
        )


def _dataset_geometry(dataset_name: str) -> tuple[int, int, int, int]:
    normalized = dataset_name.lower().strip()
    if normalized == "femnist":
        return 1, 28, 28, 62
    if normalized == "cifar10":
        return 3, 32, 32, 10
    if normalized == "cifar100":
        return 3, 32, 32, 100
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _build_experiment_config(
    method: str,
    dataset_name: str,
    alpha: float,
    seed: int,
    global_cfg: dict[str, Any],
    rounds: int,
    local_epochs: int,
    max_samples_per_client: int,
) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=f"clean_{method}_{dataset_name}_a{alpha}_s{seed}",
        dataset_name=dataset_name,
        num_clients=10,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
    )
    cfg.model.client_backbones = ["small_cnn"] * cfg.num_clients
    cfg.model.shared_dim = 64
    _, _, _, num_classes = _dataset_geometry(dataset_name)
    cfg.model.num_classes = num_classes

    cfg.training.rounds = int(rounds)
    cfg.training.local_epochs = int(local_epochs)
    cfg.training.cluster_aware_epochs = int(local_epochs)
    cfg.training.learning_rate = float(global_cfg["learning_rate"])
    cfg.training.batch_size = int(global_cfg["batch_size"])
    cfg.training.weight_decay = float(global_cfg["weight_decay"])
    cfg.training.max_samples_per_client = int(max_samples_per_client)
    cfg.training.aggregation_mode = "prototype" if method in {"flexfl", "your_method"} else method
    cfg.training.early_stopping_enabled = False

    # Keep execution infrastructure independent from method-specific hyperparameter overrides.
    cfg.training.lambda_cluster = 0.02
    cfg.training.lambda_cluster_center = 0.0
    cfg.training.cluster_center_warmup_rounds = 5
    cfg.clustering.num_clusters = 2
    cfg.similarity.sigma = 1.0
    cfg.validate()
    return cfg


def _model_init_hash(dataset_name: str, seed: int) -> str:
    set_global_seed(seed, deterministic=True)
    _, _, _, num_classes = _dataset_geometry(dataset_name)
    cfg = ExperimentConfig(experiment_name="init_hash", dataset_name=dataset_name, num_clients=10, random_seed=seed)
    cfg.model.client_backbones = ["small_cnn"] * cfg.num_clients
    cfg.model.num_classes = num_classes
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=cfg.model,
        dataset_name=dataset_name,
    )
    state = model.state_dict()
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _extract_run_schema(
    sim: FederatedSimulator,
    schema: dict[str, Any],
    method: str,
    dataset_name: str,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    rounds = schema.get("rounds", [])
    final = rounds[-1].get("global_metrics", {}) if rounds else {}
    total_bytes = float(sum(float(r.get("communication", {}).get("round_total_bytes", 0.0)) for r in rounds))
    run_summary = schema.get("final_summary", {}).get("run_summary", {})
    status = "SUCCESS" if str(run_summary.get("run_status", "SUCCESS")) == "SUCCESS" else "FAIL"
    error = run_summary.get("error_message")

    payload = {
        "run_id": f"{method}_{dataset_name}_a{alpha}_s{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "method": method,
        "dataset": dataset_name,
        "alpha": alpha,
        "seed": seed,
        "status": status,
        "rounds_expected": int(sim.config.training.rounds),
        "rounds_completed": len(rounds),
        "error": error,
        "dataset_hash": schema.get("partition_fingerprint"),
        "preprocessing": "dataset_loader_default",
        "metrics": {
            "per_round": rounds,
            "final": final,
        },
        "communication": {
            "total_bytes": total_bytes,
        },
    }
    return payload


def _run_core_method(
    method: str,
    dataset_name: str,
    alpha: float,
    seed: int,
    global_cfg: dict[str, Any],
    rounds: int,
    local_epochs: int,
    max_samples_per_client: int,
) -> dict[str, Any]:
    set_global_seed(seed, deterministic=True)
    cfg = _build_experiment_config(
        method=method,
        dataset_name=dataset_name,
        alpha=alpha,
        seed=seed,
        global_cfg=global_cfg,
        rounds=rounds,
        local_epochs=local_epochs,
        max_samples_per_client=max_samples_per_client,
    )
    sim = FederatedSimulator(workspace_root=WORKSPACE, config=cfg)
    schema = sim.build_run_schema(sim.run_experiment())
    return _extract_run_schema(
        sim=sim,
        schema=schema,
        method=method,
        dataset_name=dataset_name,
        alpha=alpha,
        seed=seed,
    )


def _run_baseline_method(
    method: str,
    dataset_name: str,
    alpha: float,
    seed: int,
    global_cfg: dict[str, Any],
    rounds: int,
    local_epochs: int,
    max_samples_per_client: int,
) -> dict[str, Any]:
    num_clients = 10
    max_samples = int(max_samples_per_client) * num_clients
    if method == "moon" and run_moon is not None:
        _, _, _, num_classes = _dataset_geometry(dataset_name)
        result = run_moon(
            dataset_name=dataset_name,
            num_classes=num_classes,
            num_clients=num_clients,
            rounds=int(rounds),
            local_epochs=int(local_epochs),
            seed=seed,
            alpha=alpha,
            lr=float(global_cfg["learning_rate"]),
            batch_size=int(global_cfg["batch_size"]),
            max_samples=max_samples,
            return_trace=True,
        )
    elif method == "scaffold" and run_scaffold is not None:
        _, _, _, num_classes = _dataset_geometry(dataset_name)
        result = run_scaffold(
            dataset_name=dataset_name,
            num_classes=num_classes,
            num_clients=num_clients,
            rounds=int(rounds),
            local_epochs=int(local_epochs),
            seed=seed,
            alpha=alpha,
            lr=float(global_cfg["learning_rate"]),
            batch_size=int(global_cfg["batch_size"]),
            max_samples=max_samples,
            return_trace=True,
        )
    else:
        return {
            "run_id": f"{method}_{dataset_name}_a{alpha}_s{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "method": method,
            "alpha": alpha,
            "dataset": dataset_name,
            "seed": seed,
            "status": "FAIL",
            "rounds_expected": int(rounds),
            "rounds_completed": 0,
            "error": f"{method} runner unavailable",
            "dataset_hash": None,
            "preprocessing": "dataset_loader_default",
            "metrics": {"per_round": [], "final": {}},
            "communication": {"total_bytes": 0.0},
        }

    per_round = result.get("per_round", [])
    total_bytes = float(result.get("total_communication_bytes", 0.0))
    final = per_round[-1].get("global_metrics", {}) if per_round else {
        "mean_client_accuracy": float(result.get("mean_accuracy", 0.0)),
        "worst_client_accuracy": float(result.get("worst_accuracy", 0.0)),
        "client_accuracies": {str(k): float(v) for k, v in result.get("client_accuracies", {}).items()},
    }
    status = "SUCCESS" if per_round and total_bytes > 0.0 else "FAIL"
    error = None if status == "SUCCESS" else "baseline adapter missing per-round trace and normalized communication accounting"

    return {
        "run_id": f"{method}_{dataset_name}_a{alpha}_s{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "method": method,
        "alpha": alpha,
        "dataset": dataset_name,
        "seed": seed,
        "status": status,
        "rounds_expected": int(rounds),
        "rounds_completed": len(per_round),
        "error": error,
        "dataset_hash": result.get("partition_fingerprint"),
        "preprocessing": "dataset_loader_default",
        "metrics": {
            "per_round": per_round,
            "final": final,
        },
        "communication": {"total_bytes": total_bytes},
    }


def _configure_baseline_runtime_device() -> None:
    """Ensure external baseline runners use an available device.

    The baseline module may default to CUDA even when unavailable, which is an
    execution-environment concern rather than model-logic behavior.
    """
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    for fn in (run_moon, run_scaffold):
        if fn is None:
            continue
        fn.__globals__["DEVICE"] = target_device


def _effect_size(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    d = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    s = float(np.std(d, ddof=1))
    if s <= 0:
        return 0.0
    return float(np.mean(d) / s)


def _compute_statistics(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[float, str], dict[int, float]] = defaultdict(dict)
    for row in run_rows:
        if row.get("status") != "SUCCESS":
            continue
        alpha = float(row["alpha"])
        method = str(row["method"])
        seed = int(row["seed"])
        final = row.get("metrics", {}).get("final", {})
        by_key[(alpha, method)][seed] = float(final.get("worst_client_accuracy", 0.0))

    stats_rows: list[dict[str, Any]] = []
    for alpha in [1.0, 0.5, 0.1]:
        ours = by_key.get((alpha, "flexfl"), {})
        for baseline in ["moon", "scaffold"]:
            base = by_key.get((alpha, baseline), {})
            common = sorted(set(ours.keys()).intersection(base.keys()))
            if len(common) < 2:
                stats_rows.append(
                    {
                        "alpha": alpha,
                        "baseline": baseline,
                        "n": len(common),
                        "p_ttest": None,
                        "p_wilcoxon": None,
                        "effect_size": None,
                        "status": "INSUFFICIENT_DATA",
                    }
                )
                continue
            a = [ours[s] for s in common]
            b = [base[s] for s in common]
            p_t = float(ttest_rel(a, b).pvalue) if ttest_rel is not None else None
            p_w = float(wilcoxon(a, b).pvalue) if wilcoxon is not None else None
            stats_rows.append(
                {
                    "alpha": alpha,
                    "baseline": baseline,
                    "n": len(common),
                    "mean_delta_worst": float(np.mean(np.array(a) - np.array(b))),
                    "p_ttest": p_t,
                    "p_wilcoxon": p_w,
                    "effect_size": _effect_size(a, b),
                    "status": "OK",
                }
            )

    return {"created_at": _now(), "statistics": stats_rows}


def _build_traceability(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in run_rows:
        run_id = str(row["run_id"])
        path = f"artifacts/runs/{run_id}.json"
        table.append(
            {
                "table_row": f"{row['method']}_a{row['alpha']}_s{row['seed']}",
                "run_id": run_id,
                "seed": int(row["seed"]),
                "file_path": path,
            }
        )
    return table


def _build_final_claim(run_rows: list[dict[str, Any]], stats_payload: dict[str, Any]) -> dict[str, Any]:
    wins: list[dict[str, Any]] = []
    failures = [r for r in run_rows if r.get("status") != "SUCCESS"]

    for alpha in [1.0, 0.5, 0.1]:
        ours = [
            float(r.get("metrics", {}).get("final", {}).get("worst_client_accuracy", 0.0))
            for r in run_rows
            if r.get("status") == "SUCCESS" and r["method"] == "flexfl" and float(r["alpha"]) == alpha
        ]
        moon = [
            float(r.get("metrics", {}).get("final", {}).get("worst_client_accuracy", 0.0))
            for r in run_rows
            if r.get("status") == "SUCCESS" and r["method"] == "moon" and float(r["alpha"]) == alpha
        ]
        scaffold = [
            float(r.get("metrics", {}).get("final", {}).get("worst_client_accuracy", 0.0))
            for r in run_rows
            if r.get("status") == "SUCCESS" and r["method"] == "scaffold" and float(r["alpha"]) == alpha
        ]
        if ours and moon:
            wins.append(
                {
                    "alpha": alpha,
                    "worst_delta_vs_moon": float(np.mean(ours) - np.mean(moon)),
                    "worst_delta_vs_scaffold": float(np.mean(ours) - np.mean(scaffold)) if scaffold else None,
                }
            )

    return {
        "created_at": _now(),
        "where_method_wins": wins,
        "statistical_summary": stats_payload,
        "cost": "See per-run communication.total_bytes in artifacts/runs",
        "failure_modes": {
            "failed_runs": len(failures),
            "examples": [
                {
                    "run_id": r.get("run_id"),
                    "method": r.get("method"),
                    "alpha": r.get("alpha"),
                    "seed": r.get("seed"),
                    "error": r.get("error"),
                }
                for r in failures[:10]
            ],
        },
    }


def _tune_baselines(global_cfg: dict[str, Any], seeds: list[int]) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    if run_moon is None or run_scaffold is None:
        _write_json(BASELINE_DIR / "best_configs.json", {"error": "baseline functions unavailable"})
        return

    dataset_name = str(global_cfg.get("dataset_name", "cifar10"))
    _, _, _, num_classes = _dataset_geometry(dataset_name)
    moon_grid = [0.01, 0.1, 1.0]
    scaffold_grid = [0.001, 0.003, 0.01]

    best_moon = {"mu": None, "score": -1.0}
    for mu in moon_grid:
        accs = []
        for seed in seeds:
            row = run_moon(
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_clients=10,
                rounds=int(global_cfg["rounds"]),
                local_epochs=int(global_cfg["local_epochs"]),
                seed=seed,
                alpha=0.5,
                lr=float(global_cfg["learning_rate"]),
                batch_size=int(global_cfg["batch_size"]),
                max_samples=2000,
                mu=mu,
            )
            accs.append(float(row.get("mean_accuracy", 0.0)))
        score = float(np.mean(accs)) if accs else 0.0
        if score > best_moon["score"]:
            best_moon = {"mu": mu, "score": score}

    best_scaffold = {"lr": None, "score": -1.0}
    for lr in scaffold_grid:
        accs = []
        for seed in seeds:
            row = run_scaffold(
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_clients=10,
                rounds=int(global_cfg["rounds"]),
                local_epochs=int(global_cfg["local_epochs"]),
                seed=seed,
                alpha=0.5,
                lr=lr,
                batch_size=int(global_cfg["batch_size"]),
                max_samples=2000,
            )
            accs.append(float(row.get("mean_accuracy", 0.0)))
        score = float(np.mean(accs)) if accs else 0.0
        if score > best_scaffold["score"]:
            best_scaffold = {"lr": lr, "score": score}

    _write_json(BASELINE_DIR / "best_configs.json", {"moon": best_moon, "scaffold": best_scaffold})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full fair federated matrix with strict validation")
    parser.add_argument("--mode", choices=list(MODE_PRESETS.keys()), default=MODE)
    parser.add_argument("--phase", default="full", help="Execution phase: sanity|baselines|full (or a|b|c)")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--tune-baselines", action="store_true")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--rerun-failed", action="store_true", help="Requeue previously failed run_ids")
    parser.add_argument("--abort-on-failure", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def run_single_experiment(config: dict[str, Any]) -> dict[str, Any]:
    try:
        _install_worker_dataset_cache()
        _ensure_gpu_ready(require_gpu=bool(config.get("require_gpu", False)))
        method = str(config["method"])
        dataset_name = str(config["dataset"])
        alpha = float(config["alpha"])
        seed = int(config["seed"])
        rounds = int(config["rounds"])
        local_epochs = int(config["local_epochs"])
        max_samples_per_client = int(config["max_samples_per_client"])
        global_cfg = _load_global_config()
        _configure_baseline_runtime_device()

        if method in {"fedavg", "flexfl"}:
            row = _run_core_method(
                method=method,
                dataset_name=dataset_name,
                alpha=alpha,
                seed=seed,
                global_cfg=global_cfg,
                rounds=rounds,
                local_epochs=local_epochs,
                max_samples_per_client=max_samples_per_client,
            )
        else:
            row = _run_baseline_method(
                method=method,
                dataset_name=dataset_name,
                alpha=alpha,
                seed=seed,
                global_cfg=global_cfg,
                rounds=rounds,
                local_epochs=local_epochs,
                max_samples_per_client=max_samples_per_client,
            )

        row["config"] = config
        return {
            "status": "success",
            "config": config,
            "result": row,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "result": _minimal_failure_row(config, f"{type(exc).__name__}: {exc}"),
        }


def run_with_retry(config: dict[str, Any]) -> dict[str, Any]:
    last_error: str = "unknown"
    best_result: dict[str, Any] | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        result = run_single_experiment(config)
        best_result = result
        if result["status"] == "success":
            try:
                validate_result(result)
                result["attempts"] = attempt
                return result
            except Exception as exc:
                last_error = f"ValidationError: {exc}"
                result["status"] = "failed"
                result["error"] = last_error
                if isinstance(result.get("result"), dict):
                    result["result"]["status"] = "FAIL"
                    result["result"]["error"] = last_error
                continue
        else:
            last_error = str(result.get("error", "run failed"))

    return {
        "status": "failed_final",
        "config": config,
        "error": last_error,
        "attempts": MAX_RETRIES,
        "result": best_result.get("result") if isinstance(best_result, dict) else _minimal_failure_row(config, last_error),
    }


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    SUCCESS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    phase = _phase_label(args.phase)
    _ensure_gpu_ready(require_gpu=bool(args.require_gpu))
    global_cfg = _load_global_config()
    preset = MODE_PRESETS[args.mode]
    methods = list(args.methods) if args.methods is not None else list(preset["methods"])
    datasets = list(args.datasets) if hasattr(args, "datasets") and args.datasets is not None else list(preset["datasets"])
    alphas = [float(a) for a in args.alphas] if args.alphas is not None else [float(a) for a in preset["alphas"]]
    seeds = [int(s) for s in args.seeds] if args.seeds is not None else [int(s) for s in preset["seeds"]]
    rounds = int(args.rounds) if args.rounds is not None else int(preset["rounds"])
    local_epochs = int(args.local_epochs) if args.local_epochs is not None else int(preset["local_epochs"])
    max_samples_per_client = (
        int(args.max_samples_per_client)
        if args.max_samples_per_client is not None
        else int(preset["max_samples_per_client"])
    )

    if phase == "sanity":
        methods = ["flexfl"] if "flexfl" in methods else methods[:1]
        alphas = [alphas[0]]
        seeds = seeds[:2]
        rounds = 5
    elif phase == "baselines":
        method_set = set(methods)
        method_set.update({"fedavg", "moon", "scaffold"})
        methods = [m for m in ["fedavg", "moon", "scaffold"] if m in method_set]
        if args.seeds is None:
            seeds = _load_seed_list()
    else:
        if args.mode == "full" and args.methods is None:
            methods = ["flexfl", "fedavg", "moon", "scaffold"]
        if args.mode == "full" and args.seeds is None:
            seeds = _load_seed_list()

    generate_manifest_main(datasets=datasets)
    manifest_path = WORKSPACE / "artifacts" / "datasets" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Dataset manifest missing after generation")

    run_configs = _build_run_configs(
        methods=methods,
        datasets=datasets,
        alphas=alphas,
        seeds=seeds,
        rounds=rounds,
        local_epochs=local_epochs,
        max_samples_per_client=max_samples_per_client,
        require_gpu=bool(args.require_gpu),
    )
    if args.num_shards is not None:
        if args.shard is None:
            raise ValueError("--shard is required when --num-shards is set")
        if args.shard < 0 or args.shard >= args.num_shards:
            raise ValueError("--shard must be in the range [0, --num-shards)")
        run_configs = [cfg for index, cfg in enumerate(run_configs) if index % args.num_shards == args.shard]
    if args.max_runs is not None:
        run_configs = run_configs[: max(0, int(args.max_runs))]

    if args.rerun_failed:
        completed = set(_load_registry(SUCCESS_REGISTRY))
    else:
        completed = load_completed()
    pending = [cfg for cfg in run_configs if get_run_id(cfg) not in completed]
    target_run_ids = {get_run_id(cfg) for cfg in run_configs}

    print(
        f"Phase={phase} | planned={len(run_configs)} | completed={len(completed.intersection(target_run_ids))} | pending={len(pending)}",
        flush=True,
    )

    success_counter = 0
    failure_counter = 0
    done_counter = 0
    max_workers = int(args.max_workers) if args.max_workers is not None else MAX_WORKERS
    max_workers = max(1, min(max_workers, max(1, len(pending))))

    _write_run_plan(
        phase=phase,
        mode=args.mode,
        run_configs=run_configs,
        pending_configs=pending,
        max_workers=max_workers,
    )
    _write_live_progress(
        phase=phase,
        mode=args.mode,
        requested_runs=len(run_configs),
        pending_runs=len(pending),
        done_runs=0,
        success_runs=0,
        failed_runs=0,
        max_workers=max_workers,
        active=bool(pending),
    )

    if pending:
        use_tqdm = tqdm is not None
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            future_to_cfg = {executor.submit(run_with_retry, cfg): cfg for cfg in pending}
            iterator = as_completed(future_to_cfg)
            if use_tqdm:
                iterator = tqdm(iterator, total=len(future_to_cfg), desc="Experiment runs", unit="run")

            for future in iterator:
                cfg = future_to_cfg[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed_final",
                        "config": cfg,
                        "error": f"WorkerCrash: {type(exc).__name__}: {exc}",
                        "attempts": MAX_RETRIES,
                        "result": _minimal_failure_row(cfg, f"WorkerCrash: {type(exc).__name__}: {exc}"),
                    }

                row = save_result(result)
                validation_errors = validate_run_schema(RUNS_DIR / f"{get_run_id(cfg)}.json", expected_clients=10)
                if validation_errors:
                    row["status"] = "FAIL"
                    prior_error = str(row.get("error") or result.get("error") or "").strip()
                    schema_error = "; ".join(validation_errors)
                    row["error"] = f"{prior_error}; {schema_error}" if prior_error else schema_error
                    _write_json(RUNS_DIR / f"{get_run_id(cfg)}.json", row)
                    update_registry(get_run_id(cfg), "failed")

                done_counter += 1
                if row.get("status") == "SUCCESS":
                    success_counter += 1
                else:
                    failure_counter += 1

                print(f"Completed: {done_counter}/{len(pending)} | Failed: {failure_counter}", flush=True)
                _write_live_progress(
                    phase=phase,
                    mode=args.mode,
                    requested_runs=len(run_configs),
                    pending_runs=max(len(pending) - done_counter, 0),
                    done_runs=done_counter,
                    success_runs=success_counter,
                    failed_runs=failure_counter,
                    max_workers=max_workers,
                    active=True,
                )

                if args.abort_on_failure and row.get("status") != "SUCCESS":
                    for fut in future_to_cfg:
                        fut.cancel()
                    break
    else:
        print("No pending runs. Resume state already complete for selected config grid.", flush=True)
        _write_live_progress(
            phase=phase,
            mode=args.mode,
            requested_runs=len(run_configs),
            pending_runs=0,
            done_runs=0,
            success_runs=0,
            failed_runs=0,
            max_workers=max_workers,
            active=False,
        )

    run_rows = _read_run_artifacts(target_run_ids)
    run_counter = len(run_rows)

    if args.tune_baselines:
        _tune_baselines(global_cfg=global_cfg, seeds=seeds)

    stats_payload = _compute_statistics(run_rows)
    _write_json(STATS_DIR / "statistics.json", stats_payload)

    traceability = _build_traceability(run_rows)
    _write_json(TRACE_DIR / "traceability_table.json", traceability)

    claim = _build_final_claim(run_rows, stats_payload=stats_payload)
    _write_json(WORKSPACE / "artifacts" / "final_claim.json", claim)

    baseline_sanity = {
        "created_at": _now(),
        "rule": "SCAFFOLD >= FedAvg on non-IID (alpha <= 0.5)",
        "violations": [
            r
            for r in stats_payload["statistics"]
            if r.get("baseline") == "scaffold" and r.get("alpha") in (0.5, 0.1) and r.get("mean_delta_worst", 0.0) <= 0.0
        ],
    }
    _write_json(STATS_DIR / "baseline_sanity.json", baseline_sanity)

    summary = {
        "created_at": _now(),
        "requested_runs": len(run_configs),
        "executed_runs": run_counter,
        "success_runs": sum(1 for r in run_rows if r.get("status") == "SUCCESS"),
        "failed_runs": sum(1 for r in run_rows if r.get("status") != "SUCCESS"),
        "mode": args.mode,
        "phase": phase,
        "max_workers": max_workers,
        "shard": args.shard,
        "num_shards": args.num_shards,
    }
    _write_json(STATS_DIR / "execution_summary.json", summary)
    _write_live_progress(
        phase=phase,
        mode=args.mode,
        requested_runs=len(run_configs),
        pending_runs=0,
        done_runs=run_counter,
        success_runs=summary["success_runs"],
        failed_runs=summary["failed_runs"],
        max_workers=max_workers,
        active=False,
    )

    failed_rows = [r for r in run_rows if r.get("status") != "SUCCESS"]
    oom_failures = [r for r in failed_rows if "out of memory" in str(r.get("error", "")).lower()]
    zero_comm = [r for r in run_rows if float(r.get("communication", {}).get("total_bytes", 0.0)) <= 0.0]
    missing_rounds = [
        r
        for r in run_rows
        if int(r.get("rounds_completed", -1)) != int(r.get("rounds_expected", 0))
    ]

    if run_rows and len(failed_rows) == len(run_rows):
        raise RuntimeError("All runs failed validation/execution")
    if oom_failures and len(oom_failures) >= min(3, max(1, len(run_rows))):
        raise RuntimeError(f"GPU OOM repeatedly detected: {len(oom_failures)} failures")
    if zero_comm and len(zero_comm) == len(run_rows):
        raise RuntimeError("Communication remains zero across all runs")
    if missing_rounds and len(missing_rounds) == len(run_rows):
        raise RuntimeError("Rounds missing across all runs")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
