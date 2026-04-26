import json
import math
import os
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from scripts.phase2_q1_validation import run_moon, run_scaffold

# LOCKED CONFIGURATION
DATASET = "cifar10"
CLIENTS = 10
SEEDS = [42, 123, 456]
ROUNDS = int(os.environ.get("LOCKED_GRID_ROUNDS", "20"))
LOCAL_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.003
OPTIMIZER = "Adam"
ALPHAS = [0.1, 1.0]
METHODS = ["FedAvg", "MOON", "SCAFFOLD", "FLEX"]
MAX_SAMPLES = 20000
CENTRALIZED_REF = 0.87

OUT_ROOT = PROJECT_ROOT / "outputs" / "locked_cifar10_grid"
RUNS_DIR = OUT_ROOT / "runs"
SUMMARY_PATH = OUT_ROOT / "summary.json"
REPORT_PATH = OUT_ROOT / "report_sections.txt"

REQUIRED_RUN_KEYS = {
    "run_id",
    "method",
    "seed",
    "alpha",
    "rounds",
    "round_accuracy",
    "final_accuracy",
    "client_accuracies",
    "mean_client_accuracy",
    "worst_client_accuracy",
    "total_bytes_sent",
    "total_bytes_received",
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _run_id(method: str, alpha: float, seed: int) -> str:
    return f"{method.lower()}_a{alpha}_s{seed}"


def _run_file(method: str, alpha: float, seed: int) -> Path:
    return RUNS_DIR / f"{_run_id(method, alpha, seed)}.json"


def _is_valid_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if not REQUIRED_RUN_KEYS.issubset(set(data.keys())):
        return False
    if data.get("method") not in METHODS:
        return False
    if int(data.get("seed", -1)) not in SEEDS:
        return False
    if float(data.get("alpha", -1.0)) not in ALPHAS:
        return False
    if int(data.get("rounds", -1)) != ROUNDS:
        return False
    curve = data.get("round_accuracy", [])
    if not isinstance(curve, list):
        return False
    if len(curve) != ROUNDS:
        return False
    if not isinstance(data.get("client_accuracies", []), list):
        return False
    try:
        _ = [float(v) for v in curve]
        _ = [float(v) for v in data.get("client_accuracies", [])]
        float(data.get("final_accuracy", 0.0))
        float(data.get("mean_client_accuracy", 0.0))
        float(data.get("worst_client_accuracy", 0.0))
        int(data.get("total_bytes_sent", 0))
        int(data.get("total_bytes_received", 0))
    except Exception:
        return False
    return True


def _cleanup_after_run() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _execute_with_retry(method: str, alpha: float, seed: int) -> dict[str, Any]:
    try:
        out = _execute(method, alpha, seed)
        _cleanup_after_run()
        return out
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" not in msg and "cuda" not in msg:
            _cleanup_after_run()
            raise
        _cleanup_after_run()
        out = _execute(method, alpha, seed)
        _cleanup_after_run()
        return out


def _sim_config(method: str, alpha: float, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=_run_id(method, alpha, seed),
        dataset_name=DATASET,
        num_clients=CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUT_ROOT),
    )
    cfg.model.num_classes = 10
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.learning_rate = LEARNING_RATE
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // CLIENTS
    if method == "FedAvg":
        cfg.training.aggregation_mode = "fedavg"
    elif method == "FLEX":
        cfg.training.aggregation_mode = "prototype"
    else:
        raise ValueError(method)
    return cfg


def _from_simulator(method: str, alpha: float, seed: int) -> dict[str, Any]:
    cfg = _sim_config(method, alpha, seed)
    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()
    schema = sim.build_run_schema(history)
    rounds = schema.get("rounds", [])
    final_round = rounds[-1] if rounds else {"global_metrics": {}, "communication": {}}
    final_metrics = final_round.get("global_metrics", {})

    round_accuracy = [float(r["global_metrics"]["mean_client_accuracy"]) for r in rounds]
    client_acc_map = final_metrics.get("client_accuracies", {})
    client_accuracies = [float(v) for _, v in sorted(client_acc_map.items(), key=lambda kv: int(kv[0]))]

    total_sent = int(sum(int(r["communication"]["round_client_to_server_bytes"]) for r in rounds))
    total_recv = int(sum(int(r["communication"]["round_server_to_client_bytes"]) for r in rounds))

    return {
        "run_id": _run_id(method, alpha, seed),
        "method": method,
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": round_accuracy,
        "final_accuracy": float(round_accuracy[-1]) if round_accuracy else 0.0,
        "client_accuracies": client_accuracies,
        "mean_client_accuracy": float(final_metrics.get("mean_client_accuracy", 0.0)),
        "worst_client_accuracy": float(final_metrics.get("worst_client_accuracy", 0.0)),
        "total_bytes_sent": total_sent,
        "total_bytes_received": total_recv,
        "partition_fingerprint": str(schema.get("partition_fingerprint", "")),
    }


def _from_moon(alpha: float, seed: int) -> dict[str, Any]:
    r = run_moon(
        dataset_name=DATASET,
        num_classes=10,
        num_clients=CLIENTS,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        seed=seed,
        alpha=alpha,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
        mu=1.0,
        temperature=0.5,
        return_trace=True,
    )
    per_round = r.get("per_round", [])
    last = per_round[-1]["global_metrics"] if per_round else {}
    client_map = last.get("client_accuracies", {})

    return {
        "run_id": _run_id("MOON", alpha, seed),
        "method": "MOON",
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": [float(x["global_metrics"]["mean_client_accuracy"]) for x in per_round],
        "final_accuracy": float(r.get("mean_accuracy", 0.0)),
        "client_accuracies": [float(v) for _, v in sorted(client_map.items(), key=lambda kv: int(kv[0]))],
        "mean_client_accuracy": float(r.get("mean_accuracy", 0.0)),
        "worst_client_accuracy": float(r.get("worst_accuracy", 0.0)),
        "total_bytes_sent": int(sum(int(x["communication"]["round_client_to_server_bytes"]) for x in per_round)),
        "total_bytes_received": int(sum(int(x["communication"]["round_server_to_client_bytes"]) for x in per_round)),
        "partition_fingerprint": str(r.get("partition_fingerprint", "")),
    }


def _from_scaffold(alpha: float, seed: int) -> dict[str, Any]:
    r = run_scaffold(
        dataset_name=DATASET,
        num_classes=10,
        num_clients=CLIENTS,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        seed=seed,
        alpha=alpha,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
        zero_control=False,
        control_strength=1.0,
        apply_control=True,
        use_control_scaling=False,
        optimizer_name="adam",
        control_in_parameter_space=False,
        return_trace=True,
    )
    per_round = r.get("per_round", [])
    last = per_round[-1]["global_metrics"] if per_round else {}
    client_map = last.get("client_accuracies", {})

    return {
        "run_id": _run_id("SCAFFOLD", alpha, seed),
        "method": "SCAFFOLD",
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": [float(x["global_metrics"]["mean_client_accuracy"]) for x in per_round],
        "final_accuracy": float(r.get("mean_accuracy", 0.0)),
        "client_accuracies": [float(v) for _, v in sorted(client_map.items(), key=lambda kv: int(kv[0]))],
        "mean_client_accuracy": float(r.get("mean_accuracy", 0.0)),
        "worst_client_accuracy": float(r.get("worst_accuracy", 0.0)),
        "total_bytes_sent": int(sum(int(x["communication"]["round_client_to_server_bytes"]) for x in per_round)),
        "total_bytes_received": int(sum(int(x["communication"]["round_server_to_client_bytes"]) for x in per_round)),
        "partition_fingerprint": str(r.get("partition_fingerprint", "")),
    }


def _execute(method: str, alpha: float, seed: int) -> dict[str, Any]:
    if method == "FedAvg":
        return _from_simulator(method, alpha, seed)
    if method == "FLEX":
        return _from_simulator(method, alpha, seed)
    if method == "MOON":
        return _from_moon(alpha, seed)
    if method == "SCAFFOLD":
        return _from_scaffold(alpha, seed)
    raise ValueError(method)


def _run_to_50(curve: list[float]) -> int | None:
    for i, v in enumerate(curve, start=1):
        if float(v) >= 0.5:
            return i
    return None


def _section_report(all_runs: list[dict[str, Any]]) -> str:
    methods = ["FedAvg", "MOON", "SCAFFOLD", "FLEX"]
    by_alpha = {a: {m: [] for m in methods} for a in ALPHAS}
    for r in all_runs:
        by_alpha[float(r["alpha"])][r["method"]].append(r)

    lines: list[str] = []

    lines.append("## SECTION 1 — CORE RESULTS")
    for alpha in ALPHAS:
        lines.append(f"alpha = {alpha}")
        for m in methods:
            rows = sorted(by_alpha[alpha][m], key=lambda x: int(x["seed"]))
            vals = [float(x["final_accuracy"]) for x in rows]
            mean = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if vals else 0.0
            lines.append(f"\n{m}:\nmean = {mean}\nstd = {std}\nper_seed = {vals}")
        lines.append("")

    lines.append("## SECTION 2 — CONVERGENCE (seed=42)")
    alpha_for_curves = 1.0
    for m in methods:
        row = next((r for r in by_alpha[alpha_for_curves][m] if int(r["seed"]) == 42), None)
        curve = [float(v) for v in (row["round_accuracy"] if row else [])]
        lines.append(f"{m}: {curve}")

    lines.append("\n## SECTION 3 — FAIRNESS")
    for m in methods:
        vals = []
        for alpha in ALPHAS:
            vals.extend(by_alpha[alpha][m])
        mean_client = float(np.mean([float(r["mean_client_accuracy"]) for r in vals])) if vals else 0.0
        worst_client = float(np.mean([float(r["worst_client_accuracy"]) for r in vals])) if vals else 0.0
        all_client_accs = [float(v) for r in vals for v in r["client_accuracies"]]
        p10 = float(np.percentile(all_client_accs, 10)) if all_client_accs else 0.0
        p90 = float(np.percentile(all_client_accs, 90)) if all_client_accs else 0.0
        lines.append(f"{m}:\nmean_client = {mean_client}\nworst_client = {worst_client}\np10 = {p10}\np90 = {p90}")

    lines.append("\n## SECTION 4 — SPEED")
    for m in methods:
        rounds = []
        for alpha in ALPHAS:
            for r in by_alpha[alpha][m]:
                rounds.append(_run_to_50([float(v) for v in r["round_accuracy"]]))
        valid = [x for x in rounds if x is not None]
        metric = float(np.mean(valid)) if valid else None
        lines.append(f"round_to_50_percent_accuracy_{m.lower()} = {metric}")

    lines.append("\n## SECTION 5 — GAP ANALYSIS")
    for m in methods:
        vals = [float(r["final_accuracy"]) for alpha in ALPHAS for r in by_alpha[alpha][m]]
        mean = float(np.mean(vals)) if vals else 0.0
        key = m.lower()
        lines.append(f"gap_{key} = {CENTRALIZED_REF - mean}")

    lines.append("\n## SECTION 6 — STATISTICAL TESTS")
    flex_vals = [float(r["final_accuracy"]) for alpha in ALPHAS for r in by_alpha[alpha]["FLEX"]]
    for baseline in ["FedAvg", "MOON", "SCAFFOLD"]:
        base_vals = [float(r["final_accuracy"]) for alpha in ALPHAS for r in by_alpha[alpha][baseline]]
        t = ttest_ind(flex_vals, base_vals, equal_var=False)
        lines.append(f"FLEX vs {baseline}:\np-value = {float(t.pvalue)}\nt-stat = {float(t.statistic)}")

    lines.append("\n## SECTION 7 — SANITY FLAGS")
    any_failed = any(len(r["round_accuracy"]) != ROUNDS for r in all_runs)
    any_nan = any(any((not math.isfinite(float(v))) for v in r["round_accuracy"]) for r in all_runs)

    def collapsed(method: str) -> bool:
        vals = [float(r["final_accuracy"]) for alpha in ALPHAS for r in by_alpha[alpha][method]]
        return bool(vals) and float(np.mean(vals)) < 0.1

    scaffold_seed42 = next((r for r in by_alpha[1.0]["SCAFFOLD"] if int(r["seed"]) == 42), None)
    moon_seed42 = next((r for r in by_alpha[1.0]["MOON"] if int(r["seed"]) == 42), None)

    scaffold_stable = "NO"
    if scaffold_seed42:
        c = [float(v) for v in scaffold_seed42["round_accuracy"]]
        scaffold_stable = "YES" if len(c) >= 3 and c[2] > 0.1 else "NO"

    moon_stable = "NO"
    if moon_seed42:
        c = [float(v) for v in moon_seed42["round_accuracy"]]
        moon_stable = "YES" if len(c) >= 3 and c[2] > 0.1 else "NO"

    lines.append(f"1. Any run failed? {'YES' if any_failed else 'NO'}")
    lines.append(f"2. Any NaN/inf? {'YES' if any_nan else 'NO'}")
    lines.append(f"3. Any method collapsing? {'YES' if any(collapsed(m) for m in methods) else 'NO'}")
    lines.append(f"4. SCAFFOLD stable? {scaffold_stable}")
    lines.append(f"5. MOON stable? {moon_stable}")

    return "\n".join(lines)


def _phase_specs(alpha: float) -> list[tuple[str, float, int]]:
    specs = []
    for method in METHODS:
        for seed in SEEDS:
            specs.append((method, alpha, seed))
    return specs


def _execute_phase(specs: list[tuple[str, float, int]], workers: int, start_index: int) -> None:
    pending = []
    total = len(ALPHAS) * len(METHODS) * len(SEEDS)
    idx = start_index
    for method, alpha, seed in specs:
        out_file = _run_file(method, alpha, seed)
        if _is_valid_checkpoint(out_file):
            idx += 1
            continue
        pending.append((idx, method, alpha, seed, out_file))
        idx += 1

    if not pending:
        return

    if workers <= 1:
        for i, method, alpha, seed, out_file in pending:
            print(f"[RUN {i}/{total}] method={method}, seed={seed}, alpha={alpha}", flush=True)
            result = _execute_with_retry(method, alpha, seed)
            _atomic_write_json(out_file, result)
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {
            ex.submit(_execute_with_retry, method, alpha, seed): (i, method, alpha, seed, out_file)
            for i, method, alpha, seed, out_file in pending
        }
        for fut in as_completed(fut_map):
            i, method, alpha, seed, out_file = fut_map[fut]
            print(f"[RUN {i}/{total}] method={method}, seed={seed}, alpha={alpha}", flush=True)
            result = fut.result()
            _atomic_write_json(out_file, result)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    workers = 2

    # Pre-run validation
    pre_method, pre_seed, pre_alpha = "FedAvg", 42, 0.1
    pre_file = _run_file(pre_method, pre_alpha, pre_seed)
    if not _is_valid_checkpoint(pre_file):
        pre_result = _execute(pre_method, pre_alpha, pre_seed)
        _atomic_write_json(pre_file, pre_result)
    else:
        pre_result = json.loads(pre_file.read_text(encoding="utf-8"))

    if not _is_valid_checkpoint(pre_file):
        raise RuntimeError("Pre-run validation failed: output file not created")
    curve = [float(v) for v in pre_result.get("round_accuracy", [])]
    if not curve or max(curve) <= 0.3 or curve[-1] <= curve[0]:
        raise RuntimeError("Pre-run validation failed: unreasonable FedAvg curve")

    # Phase A
    _execute_phase(_phase_specs(0.1), workers=workers, start_index=1)

    # Phase B
    _execute_phase(_phase_specs(1.0), workers=workers, start_index=13)

    # Phase C aggregation
    all_runs = []
    for alpha in ALPHAS:
        for method in METHODS:
            for seed in SEEDS:
                p = _run_file(method, alpha, seed)
                if not _is_valid_checkpoint(p):
                    raise RuntimeError(f"Missing run file: {p}")
                all_runs.append(json.loads(p.read_text(encoding="utf-8")))

    summary = {
        "config": {
            "dataset": DATASET,
            "clients": CLIENTS,
            "seeds": SEEDS,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER,
            "alphas": ALPHAS,
            "methods": METHODS,
            "max_samples": MAX_SAMPLES,
            "workers": workers,
            "device": "cuda" if use_cuda else "cpu",
        },
        "runs": all_runs,
    }
    _atomic_write_json(SUMMARY_PATH, summary)

    report = _section_report(all_runs)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
        f.flush()
        os.fsync(f.fileno())

    print("\n" + report)


if __name__ == "__main__":
    main()
