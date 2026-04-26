import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.federated.simulator import FederatedSimulator
from scripts.phase2_q1_validation import run_moon, run_scaffold

try:
    from scipy.stats import ttest_ind
except Exception:  # pragma: no cover
    ttest_ind = None

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "locked_cifar10_grid"
RUNS_DIR = OUTPUT_ROOT / "runs"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
REPORT_TXT = OUTPUT_ROOT / "report_sections.txt"

DATASET = "cifar10"
NUM_CLASSES = 10
CLIENTS = 10
SEEDS = [42, 123, 456]
ROUNDS = 20
LOCAL_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.003
ALPHAS = [0.1, 1.0]
METHODS = ["fedavg", "moon", "scaffold", "flex"]
MAX_SAMPLES = 20000
CENTRALIZED_REF = 0.87
WORKERS = 2


@dataclass(frozen=True)
class RunSpec:
    method: str
    alpha: float
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.method}_a{self.alpha}_s{self.seed}"

    @property
    def out_file(self) -> Path:
        return RUNS_DIR / f"{self.run_id}.json"


def _partition_fingerprint(alpha: float, seed: int) -> str:
    cfg = ExperimentConfig(
        experiment_name=f"fingerprint_a{alpha}_s{seed}",
        dataset_name=DATASET,
        num_clients=CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_ROOT),
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // CLIENTS
    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()
    return dm.partition_fingerprint(bundles)


def _build_sim_config(method: str, alpha: float, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=f"locked_{method}_a{alpha}_s{seed}",
        dataset_name=DATASET,
        num_clients=CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_ROOT),
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // CLIENTS
    if method == "fedavg":
        cfg.training.aggregation_mode = "fedavg"
    elif method == "flex":
        cfg.training.aggregation_mode = "prototype"
    else:
        raise ValueError(method)
    return cfg


def _run_fedavg_or_flex(method: str, alpha: float, seed: int) -> dict[str, Any]:
    cfg = _build_sim_config(method, alpha, seed)
    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()
    schema = sim.build_run_schema(history)

    rounds = schema["rounds"]
    round_accuracy = [float(r["global_metrics"]["mean_client_accuracy"]) for r in rounds]
    final_global = rounds[-1]["global_metrics"] if rounds else {}
    client_acc_dict = final_global.get("client_accuracies", {})
    client_accs = [float(v) for _, v in sorted(client_acc_dict.items(), key=lambda kv: int(kv[0]))]

    sent = int(sum(int(r["communication"]["round_client_to_server_bytes"]) for r in rounds))
    recv = int(sum(int(r["communication"]["round_server_to_client_bytes"]) for r in rounds))

    return {
        "run_id": f"{method}_a{alpha}_s{seed}",
        "method": method,
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": round_accuracy,
        "final_accuracy": float(round_accuracy[-1]) if round_accuracy else 0.0,
        "client_accuracies": client_accs,
        "mean_client_accuracy": float(final_global.get("mean_client_accuracy", 0.0)),
        "worst_client_accuracy": float(final_global.get("worst_client_accuracy", 0.0)),
        "total_bytes_sent": sent,
        "total_bytes_received": recv,
        "partition_fingerprint": str(schema["partition_fingerprint"]),
    }


def _run_moon(alpha: float, seed: int) -> dict[str, Any]:
    r = run_moon(
        dataset_name=DATASET,
        num_classes=NUM_CLASSES,
        num_clients=CLIENTS,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        seed=seed,
        alpha=alpha,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
        mu=1.0,
        temperature=0.5,
        return_trace=True,
    )
    per_round = r["per_round"]
    round_accuracy = [float(x["global_metrics"]["mean_client_accuracy"]) for x in per_round]
    last = per_round[-1]["global_metrics"] if per_round else {}
    client_accs = [float(v) for _, v in sorted(last.get("client_accuracies", {}).items(), key=lambda kv: int(kv[0]))]
    sent = int(sum(int(x["communication"]["round_client_to_server_bytes"]) for x in per_round))
    recv = int(sum(int(x["communication"]["round_server_to_client_bytes"]) for x in per_round))
    return {
        "run_id": f"moon_a{alpha}_s{seed}",
        "method": "moon",
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": round_accuracy,
        "final_accuracy": float(r["mean_accuracy"]),
        "client_accuracies": client_accs,
        "mean_client_accuracy": float(r["mean_accuracy"]),
        "worst_client_accuracy": float(r["worst_accuracy"]),
        "total_bytes_sent": sent,
        "total_bytes_received": recv,
        "partition_fingerprint": str(r["partition_fingerprint"]),
    }


def _run_scaffold(alpha: float, seed: int) -> dict[str, Any]:
    r = run_scaffold(
        dataset_name=DATASET,
        num_classes=NUM_CLASSES,
        num_clients=CLIENTS,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        seed=seed,
        alpha=alpha,
        lr=LR,
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
    per_round = r["per_round"]
    round_accuracy = [float(x["global_metrics"]["mean_client_accuracy"]) for x in per_round]
    last = per_round[-1]["global_metrics"] if per_round else {}
    client_accs = [float(v) for _, v in sorted(last.get("client_accuracies", {}).items(), key=lambda kv: int(kv[0]))]
    sent = int(sum(int(x["communication"]["round_client_to_server_bytes"]) for x in per_round))
    recv = int(sum(int(x["communication"]["round_server_to_client_bytes"]) for x in per_round))
    return {
        "run_id": f"scaffold_a{alpha}_s{seed}",
        "method": "scaffold",
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": round_accuracy,
        "final_accuracy": float(r["mean_accuracy"]),
        "client_accuracies": client_accs,
        "mean_client_accuracy": float(r["mean_accuracy"]),
        "worst_client_accuracy": float(r["worst_accuracy"]),
        "total_bytes_sent": sent,
        "total_bytes_received": recv,
        "partition_fingerprint": str(r["partition_fingerprint"]),
    }


def _run_one(spec: RunSpec) -> dict[str, Any]:
    if spec.method == "fedavg":
        return _run_fedavg_or_flex("fedavg", spec.alpha, spec.seed)
    if spec.method == "flex":
        return _run_fedavg_or_flex("flex", spec.alpha, spec.seed)
    if spec.method == "moon":
        return _run_moon(spec.alpha, spec.seed)
    if spec.method == "scaffold":
        return _run_scaffold(spec.alpha, spec.seed)
    raise ValueError(spec.method)


def _round_to_target(curve: list[float], threshold: float = 0.5) -> int | None:
    for idx, v in enumerate(curve, start=1):
        if v >= threshold:
            return idx
    return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _p10_p90(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, 10)), float(np.percentile(arr, 90))


def _ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    if ttest_ind is None or len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = ttest_ind(a, b, equal_var=False)
    return float(res.pvalue), float(res.statistic)


def _section_report(runs: list[dict[str, Any]]) -> str:
    by_alpha_method: dict[float, dict[str, list[dict[str, Any]]]] = {}
    by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in METHODS}
    for r in runs:
        by_alpha_method.setdefault(r["alpha"], {}).setdefault(r["method"], []).append(r)
        by_method[r["method"]].append(r)

    lines: list[str] = []

    # SECTION 1
    lines.append("SECTION 1 - CORE RESULTS")
    for alpha in sorted(by_alpha_method.keys()):
        lines.append(f"alpha = {alpha}")
        for method in ["fedavg", "moon", "scaffold", "flex"]:
            rows = sorted(by_alpha_method[alpha].get(method, []), key=lambda x: x["seed"])
            vals = [float(x["final_accuracy"]) for x in rows]
            mean, std = _mean_std(vals)
            lines.append(f"{method.upper()}:\nmean = {mean}\nstd = {std}\nper_seed = {vals}")

    # SECTION 2
    lines.append("\nSECTION 2 - CONVERGENCE")
    for alpha in sorted(by_alpha_method.keys()):
        lines.append(f"alpha = {alpha}, seed = 42")
        for method in ["fedavg", "moon", "scaffold", "flex"]:
            row = next((x for x in by_alpha_method[alpha].get(method, []) if x["seed"] == 42), None)
            curve = row["round_accuracy"] if row else []
            lines.append(f"{method.upper()}:\n{curve}")

    # SECTION 3
    lines.append("\nSECTION 3 - FAIRNESS")
    for alpha in sorted(by_alpha_method.keys()):
        lines.append(f"alpha = {alpha}")
        for method in ["fedavg", "moon", "scaffold", "flex"]:
            rows = by_alpha_method[alpha].get(method, [])
            means = [float(r["mean_client_accuracy"]) for r in rows]
            worsts = [float(r["worst_client_accuracy"]) for r in rows]
            p10s = []
            p90s = []
            for r in rows:
                p10, p90 = _p10_p90([float(v) for v in r["client_accuracies"]])
                p10s.append(p10)
                p90s.append(p90)
            lines.append(
                f"{method.upper()}:\n"
                f"mean_client = {float(np.mean(means)) if means else 0.0}\n"
                f"worst_client = {float(np.mean(worsts)) if worsts else 0.0}\n"
                f"p10 = {float(np.mean(p10s)) if p10s else 0.0}\n"
                f"p90 = {float(np.mean(p90s)) if p90s else 0.0}"
            )

    # SECTION 4
    lines.append("\nSECTION 4 - SPEED METRIC")
    lines.append("round_to_50_percent_accuracy:")
    for method in ["fedavg", "moon", "scaffold", "flex"]:
        rounds_to = []
        for r in by_method[method]:
            rounds_to.append(_round_to_target([float(v) for v in r["round_accuracy"]], 0.5))
        valid = [x for x in rounds_to if x is not None]
        metric = float(np.mean(valid)) if valid else None
        lines.append(f"{method.upper()} = {metric}")

    # SECTION 5
    lines.append("\nSECTION 5 - GAP ANALYSIS")
    for method in ["fedavg", "moon", "scaffold", "flex"]:
        vals = [float(r["final_accuracy"]) for r in by_method[method]]
        mean_val = float(np.mean(vals)) if vals else 0.0
        lines.append(f"gap_{method} = {CENTRALIZED_REF - mean_val}")

    # SECTION 6
    lines.append("\nSECTION 6 - STATISTICAL TESTS")
    flex_vals = [float(r["final_accuracy"]) for r in by_method["flex"]]
    for baseline in ["fedavg", "moon", "scaffold"]:
        base_vals = [float(r["final_accuracy"]) for r in by_method[baseline]]
        pval, tstat = _ttest(flex_vals, base_vals)
        lines.append(f"FLEX vs {baseline.upper()}:\np-value = {pval}\nt-stat = {tstat}")

    # SECTION 7
    lines.append("\nSECTION 7 - SANITY FLAGS")
    any_failed = any(len(r["round_accuracy"]) != ROUNDS for r in runs)
    any_nan = any(any((not math.isfinite(float(v))) for v in r["round_accuracy"]) for r in runs)

    def _collapsed(method: str) -> bool:
        vals = [float(r["final_accuracy"]) for r in by_method[method]]
        return bool(vals) and float(np.mean(vals)) < 0.1

    scaffold_seed42_alpha1 = next(
        (r for r in runs if r["method"] == "scaffold" and r["seed"] == 42 and float(r["alpha"]) == 1.0),
        None,
    )
    moon_seed42_alpha1 = next(
        (r for r in runs if r["method"] == "moon" and r["seed"] == 42 and float(r["alpha"]) == 1.0),
        None,
    )

    scaffold_stable = "NO"
    if scaffold_seed42_alpha1 is not None:
        curve = [float(v) for v in scaffold_seed42_alpha1["round_accuracy"]]
        scaffold_stable = "YES" if len(curve) >= 3 and curve[1] > 0.05 and curve[2] > 0.1 else "NO"

    moon_stable = "NO"
    if moon_seed42_alpha1 is not None:
        curve = [float(v) for v in moon_seed42_alpha1["round_accuracy"]]
        moon_stable = "YES" if len(curve) >= 3 and curve[1] > 0.05 and curve[2] > 0.1 else "NO"

    lines.append(f"1. Any run failed? {'YES' if any_failed else 'NO'}")
    lines.append(f"2. Any NaN/inf? {'YES' if any_nan else 'NO'}")
    lines.append(f"3. Any method collapsing? {'YES' if any(_collapsed(m) for m in METHODS) else 'NO'}")
    lines.append(f"4. SCAFFOLD stable (no early collapse)? {scaffold_stable}")
    lines.append(f"5. MOON stable? {moon_stable}")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    specs = [RunSpec(method=m, alpha=a, seed=s) for a in ALPHAS for m in METHODS for s in SEEDS]

    # Checkpointing: skip completed runs.
    completed = set()
    for spec in specs:
        if spec.out_file.exists():
            completed.add(spec.run_id)

    remaining = [s for s in specs if s.run_id not in completed]
    print(f"Total runs: {len(specs)} | Completed: {len(completed)} | Remaining: {len(remaining)}")

    if remaining:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            fut_to_spec = {ex.submit(_run_one, spec): spec for spec in remaining}
            for fut in as_completed(fut_to_spec):
                spec = fut_to_spec[fut]
                run = fut.result()
                spec.out_file.write_text(json.dumps(run, indent=2), encoding="utf-8")
                print(f"[DONE] {spec.run_id}")

    # Load all runs.
    all_runs = []
    for spec in specs:
        if not spec.out_file.exists():
            raise RuntimeError(f"Missing run output: {spec.out_file}")
        all_runs.append(json.loads(spec.out_file.read_text(encoding="utf-8")))

    # Data integrity: fingerprint consistency per (alpha, seed).
    for alpha in ALPHAS:
        for seed in SEEDS:
            expected = _partition_fingerprint(alpha, seed)
            subset = [r for r in all_runs if float(r["alpha"]) == float(alpha) and int(r["seed"]) == int(seed)]
            for r in subset:
                if str(r.get("partition_fingerprint", "")) != str(expected):
                    raise AssertionError(
                        f"Partition fingerprint mismatch for alpha={alpha}, seed={seed}, method={r['method']}"
                    )

    summary = {
        "config": {
            "dataset": DATASET,
            "clients": CLIENTS,
            "seeds": SEEDS,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "optimizer": "Adam",
            "alphas": ALPHAS,
            "methods": METHODS,
            "max_samples": MAX_SAMPLES,
        },
        "runs": all_runs,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = _section_report(all_runs)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
