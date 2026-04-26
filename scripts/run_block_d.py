#!/usr/bin/env python3
"""Block D: Heterogeneity Sweep Execution.

Determines whether FLEX-Persona's performance advantage is caused by
its ability to handle non-IID data (heterogeneity).

Configuration (LOCKED):
- Dataset: CIFAR-10
- Clients: 10
- Seeds: [42, 43, 44]
- Rounds: 20
- Local epochs: 5
- Cluster-aware epochs: 0
- Batch size: 64
- LR: 0.001
- Samples per client: 2000
- Alpha values: [0.05, 0.1, 0.5, 1.0, 10.0]
- Methods: flex_no_extra, fedavg_sgd
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_failure_mode_coverage import (
    run_flex_simulator,
    run_fedavg_manual,
    COVERAGE_DIR,
)

# ── CONFIGURATION ─────────────────────────────────────────────────
DATASET_NAME = "cifar10"
NUM_CLASSES = 10
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
CLUSTER_AWARE_EPOCHS = 0
BATCH_SIZE = 64
LR = 0.001
MAX_SAMPLES = 20000  # 2000 per client
SEEDS = [42, 43, 44]
ALPHA_VALUES = [0.05, 0.1, 0.5, 1.0, 10.0]
METHODS = ["flex_no_extra", "fedavg_sgd"]

D_RESULTS_PATH = COVERAGE_DIR / "D_results.jsonl"
BLOCK_D_MD_PATH = COVERAGE_DIR / "BLOCK_D.md"


def _load_existing() -> set[tuple[float, int, str]]:
    """Read D_results.jsonl and return set of completed (alpha, seed, method)."""
    completed = set()
    if not D_RESULTS_PATH.exists():
        return completed
    with open(D_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("block") == "D":
                    alpha = float(r.get("alpha"))
                    seed = int(r.get("seed"))
                    method = str(r.get("method"))
                    completed.add((alpha, seed, method))
            except Exception:
                continue
    return completed


def _log_to_md(text: str) -> None:
    """Append line to BLOCK_D.md."""
    BLOCK_D_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BLOCK_D_MD_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _log_run(fields: dict) -> None:
    """Append JSON line to D_results.jsonl."""
    D_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(D_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(fields) + "\n")


def main() -> None:
    overall_start = time.time()

    # ── INITIAL STATE ─────────────────────────────────────────────
    existing = _load_existing()
    total_runs = len(ALPHA_VALUES) * len(SEEDS) * len(METHODS)
    remaining = total_runs - len(existing)

    _log_to_md("# Block D Execution Log")
    _log_to_md("")
    _log_to_md(f"**Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log_to_md(f"**Total planned runs:** {total_runs}")
    _log_to_md(f"**Already completed:** {len(existing)}")
    _log_to_md(f"**Remaining:** {remaining}")
    _log_to_md("")

    if remaining == 0:
        _log_to_md("All runs already complete. Nothing to do.")
        print("All Block D runs already complete.")
        return

    _log_to_md("## Planned Runs vs Skipped Runs")
    _log_to_md("")
    _log_to_md("| Alpha | Seed | Method | Status |")
    _log_to_md("|-------|------|--------|--------|")

    todo = []
    for alpha in ALPHA_VALUES:
        for seed in SEEDS:
            for method in METHODS:
                key = (alpha, seed, method)
                if key in existing:
                    _log_to_md(f"| {alpha} | {seed} | {method} | SKIP (already done) |")
                else:
                    _log_to_md(f"| {alpha} | {seed} | {method} | RUN |")
                    todo.append((alpha, seed, method))

    _log_to_md("")
    _log_to_md(f"**Runs to execute:** {len(todo)}")
    _log_to_md("")

    # ── EXECUTION ─────────────────────────────────────────────────
    _log_to_md("## Per-Run Execution Logs")
    _log_to_md("")

    for run_idx, (alpha, seed, method) in enumerate(todo, start=1):
        run_start = time.time()
        _log_to_md(f"### Run {run_idx}/{len(todo)}")
        _log_to_md(f"- **Alpha:** {alpha}")
        _log_to_md(f"- **Seed:** {seed}")
        _log_to_md(f"- **Method:** {method}")
        _log_to_md(f"- **Started:** {time.strftime('%H:%M:%S')}")

        print(f"\n{'='*60}")
        print(f"  Run {run_idx}/{len(todo)} | alpha={alpha} | seed={seed} | method={method}")
        print(f"{'='*60}")

        try:
            if method == "flex_no_extra":
                result = run_flex_simulator(
                    dataset_name=DATASET_NAME,
                    num_classes=NUM_CLASSES,
                    num_clients=NUM_CLIENTS,
                    rounds=ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    cluster_aware_epochs=CLUSTER_AWARE_EPOCHS,
                    seed=seed,
                    alpha=alpha,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    max_samples=MAX_SAMPLES,
                )
            else:  # fedavg_sgd
                result = run_fedavg_manual(
                    dataset_name=DATASET_NAME,
                    num_classes=NUM_CLASSES,
                    num_clients=NUM_CLIENTS,
                    rounds=ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    seed=seed,
                    alpha=alpha,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    max_samples=MAX_SAMPLES,
                    optimizer_name="sgd",
                )

            # Standardize output format
            record = {
                "block": "D",
                "alpha": float(alpha),
                "seed": int(seed),
                "method": method,
                "samples_per_client": MAX_SAMPLES // NUM_CLIENTS,
                "mean_accuracy": float(result["mean_accuracy"]),
                "worst_accuracy": float(result["worst_accuracy"]),
                "std": float(result["std_across_clients"]),
                "p10": float(np.percentile(list(result["client_accuracies"].values()), 10)),
            }

            _log_run(record)

            elapsed = time.time() - run_start
            _log_to_md(f"- **Completed:** {time.strftime('%H:%M:%S')}")
            _log_to_md(f"- **Duration:** {elapsed:.1f}s")
            _log_to_md(f"- **Mean accuracy:** {record['mean_accuracy']:.4f}")
            _log_to_md(f"- **Worst accuracy:** {record['worst_accuracy']:.4f}")
            _log_to_md(f"- **Std:** {record['std']:.4f}")
            _log_to_md(f"- **P10:** {record['p10']:.4f}")
            _log_to_md("")

            print(f"  mean={record['mean_accuracy']:.4f}  "
                  f"worst={record['worst_accuracy']:.4f}  "
                  f"std={record['std']:.4f}  "
                  f"p10={record['p10']:.4f}  "
                  f"time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - run_start
            _log_to_md(f"- **ERROR:** {str(e)}")
            _log_to_md(f"- **Duration:** {elapsed:.1f}s")
            _log_to_md("")
            print(f"  ERROR: {e}")

    # ── SUMMARY ───────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    _log_to_md("## Execution Summary")
    _log_to_md("")
    _log_to_md(f"- **Total runs executed:** {len(todo)}")
    _log_to_md(f"- **Total elapsed time:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    _log_to_md(f"- **Finished:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log_to_md("")

    print(f"\n{'█'*60}")
    print(f"  BLOCK D COMPLETE")
    print(f"  Runs: {len(todo)}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'█'*60}")


if __name__ == "__main__":
    main()
