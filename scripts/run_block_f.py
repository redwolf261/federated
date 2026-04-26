#!/usr/bin/env python3
"""Block F: Mechanism Ablation Study Execution.

Identifies which components of FLEX are responsible for its performance gains
by isolating and evaluating: prototype exchange, clustering, and cluster-aware guidance.

Configuration (LOCKED):
- Dataset: CIFAR-10
- Clients: 10
- Seeds: [42, 43, 44]
- Rounds: 20
- Local epochs: 5
- Batch size: 64
- LR: 0.001
- Samples per client: 2000
- Alpha: 0.1
- Methods: fedavg_sgd, flex_full, flex_no_clustering, flex_random_clusters, flex_no_guidance
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
BATCH_SIZE = 64
LR = 0.001
MAX_SAMPLES = 20000  # 2000 per client
SEEDS = [42, 43, 44]
ALPHA = 0.1

METHODS = [
    {"name": "fedavg_sgd", "type": "fedavg", "optimizer": "sgd"},
    {"name": "flex_full", "type": "flex", "ablation_mode": "full", "cluster_aware": 2},
    {"name": "flex_no_clustering", "type": "flex", "ablation_mode": "no_clustering", "cluster_aware": 2},
    {"name": "flex_random_clusters", "type": "flex", "ablation_mode": "random_clusters", "cluster_aware": 2},
    {"name": "flex_no_guidance", "type": "flex", "ablation_mode": "full", "cluster_aware": 0},
]

F_RESULTS_PATH = COVERAGE_DIR / "F_results.jsonl"
BLOCK_F_MD_PATH = COVERAGE_DIR / "BLOCK_F.md"


def _load_existing() -> set[tuple[str, int]]:
    """Read F_results.jsonl and return set of completed (method_name, seed)."""
    completed = set()
    if not F_RESULTS_PATH.exists():
        return completed
    with open(F_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("block") == "F":
                    method = str(r.get("method"))
                    seed = int(r.get("seed"))
                    completed.add((method, seed))
            except Exception:
                continue
    return completed


def _log_to_md(text: str) -> None:
    """Append line to BLOCK_F.md."""
    BLOCK_F_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BLOCK_F_MD_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _log_run(fields: dict) -> None:
    """Append JSON line to F_results.jsonl."""
    F_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(F_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(fields) + "\n")


def main() -> None:
    overall_start = time.time()

    # ── INITIAL STATE ─────────────────────────────────────────────
    existing = _load_existing()
    total_runs = len(METHODS) * len(SEEDS)
    remaining = total_runs - len(existing)

    _log_to_md("# Block F: Mechanism Ablation Study Execution Log")
    _log_to_md("")
    _log_to_md(f"**Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log_to_md(f"**Objective:** Identify which FLEX components drive performance gains")
    _log_to_md(f"**Total planned runs:** {total_runs}")
    _log_to_md(f"**Already completed:** {len(existing)}")
    _log_to_md(f"**Remaining:** {remaining}")
    _log_to_md("")

    if remaining == 0:
        _log_to_md("All runs already complete. Nothing to do.")
        print("All Block F runs already complete.")
        return

    _log_to_md("## Planned Runs vs Skipped Runs")
    _log_to_md("")
    _log_to_md("| Method | Seed | Status |")
    _log_to_md("|--------|------|--------|")

    todo = []
    for method_cfg in METHODS:
        for seed in SEEDS:
            key = (method_cfg["name"], seed)
            if key in existing:
                _log_to_md(f"| {method_cfg['name']} | {seed} | SKIP (already done) |")
            else:
                _log_to_md(f"| {method_cfg['name']} | {seed} | RUN |")
                todo.append((method_cfg, seed))

    _log_to_md("")
    _log_to_md(f"**Runs to execute:** {len(todo)}")
    _log_to_md("")

    # ── EXECUTION ─────────────────────────────────────────────────
    _log_to_md("## Per-Run Execution Logs")
    _log_to_md("")

    for run_idx, (method_cfg, seed) in enumerate(todo, start=1):
        run_start = time.time()
        _log_to_md(f"### Run {run_idx}/{len(todo)}")
        _log_to_md(f"- **Method:** {method_cfg['name']}")
        _log_to_md(f"- **Seed:** {seed}")
        _log_to_md(f"- **Started:** {time.strftime('%H:%M:%S')}")

        print(f"\n{'='*60}")
        print(f"  Run {run_idx}/{len(todo)} | method={method_cfg['name']} | seed={seed}")
        print(f"{'='*60}")

        try:
            if method_cfg["type"] == "fedavg":
                result = run_fedavg_manual(
                    dataset_name=DATASET_NAME,
                    num_classes=NUM_CLASSES,
                    num_clients=NUM_CLIENTS,
                    rounds=ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    seed=seed,
                    alpha=ALPHA,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    max_samples=MAX_SAMPLES,
                    optimizer_name=method_cfg["optimizer"],
                )
            else:  # flex
                result = run_flex_simulator(
                    dataset_name=DATASET_NAME,
                    num_classes=NUM_CLASSES,
                    num_clients=NUM_CLIENTS,
                    rounds=ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    cluster_aware_epochs=method_cfg["cluster_aware"],
                    seed=seed,
                    alpha=ALPHA,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    max_samples=MAX_SAMPLES,
                    ablation_mode=method_cfg["ablation_mode"],
                )

            # Standardize output format
            client_accs = result.get("client_accuracies", {})
            p10 = float(np.percentile(list(client_accs.values()), 10)) if client_accs else 0.0

            record = {
                "block": "F",
                "method": method_cfg["name"],
                "seed": int(seed),
                "alpha": float(ALPHA),
                "samples_per_client": MAX_SAMPLES // NUM_CLIENTS,
                "mean_accuracy": float(result["mean_accuracy"]),
                "worst_accuracy": float(result["worst_accuracy"]),
                "std": float(result["std_across_clients"]),
                "p10": p10,
                "ablation_mode": method_cfg.get("ablation_mode", "N/A"),
                "cluster_aware_epochs": method_cfg.get("cluster_aware", 0),
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
    print(f"  BLOCK F EXECUTION COMPLETE")
    print(f"  Runs: {len(todo)}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'█'*60}")


if __name__ == "__main__":
    main()
