#!/usr/bin/env python3
"""Block I: Signal Nature Analysis.

Identifies the TRUE PROPERTY responsible for FLEX's performance gain
by testing different types of alignment signals during cluster-aware training.

We are no longer testing components (Blocks F-H) — we are testing the
nature of the training signal itself.

Methods tested:
  1. flex_full               — reference: cluster prototype alignment
  2. class_centroid_alignment — per-class centroids from client's own data
  3. global_centroid_alignment— single global centroid (no class structure)
  4. random_centroid_alignment— random fixed centroids per class
  5. feature_norm_only        — L2 normalization loss (geometry/scale)
  6. variance_minimization    — minimize intra-batch feature variance
  7. fedavg_sgd               — baseline

Interpretation rules:
  CASE 1: Only class_centroid works  → DRIVER = class structure
  CASE 2: global + variance also work → DRIVER = regularization
  CASE 3: norm_only works             → DRIVER = geometry/scale
  CASE 4: everything similar          → DRIVER = architecture bias

Output: outputs/failure_mode_coverage/block_I_results.json
        outputs/failure_mode_coverage/block_I.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from scripts.run_failure_mode_coverage import run_fedavg_manual, COVERAGE_DIR
from scripts.phase2_q1_validation import set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_JSON = COVERAGE_DIR / "block_I_results.json"
REPORT_MD = COVERAGE_DIR / "block_I.md"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment configuration (IDENTICAL to Block H)
# ---------------------------------------------------------------------------
DATASET_NAME = "cifar10"
NUM_CLASSES = 10
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
CLUSTER_AWARE_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001
MAX_SAMPLES = 20000          # 2000 samples/client
ALPHA = 0.1
SEEDS = [42, 43, 44]

# Map method name → alignment_mode config value
METHOD_TO_ALIGNMENT_MODE: dict[str, str] = {
    "flex_full":                "cluster_prototype",
    "class_centroid_alignment": "class_centroid",
    "global_centroid_alignment": "global_centroid",
    "random_centroid_alignment": "random_centroid",
    "feature_norm_only":        "feature_norm",
    "variance_minimization":    "variance_min",
}

FLEX_METHODS = list(METHOD_TO_ALIGNMENT_MODE.keys())

ALL_METHODS = FLEX_METHODS + ["fedavg_sgd"]

METHOD_LABELS: dict[str, str] = {
    "flex_full":                "FLEX Full (Reference)",
    "class_centroid_alignment": "Class Centroid Alignment",
    "global_centroid_alignment": "Global Centroid Alignment",
    "random_centroid_alignment": "Random Centroid Alignment",
    "feature_norm_only":        "Feature Norm Only",
    "variance_minimization":    "Variance Minimization",
    "fedavg_sgd":               "FedAvg SGD (Baseline)",
}


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------

def load_existing_results() -> list[dict]:
    """Load existing results from JSON file (for resume support)."""
    if not RESULTS_JSON.exists():
        return []
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: list[dict]) -> None:
    """Persist results to JSON file after every run."""
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def is_run_complete(results: list[dict], method: str, seed: int) -> bool:
    """Return True if this (method, seed) pair is already recorded."""
    for entry in results:
        if entry.get("method") == method and entry.get("seed") == seed:
            return True
    return False


def compute_p10(values: list[float]) -> float:
    """Return the 10th-percentile accuracy across clients."""
    return float(np.percentile(np.array(values), 10))


# ---------------------------------------------------------------------------
# FLEX variant runner
# ---------------------------------------------------------------------------

def run_flex_variant(variant: str, seed: int) -> dict:
    """Run a single FLEX alignment-mode variant for Block I."""
    alignment_mode = METHOD_TO_ALIGNMENT_MODE[variant]

    print(f"\n    Running: {variant} | alignment_mode={alignment_mode} | seed={seed}")

    cfg = ExperimentConfig(
        experiment_name=f"block_i_{variant}_s{seed}",
        dataset_name=DATASET_NAME,
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "prototype"
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = CLUSTER_AWARE_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.lambda_cluster = 0.1
    cfg.training.lambda_cluster_center = 0.01
    cfg.training.ablation_mode = "full"

    # KEY: Set the alignment mode for this variant
    cfg.training.alignment_mode = alignment_mode

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    # Collect per-client accuracies
    client_accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
    vals = list(client_accs.values())

    # Extract alignment loss from the final round (if logged)
    alignment_loss: Optional[float] = None
    if history:
        final_state = history[-1]
        alignments = []
        for _cid, metrics in final_state.local_metrics.items():
            if "cluster_alignment_loss" in metrics:
                alignments.append(float(metrics["cluster_alignment_loss"]))
        if alignments:
            alignment_loss = float(np.mean(alignments))

    result = {
        "method": variant,
        "seed": seed,
        "block": "I",
        "alignment_mode": alignment_mode,
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std": float(np.std(vals)),
        "p10": compute_p10(vals),
        "client_accuracies": {str(k): float(v) for k, v in client_accs.items()},
        "alignment_loss": alignment_loss,
        "config": {
            "lambda_cluster": cfg.training.lambda_cluster,
            "cluster_aware_epochs": cfg.training.cluster_aware_epochs,
            "local_epochs": cfg.training.local_epochs,
            "rounds": cfg.training.rounds,
            "alpha": ALPHA,
            "max_samples_per_client": MAX_SAMPLES // NUM_CLIENTS,
        },
    }

    print(
        f"      mean={result['mean_accuracy']:.4f}  "
        f"worst={result['worst_accuracy']:.4f}  "
        f"std={result['std']:.4f}  "
        f"p10={result['p10']:.4f}"
    )
    if alignment_loss is not None:
        print(f"      alignment_loss={alignment_loss:.6f}")

    return result


# ---------------------------------------------------------------------------
# FedAvg baseline runner
# ---------------------------------------------------------------------------

def run_fedavg_variant(seed: int) -> dict:
    """Run the FedAvg SGD baseline for Block I."""
    print(f"\n    Running: fedavg_sgd | seed={seed}")

    raw = run_fedavg_manual(
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
        optimizer_name="adam",
    )

    vals = list(raw["client_accuracies"].values())
    result = {
        "method": "fedavg_sgd",
        "seed": seed,
        "block": "I",
        "alignment_mode": "none",
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std": float(np.std(vals)),
        "p10": compute_p10(vals),
        "client_accuracies": {str(k): float(v) for k, v in raw["client_accuracies"].items()},
        "alignment_loss": None,
        "config": {
            "local_epochs": LOCAL_EPOCHS,
            "rounds": ROUNDS,
            "alpha": ALPHA,
            "max_samples_per_client": MAX_SAMPLES // NUM_CLIENTS,
        },
    }

    print(
        f"      mean={result['mean_accuracy']:.4f}  "
        f"worst={result['worst_accuracy']:.4f}  "
        f"std={result['std']:.4f}  "
        f"p10={result['p10']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Single run dispatcher
# ---------------------------------------------------------------------------

def run_single(method: str, seed: int) -> dict:
    """Dispatch a single (method, seed) run."""
    if method == "fedavg_sgd":
        return run_fedavg_variant(seed)
    return run_flex_variant(method, seed)


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict]) -> dict[str, dict]:
    """Aggregate per-run results by method, computing mean/std/worst/p10."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["method"]].append(r)

    stats: dict[str, dict] = {}
    for method, runs in groups.items():
        means = [r["mean_accuracy"] for r in runs]
        worsts = [r["worst_accuracy"] for r in runs]
        p10s = [r["p10"] for r in runs]
        stds = [r["std"] for r in runs]
        stats[method] = {
            "n": len(runs),
            "mean": float(np.mean(means)),
            "std_across_seeds": float(np.std(means)),
            "worst": float(np.mean(worsts)),
            "p10": float(np.mean(p10s)),
            "mean_per_client_std": float(np.mean(stds)),
            "seeds": [r["seed"] for r in runs],
            "per_seed_means": means,
        }
    return stats


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print an aggregated summary table to stdout."""
    stats = aggregate_results(results)
    flex_full_mean = stats.get("flex_full", {}).get("mean", 0.0)

    print(f"\n{'='*80}")
    print("  BLOCK I SUMMARY (Aggregated across seeds)")
    print(f"  {'Method':<30} {'Mean':>8} {'Std':>8} {'Worst':>8} {'P10':>8}  Drop")
    print(f"  {'-'*72}")

    for method in ALL_METHODS:
        if method not in stats:
            continue
        s = stats[method]
        drop = flex_full_mean - s["mean"]
        drop_str = f"{drop:+.4f}"
        print(
            f"  {method:<30} {s['mean']:>8.4f} {s['std_across_seeds']:>8.4f} "
            f"{s['worst']:>8.4f} {s['p10']:>8.4f}  {drop_str}"
        )
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Final report generator
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], stats: dict[str, dict]) -> str:
    """Generate the block_I.md report with causal conclusion."""
    flex_full = stats.get("flex_full", {})
    fedavg = stats.get("fedavg_sgd", {})
    flex_mean = flex_full.get("mean", 0.0)
    fedavg_mean = fedavg.get("mean", 0.0)
    total_gap = flex_mean - fedavg_mean

    lines: list[str] = []

    # Header
    lines += [
        "# Block I: Signal Nature Analysis",
        "",
        "**Experiment:** What is the TRUE PROPERTY responsible for FLEX's performance gain?",
        "",
        "**Setup (identical to Block H):**",
        f"- Dataset: CIFAR-10",
        f"- Clients: {NUM_CLIENTS}",
        f"- Samples/client: {MAX_SAMPLES // NUM_CLIENTS}",
        f"- Dirichlet α = {ALPHA}",
        f"- Rounds: {ROUNDS}",
        f"- Local epochs: {LOCAL_EPOCHS}",
        f"- Cluster-aware epochs: {CLUSTER_AWARE_EPOCHS}",
        f"- Seeds: {SEEDS}",
        f"- Total runs: {len(results)}",
        "",
        "---",
        "",
    ]

    # Methods table
    lines += [
        "## Methods",
        "",
        "| # | Method | Signal Type | Purpose |",
        "|---|--------|-------------|---------|",
        "| 1 | `flex_full` | Cluster prototypes (cross-client) | Reference |",
        "| 2 | `class_centroid_alignment` | Per-class centroids from **own** data | Remove cross-client info, keep class structure |",
        "| 3 | `global_centroid_alignment` | Single global centroid | Remove class structure, keep regularization |",
        "| 4 | `random_centroid_alignment` | Random fixed centroids per class | Remove semantic meaning |",
        "| 5 | `feature_norm_only` | L2 norm constraint on features | Test geometry/scale alone |",
        "| 6 | `variance_minimization` | Minimize intra-batch feature variance | Test generic regularization |",
        "| 7 | `fedavg_sgd` | None | Baseline |",
        "",
        "---",
        "",
    ]

    # Results table
    lines += [
        "## Results",
        "",
        "### Summary Table (Averaged Across Seeds)",
        "",
        "| Method | Mean Acc ± Std | Worst | P10 | Drop vs Full | Interpretation |",
        "|--------|---------------|-------|-----|--------------|----------------|",
    ]

    sorted_methods = sorted(
        ALL_METHODS,
        key=lambda m: stats.get(m, {}).get("mean", 0.0),
        reverse=True,
    )

    drops: dict[str, float] = {}
    for method in sorted_methods:
        if method not in stats:
            continue
        s = stats[method]
        mean = s["mean"]
        std = s["std_across_seeds"]
        worst = s["worst"]
        p10 = s["p10"]
        drop = flex_mean - mean
        drops[method] = drop
        drop_pct = (drop / flex_mean * 100) if flex_mean > 0 else 0.0

        if method == "flex_full":
            interp = "🔵 Reference"
        elif method == "fedavg_sgd":
            interp = "⬛ Baseline"
        elif abs(drop) < 0.02:
            interp = "✅ Equivalent — property preserved"
        elif drop < 0.05:
            interp = "⚠️ Minor degradation"
        elif drop < 0.10:
            interp = "🟡 Moderate loss"
        else:
            interp = "❌ Significant loss"

        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {mean:.4f} ± {std:.4f} | "
            f"{worst:.4f} | {p10:.4f} | {drop:+.4f} ({drop_pct:+.1f}%) | {interp} |"
        )

    lines += ["", "---", ""]

    # Per-seed raw table
    lines += [
        "### Per-Seed Raw Results",
        "",
        "| Method | Seed 42 | Seed 43 | Seed 44 | Mean |",
        "|--------|---------|---------|---------|------|",
    ]
    for method in ALL_METHODS:
        if method not in stats:
            continue
        s = stats[method]
        per_seed = {r["seed"]: r["mean_accuracy"] for r in results if r["method"] == method}
        s42 = per_seed.get(42, float("nan"))
        s43 = per_seed.get(43, float("nan"))
        s44 = per_seed.get(44, float("nan"))
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {s42:.4f} | {s43:.4f} | {s44:.4f} | {s['mean']:.4f} |"
        )

    lines += ["", "---", ""]

    # Causal analysis
    lines += ["## Causal Analysis", ""]

    # Compute relevant drops (excluding flex_full and fedavg_sgd)
    ablation_drops = {
        m: drops.get(m, 0.0)
        for m in [
            "class_centroid_alignment",
            "global_centroid_alignment",
            "random_centroid_alignment",
            "feature_norm_only",
            "variance_minimization",
        ]
        if m in stats
    }

    # Thresholds for judgment
    SIGNIFICANT_DROP = 0.05   # >5pp drop = mechanism matters
    MINOR_DROP = 0.02         # <2pp = virtually equivalent

    class_centroid_drop = ablation_drops.get("class_centroid_alignment", float("nan"))
    global_centroid_drop = ablation_drops.get("global_centroid_alignment", float("nan"))
    random_centroid_drop = ablation_drops.get("random_centroid_alignment", float("nan"))
    feature_norm_drop = ablation_drops.get("feature_norm_only", float("nan"))
    variance_drop = ablation_drops.get("variance_minimization", float("nan"))

    lines += [
        "### Drop Analysis",
        "",
        f"Total FLEX vs FedAvg gap: **{total_gap:+.4f}** ({total_gap/fedavg_mean*100:+.1f}% relative to baseline)",
        "",
        f"| Signal | Drop vs Full | Drop > 5pp? | Interpretation |",
        f"|--------|-------------|-------------|----------------|",
        f"| Class Centroid | {class_centroid_drop:+.4f} | {'YES ❌' if class_centroid_drop > SIGNIFICANT_DROP else 'no ✅'} | {'Cross-client class info matters' if class_centroid_drop > SIGNIFICANT_DROP else 'Own-data class centroids are sufficient'} |",
        f"| Global Centroid | {global_centroid_drop:+.4f} | {'YES ❌' if global_centroid_drop > SIGNIFICANT_DROP else 'no ✅'} | {'Class structure is required' if global_centroid_drop > SIGNIFICANT_DROP else 'Class structure not needed — regularization alone works'} |",
        f"| Random Centroid | {random_centroid_drop:+.4f} | {'YES ❌' if random_centroid_drop > SIGNIFICANT_DROP else 'no ✅'} | {'Semantic meaning matters' if random_centroid_drop > SIGNIFICANT_DROP else 'Even random targets work — not about information content'} |",
        f"| Feature Norm | {feature_norm_drop:+.4f} | {'YES ❌' if feature_norm_drop > SIGNIFICANT_DROP else 'no ✅'} | {'Geometry/scale insufficient' if feature_norm_drop > SIGNIFICANT_DROP else 'Geometry/scale alone explains gains'} |",
        f"| Variance Min | {variance_drop:+.4f} | {'YES ❌' if variance_drop > SIGNIFICANT_DROP else 'no ✅'} | {'Generic regularization insufficient' if variance_drop > SIGNIFICANT_DROP else 'Generic regularization explains gains'} |",
        "",
    ]

    # Determine causal verdict
    all_minor = all(d < MINOR_DROP for d in ablation_drops.values())
    class_only_matters = (class_centroid_drop < MINOR_DROP and
                          global_centroid_drop > SIGNIFICANT_DROP and
                          random_centroid_drop > SIGNIFICANT_DROP)
    regularization_drives = (global_centroid_drop < MINOR_DROP and
                              variance_drop < MINOR_DROP)
    geometry_drives = feature_norm_drop < MINOR_DROP

    lines += ["### Verdict", ""]

    if all_minor:
        case = "CASE 4"
        primary_driver = "Architecture Bias"
        mechanism_type = "architecture"
        rejected = "class-structure, regularization, geometry, cross-client signal"
        explanation = (
            "ALL ablation variants perform equivalently to flex_full. "
            "No specific property of the alignment signal matters — the gains are driven "
            "entirely by the backbone+adapter architecture. The training signal is irrelevant."
        )
    elif class_only_matters:
        case = "CASE 1"
        primary_driver = "Class Structure"
        mechanism_type = "class-structure"
        rejected = "generic regularization, geometry/scale, architecture-only"
        explanation = (
            "Class-centroid alignment (using the client's own data) preserves performance, "
            "but removing class structure (global centroid) causes a significant drop. "
            "The CLASS-DISCRIMINATIVE structure of the alignment signal is the key driver."
        )
    elif regularization_drives:
        case = "CASE 2"
        primary_driver = "Regularization"
        mechanism_type = "regularization"
        rejected = "cross-client information, class-structure specificity"
        explanation = (
            "Global centroid alignment (no class structure) and variance minimization "
            "both preserve performance. The key mechanism is generic representation "
            "regularization, not the specific semantic content of the prototypes."
        )
    elif geometry_drives:
        case = "CASE 3"
        primary_driver = "Geometry / Scale"
        mechanism_type = "geometry"
        rejected = "class-structure, cross-client signal, generic regularization"
        explanation = (
            "Feature norm constraint alone preserves performance. "
            "The geometry and scale of the shared feature space is the true driver."
        )
    else:
        case = "MIXED"
        primary_driver = "Multiple Factors"
        mechanism_type = "mixed"
        rejected = "single-cause hypothesis"
        explanation = (
            "Results are mixed — some ablations fail while others succeed. "
            "Multiple mechanisms may contribute. "
            "See individual drop values above for details."
        )

    lines += [
        f"**{case}**: {explanation}",
        "",
        "---",
        "",
        "## Final Conclusion",
        "",
        "```",
        f'Primary driver: {primary_driver}',
        f'Mechanism type: ({mechanism_type})',
        f'Rejected hypotheses: {rejected}',
        "```",
        "",
        f"**Reference (flex_full):** {flex_mean:.4f} ± {flex_full.get('std_across_seeds', 0.0):.4f}",
        f"**Baseline (fedavg_sgd):** {fedavg_mean:.4f} ± {fedavg.get('std_across_seeds', 0.0):.4f}",
        f"**Total gap explained by architecture:** ~{total_gap:.4f} ({total_gap/fedavg_mean*100:.1f}% above baseline)",
        "",
        "---",
        "",
    ]

    # Raw JSON appendix
    lines += [
        "## Appendix: Aggregated Statistics (JSON)",
        "",
        "```json",
        json.dumps(
            {
                method: {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in s.items()
                }
                for method, s in stats.items()
            },
            indent=2,
        ),
        "```",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "█" * 70)
    print("  BLOCK I: SIGNAL NATURE ANALYSIS")
    print("  Identifying the TRUE PROPERTY behind FLEX's performance gain")
    print("█" * 70)
    print(f"\n  Configuration:")
    print(f"    Dataset:              {DATASET_NAME}")
    print(f"    Clients:              {NUM_CLIENTS}")
    print(f"    Samples/client:       {MAX_SAMPLES // NUM_CLIENTS}")
    print(f"    Dirichlet α:          {ALPHA}")
    print(f"    Rounds:               {ROUNDS}")
    print(f"    Local epochs:         {LOCAL_EPOCHS}")
    print(f"    Cluster-aware epochs: {CLUSTER_AWARE_EPOCHS}")
    print(f"    Learning rate:        {LR}")
    print(f"    Batch size:           {BATCH_SIZE}")
    print(f"    Seeds:                {SEEDS}")
    print(f"    Methods:              {len(ALL_METHODS)}")
    print(f"    Total runs:           {len(ALL_METHODS) * len(SEEDS)}")

    # Load any existing results (for resume support)
    results = load_existing_results()
    print(f"\n  Loaded {len(results)} existing results (skipping completed runs)")

    total_start = time.time()
    completed = 0
    skipped = 0

    for seed in SEEDS:
        print(f"\n  {'─'*60}")
        print(f"  Seed {seed}")
        print(f"  {'─'*60}")

        for method in ALL_METHODS:
            if is_run_complete(results, method, seed):
                print(f"    SKIP: {method} | seed={seed} (already complete)")
                skipped += 1
                continue

            run_start = time.time()
            try:
                result = run_single(method, seed)
            except Exception as exc:
                print(f"    ERROR: {method} | seed={seed} → {type(exc).__name__}: {exc}")
                raise

            result["wall_time_seconds"] = round(time.time() - run_start, 2)
            results.append(result)
            save_results(results)
            completed += 1
            print(f"      Completed in {result['wall_time_seconds']:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  BLOCK I COMPLETE")
    print(f"  Completed: {completed} new runs")
    print(f"  Skipped:   {skipped} cached runs")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'='*70}")

    # Print summary table
    print_summary(results)

    # Generate and save markdown report
    stats = aggregate_results(results)
    report = generate_report(results, stats)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Results JSON: {RESULTS_JSON}")
    print(f"  Report MD:    {REPORT_MD}")
    print(f"\n  Done! Run 'python scripts/generate_block_i_report.py' for the full analysis.")


if __name__ == "__main__":
    main()
