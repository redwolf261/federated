#!/usr/bin/env python3
"""Block K: Extra-Epoch Control Experiment.

Purpose: Close the final causal ambiguity.

After Blocks I+J we know:
  - Prototype content is irrelevant
  - Adapter bottleneck (8192→64 + L2 norm) drives +12pp gain
  - The remaining gap (flex_full +39pp vs random_proj +12pp) is unattributed

Hypothesis: The remaining ~27pp gap comes from EXTRA TRAINING STEPS (2
cluster-aware epochs) not from guidance signal quality.

Methods (all use backbone+adapter architecture, identical hyperparams):
  1. flex_full          - 5 local + 2 guidance epochs (reference)
  2. extra_local_7ep   - 7 local epochs, no guidance, no exchange
                         → MATCHED COMPUTE, different structure
  3. random_guidance   - 5 local + 2 guidance w/ random targets
                         → SAME STRUCTURE, meaningless signal
  4. local_only_5ep    - 5 local epochs, no guidance, no exchange
                         → BASELINE compute

Decision table:
  flex_full ≈ extra_local_7ep → EXTRA TRAINING is the mechanism
  flex_full ≈ random_guidance → STRUCTURE matters (not content)
  extra_local_7ep ≈ random_guidance → both are equivalent (training > signal)
  local_only_5ep << all others → extra epochs matter

Output:
  outputs/failure_mode_coverage/block_K_results.json
  outputs/failure_mode_coverage/block_K.md
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from scripts.phase2_q1_validation import set_seed

# ---------------------------------------------------------------------------
COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = COVERAGE_DIR / "block_K_results.json"
REPORT_MD    = COVERAGE_DIR / "block_K.md"

DATASET      = "cifar10"
NUM_CLASSES  = 10
NUM_CLIENTS  = 10
ROUNDS       = 20
LOCAL_EPOCHS = 5
BATCH_SIZE   = 64
LR           = 0.001
MAX_SAMPLES  = 20_000
ALPHA        = 0.1
SEEDS        = [42, 43, 44]
LAMBDA_CLUSTER = 0.1

ALL_METHODS = [
    "flex_full",
    "extra_local_7ep",
    "random_guidance",
    "local_only_5ep",
]

LABELS = {
    "flex_full":       "FLEX Full (5+2 epochs, real guidance)",
    "extra_local_7ep": "Extra Local Epochs (7+0, no guidance)",
    "random_guidance": "Random Guidance (5+2, random targets)",
    "local_only_5ep":  "Local Only (5+0, no guidance)",
}


# ---------------------------------------------------------------------------
def _base_config(name: str, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=name,
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes              = NUM_CLASSES
    cfg.model.client_backbones         = ["small_cnn"]
    cfg.model.shared_dim               = 64
    cfg.training.rounds                = ROUNDS
    cfg.training.learning_rate         = LR
    cfg.training.batch_size            = BATCH_SIZE
    cfg.training.weight_decay          = 1e-5
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.lambda_cluster        = LAMBDA_CLUSTER
    cfg.training.lambda_cluster_center = 0.01
    return cfg


def _run(method: str, seed: int) -> dict:
    set_seed(seed)
    name = f"block_k_{method}_s{seed}"

    if method == "flex_full":
        cfg = _base_config(name, seed)
        cfg.training.aggregation_mode    = "prototype"
        cfg.training.local_epochs        = LOCAL_EPOCHS
        cfg.training.cluster_aware_epochs = 2
        cfg.training.ablation_mode       = "full"
        cfg.training.alignment_mode      = "cluster_prototype"

    elif method == "extra_local_7ep":
        # 7 local CE epochs, no cluster guidance, no prototype exchange
        cfg = _base_config(name, seed)
        cfg.training.aggregation_mode    = "prototype"
        cfg.training.local_epochs        = LOCAL_EPOCHS + 2   # = 7
        cfg.training.cluster_aware_epochs = 0
        cfg.training.ablation_mode       = "no_prototype_sharing"

    elif method == "random_guidance":
        cfg = _base_config(name, seed)
        cfg.training.aggregation_mode    = "prototype"
        cfg.training.local_epochs        = LOCAL_EPOCHS
        cfg.training.cluster_aware_epochs = 2
        cfg.training.ablation_mode       = "full"
        cfg.training.alignment_mode      = "random_centroid"

    elif method == "local_only_5ep":
        # 5 local CE epochs, no cluster guidance, no prototype exchange
        cfg = _base_config(name, seed)
        cfg.training.aggregation_mode    = "prototype"
        cfg.training.local_epochs        = LOCAL_EPOCHS       # = 5
        cfg.training.cluster_aware_epochs = 0
        cfg.training.ablation_mode       = "no_prototype_sharing"

    else:
        raise ValueError(f"Unknown method: {method}")

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    sim.run_experiment()

    client_accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
    vals = list(client_accs.values())
    return {
        "method": method, "seed": seed, "block": "K",
        "mean_accuracy":  float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std":            float(np.std(vals)),
        "p10":            float(np.percentile(vals, 10)),
        "client_accuracies": {str(k): float(v) for k, v in client_accs.items()},
    }


# ---------------------------------------------------------------------------
def _load() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    return json.loads(RESULTS_JSON.read_text())


def _save(r: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(r, indent=2))


def _done(r: list[dict], method: str, seed: int) -> bool:
    return any(x["method"] == method and x["seed"] == seed for x in r)


def _agg(results: list[dict]) -> dict[str, dict]:
    g: dict[str, list] = defaultdict(list)
    for r in results:
        g[r["method"]].append(r)
    out = {}
    for m, runs in g.items():
        means  = [r["mean_accuracy"]  for r in runs]
        worsts = [r["worst_accuracy"] for r in runs]
        p10s   = [r["p10"]            for r in runs]
        out[m] = {"n": len(runs), "mean": float(np.mean(means)),
                  "std": float(np.std(means)), "worst": float(np.mean(worsts)),
                  "p10": float(np.mean(p10s)),
                  "seeds": sorted(r["seed"] for r in runs)}
    return out


# ---------------------------------------------------------------------------
def _report(results: list[dict], stats: dict[str, dict]) -> str:
    flex  = stats.get("flex_full", {}).get("mean", float("nan"))
    local = stats.get("local_only_5ep", {}).get("mean", float("nan"))
    extra = stats.get("extra_local_7ep", {}).get("mean", float("nan"))
    rand  = stats.get("random_guidance", {}).get("mean", float("nan"))

    lines = [
        "# Block K: Extra-Epoch Control Experiment",
        "",
        "**Question:** Does the cluster-aware phase benefit performance through",
        "(A) **extra training steps** or (B) **guidance signal structure**?",
        "",
        f"FLEX Full (reference): {flex:.4f}",
        f"Total gap (flex vs local_only): {flex-local:+.4f}",
        "",
        "---",
        "## Results Table",
        "",
        "| Method | Mean ± Std | Worst | P10 | Δ vs FLEX | Δ vs local_only |",
        "|--------|-----------|-------|-----|-----------|----------------|",
    ]

    for m in ALL_METHODS:
        if m not in stats:
            continue
        s = stats[m]
        df = s["mean"] - flex
        dl = s["mean"] - local
        lines.append(
            f"| {LABELS[m]} | {s['mean']:.4f} ± {s['std']:.4f} | "
            f"{s['worst']:.4f} | {s['p10']:.4f} | {df:+.4f} | {dl:+.4f} |"
        )

    lines += ["", "---", "## Causal Decision", ""]

    THR = 0.02
    extra_vs_flex  = abs(extra - flex)
    rand_vs_flex   = abs(rand  - flex)
    extra_vs_rand  = abs(extra - rand)
    extra_vs_local = extra - local

    lines.append("### Key Comparisons")
    lines.append("")
    lines.append(f"- `extra_local_7ep` vs `flex_full`:     {extra-flex:+.4f} "
                 f"({'≈ same' if extra_vs_flex < THR else 'DIFFERENT'})")
    lines.append(f"- `random_guidance` vs `flex_full`:     {rand-flex:+.4f} "
                 f"({'≈ same' if rand_vs_flex < THR else 'DIFFERENT'})")
    lines.append(f"- `extra_local_7ep` vs `random_guidance`: {extra-rand:+.4f} "
                 f"({'≈ same' if extra_vs_rand < THR else 'DIFFERENT'})")
    lines.append(f"- `extra_local_7ep` vs `local_only_5ep`: {extra_vs_local:+.4f} "
                 f"({'extra epochs help' if extra_vs_local > THR else 'no difference'})")
    lines.append("")

    # Verdict
    if extra_vs_flex < THR:
        verdict = "EXTRA TRAINING STEPS"
        explanation = (
            "extra_local_7ep ≈ flex_full — the cluster-aware phase helps because "
            "it provides extra gradient steps, NOT because of guidance signal content. "
            "Replacing guidance with plain CE epochs achieves the same result."
        )
        causal = (
            "Primary driver: Additional optimization steps (7 vs 5 epochs/round).\n"
            "Secondary driver: Adapter geometry conditioning (from Block J).\n"
            "Rejected: Guidance signal content, prototype semantics, cluster structure."
        )
    elif rand_vs_flex < THR and extra_vs_rand > THR:
        verdict = "CLUSTER-AWARE STRUCTURE"
        explanation = (
            "random_guidance ≈ flex_full but extra_local_7ep << flex_full — the "
            "cluster-aware training phase structure helps beyond just extra epochs. "
            "The auxiliary loss formulation (even with random targets) matters."
        )
        causal = (
            "Primary driver: Cluster-aware training structure (auxiliary loss phase).\n"
            "Secondary driver: Adapter geometry conditioning.\n"
            "Rejected: Prototype semantic content."
        )
    else:
        verdict = "MIXED — BOTH CONTRIBUTE"
        explanation = (
            "Neither extra_local_7ep nor random_guidance fully matches flex_full. "
            "Both extra training steps and cluster-aware structure contribute independently."
        )
        causal = (
            "Primary driver: Combination of extra epochs + auxiliary loss structure.\n"
            "Secondary driver: Adapter geometry conditioning.\n"
            "Rejected: Prototype semantic content."
        )

    lines += [
        f"### Verdict: **{verdict}**",
        "",
        explanation,
        "",
        "```",
        causal,
        "```",
        "",
        "---",
        "## Per-Seed Raw Results",
        "",
        "| Method | Seed 42 | Seed 43 | Seed 44 |",
        "|--------|---------|---------|---------|",
    ]

    for m in ALL_METHODS:
        per = {r["seed"]: r["mean_accuracy"]
               for r in results if r["method"] == m}
        s42 = f"{per.get(42, float('nan')):.4f}"
        s43 = f"{per.get(43, float('nan')):.4f}"
        s44 = f"{per.get(44, float('nan')):.4f}"
        lines.append(f"| {LABELS[m]} | {s42} | {s43} | {s44} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
def _print_summary(results: list[dict]) -> None:
    stats = _agg(results)
    flex_mean = stats.get("flex_full", {}).get("mean", 0.0)
    print(f"\n{'='*68}")
    print("  BLOCK K SUMMARY")
    print(f"  {'Method':<36} {'Done':>4}  {'Mean':>7}  {'Δ FLEX':>8}")
    print(f"  {'-'*58}")
    for m in ALL_METHODS:
        if m not in stats:
            print(f"  {m:<36} {'0/3':>4}  {'---':>7}  {'---':>8}")
            continue
        s = stats[m]
        d = s["mean"] - flex_mean
        print(f"  {m:<36} {str(s['n'])+'/3':>4}  {s['mean']:>7.4f}  {d:>+8.4f}")
    print(f"{'='*68}\n")


# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 68)
    print("  BLOCK K: EXTRA-EPOCH CONTROL")
    print("  Final causal closure: extra training vs guidance structure")
    print(f"  Methods: {len(ALL_METHODS)}   Seeds: {SEEDS}   Total: {len(ALL_METHODS)*len(SEEDS)}")
    print("=" * 68)

    results = _load()
    print(f"\n  Loaded {len(results)} existing results\n")

    total_start = time.time()
    completed = skipped = 0

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        for method in ALL_METHODS:
            if _done(results, method, seed):
                print(f"  SKIP {method} s{seed}")
                skipped += 1
                continue

            print(f"\n  RUN  {method} | seed={seed}")
            t0 = time.time()
            result = _run(method, seed)
            result["wall_time_s"] = round(time.time() - t0, 1)
            results.append(result)
            _save(results)
            completed += 1
            print(f"  => mean={result['mean_accuracy']:.4f}  "
                  f"worst={result['worst_accuracy']:.4f}  "
                  f"({result['wall_time_s']:.0f}s)")

    elapsed = time.time() - total_start
    print(f"\n  Done. {completed} new | {skipped} cached | {elapsed/60:.1f} min total")

    _print_summary(results)

    stats  = _agg(results)
    report = _report(results, stats)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"  JSON: {RESULTS_JSON}")
    print(f"  MD:   {REPORT_MD}\n")


if __name__ == "__main__":
    main()
