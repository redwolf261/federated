#!/usr/bin/env python3
"""Block M — Long-Horizon Convergence Validation (200 rounds, compute-equalized).

Methods (all compute-equalized to 7 effective epochs/round):
  flex_full      : 5 local + 2 cluster-aware, prototype mode (FLEX reference)
  fedavg_7ep     : 7 local epochs, FedAvg aggregation
  scaffold_7ep   : 7 local epochs, SCAFFOLD control variates
  moon_7ep       : 7 local epochs, MOON contrastive loss
  pure_local_7ep : 7 local epochs, no server interaction
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from block_m_methods import run_pure_local, run_scaffold, run_moon, _get_data
from phase2_q1_validation import set_seed

# ---------------------------------------------------------------------------
COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"
FIG_DIR      = COVERAGE_DIR / "block_M_figures"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON  = COVERAGE_DIR / "block_M_results.json"
REPORT_MD     = COVERAGE_DIR / "block_M.md"

DATASET         = "cifar10"
NUM_CLIENTS     = 10
ROUNDS          = 200
LOCAL_EPOCHS    = 7       # equalized budget for baselines
FLEX_LOCAL      = 5
FLEX_CLUSTER    = 2
BATCH_SIZE      = 64
LR              = 0.001
ALPHA           = 0.1
MAX_PER_CLIENT  = 2000
SEEDS           = [42, 43, 44]
ALL_METHODS     = ["flex_full", "fedavg_7ep", "scaffold_7ep", "moon_7ep", "pure_local_7ep"]


# ---------------------------------------------------------------------------
# FLEX and FedAvg via simulator
# ---------------------------------------------------------------------------

def _base_cfg(name: str, seed: int):
    from flex_persona.config.experiment_config import ExperimentConfig
    cfg = ExperimentConfig(
        experiment_name=name, dataset_name=DATASET,
        num_clients=NUM_CLIENTS, random_seed=seed,
        partition_mode="dirichlet", dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes = 10
    cfg.model.client_backbones = ["small_cnn"]
    cfg.model.shared_dim = 64
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.weight_decay = 1e-5
    cfg.training.max_samples_per_client = MAX_PER_CLIENT
    cfg.training.lambda_cluster = 0.1
    cfg.training.lambda_cluster_center = 0.01
    return cfg


def _run_sim(method: str, seed: int) -> list[dict]:
    """Run FLEX or FedAvg; extract per-round metrics from RoundState metadata."""
    import numpy as np
    from flex_persona.federated.simulator import FederatedSimulator
    set_seed(seed)
    cfg = _base_cfg(f"block_m_{method}_s{seed}", seed)
    if method == "flex_full":
        cfg.training.aggregation_mode     = "prototype"
        cfg.training.local_epochs         = FLEX_LOCAL
        cfg.training.cluster_aware_epochs = FLEX_CLUSTER
    else:
        cfg.training.aggregation_mode     = "fedavg"
        cfg.training.local_epochs         = LOCAL_EPOCHS
        cfg.training.cluster_aware_epochs = 0
    cfg.training.rounds = ROUNDS

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    round_states = sim.run_experiment()  # returns list[RoundState]

    history = []
    for i, rs in enumerate(round_states):
        ev = rs.metadata.get("evaluation", {})
        client_accs = ev.get("client_accuracies", {})
        accs = list(client_accs.values()) if client_accs else []
        entry = {
            "round":     i + 1,
            "mean":      float(ev.get("mean_client_accuracy",  0.0)),
            "worst":     float(ev.get("worst_client_accuracy", 0.0)),
            "std":       float(np.std(accs)) if accs else 0.0,
            "p10":       float(ev.get("p10_client_accuracy",   0.0)),
            "per_client": [float(v) for v in accs],
        }
        history.append(entry)
    return history


def _run_flex(seed: int) -> list[dict]:
    return _run_sim("flex_full", seed)


def _run_fedavg(seed: int) -> list[dict]:
    return _run_sim("fedavg_7ep", seed)


def _run_custom(method: str, seed: int) -> list[dict]:
    """Run SCAFFOLD, MOON, or PureLocal using custom loops with shared data pipeline."""
    set_seed(seed)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = _get_data(PROJECT_ROOT, seed, NUM_CLIENTS, ALPHA,
                        BATCH_SIZE, MAX_PER_CLIENT, DATASET)
    if method == "scaffold_7ep":
        return run_scaffold(loaders, ROUNDS, LOCAL_EPOCHS, LR, device, seed)
    elif method == "moon_7ep":
        return run_moon(loaders, ROUNDS, LOCAL_EPOCHS, LR, device, seed)
    elif method == "pure_local_7ep":
        return run_pure_local(loaders, ROUNDS, LOCAL_EPOCHS, LR, device, seed)
    else:
        raise ValueError(f"Unknown method: {method}")


def _dispatch(method: str, seed: int) -> list[dict]:
    if method == "flex_full":
        return _run_flex(seed)
    elif method == "fedavg_7ep":
        return _run_fedavg(seed)
    else:
        return _run_custom(method, seed)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    return json.loads(RESULTS_JSON.read_text())


def _save(r: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(r, indent=2))


def _done(r: list[dict], method: str, seed: int) -> bool:
    return any(x["method"] == method and x["seed"] == seed for x in r)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "flex_full":      "#e74c3c",
    "fedavg_7ep":     "#3498db",
    "scaffold_7ep":   "#2ecc71",
    "moon_7ep":       "#f39c12",
    "pure_local_7ep": "#9b59b6",
}

STYLE = {
    "flex_full":      "-",
    "fedavg_7ep":     "--",
    "scaffold_7ep":   "-.",
    "moon_7ep":       ":",
    "pure_local_7ep": "-",
}


def _get_curve(results: list[dict], method: str, metric: str) -> tuple[list, list]:
    runs = [r for r in results if r["method"] == method]
    if not runs:
        return [], []
    # Each run has "history" key
    all_seeds = [r["history"] for r in runs]
    # Find max common rounds
    min_len = min(len(h) for h in all_seeds)
    rounds = [h["round"] for h in all_seeds[0][:min_len]]
    means  = [float(np.mean([h[metric] for h in [s[i] for s in all_seeds]]))
              for i in range(min_len)]
    return rounds, means


def _plot(results: list[dict]) -> None:
    for metric, title, fname in [
        ("mean",  "Mean Client Accuracy vs Round",   "fig1_mean_accuracy.png"),
        ("worst", "Worst Client Accuracy vs Round",  "fig2_worst_accuracy.png"),
        ("std",   "Client Accuracy Std vs Round",    "fig3_client_variance.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in ALL_METHODS:
            rounds, curve = _get_curve(results, m, metric)
            if not rounds:
                continue
            ax.plot(rounds, curve, color=COLORS[m], linestyle=STYLE[m],
                    label=m, linewidth=1.8, alpha=0.9)
        ax.set_xlabel("Round"); ax.set_ylabel(metric.title())
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)

    # Plot 4 — FLEX vs Pure Local
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["flex_full", "pure_local_7ep"]:
        rounds, curve = _get_curve(results, m, "mean")
        if rounds:
            ax.plot(rounds, curve, color=COLORS[m], label=m, linewidth=2)
    ax.set_xlabel("Round"); ax.set_ylabel("Mean Accuracy")
    ax.set_title("FLEX vs Pure Local Training")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_flex_vs_purelocal.png", dpi=150)
    plt.close(fig)
    print(f"  Figures saved to {FIG_DIR}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _final_stats(results: list[dict]) -> dict[str, dict]:
    out = {}
    for m in ALL_METHODS:
        runs = [r for r in results if r["method"] == m]
        if not runs:
            continue
        final_means  = [r["history"][-1]["mean"]  for r in runs if r.get("history")]
        final_worsts = [r["history"][-1]["worst"] for r in runs if r.get("history")]
        final_p10s   = [r["history"][-1]["p10"]   for r in runs if r.get("history")]
        if not final_means:
            continue
        out[m] = {
            "mean":  float(np.mean(final_means)),
            "std":   float(np.std(final_means)),
            "worst": float(np.mean(final_worsts)),
            "p10":   float(np.mean(final_p10s)),
            "n":     len(final_means),
        }
    return out


def _write_report(results: list[dict]) -> None:
    stats = _final_stats(results)
    ref   = stats.get("flex_full",      {}).get("mean", float("nan"))
    pl    = stats.get("pure_local_7ep", {}).get("mean", float("nan"))
    fa    = stats.get("fedavg_7ep",     {}).get("mean", float("nan"))

    lines = [
        "# Block M — Long-Horizon Convergence Validation (200 Rounds)",
        "",
        "## 1. Objective",
        "Determine whether FLEX advantages persist under extended optimization",
        "with fully compute-equalized training budgets (7 effective epochs/round).",
        "",
        "**Central hypothesis:** FLEX gains arise primarily from avoiding destructive",
        "cross-client parameter averaging under non-IID data.",
        "",
        "## 2. Experimental Setup",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Dataset | CIFAR-10 |",
        f"| Clients | {NUM_CLIENTS} |",
        f"| Dirichlet α | {ALPHA} |",
        f"| Rounds | {ROUNDS} |",
        f"| Local epochs (baselines) | {LOCAL_EPOCHS} |",
        f"| FLEX epochs | {FLEX_LOCAL} local + {FLEX_CLUSTER} cluster-aware |",
        f"| Seeds | {SEEDS} |",
        "",
        "## 3. Methods",
        "- **flex_full**: 5+2 epochs, prototype mode (reference)",
        "- **fedavg_7ep**: 7 epochs, FedAvg weight averaging",
        "- **scaffold_7ep**: 7 epochs, SCAFFOLD control variates",
        "- **moon_7ep**: 7 epochs, MOON contrastive loss (μ=5.0)",
        "- **pure_local_7ep**: 7 epochs, no server interaction",
        "",
        "## 4. Final Results Table (Round 200)",
        "",
        "| Method | Mean ± Std | Worst | P10 | Δ vs FLEX | Δ vs PureLocal |",
        "|--------|-----------|-------|-----|-----------|----------------|",
    ]

    for m in ALL_METHODS:
        if m not in stats:
            lines.append(f"| {m} | — | — | — | — | — |")
            continue
        s = stats[m]
        lines.append(
            f"| {m} | {s['mean']:.4f} ± {s['std']:.4f} | {s['worst']:.4f} | "
            f"{s['p10']:.4f} | {s['mean']-ref:+.4f} | {s['mean']-pl:+.4f} |"
        )

    # Analysis sections
    flex_pl_gap = ref - pl
    pl_fa_gap   = pl  - fa

    lines += [
        "",
        "## 5. Convergence Analysis",
        "See figures in `block_M_figures/`.",
        "",
        "## 6. Pure Local Comparison",
        f"- FLEX vs PureLocal: {flex_pl_gap:+.4f}",
        f"- PureLocal vs FedAvg: {pl_fa_gap:+.4f}",
        "",
        _interpret_pure_local(flex_pl_gap),
        "",
        "## 7. Averaging Damage Analysis",
        _averaging_damage(stats),
        "",
        "## 8. Statistical Interpretation",
        _statistical_interpretation(stats, ref, pl, fa),
        "",
        "## 9. Final Causal Conclusion",
        _causal_conclusion(stats, ref, pl, fa),
        "",
        "## 10. Limitations",
        "- SCAFFOLD and MOON implemented from scratch; minor implementation",
        "  differences from original papers may exist.",
        "- FedAvg per-round curves use final-round evaluation only for",
        "  the simulator path; custom loops provide full per-round curves.",
        "- 200 rounds may still be insufficient for FedAvg to fully converge",
        "  under α=0.1 heterogeneity.",
    ]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {REPORT_MD}")


def _interpret_pure_local(gap: float) -> str:
    thr = 0.02
    if abs(gap) < thr:
        return ("**CASE A confirmed**: FLEX ≈ PureLocal. Prototype collaboration "
                "is causally negligible. Avoiding parameter averaging is the primary mechanism.")
    elif gap > thr:
        return (f"**CASE B**: FLEX > PureLocal by {gap:.4f}. A weak auxiliary "
                "collaborative effect survives at 200 rounds.")
    else:
        return (f"**Unexpected**: PureLocal > FLEX by {-gap:.4f}. "
                "Independent optimization outperforms federated collaboration.")


def _averaging_damage(stats: dict) -> str:
    fa = stats.get("fedavg_7ep",     {}).get("mean", float("nan"))
    pl = stats.get("pure_local_7ep", {}).get("mean", float("nan"))
    sc = stats.get("scaffold_7ep",   {}).get("mean", float("nan"))
    mn = stats.get("moon_7ep",       {}).get("mean", float("nan"))
    lines = [
        f"PureLocal (no averaging) at round 200:  {pl:.4f}",
        f"FedAvg (repeated averaging) at round 200: {fa:.4f}",
        f"SCAFFOLD (corrected averaging):          {sc:.4f}",
        f"MOON (contrastive + averaging):          {mn:.4f}",
        "",
        f"Averaging damage (PureLocal - FedAvg): {pl-fa:+.4f}",
        f"SCAFFOLD mitigation vs FedAvg: {sc-fa:+.4f}",
        f"MOON mitigation vs FedAvg: {mn-fa:+.4f}",
    ]
    return "\n".join(lines)


def _statistical_interpretation(stats, ref, pl, fa) -> str:
    sc = stats.get("scaffold_7ep", {}).get("mean", float("nan"))
    mn = stats.get("moon_7ep",     {}).get("mean", float("nan"))
    return "\n".join([
        f"Q1 (Does extended optimization eliminate FLEX advantage?): "
        f"FLEX={ref:.4f} PureLocal={pl:.4f} gap={ref-pl:+.4f}",
        f"Q2 (Do averaging methods asymptotically recover?): "
        f"FedAvg={fa:.4f} vs PureLocal={pl:.4f} gap={fa-pl:+.4f}",
        f"Q3 (Is avoiding averaging still dominant at 200 rounds?): "
        f"All non-averaging methods cluster at ~{pl:.3f} vs FedAvg={fa:.4f}",
        f"Q4 (Does PureLocal match FLEX?): gap={ref-pl:+.4f}",
    ])


def _causal_conclusion(stats, ref, pl, fa) -> str:
    gap = ref - pl
    thr = 0.02
    if abs(gap) < thr and pl - fa > 0.10:
        return (
            "**Primary mechanism confirmed**: Avoiding destructive cross-client "
            "parameter averaging under non-IID data (Dirichlet α=0.1) is the dominant "
            "causal driver of FLEX performance. Prototype collaboration contributes "
            "negligibly. This finding is robust across 200 rounds of optimization."
        )
    elif gap > thr:
        return (
            f"**Weak collaborative signal survives**: FLEX maintains a {gap:.4f} "
            "advantage over pure local training at 200 rounds, suggesting prototype "
            "exchange provides a marginal benefit beyond averaging avoidance."
        )
    else:
        return (
            "**Mixed result**: Requires further analysis. See per-round curves "
            "in block_M_figures/ for convergence trajectory interpretation."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total = len(ALL_METHODS) * len(SEEDS)
    print(f"\n{'='*70}")
    print("  BLOCK M: LONG-HORIZON CONVERGENCE VALIDATION")
    print(f"  Methods: {len(ALL_METHODS)}  Seeds: {SEEDS}  Rounds: {ROUNDS}")
    print(f"  Total runs: {total}")
    print(f"{'='*70}\n")

    all_results = _load()
    print(f"  Loaded {len(all_results)} existing results\n")

    total_start = time.time()
    completed = skipped = 0

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        for method in ALL_METHODS:
            if _done(all_results, method, seed):
                print(f"  SKIP {method} s{seed}")
                skipped += 1
                continue
            print(f"\n  RUN  {method} | seed={seed}")
            t0 = time.time()
            history = _dispatch(method, seed)
            entry = {
                "method": method, "seed": seed, "block": "M",
                "rounds_completed": len(history),
                "history": history,
                "wall_time_s": round(time.time() - t0, 1),
            }
            if history:
                entry["final_mean"]  = history[-1]["mean"]
                entry["final_worst"] = history[-1].get("worst", float("nan"))
            all_results.append(entry)
            _save(all_results)
            completed += 1
            print(f"  => final_mean={entry.get('final_mean', 'nan'):.4f}  "
                  f"rounds={len(history)}  ({entry['wall_time_s']:.0f}s)")

    elapsed = time.time() - total_start
    print(f"\n  Done. {completed} new | {skipped} cached | {elapsed/60:.1f} min total\n")

    # Final summary
    stats = _final_stats(all_results)
    ref = stats.get("flex_full", {}).get("mean", 0.0)
    print(f"  {'Method':<22} {'N':>3}  {'Mean':>7}  {'Δ FLEX':>8}")
    print(f"  {'-'*46}")
    for m in ALL_METHODS:
        if m not in stats:
            print(f"  {m:<22} {'0':>3}  {'---':>7}  {'---':>8}")
            continue
        s = stats[m]
        print(f"  {m:<22} {s['n']:>3}  {s['mean']:>7.4f}  {s['mean']-ref:>+8.4f}")

    _plot(all_results)
    _write_report(all_results)
    print(f"\n  JSON:   {RESULTS_JSON}")
    print(f"  Report: {REPORT_MD}")
    print(f"  Figs:   {FIG_DIR}\n")


if __name__ == "__main__":
    main()
