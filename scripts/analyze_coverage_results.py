#!/usr/bin/env python3
"""Analyze and format failure mode coverage experiment results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"


def load_block_results(block: str) -> list[dict]:
    """Load all results for a given block from JSONL file."""
    jsonl_path = COVERAGE_DIR / f"{block}_results.jsonl"
    if not jsonl_path.exists():
        return []
    results = []
    with open(jsonl_path, encoding="utf-8") as f:

        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def format_block_a(results: list[dict]) -> str:
    """Format Block A (Optimizer Validity) results."""
    lines = ["[BLOCK A RESULTS]", "Optimizer Validity: Is SCAFFOLD failure due to Adam misuse?", ""]
    lines.append(f"{'Method':<12} {'Optimizer':<8} {'LR':>8} {'Mean Acc':>10} {'Worst':>8} {'Control/Grad':>12}")
    lines.append("-" * 65)

    for r in results:
        method = r.get("method", "N/A")
        opt = r.get("optimizer", "N/A")
        lr = r.get("lr", 0.0)
        mean_acc = r.get("mean_accuracy", 0.0)
        worst = r.get("worst_accuracy", 0.0)
        ratio = r.get("ratio_control_to_grad", "N/A")

        ratio_str = f"{ratio:.1f}" if isinstance(ratio, float) else "N/A"
        lines.append(f"{method:<12} {opt:<8} {lr:>8.4f} {mean_acc:>10.4f} {worst:>8.4f} {ratio_str:>12}")

    lines.append("")

    # Interpretation
    scaffold_results = [r for r in results if r.get("method") == "scaffold"]
    adam_scaffold = [r for r in scaffold_results if r.get("optimizer") == "adam"]
    sgd_scaffold = [r for r in scaffold_results if r.get("optimizer") == "sgd"]

    if adam_scaffold and sgd_scaffold:
        adam_acc = adam_scaffold[0].get("mean_accuracy", 0)
        best_sgd_acc = max(r.get("mean_accuracy", 0) for r in sgd_scaffold)
        if best_sgd_acc > adam_acc + 0.05:
            lines.append("INTERPRETATION: SCAFFOLD improves with SGD → regime-specific failure")
        else:
            lines.append("INTERPRETATION: SCAFFOLD fails across optimizers → fundamental sensitivity")

    lines.append("")
    return "\n".join(lines)


def format_block_b(results: list[dict]) -> str:
    """Format Block B (Compute Fairness) results."""
    lines = ["[BLOCK B RESULTS]", "Compute Fairness: Is FLEX benefiting from extra compute?", ""]

    lines.append(f"{'Variant':<20} {'LocalEp':>8} {'Cluster':>8} {'Mean Acc':>10} {'Worst':>8}")
    lines.append("-" * 60)

    for r in results:
        name = r.get("variant_name", r.get("method", "N/A"))
        le = r.get("local_epochs", 0)
        cae = r.get("cluster_aware_epochs", 0)
        mean_acc = r.get("mean_accuracy", 0.0)
        worst = r.get("worst_accuracy", 0.0)
        lines.append(f"{name:<20} {le:>8} {cae:>8} {mean_acc:>10.4f} {worst:>8.4f}")

    lines.append("")

    # Interpretation
    flex_full = [r for r in results if "full" in r.get("variant_name", "").lower()]
    flex_no_extra = [r for r in results if "no_extra" in r.get("variant_name", "").lower()]
    fedavg_7 = [r for r in results if "7epochs" in r.get("variant_name", "").lower()]

    if flex_full and flex_no_extra and fedavg_7:
        full_acc = flex_full[0].get("mean_accuracy", 0)
        no_extra_acc = flex_no_extra[0].get("mean_accuracy", 0)
        fed7_acc = fedavg_7[0].get("mean_accuracy", 0)

        lines.append(f"FLEX_full ({full_acc:.4f}) vs FLEX_no_extra ({no_extra_acc:.4f}):")
        if no_extra_acc > full_acc - 0.03:
            lines.append("  → Clustering adds minimal benefit (core is local training)")
        else:
            lines.append("  → Clustering provides significant benefit")

        lines.append(f"FLEX_no_extra ({no_extra_acc:.4f}) vs FedAvg_7epochs ({fed7_acc:.4f}):")
        if no_extra_acc > fed7_acc + 0.05:
            lines.append("  → FLEX outperforms compute-matched FedAvg (not just compute advantage)")
        elif abs(no_extra_acc - fed7_acc) < 0.03:
            lines.append("  → Equivalent to compute-matched FedAvg (compute advantage is main factor)")
        else:
            lines.append("  → Worse than compute-matched FedAvg")

    lines.append("")
    return "\n".join(lines)


def format_block_c(results: list[dict]) -> str:
    """Format Block C (Data Regime) results."""
    lines = ["[BLOCK C RESULTS]", "Data Regime: Does FLEX advantage disappear with more data?", ""]

    lines.append(f"{'Samples/Client':<15} {'Method':<10} {'Mean Acc':>10} {'Worst':>8}")
    lines.append("-" * 50)

    for r in results:
        samples = r.get("samples_per_client", 0)
        method = r.get("method", "N/A")
        mean_acc = r.get("mean_accuracy", 0.0)
        worst = r.get("worst_accuracy", 0.0)
        lines.append(f"{samples:<15} {method:<10} {mean_acc:>10.4f} {worst:>8.4f}")

    lines.append("")

    # Check if FedAvg catches up
    by_samples = {}
    for r in results:
        s = r.get("samples_per_client", 0)
        if s not in by_samples:
            by_samples[s] = {}
        by_samples[s][r.get("method")] = r.get("mean_accuracy", 0)

    if by_samples:
        lines.append("Trend analysis:")
        for samples in sorted(by_samples.keys()):
            methods = by_samples[samples]
            flex_acc = next((v for k, v in methods.items() if "flex" in k), 0)
            fedavg_acc = next((v for k, v in methods.items() if "fedavg" in k), 0)
            gap = flex_acc - fedavg_acc
            lines.append(f"  {samples} samples/client: FLEX-FedAvg gap = {gap:+.4f}")


    lines.append("")
    return "\n".join(lines)


def format_block_d(results: list[dict]) -> str:
    """Format Block D (Heterogeneity Sweep) results."""
    lines = ["[BLOCK D RESULTS]", "Heterogeneity Sweep: Behavior across regimes", ""]

    lines.append(f"{'Alpha':>8} {'Method':<10} {'Mean Acc':>10} {'Worst':>8}")
    lines.append("-" * 40)

    for r in results:
        alpha = r.get("alpha", 0.0)
        method = r.get("method", "N/A")
        mean_acc = r.get("mean_accuracy", 0.0)
        worst = r.get("worst_accuracy", 0.0)
        lines.append(f"{alpha:>8.2f} {method:<10} {mean_acc:>10.4f} {worst:>8.4f}")

    lines.append("")

    # Curve analysis
    by_alpha = {}
    for r in results:
        a = r.get("alpha", 0)
        if a not in by_alpha:
            by_alpha[a] = {}
        by_alpha[a][r.get("method")] = r.get("mean_accuracy", 0)

    if by_alpha:
        lines.append("Cross-method comparison at each alpha:")
        for alpha in sorted(by_alpha.keys()):
            methods = by_alpha[alpha]
            flex_acc = next((v for k, v in methods.items() if "flex" in k), 0)
            fedavg_acc = next((v for k, v in methods.items() if "fedavg" in k), 0)
            lines.append(f"  α={alpha:.2f}: FLEX={flex_acc:.4f}, FedAvg={fedavg_acc:.4f}, gap={flex_acc-fedavg_acc:+.4f}")


    lines.append("")
    return "\n".join(lines)


def format_block_e(results: list[dict]) -> str:
    """Format Block E (SCAFFOLD Failure) results."""
    if not results:
        return "[BLOCK E RESULTS]\nNo data available\n"

    r = results[0]
    round_logs = r.get("per_round_diagnostics", [])

    lines = ["[BLOCK E RESULTS]", "SCAFFOLD Internal Failure Analysis", ""]

    lines.append(f"{'Round':>6} {'Acc':>7} {'GradNorm':>10} {'ControlNorm':>12} {'Ratio':>8}")
    lines.append("-" * 50)

    for rl in round_logs:
        ratio = rl.get("control_norm", 0) / max(rl.get("grad_norm", 1e-12), 1e-12)
        lines.append(f"{rl.get('round', 0):>6} {rl.get('accuracy', 0):>7.4f} "
                    f"{rl.get('grad_norm', 0):>10.4f} {rl.get('control_norm', 0):>12.4f} {ratio:>8.1f}")

    lines.append("")

    if round_logs:
        first_ratio = round_logs[0].get("control_norm", 0) / max(round_logs[0].get("grad_norm", 1e-12), 1e-12)
        last_ratio = round_logs[-1].get("control_norm", 0) / max(round_logs[-1].get("grad_norm", 1e-12), 1e-12)
        lines.append(f"First round ratio: {first_ratio:.1f}")
        lines.append(f"Last round ratio: {last_ratio:.1f}")

        if last_ratio > 10:
            lines.append("CONFIRMED: Control term dominates throughout → learning collapse")
        elif last_ratio > 2:
            lines.append("Control term is elevated but not catastrophic")
        else:
            lines.append("Control term is well-behaved")

    lines.append("")
    return "\n".join(lines)


def format_block_f(results: list[dict]) -> str:
    """Format Block F (FLEX Ablation) results."""
    lines = ["[BLOCK F RESULTS]", "FLEX Ablation: Which components matter?", ""]

    lines.append(f"{'Variant':<25} {'Cluster':>8} {'Guidance':>8} {'Mean Acc':>10} {'Worst':>8}")
    lines.append("-" * 65)

    for r in results:
        name = r.get("variant_name", r.get("method", "N/A"))
        clustering = r.get("use_clustering", True)
        guidance = r.get("use_guidance", True)
        mean_acc = r.get("mean_accuracy", 0.0)
        worst = r.get("worst_accuracy", 0.0)
        lines.append(f"{name:<25} {str(clustering):>8} {str(guidance):>8} {mean_acc:>10.4f} {worst:>8.4f}")

    lines.append("")

    # Component importance
    full = [r for r in results if "full" in r.get("variant_name", "").lower()]
    no_cluster = [r for r in results if "no_clustering" in r.get("variant_name", "").lower()]
    no_guide = [r for r in results if "no_guidance" in r.get("variant_name", "").lower()]
    no_proto = [r for r in results if "no_prototypes" in r.get("variant_name", "").lower()]

    if full and no_cluster:
        drop = full[0].get("mean_accuracy", 0) - no_cluster[0].get("mean_accuracy", 0)
        lines.append(f"Removing clustering: performance drop = {drop:.4f}")
        if drop > 0.05:
            lines.append("  → Clustering is critical")
        elif drop > 0.02:
            lines.append("  → Clustering provides moderate benefit")
        else:
            lines.append("  → Clustering has minimal impact")

    if full and no_guide:
        drop = full[0].get("mean_accuracy", 0) - no_guide[0].get("mean_accuracy", 0)
        lines.append(f"Removing guidance: performance drop = {drop:.4f}")

    if full and no_proto:
        drop = full[0].get("mean_accuracy", 0) - no_proto[0].get("mean_accuracy", 0)
        lines.append(f"Removing prototypes (FedAvg-like): performance drop = {drop:.4f}")
        if drop > 0.05:
            lines.append("  → Prototype-based collaboration is essential")
        else:
            lines.append("  → Prototypes provide modest benefit")

    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("FAILURE MODE COVERAGE RESULTS")
    print("=" * 70)
    print()

    blocks = {
        "A": format_block_a,
        "B": format_block_b,
        "C": format_block_c,
        "D": format_block_d,
        "E": format_block_e,
        "F": format_block_f,
    }

    for block, formatter in blocks.items():
        results = load_block_results(block)
        if results:
            print(formatter(results))
        else:
            print(f"[BLOCK {block} RESULTS]")
            print("No data available yet")
            print()

    # Save formatted report
    report_path = COVERAGE_DIR / "formatted_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:

        for block, formatter in blocks.items():
            results = load_block_results(block)
            if results:
                f.write(formatter(results) + "\n")
            else:
                f.write(f"[BLOCK {block} RESULTS]\nNo data available yet\n\n")

    print(f"\nFormatted report saved to: {report_path}")


if __name__ == "__main__":
    main()
