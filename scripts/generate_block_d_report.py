#!/usr/bin/env python3
"""Generate Block D report from D_results.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"
D_RESULTS_PATH = COVERAGE_DIR / "D_results.jsonl"
REPORT_JSON_PATH = COVERAGE_DIR / "block_D_report.json"
REPORT_MD_PATH = COVERAGE_DIR / "BLOCK_D_HETEROGENEITY_ANALYSIS.md"


def load_results() -> list[dict]:
    results = []
    if not D_RESULTS_PATH.exists():
        return results
    with open(D_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("block") == "D":
                    results.append(r)
            except Exception:
                continue
    return results


def aggregate_by_alpha(results: list[dict]) -> dict:
    """Aggregate results per alpha across seeds."""
    # Group by (alpha, method)
    grouped = defaultdict(list)
    for r in results:
        key = (float(r["alpha"]), str(r["method"]))
        grouped[key].append(r)

    aggregated = {}
    for (alpha, method), runs in grouped.items():
        mean_accs = [run["mean_accuracy"] for run in runs]
        worst_accs = [run["worst_accuracy"] for run in runs]
        p10s = [run.get("p10", 0.0) for run in runs]
        aggregated[(alpha, method)] = {
            "mean_accuracy": float(np.mean(mean_accs)),
            "std_across_seeds": float(np.std(mean_accs)),
            "worst_accuracy": float(np.mean(worst_accs)),
            "p10": float(np.mean(p10s)),
            "n_seeds": len(runs),
        }
    return aggregated


def compute_gains(aggregated: dict) -> dict:
    """Compute FLEX vs FedAvg gains per alpha."""
    alphas = sorted({k[0] for k in aggregated.keys()})
    gains = {}
    for alpha in alphas:
        flex = aggregated.get((alpha, "flex_no_extra"))
        fedavg = aggregated.get((alpha, "fedavg_sgd"))
        if flex and fedavg:
            abs_gain = flex["mean_accuracy"] - fedavg["mean_accuracy"]
            rel_gain = (abs_gain / fedavg["mean_accuracy"]) * 100.0 if fedavg["mean_accuracy"] > 0 else 0.0
            worst_gain = flex["worst_accuracy"] - fedavg["worst_accuracy"]
            gains[alpha] = {
                "flex_mean": flex["mean_accuracy"],
                "fedavg_mean": fedavg["mean_accuracy"],
                "absolute_gain": abs_gain,
                "relative_gain": rel_gain,
                "worst_gain": worst_gain,
                "flex_std": flex["std_across_seeds"],
                "fedavg_std": fedavg["std_across_seeds"],
            }
    return gains


def detect_pattern(gains: dict) -> str:
    """Classify the pattern of gains vs alpha."""
    alphas = sorted(gains.keys())
    if len(alphas) < 2:
        return "insufficient_data"

    gains_list = [gains[a]["absolute_gain"] for a in alphas]
    # Check if monotonically decreasing
    decreasing = all(gains_list[i] >= gains_list[i+1] for i in range(len(gains_list)-1))
    # Check if monotonically increasing
    increasing = all(gains_list[i] <= gains_list[i+1] for i in range(len(gains_list)-1))
    # Check if roughly constant (all within 0.05)
    constant = max(gains_list) - min(gains_list) < 0.05

    if decreasing:
        return "heterogeneity_dependent"
    elif constant:
        return "heterogeneity_invariant"
    elif increasing:
        return "unexpected_pattern"
    else:
        return "mixed_pattern"


def generate_json_report(aggregated: dict, gains: dict, pattern: str) -> None:
    report = {
        "block": "D",
        "description": "Heterogeneity Sweep",
        "alpha_values": sorted({k[0] for k in aggregated.keys()}),
        "per_alpha": {},
        "gains": {},
        "pattern": pattern,
        "interpretation": {
            "heterogeneity_dependent": "FLEX advantage decreases as heterogeneity decreases",
            "heterogeneity_invariant": "FLEX advantage is constant regardless of heterogeneity",
            "unexpected_pattern": "FLEX advantage increases as heterogeneity decreases",
            "mixed_pattern": "No clear monotonic relationship",
            "insufficient_data": "Not enough data points to classify",
        }.get(pattern, "unknown"),
    }

    for alpha in sorted({k[0] for k in aggregated.keys()}):
        alpha_str = str(alpha)
        report["per_alpha"][alpha_str] = {}
        for method in ["flex_no_extra", "fedavg_sgd"]:
            key = (alpha, method)
            if key in aggregated:
                report["per_alpha"][alpha_str][method] = aggregated[key]

    for alpha in sorted(gains.keys()):
        report["gains"][str(alpha)] = gains[alpha]

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved to: {REPORT_JSON_PATH}")


def generate_md_report(aggregated: dict, gains: dict, pattern: str, raw_results: list[dict]) -> None:
    lines = []
    lines.append("# Block D: Heterogeneity Sweep Analysis")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Determine whether FLEX-Persona's performance advantage is caused by its ability to handle non-IID data (heterogeneity).")
    lines.append("")
    lines.append("## Experimental Design")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append("| Dataset | CIFAR-10 |")
    lines.append("| Clients | 10 |")
    lines.append("| Seeds | [42, 43, 44] |")
    lines.append("| Rounds | 20 |")
    lines.append("| Local epochs | 5 |")
    lines.append("| Cluster-aware epochs | 0 |")
    lines.append("| Batch size | 64 |")
    lines.append("| Learning rate | 0.001 |")
    lines.append("| Samples per client | 2000 |")
    lines.append("| Alpha values | [0.05, 0.1, 0.5, 1.0, 10.0] |")
    lines.append("| Methods | flex_no_extra, fedavg_sgd |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Aggregate Performance by Alpha")
    lines.append("")
    lines.append("| Alpha | FLEX Mean | FLEX Std | FedAvg Mean | FedAvg Std |")
    lines.append("|---|---|---|---|---|")

    for alpha in sorted({k[0] for k in aggregated.keys()}):
        flex = aggregated.get((alpha, "flex_no_extra"), {})
        fedavg = aggregated.get((alpha, "fedavg_sgd"), {})
        lines.append(f"| {alpha} | {flex.get('mean_accuracy', 0):.4f} | {flex.get('std_across_seeds', 0):.4f} | "
                     f"{fedavg.get('mean_accuracy', 0):.4f} | {fedavg.get('std_across_seeds', 0):.4f} |")

    lines.append("")
    lines.append("### Gain Analysis")
    lines.append("")
    lines.append("| Alpha | FLEX Mean | FedAvg Mean | Abs Gain | Rel Gain | Worst Gain |")
    lines.append("|---|---|---|---|---|---|")

    for alpha in sorted(gains.keys()):
        g = gains[alpha]
        lines.append(f"| {alpha} | {g['flex_mean']:.4f} | {g['fedavg_mean']:.4f} | "
                     f"{g['absolute_gain']:+.4f} | {g['relative_gain']:+.1f}% | {g['worst_gain']:+.4f} |")

    lines.append("")
    lines.append("### Pattern Classification")
    lines.append("")
    lines.append(f"**Pattern detected:** `{pattern}`")
    lines.append("")
    interpretation = {
        "heterogeneity_dependent": "FLEX advantage decreases as heterogeneity decreases. This confirms that FLEX's benefit is specifically tied to its ability to handle non-IID data.",
        "heterogeneity_invariant": "FLEX advantage is constant regardless of heterogeneity. This suggests FLEX provides a universal benefit independent of data distribution.",
        "unexpected_pattern": "FLEX advantage increases as heterogeneity decreases. This contradicts expectations and requires further investigation.",
        "mixed_pattern": "No clear monotonic relationship between heterogeneity and FLEX advantage.",
        "insufficient_data": "Not enough data points to classify the pattern.",
    }.get(pattern, "Unknown pattern.")
    lines.append(f"**Interpretation:** {interpretation}")
    lines.append("")

    # Critical validation checks
    lines.append("### Critical Validation Checks")
    lines.append("")
    check_10 = gains.get(10.0)
    if check_10:
        fedavg_10 = check_10["fedavg_mean"]
        flex_10 = check_10["flex_mean"]
        gap_10 = check_10["absolute_gain"]
        lines.append(f"1. **Alpha = 10 (near-IID):** FedAvg = {fedavg_10:.4f}, FLEX = {flex_10:.4f}, gap = {gap_10:+.4f}")
        if fedavg_10 > 0.7:
            lines.append("   ✅ FedAvg improved significantly in near-IID regime.")
        else:
            lines.append("   ⚠️ FedAvg did not improve as expected in near-IID regime.")
        if gap_10 < 0.1:
            lines.append("   ✅ FLEX-FedAvg gap shrank to near-zero in near-IID regime.")
        else:
            lines.append("   ⚠️ FLEX-FedAvg gap remained large in near-IID regime.")
    else:
        lines.append("1. **Alpha = 10:** No data available.")

    # Check gap shrinkage
    alpha_low = min(gains.keys())
    alpha_high = max(gains.keys())
    gap_low = gains[alpha_low]["absolute_gain"]
    gap_high = gains[alpha_high]["absolute_gain"]
    lines.append(f"2. **Gap shrinkage:** Low-alpha gap = {gap_low:+.4f}, High-alpha gap = {gap_high:+.4f}")
    if gap_high < gap_low:
        lines.append("   ✅ Gap decreased as alpha increased.")
    else:
        lines.append("   ⚠️ Gap did not decrease as alpha increased.")

    # Worst-case check
    worst_low = gains[alpha_low]["worst_gain"]
    lines.append(f"3. **Worst-case improvement at low alpha:** {worst_low:+.4f}")
    if worst_low > 0:
        lines.append("   ✅ FLEX consistently outperforms in high-heterogeneity regime.")
    else:
        lines.append("   ⚠️ FLEX does not improve worst-case at low alpha.")

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if pattern == "heterogeneity_dependent":
        lines.append("The results confirm that FLEX-Persona's advantage is **heterogeneity-dependent**. "
                     "As data becomes more IID (higher alpha), the performance gap between FLEX and FedAvg shrinks. "
                     "This validates the core hypothesis: FLEX's representation-based collaboration mechanism "
                     "is specifically beneficial for non-IID data distributions.")
    elif pattern == "heterogeneity_invariant":
        lines.append("The results show that FLEX-Persona's advantage is **heterogeneity-invariant**. "
                     "The performance gap remains constant across all heterogeneity levels, suggesting "
                     "FLEX provides universal benefits beyond just handling non-IID data.")
    elif pattern == "unexpected_pattern":
        lines.append("The results show an **unexpected pattern**: FLEX advantage increases as heterogeneity decreases. "
                     "This contradicts the expected behavior and requires further investigation.")
    else:
        lines.append("The results show a **mixed pattern** with no clear monotonic relationship "
                     "between heterogeneity and FLEX advantage.")

    lines.append("")
    lines.append("## Raw Results")
    lines.append("")
    lines.append("| Alpha | Seed | Method | Mean | Worst | Std | P10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(raw_results, key=lambda x: (x["alpha"], x["seed"], x["method"])):
        lines.append(f"| {r['alpha']} | {r['seed']} | {r['method']} | "
                     f"{r['mean_accuracy']:.4f} | {r['worst_accuracy']:.4f} | "
                     f"{r.get('std', 0):.4f} | {r.get('p10', 0):.4f} |")

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown report saved to: {REPORT_MD_PATH}")


def main():
    print("=" * 60)
    print("BLOCK D REPORT GENERATION")
    print("=" * 60)

    results = load_results()
    print(f"Loaded {len(results)} Block D results")

    if not results:
        print("No results found. Exiting.")
        return

    aggregated = aggregate_by_alpha(results)
    gains = compute_gains(aggregated)
    pattern = detect_pattern(gains)

    print(f"\nPattern detected: {pattern}")
    print(f"\nGains by alpha:")
    for alpha in sorted(gains.keys()):
        g = gains[alpha]
        print(f"  alpha={alpha}: abs={g['absolute_gain']:+.4f}, rel={g['relative_gain']:+.1f}%, worst={g['worst_gain']:+.4f}")

    generate_json_report(aggregated, gains, pattern)
    generate_md_report(aggregated, gains, pattern, results)

    print("\n" + "=" * 60)
    print("BLOCK D REPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
