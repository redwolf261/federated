#!/usr/bin/env python3
"""Generate Block I Signal Nature Analysis Report.

Reads outputs/failure_mode_coverage/block_I_results.json
and generates an analysis report in markdown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Input / Output paths
RESULTS_JSON = PROJECT_ROOT / "outputs" / "failure_mode_coverage" / "block_I_results.json"
REPORT_MD = PROJECT_ROOT / "outputs" / "failure_mode_coverage" / "BLOCK_I_SIGNAL_NATURE_ANALYSIS.md"

METHOD_LABELS = {
    "flex_full": "Flex Full (Reference)",
    "class_centroid_alignment": "Class Centroid Alignment",
    "global_centroid_alignment": "Global Centroid Alignment",
    "random_centroid_alignment": "Random Centroid Alignment",
    "feature_norm_only": "Feature Norm Only",
    "variance_minimization": "Variance Minimization",
    "fedavg_sgd": "FedAvg SGD (Baseline)",
}


def load_results() -> list[dict]:
    """Load experiment results from JSON."""
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"Results not found: {RESULTS_JSON}\n"
            "Run scripts/run_block_i.py first."
        )
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(results: list[dict]) -> dict[str, dict]:
    """Aggregate results by method."""
    from collections import defaultdict

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
            "std": float(np.std(means)),
            "worst": float(np.mean(worsts)),
            "p10": float(np.mean(p10s)),
            "min": float(np.min(means)),
            "max": float(np.max(means)),
        }
    return stats


def generate_report(results: list[dict], stats: dict[str, dict]) -> str:
    """Generate markdown report."""
    # Get reference values
    flex_full = stats.get("flex_full", {})
    fedavg_sgd = stats.get("fedavg_sgd", {})
    flex_mean = flex_full.get("mean", 0.0)

    lines = [
        "# Block I: Signal Nature Analysis Report",
        "",
        f"**Results file**: `{RESULTS_JSON}`",
        f"**Total runs**: {len(results)}",
        "",
        "## Purpose",
        "",
        "Block I investigates what is the TRUE PROPERTY of the server-provided training signal.",
        "If FLEX performs well because the prototypes provide useful information, then",
        "variants that ablate the information should perform worse.",
        "",
        "If FLEX performs well due to some other property (geometry, spatial properties, etc.)",
        "then the ablated variants should perform similarly well.",
        "",
        "## Methods",
        "",
        "| # | Method | Description |",
        "|---|--------|-------------|",
        "| 1 | flex_full | Standard cluster prototype alignment (reference) |",
        "| 2 | class_centroid_alignment | Align to per-class centroids computed from client data |",
        "| 3 | global_centroid_alignment | Align to single global centroid from client data |",
        "| 4 | random_centroid_alignment | Align to random fixed centroids per class |",
        "| 5 | feature_norm_only | L2 normalize features in shared space |",
        "| 6 | variance_minimization | Minimize intra-batch feature variance |",
        "| 7 | fedavg_sgd | Standard FedAvg SGD baseline |",
        "",
        "---",
        "",
        "## Results",
        "",
    ]

    # Table of results
    lines.append("### Summary Table (Aggregated Across Seeds)")
    lines.append("")
    lines.append(
        "| Rank | Method | Mean Acc ± Std | Worst | P10 | Drop vs Full | Verdict |"
    )
    lines.append(
        "|------|--------|---------------|-------|-----|--------------|--------|"
    )

    # Sort by mean accuracy descending
    sorted_methods = sorted(
        stats.items(),
        key=lambda item: item[1].get("mean", 0.0),
        reverse=True,
    )

    for rank, (method, stat) in enumerate(sorted_methods, start=1):
        label = METHOD_LABELS.get(method, method)
        mean = stat["mean"]
        std = stat["std"]
        worst = stat["worst"]
        p10 = stat["p10"]
        drop = flex_mean - mean
        n = stat["n"]

        if flex_mean > 0:
            drop_pct = (drop / flex_mean) * 100
            drop_str = f"{drop:+.4f} ({drop_pct:+.1f}%)"
        else:
            drop_str = f"{drop:+.4f}"

        # Verdict
        if method == "flex_full":
            verdict = "Reference"
        elif method == "fedavg_sgd":
            verdict = "Baseline"
        elif drop_pct > 10:
            verdict = "**FAIL** — loss matters"
        elif drop < 0.05:
            verdict = "PASS — property preserved"
        else:
            verdict = "? Partial loss"

        lines.append(
            f"| {rank} | {label} | {mean:.4f} ± {std:.4f} ({n}) | {worst:.4f} | {p10:.4f} | {drop_str} | {verdict} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Drop comparison table
    lines.append("### Drop vs Reference")
    lines.append("")
    lines.append(
        "| Method | Metric | Reference | Variant | Drop | Property Retained? |"
    )
    lines.append(
        "|--------|--------|-----------|---------|------|---------------------|"
    )

    for method, stat in sorted_methods:
        if method == "flex_full" or method == "fedavg_sgd":
            continue

        label = METHOD_LABELS.get(method, method)
        variant_mean = stat["mean"]
        drop = flex_mean - variant_mean
        if flex_mean > 0:
            drop_pct = (drop / flex_mean) * 100
            retained = "YES" if drop_pct < 10 else "NO"
        else:
            drop_pct = 0.0
            retained = "?"

        lines.append(
            f"| {label} | Mean Acc | {flex_mean:.4f} | {variant_mean:.4f} | {drop:+.4f} ({drop_pct:+.1f}%) | {retained} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Collect drops
    drops = {}
    for method in [
        "class_centroid_alignment",
        "global_centroid_alignment",
        "random_centroid_alignment",
        "feature_norm_only",
        "variance_minimization",
    ]:
        if method in stats:
            drop = flex_mean - stats[method]["mean"]
            drops[method] = drop

    if drops:
        max_drop_method = max(drops, key=drops.get)
        max_drop = drops[max_drop_method]
        max_drop_label = METHOD_LABELS.get(max_drop_method, max_drop_method)
        lines.append(
            f"**Largest drop**: {max_drop_label} with {max_drop:+.4f}"
        )
        lines.append("")

    if all(drop < 0.05 for drop in drops.values()):
        lines.append(
            "**Verdict: ALL variants pass** — the property is NOT about "
            "information content. It is about some geometric/procedural property."
        )
    elif any(drop > 0.1 for drop in drops.values()):
        lines.append(
            "**Verdict: At least one variant fails** — the property IS about "
            "information content (accurate, meaningful prototypes from similar clients)."
        )
    else:
        lines.append(
            "**Verdict: MIXED** — some variants retain the property while others lose it."
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    if all(drop < 0.05 for drop in drops.values()):
        lines.append(
            "The signal does not provide INFORMATION. It provides some other property."
            " The ~30pp gap to FedAvg is driven by the architecture, not by meaningful "
            "cross-client information sharing."
        )
    else:
        lines.append(
            "The signal provides meaningful cross-client INFORMATION. "
            "Removing or corrupting the signal decreases performance significantly."
        )

    lines.append("")
    lines.append(
        f"**Reference**: {flex_full.get('mean', 0.0):.4f} ± {flex_full.get('std', 0.0):.4f}"
    )
    lines.append(
        f"**Baseline**: {fedavg_sgd.get('mean', 0.0):.4f} ± {fedavg_sgd.get('std', 0.0):.4f}"
    )
    lines.append("")
    lines.append("-" * 50)
    lines.append("")

    # Raw data
    lines.append("## Raw Results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        method: {k: round(v, 4) if isinstance(v, float) else v
                for k, v in stat.items()}
        for method, stat in stats.items()
    }, indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("Generating Block I Signal Nature Analysis Report...")

    results = load_results()
    print(f"  Loaded {len(results)} raw results")

    stats = aggregate(results)
    print(f"  Aggregated {len(stats)} methods")

    report = generate_report(results, stats)

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  Report written to: {REPORT_MD}")
    print("Done!")


if __name__ == "__main__":
    main()
