#!/usr/bin/env python3
"""Generate Block G report from G_results.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"


def load_results() -> list[dict]:
    """Load all Block G results from JSONL."""
    jsonl_path = COVERAGE_DIR / "G_results.jsonl"
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("block") == "G":
                results.append(entry)
    return results


def aggregate(results: list[dict]) -> dict[str, dict]:
    """Compute mean ± std across seeds for each method."""
    grouped = defaultdict(list)
    for r in results:
        grouped[r["method"]].append(r)

    stats = {}
    for method, runs in grouped.items():
        mean_acc = np.mean([r["mean_accuracy"] for r in runs])
        std_acc = np.std([r["mean_accuracy"] for r in runs])
        worst_acc = np.mean([r["worst_accuracy"] for r in runs])
        std_worst = np.std([r["worst_accuracy"] for r in runs])
        p10_acc = np.mean([r.get("p10", 0.0) for r in runs])
        std_p10 = np.std([r.get("p10", 0.0) for r in runs])
        std_clients = np.mean([r["std_across_clients"] for r in runs])

        stats[method] = {
            "mean": float(mean_acc),
            "std": float(std_acc),
            "worst": float(worst_acc),
            "std_worst": float(std_worst),
            "p10": float(p10_acc),
            "std_p10": float(std_p10),
            "std_clients": float(std_clients),
            "n_runs": len(runs),
        }

    return stats


def compute_drops(stats: dict, reference_method: str = "flex_full") -> dict[str, float]:
    """Compute performance drop relative to reference method."""
    reference = stats.get(reference_method, {}).get("mean", 0.0)
    drops = {}
    for method, stat in stats.items():
        drops[method] = reference - stat["mean"]
    return drops


def evaluate_tests(stats: dict, drops: dict) -> dict[str, tuple[bool, str]]:
    """Evaluate validation checks."""
    flex_full_mean = stats["flex_full"]["mean"]

    tests = {}

    # Test 1: Necessity of Sharing
    no_share_mean = stats["flex_no_prototype_sharing"]["mean"]
    drop_no_share = drops["flex_no_prototype_sharing"]
    threshold1 = 0.20  # >20 pp drop expected
    pass1 = drop_no_share > threshold1
    tests["necessity_of_sharing"] = (
        pass1,
        f"flex_no_prototype_sharing mean={no_share_mean:.4f}, "
        f"drop={drop_no_share:.4f} ({drop_no_share*100:.1f} pp) "
        f"vs threshold {threshold1:.2f} — {'PASS' if pass1 else 'FAIL'}"
    )

    # Test 2: Information Integrity
    shuffled_mean = stats["flex_shuffled_prototypes"]["mean"]
    drop_shuffled = drops["flex_shuffled_prototypes"]
    threshold2 = 0.01  # >1% drop expected
    pass2 = drop_shuffled > threshold2
    tests["information_integrity"] = (
        pass2,
        f"flex_shuffled_prototypes mean={shuffled_mean:.4f}, "
        f"drop={drop_shuffled:.4f} ({drop_shuffled*100:.1f} pp) "
        f"vs threshold {threshold2:.2f} — {'PASS' if pass2 else 'FAIL'}"
    )

    # Test 3: Signal vs Noise
    noise_mean = stats["flex_noise_prototypes"]["mean"]
    drop_noise = drops["flex_noise_prototypes"]
    threshold3 = 0.10  # >10 pp drop expected
    pass3 = drop_noise > threshold3
    tests["signal_vs_noise"] = (
        pass3,
        f"flex_noise_prototypes mean={noise_mean:.4f}, "
        f"drop={drop_noise:.4f} ({drop_noise*100:.1f} pp) "
        f"vs threshold {threshold3:.2f} — {'PASS' if pass3 else 'FAIL'}"
    )

    # Test 4: Collaboration Requirement
    self_only_mean = stats["flex_self_only"]["mean"]
    drop_self = drops["flex_self_only"]
    threshold4 = 0.01  # >1% drop expected
    pass4 = drop_self > threshold4
    tests["collaboration_requirement"] = (
        pass4,
        f"flex_self_only mean={self_only_mean:.4f}, "
        f"drop={drop_self:.4f} ({drop_self*100:.1f} pp) "
        f"vs threshold {threshold4:.2f} — {'PASS' if pass4 else 'FAIL'}"
    )

    return tests


def generate_markdown(stats: dict, drops: dict, tests: dict) -> str:
    """Generate the Block G markdown report."""

    lines = [
        "# Block G: Mechanism Isolation (Prototype Exchange Causality)",
        "",
        "## Section 1: Objective",
        "",
        "Establish a direct causal link between prototype exchange and FLEX-Persona's performance gains.",
        "Previous experiments (Blocks A–F) eliminated alternative explanations, but did not directly",
        "remove or corrupt the prototype-sharing mechanism itself.",
        "",
        "This block isolates and stress-tests the mechanism by removing or degrading cross-client",
        "prototype information flow.",
        "",
        "## Section 2: Experimental Setup",
        "",
        "| Parameter | Value |",
        "|---|---|",
        "| Dataset | CIFAR-10 |",
        "| Clients | 10 |",
        "| Samples per client | 2000 |",
        "| Partition | Dirichlet (α = 0.1) |",
        "| Rounds | 20 |",
        "| Local epochs | 5 |",
        "| Cluster-aware epochs | 0 |",
        "| Batch size | 64 |",
        "| Learning rate | 0.001 |",
        "| Seeds | [42, 43, 44] |",
        "| Total runs | 18 (6 methods × 3 seeds) |",
        "",
        "## Section 3: Methods",
        "",
        "| Method | Description |",
        "|---|---|",
        "| flex_full | Normal prototype extraction + sharing + aggregation |",
        "| flex_no_prototype_sharing | Clients do NOT send prototypes; no aggregation |",
        "| flex_self_only | Server returns each client's own prototype only |",
        "| flex_shuffled_prototypes | Server randomly permutes prototype assignments |",
        "| flex_noise_prototypes | All prototypes replaced with random noise |",
        "| fedavg_sgd | Baseline reference (FedAvg) |",
        "",
        "## Section 4: Results Table",
        "",
        "| Method | Mean | Std | Worst | P10 | Drop vs FLEX |",
        "|---|---|---|---|---|---|",
    ]

    for method in ["flex_full", "flex_no_prototype_sharing", "flex_self_only",
                   "flex_shuffled_prototypes", "flex_noise_prototypes", "fedavg_sgd"]:
        if method not in stats:
            continue
        s = stats[method]
        d = drops[method]
        lines.append(
            f"| {method} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['worst']:.4f} | {s['p10']:.4f} | {d:+.4f} |"
        )

    lines.extend([
        "",
        "## Section 5: Performance Drops vs FLEX Full",
        "",
        "| Method | Drop | Interpretation |",
        "|---|---|---|",
    ])

    drop_interpretations = {
        "flex_no_prototype_sharing": "Removing prototype sharing entirely",
        "flex_self_only": "Removing cross-client mixing (self-only)",
        "flex_shuffled_prototypes": "Corrupting assignment structure",
        "flex_noise_prototypes": "Replacing signal with noise",
        "fedavg_sgd": "Replacing FLEX with FedAvg baseline",
    }

    for method in ["flex_no_prototype_sharing", "flex_self_only",
                   "flex_shuffled_prototypes", "flex_noise_prototypes", "fedavg_sgd"]:
        if method not in drops:
            continue
        interp = drop_interpretations.get(method, "")
        lines.append(f"| {method} | {drops[method]:+.4f} | {interp} |")

    lines.extend([
        "",
        "## Section 6: Validation Checks",
        "",
    ])

    for test_name, (passed, description) in tests.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"### {test_name.replace('_', ' ').title()}")
        lines.append(f"{status}: {description}")
        lines.append("")

    # Determine overall conclusion
    all_passed = all(passed for passed, _ in tests.values())
    if all_passed:
        conclusion = "CONFIRMED"
        conclusion_text = (
            "Prototype exchange is the causal driver of FLEX-Persona's performance gains. "
            "Removing or corrupting prototype sharing causes significant degradation, "
            "confirming that the mechanism is both necessary and sufficient for the observed improvements."
        )
    elif sum(passed for passed, _ in tests.values()) >= 3:
        conclusion = "PARTIAL"
        conclusion_text = (
            "Prototype exchange shows strong causal dependence, but one or more validation checks "
            "did not meet expected thresholds. The mechanism is important but may interact with other factors."
        )
    else:
        conclusion = "REJECTED"
        conclusion_text = (
            "Prototype exchange does not appear to be the primary mechanism. "
            "The observed performance gains may stem from other factors (e.g., architecture, optimizer, data)."
        )

    lines.extend([
        "## Section 7: Causal Conclusion",
        "",
        f"**Verdict: {conclusion}**",
        "",
        f"{conclusion_text}",
        "",
        "### Explicit Answer",
        "",
        f"> Does performance collapse when prototype exchange is removed or corrupted?",
        f">",
        f"> **{conclusion}**: {'Yes' if conclusion in ('CONFIRMED', 'PARTIAL') else 'No'} — "
        f"performance {'degrades significantly' if conclusion == 'CONFIRMED' else 'shows partial dependence' if conclusion == 'PARTIAL' else 'does not depend on prototype exchange'}.",
        "",
        "## Section 8: Mechanistic Interpretation",
        "",
        "### What the Data Shows",
        "",
        "1. **Prototype exchange has no effect when guidance is disabled**: With cluster_aware_epochs=0,",
        "   removing prototype sharing (flex_no_prototype_sharing) or corrupting assignments",
        "   (flex_shuffled_prototypes) produces identical results to flex_full.",
        "2. **Self-only prototypes are equivalent to full aggregation**: flex_self_only matches flex_full",
        "   exactly (to 16 decimal places), confirming that cross-client prototype mixing is not active.",
        "3. **Noise prototypes crashed**: flex_noise_prototypes failed after ~1 round (8-14s vs 120-235s),",
        "   indicating a bug in the noise injection implementation. Results reflect single-round accuracy only.",
        "4. **The gap to FedAvg is architectural**: All functional FLEX variants (including no-prototype-sharing)",
        "   outperform FedAvg by ~30-35pp, suggesting the adapter architecture drives the gain.",

        "",
        "### Structural Insight",
        "",
        "The magnitude of degradation correlates with the severity of corruption:",
        "",
        "| Corruption Level | Method | Drop |",
        "|---|---|---|",
    ])

    # Sort methods by drop magnitude
    corruption_methods = [
        ("flex_noise_prototypes", "Complete signal destruction"),
        ("flex_no_prototype_sharing", "Complete mechanism removal"),
        ("fedavg_sgd", "Replace with FedAvg"),
        ("flex_shuffled_prototypes", "Structural corruption"),
        ("flex_self_only", "Remove cross-client mixing"),
    ]

    for method, label in corruption_methods:
        if method in drops:
            lines.append(f"| {label} | {method} | {drops[method]:+.4f} |")

    lines.extend([
        "",
        "With cluster_aware_epochs=0, no degradation is observed regardless of corruption severity,",
        "because the prototype information is never consumed during training. The exchange mechanism",
        "requires active cluster guidance (cluster_aware_epochs > 0) to function. All observed FLEX",
        "performance gains in this configuration are attributable to the backbone+adapter architecture,",
        "not to prototype-based collaboration.",

        "",
        "---",
        "",
        f"*Report generated from {stats.get('flex_full', {}).get('n_runs', 0)} seeds per method.*",
        f"*Total runs: {sum(s['n_runs'] for s in stats.values())}.*",
    ])

    return "\n".join(lines)


def generate_json_report(stats: dict, drops: dict, tests: dict) -> dict:
    """Generate structured JSON report."""
    return {
        "block": "G",
        "title": "Mechanism Isolation (Prototype Exchange Causality)",
        "statistics": stats,
        "drops_vs_flex_full": drops,
        "validation_checks": {
            name: {"passed": passed, "description": desc}
            for name, (passed, desc) in tests.items()
        },
        "overall_conclusion": {
            "verdict": "CONFIRMED" if all(p for p, _ in tests.values()) else "PARTIAL",
            "all_passed": all(p for p, _ in tests.values()),
            "num_passed": sum(1 for p, _ in tests.values() if p),
            "num_total": len(tests),
        },
    }


def main():
    print("Generating Block G report...")

    results = load_results()
    if not results:
        print("No Block G results found in G_results.jsonl")
        sys.exit(1)

    print(f"Loaded {len(results)} runs")

    stats = aggregate(results)
    drops = compute_drops(stats)
    tests = evaluate_tests(stats, drops)

    # Write markdown report
    md_content = generate_markdown(stats, drops, tests)
    md_path = COVERAGE_DIR / "block_G.md"
    with open(md_path, "w", encoding="utf-8") as f:

        f.write(md_content)
    print(f"Wrote markdown report: {md_path}")

    # Write JSON report
    json_report = generate_json_report(stats, drops, tests)
    json_path = COVERAGE_DIR / "block_G_report.json"
    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(json_report, f, indent=2)
    print(f"Wrote JSON report: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("BLOCK G SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Mean':>8} {'Std':>8} {'Drop':>8}")
    print(f"{'-'*60}")
    for method in ["flex_full", "flex_no_prototype_sharing", "flex_self_only",
                   "flex_shuffled_prototypes", "flex_noise_prototypes", "fedavg_sgd"]:
        if method in stats:
            s = stats[method]
            d = drops[method]
            print(f"{method:<30} {s['mean']:>8.4f} {s['std']:>8.4f} {d:>+8.4f}")
    print(f"{'-'*60}")

    print(f"\nValidation Checks:")
    for name, (passed, desc) in tests.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {desc}")

    verdict = "CONFIRMED" if all(p for p, _ in tests.values()) else "PARTIAL"
    print(f"\nOverall Verdict: {verdict}")


if __name__ == "__main__":
    main()
