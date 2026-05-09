"""Quick live summary of Block I results from JSON."""
import json, sys
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("numpy not available"); sys.exit(1)

f = Path("outputs/failure_mode_coverage/block_I_results.json")
if not f.exists():
    print("No results yet"); sys.exit(0)

data = json.loads(f.read_text())
print(f"Completed: {len(data)}/21 runs\n")

groups = defaultdict(list)
for r in data:
    groups[r["method"]].append(r)

ORDER = [
    "flex_full",
    "class_centroid_alignment",
    "global_centroid_alignment",
    "random_centroid_alignment",
    "feature_norm_only",
    "variance_minimization",
    "fedavg_sgd",
]

flex_runs = groups.get("flex_full", [])
flex_ref = float(np.mean([r["mean_accuracy"] for r in flex_runs])) if flex_runs else None

header = f"  {'Method':<32} {'Done':>5}  {'Mean':>7}  {'Drop':>7}  {'Seeds'}"
print(header)
print("  " + "-" * 68)

for m in ORDER:
    runs = groups.get(m, [])
    done = f"{len(runs)}/3"
    if not runs:
        print(f"  {m:<32} {done:>5}  {'---':>7}  {'---':>7}  []")
        continue
    means = [r["mean_accuracy"] for r in runs]
    seeds = sorted(r["seed"] for r in runs)
    avg = float(np.mean(means))
    drop_str = f"{avg - flex_ref:+.4f}" if flex_ref is not None else "---"
    print(f"  {m:<32} {done:>5}  {avg:>7.4f}  {drop_str:>7}  {seeds}")

print()
if flex_ref:
    print(f"  flex_full reference mean: {flex_ref:.4f}")
    remaining = 21 - len(data)
    print(f"  Runs remaining: {remaining}")
