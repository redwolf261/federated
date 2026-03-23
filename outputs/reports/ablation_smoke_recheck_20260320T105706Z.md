# Multi-Seed Ablation Summary

Baseline for significance: fedavg_baseline

| Variant | Mode | Mean Acc (mean±CI95) | Worst Acc (mean±CI95) | Comm Bytes (mean±CI95) | p-value vs baseline (mean acc) |
|---|---|---:|---:|---:|---:|
| fedavg_baseline | fedavg | 0.0833 ± 0.0000 | 0.0000 ± 0.0000 | 7902891 ± 0 | - |
| prototype_lambda_0_1 | prototype | 0.0556 ± 0.0544 | 0.0000 ± 0.0000 | 3843080 ± 168648 | 0.5 |

Notes:
- CI95 uses normal approximation: 1.96 * std/sqrt(n).
- p-values use Welch's t-test on per-seed mean client accuracy.