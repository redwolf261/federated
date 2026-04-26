# Multi-Seed Ablation Summary

Baseline for significance: fedavg_baseline

| Variant | Mode | Mean Acc (mean±CI95) | Worst Acc (mean±CI95) | Comm Bytes (mean±CI95) | p-value vs baseline (mean acc) |
|---|---|---:|---:|---:|---:|
| fedavg_baseline | fedavg | 0.0208 ± 0.0408 | 0.0000 ± 0.0000 | 1756198 ± 0 | - |
| prototype_lambda_0_1 | prototype | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 410638 ± 2016 | 0.5 |

Notes:
- CI95 uses normal approximation: 1.96 * std/sqrt(n).
- p-values use Welch's t-test on per-seed mean client accuracy.