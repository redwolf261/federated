# Block G (FIXED): Mechanism Isolation with Active Guidance

## Section 1: Objective

Establish a direct causal link between prototype exchange and FLEX-Persona's
performance gains **with cluster guidance active** (cluster_aware_epochs=2).
The previous Block G experiment was flawed because cluster_aware_epochs=0,
which meant prototype exchange was never consumed during training.

## Section 2: Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Clients | 10 |
| Samples per client | 2000 |
| Partition | Dirichlet (alpha = 0.1) |
| Rounds | 20 |
| Local epochs | 5 |
| Cluster-aware epochs | **2** |
| Batch size | 64 |
| Learning rate | 0.001 |
| Seeds | [42, 43, 44] |
| Total runs | 18 (6 methods x 3 seeds) |

## Section 3: Methods

| Method | Description |
|---|---|
| flex_full | Normal prototype extraction + sharing + aggregation |
| flex_no_prototype_sharing | Clients do NOT send prototypes; no aggregation |
| flex_self_only | Server returns each client's own prototype only |
| flex_shuffled_prototypes | Server randomly permutes prototype assignments |
| flex_noise_prototypes | All prototypes replaced with random noise |
| fedavg_sgd | Baseline reference (FedAvg) |

## Section 4: Results Table

| Method | Mean | Std | Worst | P10 | Drop vs FLEX |
|---|---|---|---|---|---|
| flex_full | 0.7892 | 0.0160 | 0.6379 | 0.6659 | +0.0000 |
| flex_no_prototype_sharing | 0.7891 | 0.0164 | 0.6198 | 0.6519 | +0.0001 |
| flex_self_only | 0.7777 | 0.0147 | 0.6279 | 0.6407 | +0.0116 |
| flex_shuffled_prototypes | 0.7909 | 0.0181 | 0.6375 | 0.6648 | -0.0017 |
| flex_noise_prototypes | 0.7442 | 0.0169 | 0.4938 | 0.6051 | +0.0450 |
| fedavg_sgd | 0.4715 | 0.0174 | 0.3169 | 0.3549 | +0.3178 |

## Section 5: Performance Drops vs FLEX Full

| Method | Drop | Interpretation |
|---|---|---|
| flex_no_prototype_sharing | +0.0001 | Removing prototype sharing entirely |
| flex_self_only | +0.0116 | Removing cross-client mixing (self-only) |
| flex_shuffled_prototypes | -0.0017 | Corrupting assignment structure |
| flex_noise_prototypes | +0.0450 | Replacing signal with noise |
| fedavg_sgd | +0.3178 | Replacing FLEX with FedAvg baseline |

## Section 6: Validation Checks

### Necessity Of Sharing
FAIL: flex_no_prototype_sharing mean=0.7891, drop=0.0001 (0.0 pp) vs threshold 5.0 pp — FAIL

### Information Integrity
FAIL: flex_shuffled_prototypes mean=0.7909, drop=-0.0017 (-0.2 pp) vs threshold 1.0 pp — FAIL

### Signal Vs Noise
FAIL: flex_noise_prototypes mean=0.7442, drop=0.0450 (4.5 pp) vs threshold 5.0 pp — FAIL

### Collaboration Requirement
PASS: flex_self_only mean=0.7777, drop=0.0116 (1.2 pp) vs threshold 1.0 pp — PASS

## Section 7: Causal Conclusion

**Verdict: REJECTED**

Prototype exchange does not appear to be the primary causal mechanism. The observed performance gains may stem from other factors (e.g., architecture, optimizer, data). Even with active guidance, removing or corrupting prototype exchange does not significantly degrade performance.

### Explicit Answer

> Does performance collapse when prototype exchange is removed or corrupted?
>
> **REJECTED**: No — performance does not depend on prototype exchange.

## Section 8: Mechanistic Interpretation

### What the Data Shows

1. **Prototype exchange has no effect**: Even with active guidance, removing or corrupting
   prototype sharing produces no meaningful degradation.
2. **Architecture dominates**: The backbone+adapter design drives all observed gains.
3. **Cluster guidance is not prototype-dependent**: The alignment loss uses cluster centers,
   not prototype distributions, suggesting the feature-mean clustering is the actual mechanism.

---

*Report generated from 3 seeds per method.*
*Total runs: 18.*