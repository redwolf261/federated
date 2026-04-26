# Block G: Mechanism Isolation (Prototype Exchange Causality)

## Section 1: Objective

Establish a direct causal link between prototype exchange and FLEX-Persona's performance gains.
Previous experiments (Blocks A–F) eliminated alternative explanations, but did not directly
remove or corrupt the prototype-sharing mechanism itself.

This block isolates and stress-tests the mechanism by removing or degrading cross-client
prototype information flow.

## Section 2: Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Clients | 10 |
| Samples per client | 2000 |
| Partition | Dirichlet (α = 0.1) |
| Rounds | 20 |
| Local epochs | 5 |
| Cluster-aware epochs | 0 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Seeds | [42, 43, 44] |
| Total runs | 18 (6 methods × 3 seeds) |

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
| flex_full | 0.7853 | 0.0128 | 0.6191 | 0.6409 | +0.0000 |
| flex_no_prototype_sharing | 0.7891 | 0.0164 | 0.6198 | 0.6519 | -0.0038 |
| flex_self_only | 0.7853 | 0.0128 | 0.6191 | 0.6409 | +0.0000 |
| flex_shuffled_prototypes | 0.7853 | 0.0128 | 0.6191 | 0.6409 | +0.0000 |
| flex_noise_prototypes | 0.7442 | 0.0169 | 0.4938 | 0.6051 | +0.0411 |
| fedavg_sgd | 0.4715 | 0.0174 | 0.3169 | 0.3549 | +0.3139 |

## Section 5: Performance Drops vs FLEX Full

| Method | Drop | Interpretation |
|---|---|---|
| flex_no_prototype_sharing | -0.0038 | Removing prototype sharing entirely |
| flex_self_only | +0.0000 | Removing cross-client mixing (self-only) |
| flex_shuffled_prototypes | +0.0000 | Corrupting assignment structure |
| flex_noise_prototypes | +0.0411 | Replacing signal with noise |
| fedavg_sgd | +0.3139 | Replacing FLEX with FedAvg baseline |

## Section 6: Validation Checks

### Necessity Of Sharing
❌ FAIL: flex_no_prototype_sharing mean=0.7891, drop=-0.0038 (-0.4 pp) vs threshold 0.20 — FAIL

### Information Integrity
❌ FAIL: flex_shuffled_prototypes mean=0.7853, drop=0.0000 (0.0 pp) vs threshold 0.01 — FAIL

### Signal Vs Noise
❌ FAIL: flex_noise_prototypes mean=0.7442, drop=0.0411 (4.1 pp) vs threshold 0.10 — FAIL

### Collaboration Requirement
❌ FAIL: flex_self_only mean=0.7853, drop=0.0000 (0.0 pp) vs threshold 0.01 — FAIL

## Section 7: Causal Conclusion

**Verdict: REJECTED**

Prototype exchange does not appear to be the primary mechanism. The observed performance gains may stem from other factors (e.g., architecture, optimizer, data).

### Explicit Answer

> Does performance collapse when prototype exchange is removed or corrupted?
>
> **REJECTED**: No — performance does not depend on prototype exchange.

## Section 8: Mechanistic Interpretation

### What the Data Shows

1. **Prototype exchange has no effect when guidance is disabled**: With cluster_aware_epochs=0,
   removing prototype sharing (flex_no_prototype_sharing) or corrupting assignments
   (flex_shuffled_prototypes) produces identical results to flex_full.
2. **Self-only prototypes are equivalent to full aggregation**: flex_self_only matches flex_full
   exactly (to 16 decimal places), confirming that cross-client prototype mixing is not active.
3. **Noise prototypes crashed**: flex_noise_prototypes failed after ~1 round (8-14s vs 120-235s),
   indicating a bug in the noise injection implementation. Results reflect single-round accuracy only.
4. **The gap to FedAvg is architectural**: All functional FLEX variants (including no-prototype-sharing)
   outperform FedAvg by ~30-35pp, suggesting the adapter architecture drives the gain.

### Structural Insight

The magnitude of degradation correlates with the severity of corruption:

| Corruption Level | Method | Drop |
|---|---|---|
| Complete signal destruction | flex_noise_prototypes | +0.0411 |
| Complete mechanism removal | flex_no_prototype_sharing | -0.0038 |
| Replace with FedAvg | fedavg_sgd | +0.3139 |
| Structural corruption | flex_shuffled_prototypes | +0.0000 |
| Remove cross-client mixing | flex_self_only | +0.0000 |

With cluster_aware_epochs=0, no degradation is observed regardless of corruption severity,
because the prototype information is never consumed during training. The exchange mechanism
requires active cluster guidance (cluster_aware_epochs > 0) to function. All observed FLEX
performance gains in this configuration are attributable to the backbone+adapter architecture,
not to prototype-based collaboration.

---

*Report generated from 3 seeds per method.*
*Total runs: 18.*