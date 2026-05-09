# Block I: Signal Nature Analysis

**Experiment:** What is the TRUE PROPERTY responsible for FLEX's performance gain?

**Setup (identical to Block H):**
- Dataset: CIFAR-10
- Clients: 10
- Samples/client: 2000
- Dirichlet α = 0.1
- Rounds: 20
- Local epochs: 5
- Cluster-aware epochs: 2
- Seeds: [42, 43, 44]
- Total runs: 21

---

## Methods

| # | Method | Signal Type | Purpose |
|---|--------|-------------|---------|
| 1 | `flex_full` | Cluster prototypes (cross-client) | Reference |
| 2 | `class_centroid_alignment` | Per-class centroids from **own** data | Remove cross-client info, keep class structure |
| 3 | `global_centroid_alignment` | Single global centroid | Remove class structure, keep regularization |
| 4 | `random_centroid_alignment` | Random fixed centroids per class | Remove semantic meaning |
| 5 | `feature_norm_only` | L2 norm constraint on features | Test geometry/scale alone |
| 6 | `variance_minimization` | Minimize intra-batch feature variance | Test generic regularization |
| 7 | `fedavg_sgd` | None | Baseline |

---

## Results

### Summary Table (Averaged Across Seeds)

| Method | Mean Acc ± Std | Worst | P10 | Drop vs Full | Interpretation |
|--------|---------------|-------|-----|--------------|----------------|
| FLEX Full (Reference) | 0.7892 ± 0.0160 | 0.6379 | 0.6659 | +0.0000 (+0.0%) | 🔵 Reference |
| Random Centroid Alignment | 0.7882 ± 0.0163 | 0.6102 | 0.6449 | +0.0010 (+0.1%) | ✅ Equivalent — property preserved |
| Variance Minimization | 0.7807 ± 0.0204 | 0.5968 | 0.6452 | +0.0085 (+1.1%) | ✅ Equivalent — property preserved |
| Global Centroid Alignment | 0.7804 ± 0.0174 | 0.6061 | 0.6619 | +0.0088 (+1.1%) | ✅ Equivalent — property preserved |
| Class Centroid Alignment | 0.7769 ± 0.0180 | 0.6133 | 0.6574 | +0.0123 (+1.6%) | ✅ Equivalent — property preserved |
| Feature Norm Only | 0.7765 ± 0.0181 | 0.6193 | 0.6439 | +0.0128 (+1.6%) | ✅ Equivalent — property preserved |
| FedAvg SGD (Baseline) | 0.4715 ± 0.0174 | 0.3169 | 0.3549 | +0.3178 (+40.3%) | ⬛ Baseline |

---

### Per-Seed Raw Results

| Method | Seed 42 | Seed 43 | Seed 44 | Mean |
|--------|---------|---------|---------|------|
| FLEX Full (Reference) | 0.8051 | 0.7673 | 0.7952 | 0.7892 |
| Class Centroid Alignment | 0.7909 | 0.7515 | 0.7884 | 0.7769 |
| Global Centroid Alignment | 0.7976 | 0.7565 | 0.7872 | 0.7804 |
| Random Centroid Alignment | 0.7946 | 0.7658 | 0.8042 | 0.7882 |
| Feature Norm Only | 0.7844 | 0.7514 | 0.7935 | 0.7765 |
| Variance Minimization | 0.7927 | 0.7520 | 0.7976 | 0.7807 |
| FedAvg SGD (Baseline) | 0.4490 | 0.4740 | 0.4913 | 0.4715 |

---

## Causal Analysis

### Drop Analysis

Total FLEX vs FedAvg gap: **+0.3178** (+67.4% relative to baseline)

| Signal | Drop vs Full | Drop > 5pp? | Interpretation |
|--------|-------------|-------------|----------------|
| Class Centroid | +0.0123 | no ✅ | Own-data class centroids are sufficient |
| Global Centroid | +0.0088 | no ✅ | Class structure not needed — regularization alone works |
| Random Centroid | +0.0010 | no ✅ | Even random targets work — not about information content |
| Feature Norm | +0.0128 | no ✅ | Geometry/scale alone explains gains |
| Variance Min | +0.0085 | no ✅ | Generic regularization explains gains |

### Verdict

**CASE 4**: ALL ablation variants perform equivalently to flex_full. No specific property of the alignment signal matters — the gains are driven entirely by the backbone+adapter architecture. The training signal is irrelevant.

---

## Final Conclusion

```
Primary driver: Architecture Bias
Mechanism type: (architecture)
Rejected hypotheses: class-structure, regularization, geometry, cross-client signal
```

**Reference (flex_full):** 0.7892 ± 0.0160
**Baseline (fedavg_sgd):** 0.4715 ± 0.0174
**Total gap explained by architecture:** ~0.3178 (67.4% above baseline)

---

## Appendix: Aggregated Statistics (JSON)

```json
{
  "flex_full": {
    "n": 3,
    "mean": 0.789217,
    "std_across_seeds": 0.016017,
    "worst": 0.637857,
    "p10": 0.665869,
    "mean_per_client_std": 0.099847,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.8051392484797317,
      0.7673034045393858,
      0.7952075471698113
    ]
  },
  "class_centroid_alignment": {
    "n": 3,
    "mean": 0.776928,
    "std_across_seeds": 0.018012,
    "worst": 0.613274,
    "p10": 0.657411,
    "mean_per_client_std": 0.102907,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.7908590863265477,
      0.7514938251001334,
      0.788430817610063
    ]
  },
  "global_centroid_alignment": {
    "n": 3,
    "mean": 0.780422,
    "std_across_seeds": 0.017432,
    "worst": 0.606071,
    "p10": 0.66194,
    "mean_per_client_std": 0.103034,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.7975949381688301,
      0.7565176902536715,
      0.7871540880503145
    ]
  },
  "random_centroid_alignment": {
    "n": 3,
    "mean": 0.788205,
    "std_across_seeds": 0.016343,
    "worst": 0.610238,
    "p10": 0.644857,
    "mean_per_client_std": 0.106134,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.7946209943899478,
      0.7657676902536716,
      0.8042264150943396
    ]
  },
  "feature_norm_only": {
    "n": 3,
    "mean": 0.776456,
    "std_across_seeds": 0.018069,
    "worst": 0.619299,
    "p10": 0.64393,
    "mean_per_client_std": 0.10719,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.7844481884233572,
      0.7514402536715621,
      0.7934795597484277
    ]
  },
  "variance_minimization": {
    "n": 3,
    "mean": 0.780746,
    "std_across_seeds": 0.02044,
    "worst": 0.596786,
    "p10": 0.645179,
    "mean_per_client_std": 0.109533,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.7926507297961515,
      0.7519819759679574,
      0.7976053459119496
    ]
  },
  "fedavg_sgd": {
    "n": 3,
    "mean": 0.471461,
    "std_across_seeds": 0.017387,
    "worst": 0.316944,
    "p10": 0.354944,
    "mean_per_client_std": 0.095153,
    "seeds": [
      42,
      43,
      44
    ],
    "per_seed_means": [
      0.44899197114915806,
      0.47404372496662217,
      0.49134591194968547
    ]
  }
}
```
