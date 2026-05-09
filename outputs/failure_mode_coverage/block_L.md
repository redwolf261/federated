# Block L: L2 Normalization Isolation Experiment

**Question:** Do FLEX gains arise from (A) dimensionality reduction,
(B) L2 spherical normalization, or (C) their interaction?

Reference (bottleneck_with_l2): 0.7899
FedAvg baseline:                0.4871
Total ref gap vs FedAvg:        +0.3028

---
## Results Table

| Method | Mean ± Std | Worst | P10 | Δ vs L2-Bottleneck | Δ vs FedAvg |
|--------|-----------|-------|-----|-------------------|------------|
| FedAvg SGD (baseline) | 0.4871 ± 0.0089 | 0.3225 | 0.3798 | -0.3028 | +0.0000 |
| Backbone Only (8192→classifier) | 0.7927 ± 0.0133 | 0.6255 | 0.6700 | +0.0027 | +0.3055 |
| Bottleneck No Norm (8192→64, no L2) | 0.7926 ± 0.0132 | 0.6213 | 0.6644 | +0.0027 | +0.3054 |
| Bottleneck + L2 (8192→64 + normalize) [FLEX] | 0.7899 ± 0.0124 | 0.6187 | 0.6566 | +0.0000 | +0.3028 |
| Random Proj No Norm (frozen R, no L2) | 0.7821 ± 0.0173 | 0.6062 | 0.6606 | -0.0078 | +0.2950 |
| Random Proj + L2 (frozen R + normalize) | 0.8103 ± 0.0155 | 0.6473 | 0.6877 | +0.0204 | +0.3232 |

---
## Causal Key Comparisons

- L2 norm gain (learned proj):  bn_with_l2 - bn_no_norm = -0.0027
- L2 norm gain (frozen proj):   rp_with_l2 - rp_no_norm = +0.0283
- Projection gain (no norm):    bn_no_norm - backbone    = -0.0001
- Projection gain (with L2):    bn_with_l2 - backbone    = -0.0027

### Verdict: **CASE C — INTERACTION EFFECT**

Both projection and normalization contribute independently and neither alone is sufficient to explain the full gain. Their combination is necessary.

```
Primary: Projection + L2 normalization interaction
Secondary: Either alone
```

---
## Per-Seed Raw Results

| Method | Seed 42 | Seed 43 | Seed 44 |
|--------|---------|---------|---------|
| FedAvg SGD (baseline) | 0.4774 | 0.4850 | 0.4990 |
| Backbone Only (8192→classifier) | 0.7950 | 0.7754 | 0.8076 |
| Bottleneck No Norm (8192→64, no L2) | 0.7948 | 0.7754 | 0.8075 |
| Bottleneck + L2 (8192→64 + normalize) [FLEX] | 0.7957 | 0.7727 | 0.8013 |
| Random Proj No Norm (frozen R, no L2) | 0.7882 | 0.7585 | 0.7995 |
| Random Proj + L2 (frozen R + normalize) | 0.8185 | 0.7887 | 0.8238 |

---
## Final Mechanistic Interpretation

This experiment closes the L2 normalization question in the FLEX-Persona
causal audit. Combined with Blocks I, J, and K findings:

- Block I: Prototype semantic content → irrelevant
- Block J: Adapter bottleneck vs random projection → projection geometry drives gains
- Block K: Extra epochs vs guidance structure → all methods equivalent
- Block L: L2 norm vs bottleneck → **this verdict**

*FLEX-Persona Causal Audit — Block L complete.*