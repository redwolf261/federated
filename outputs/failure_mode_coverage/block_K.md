# Block K: Extra-Epoch Control Experiment

**Question:** Does the cluster-aware phase benefit performance through
(A) **extra training steps** or (B) **guidance signal structure**?

FLEX Full (reference): 0.7801
Total gap (flex vs local_only): -0.0077

---
## Results Table

| Method | Mean ± Std | Worst | P10 | Δ vs FLEX | Δ vs local_only |
|--------|-----------|-------|-----|-----------|----------------|
| FLEX Full (5+2 epochs, real guidance) | 0.7801 ± 0.0194 | 0.6057 | 0.6321 | +0.0000 | -0.0077 |
| Extra Local Epochs (7+0, no guidance) | 0.7831 ± 0.0206 | 0.6071 | 0.6465 | +0.0030 | -0.0048 |
| Random Guidance (5+2, random targets) | 0.7901 ± 0.0194 | 0.6351 | 0.6595 | +0.0099 | +0.0022 |
| Local Only (5+0, no guidance) | 0.7879 ± 0.0166 | 0.6164 | 0.6526 | +0.0077 | +0.0000 |

---
## Causal Decision

### Key Comparisons

- `extra_local_7ep` vs `flex_full`:     +0.0030 (≈ same)
- `random_guidance` vs `flex_full`:     +0.0099 (≈ same)
- `extra_local_7ep` vs `random_guidance`: -0.0069 (≈ same)
- `extra_local_7ep` vs `local_only_5ep`: -0.0048 (no difference)

### Verdict: **EXTRA TRAINING STEPS**

extra_local_7ep ≈ flex_full — the cluster-aware phase helps because it provides extra gradient steps, NOT because of guidance signal content. Replacing guidance with plain CE epochs achieves the same result.

```
Primary driver: Additional optimization steps (7 vs 5 epochs/round).
Secondary driver: Adapter geometry conditioning (from Block J).
Rejected: Guidance signal content, prototype semantics, cluster structure.
```

---
## Per-Seed Raw Results

| Method | Seed 42 | Seed 43 | Seed 44 |
|--------|---------|---------|---------|
| FLEX Full (5+2 epochs, real guidance) | 0.7849 | 0.7543 | 0.8011 |
| Extra Local Epochs (7+0, no guidance) | 0.7905 | 0.7550 | 0.8038 |
| Random Guidance (5+2, random targets) | 0.8032 | 0.7626 | 0.8043 |
| Local Only (5+0, no guidance) | 0.7945 | 0.7650 | 0.8041 |