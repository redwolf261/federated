# Block J: Final Causal Disentanglement

**Question:** Do FLEX gains arise from (A) architecture, (B) auxiliary loss, (C) geometry/random transformation, or (D) their interaction?

**Total gap (FLEX vs FedAvg):** +0.3899 (97.6%)

---
## Results Table

| Method | Mean ± Std | Worst | P10 | Δ vs FLEX | Δ vs FedAvg |
|--------|-----------|-------|-----|-----------|-------------|
| FedAvg SGD (Baseline) | 0.3993 ± 0.0210 | 0.1603 | 0.2152 | -0.3899 | +0.0000 |
| Backbone Only (Control) | 0.3963 ± 0.0240 | 0.1562 | 0.2162 | -0.3929 | -0.0030 |
| Backbone + Adapter, No Loss | 0.4950 ± 0.0137 | 0.2797 | 0.3770 | -0.2942 | +0.0957 |
| Backbone + Random Proj (Frozen) | 0.5172 ± 0.0047 | 0.2813 | 0.3344 | -0.2720 | +0.1179 |
| Backbone + Dummy Loss | 0.4477 ± 0.0281 | 0.2250 | 0.2692 | -0.3415 | +0.0484 |
| Backbone + Adapter + Dummy Loss | 0.4857 ± 0.0195 | 0.1842 | 0.3087 | -0.3035 | +0.0864 |
| FLEX Full (Reference) | 0.7892 ± 0.0160 | 0.6379 | 0.6659 | +0.0000 | +0.3899 |

---
## Causal Classification

### Sanity Checks

- backbone_only ≈ fedavg_sgd: PASS ✅ (Δ=-0.0030)

### Case Verdict: **CASE MIXED**

Results are mixed; no single mechanism dominates cleanly.

---
## Final Causal Statement

```
Primary driver:   Inconclusive — largest: random_proj (+0.1179)
Secondary driver: See deltas above
Rejected mechanisms: none conclusively rejected
```

---
## Per-Seed Raw Results

| Method | Seed 42 | Seed 43 | Seed 44 |
|--------|---------|---------|---------|
| FedAvg SGD (Baseline) | 0.3800 | 0.3895 | 0.4285 |
| Backbone Only (Control) | 0.3710 | 0.3895 | 0.4285 |
| Backbone + Adapter, No Loss | 0.4998 | 0.5088 | 0.4763 |
| Backbone + Random Proj (Frozen) | 0.5105 | 0.5208 | 0.5203 |
| Backbone + Dummy Loss | 0.4120 | 0.4806 | 0.4505 |
| Backbone + Adapter + Dummy Loss | 0.5030 | 0.4958 | 0.4584 |
| FLEX Full (Reference) | 0.8051 | 0.7673 | 0.7952 |

---
## Aggregated Statistics (JSON)

```json
{
  "fedavg_sgd": {
    "n": 3,
    "mean": 0.399309,
    "std": 0.020983,
    "worst": 0.160319,
    "p10": 0.215202,
    "seeds": [
      42,
      43,
      44
    ]
  },
  "backbone_only": {
    "n": 3,
    "mean": 0.396319,
    "std": 0.023953,
    "worst": 0.156153,
    "p10": 0.216204,
    "seeds": [
      42,
      43,
      44
    ]
  },
  "backbone_plus_adapter_no_loss": {
    "n": 3,
    "mean": 0.494979,
    "std": 0.013729,
    "worst": 0.279722,
    "p10": 0.377001,
    "seeds": [
      42,
      43,
      44
    ]
  },
  "backbone_plus_random_proj": {
    "n": 3,
    "mean": 0.517198,
    "std": 0.004725,
    "worst": 0.281317,
    "p10": 0.334382,
    "seeds": [
      42,
      43,
      44
    ]
  },
  "backbone_plus_dummy_loss": {
    "n": 3,
    "mean": 0.447737,
    "std": 0.028081,
    "worst": 0.225,
    "p10": 0.26925,
    "seeds": [
      42,
      43,
      44
    ]
  },
  "backbone_plus_adapter_dummy": {
    "n": 3,
    "mean": 0.485742,
    "std": 0.019522,
    "worst": 0.184167,
    "p10": 0.308667,
    "seeds": [
      42,
      43,
      44
    ]
  },
  "flex_full_reference": {
    "n": 3,
    "mean": 0.789217,
    "std": 0.016017,
    "worst": 0.637857,
    "p10": 0.665869,
    "seeds": [
      42,
      43,
      44
    ]
  }
}
```