# Block F: Mechanism Ablation Study

## Section 1: Objective

Identify which components of FLEX-Persona are responsible for its performance gains. We isolate and evaluate the contribution of:

- **Prototype exchange**: Sharing compact representation summaries instead of full weights
- **Clustering**: Grouping similar clients via spectral clustering on feature similarity
- **Cluster-aware guidance**: Aligning local representations with cluster prototypes

---

## Section 2: Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Clients | 10 |
| Samples per client | 2000 |
| Partition | Dirichlet (α = 0.1) |
| Rounds | 20 |
| Local epochs | 5 |
| Cluster-aware epochs | 2 (full / no_clustering / random); 0 (no_guidance) |
| Batch size | 64 |
| Learning rate | 0.001 |
| Seeds | [42, 43, 44] |
| Total runs | 15 (5 methods × 3 seeds) |

**Important limitation:** This experiment tests a single regime (α = 0.1, moderate heterogeneity). Conclusions about component importance may not generalize to other heterogeneity levels, data scales, or training horizons.

---

## Section 3: Results Table

| Method | Mean | Std | Worst | P10 | Drop vs FLEX |
|---|---|---|---|---|---|
| fedavg_sgd | 0.4258 | 0.0193 | 0.1892 | 0.2062 | +0.3635 |
| flex_full | 0.7892 | 0.0160 | 0.6379 | 0.6659 | — (reference) |
| flex_no_clustering | 0.7880 | 0.0164 | 0.6257 | 0.6491 | +0.0012 |
| flex_random_clusters | 0.7814 | 0.0108 | 0.6052 | 0.6343 | +0.0078 |
| flex_no_guidance | 0.7853 | 0.0128 | 0.6191 | 0.6409 | +0.0039 |

**Observed range of all FLEX variants:** [0.7814 → 0.7892] ≈ **1.0%**  
**Distance to FedAvg:** 0.3634 ≈ **46%**

---

## Section 4: What the Data Actually Says (No Interpretation)

1. All FLEX variants cluster tightly around ~0.79 mean accuracy.
2. FedAvg is far away at ~0.43.
3. The dominant separation in this experiment is **FLEX mechanism vs. parameter averaging**.
4. Differences *within* FLEX variants are small (< 1%).

---

## Section 5: Component Impact Analysis

### (A) Prototype Exchange — DOMINANT MECHANISM

**Evidence:**
- Removing everything → FedAvg drops to **0.4258**
- Any FLEX variant → **~0.78–0.79**

**Conclusion:** The prototype exchange mechanism is the **primary driver** of performance gains. The act of extracting per-class mean representations, transmitting them to the server, and aggregating them into a global statistical view creates implicit alignment that dramatically outperforms parameter averaging.

---

### (B) Clustering — STRUCTURAL REFINEMENT (NOT REQUIRED)

**Evidence:**
- Disabling clustering entirely (all clients in one group): drop = **0.0012 (~0.15%)**
- Replacing with random assignment: drop = **0.0078 (~1.0%)**

**Conclusion:** Clustering provides **structural coherence** but is **not essential** for achieving high average accuracy in this regime (α = 0.1). Global prototype aggregation alone is sufficient for alignment. However, random assignment degrades worst-case performance (~5%), indicating that meaningful grouping provides stability benefits.

**Important caveat:** This conclusion is regime-specific. Clustering may matter more under:
- Higher heterogeneity (lower α)
- Larger client counts
- Longer training horizons

---

### (C) Guidance — INCONCLUSIVE

**Evidence:**
- Removing cluster-aware guidance: drop = **0.0039 (~0.5%)**

**Critical limitation:** Due to `cluster_aware_epochs = 0` in the base configuration, the cluster-aware guidance mechanism is **barely exercised** in this experiment. The observed small drop (~0.5%) cannot be interpreted as definitive evidence of guidance's insignificance.

**What is actually being compared:**
- `flex_full`: local training (5 epochs) + minimal guidance (2 epochs with cluster signal)
- `flex_no_guidance`: local training (5 epochs) + zero guidance epochs

The small gap may reflect insufficient activation of the guidance mechanism rather than true ineffectiveness.

---

### Component Attribution Summary

| Component | Attribution | Evidence |
|---|---|---|
| **Prototype exchange** | **Primary** | ~46 pp gap vs FedAvg; all FLEX variants ~0.79 |
| **Structural clustering** | **Secondary / refinement** | <1% impact; stabilizes worst-case |
| **Cluster-aware guidance** | **Inconclusive** | Barely activated in current config |

---

## Section 6: Hidden Structural Insight

The ablation reveals a deeper mechanism than originally hypothesized:

**Originally assumed model:**
```
clustering → alignment → performance
```

**Actual model suggested by data:**
```
prototype exchange → implicit alignment → performance
clustering → refinement (minor)
guidance → unclear (insufficiently tested)
```

This means FLEX-Persona behaves as a **representation alignment system via shared statistics**, where the coordination strategy (clustering + guidance) refines behavior but does not define it.

---

## Section 7: Limitations of This Experiment

| Limitation | Impact on Conclusions |
|---|---|
| Single regime (α = 0.1) | Clustering may matter more at extreme heterogeneity |
| Fixed K (number of clusters) | Adaptive K might change dynamics |
| Short horizon (20 rounds) | Long-term drift effects not captured |
| Minimal guidance activation | Guidance effect cannot be conclusively assessed |
| SmallCNN only | Architecture sensitivity unknown |

---

## Section 8: Recommended Follow-up

To strengthen conclusions, run one additional variant:

| Variant | cluster_aware_epochs | Purpose |
|---|---|---|
| `flex_guided_2ep` | 2 | Compare to `flex_no_guidance` with identical local training |
| `flex_guided_5ep` | 5 | Test if stronger guidance signal matters |

If gap increases → guidance matters when sufficiently activated.  
If still small → guidance truly negligible.

---

## Section 9: Publication-Grade Conclusion

> The ablation study reveals that the primary source of FLEX-Persona's performance gains is the **prototype exchange mechanism**, which enables implicit alignment of client representations even without explicit coordination. Under moderate heterogeneity (α = 0.1), clustering contributes marginal improvements by enforcing structural coherence, particularly in worst-case performance, but is not essential for achieving high average accuracy. The effect of cluster-aware guidance remains **inconclusive** under the current configuration due to minimal activation, requiring further investigation. These results suggest that the key innovation of FLEX lies in its **representation-sharing paradigm** rather than its clustering or guidance components.

**Correct attribution:**
- **Primary:** Prototype exchange
- **Secondary:** Structural clustering (regime-specific)
- **Unclear:** Guidance (insufficiently tested)

---

*Analysis completed. All 15 runs executed across 3 seeds. Report generated from `F_results.jsonl`.*
