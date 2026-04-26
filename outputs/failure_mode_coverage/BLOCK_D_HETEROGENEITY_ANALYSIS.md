# Block D: Heterogeneity Sweep Analysis

## Objective

Determine whether FLEX-Persona's performance advantage is caused by its ability to handle non-IID data (heterogeneity).

## Experimental Design

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Clients | 10 |
| Seeds | [42, 43, 44] |
| Rounds | 20 |
| Local epochs | 5 |
| Cluster-aware epochs | 0 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Samples per client | 2000 |
| Alpha values | [0.05, 0.1, 0.5, 1.0, 10.0] |
| Methods | flex_no_extra, fedavg_sgd |

## Results

### Aggregate Performance by Alpha

| Alpha | FLEX Mean | FLEX Std | FedAvg Mean | FedAvg Std |
|---|---|---|---|---|
| 0.05 | 0.8646 | 0.0482 | 0.3250 | 0.1226 |
| 0.1 | 0.7853 | 0.0128 | 0.4258 | 0.0194 |
| 0.5 | 0.6422 | 0.0308 | 0.4943 | 0.0163 |
| 1.0 | 0.5517 | 0.0245 | 0.5162 | 0.0070 |
| 10.0 | 0.4520 | 0.0063 | 0.5372 | 0.0075 |

### Gain Analysis

| Alpha | FLEX Mean | FedAvg Mean | Abs Gain | Rel Gain | Worst Gain |
|---|---|---|---|---|---|
| 0.05 | 0.8646 | 0.3250 | +0.5396 | +166.0% | +0.6380 |
| 0.1 | 0.7853 | 0.4258 | +0.3595 | +84.4% | +0.4291 |
| 0.5 | 0.6422 | 0.4943 | +0.1480 | +29.9% | +0.1833 |
| 1.0 | 0.5517 | 0.5162 | +0.0355 | +6.9% | +0.0342 |
| 10.0 | 0.4520 | 0.5372 | -0.0852 | -15.9% | -0.0850 |

### Pattern Classification

**Pattern detected:** `heterogeneity_dependent`

**Interpretation:** FLEX advantage decreases as heterogeneity decreases. This confirms that FLEX's benefit is specifically tied to its ability to handle non-IID data.

### Critical Validation Checks

1. **Alpha = 10 (near-IID):** FedAvg = 0.5372, FLEX = 0.4520, gap = -0.0852
   ✅ FedAvg improves monotonically with alpha (0.3250 → 0.5372), as expected.
   ✅ The gap flipped because FLEX became worse under homogeneity, not because FedAvg failed.
2. **Gap shrinkage:** Low-alpha gap = +0.5396, High-alpha gap = -0.0852
   ✅ Gap decreased monotonically as alpha increased.
3. **Worst-case improvement at low alpha:** +0.6380
   ✅ FLEX massively improves worst-client fairness in high-heterogeneity regimes.

## Mechanism-Level Explanation

| Regime | Winner | Why |
|---|---|---|
| High heterogeneity (α ≤ 0.5) | **FLEX** | Cross-client distributions differ; clustering + alignment resolves misalignment |
| Moderate (α ≈ 1.0) | ~Tie | Some heterogeneity remains, but FedAvg can still cope |
| Near-IID (α = 10) | **FedAvg** | All clients approximate the same distribution; global model = optimal. FLEX's clustering constraints become unnecessary interference |

**Analogy:** Non-IID → students studying different chapters → coordination helps. IID → everyone studying the same chapter → coordination becomes interference.

## Conclusion

FLEX-Persona is **conditionally optimal**: it provides substantial gains under heterogeneous (non-IID) data by mitigating cross-client distribution misalignment, but introduces unnecessary representation constraints under near-IID conditions, where standard parameter averaging (FedAvg) becomes optimal and outperforms FLEX.

**Key contribution:** FLEX does not just improve accuracy — it solves fairness imbalance in federated learning. At α = 0.05, worst-client accuracy improves by **+0.6380**, demonstrating that representation-based collaboration specifically protects disadvantaged clients.

**One-line research claim:** *FLEX-Persona is a heterogeneity-aware federated learning method that significantly improves performance and worst-case client outcomes under non-IID conditions, but is outperformed by standard parameter averaging in near-IID regimes due to unnecessary representation constraints.*


## Raw Results

| Alpha | Seed | Method | Mean | Worst | Std | P10 |
|---|---|---|---|---|---|---|
| 0.05 | 42 | fedavg_sgd | 0.4248 | 0.0625 | 0.2914 | 0.1030 |
| 0.05 | 42 | flex_no_extra | 0.8759 | 0.7425 | 0.0880 | 0.7448 |
| 0.05 | 43 | fedavg_sgd | 0.3979 | 0.0700 | 0.2108 | 0.1356 |
| 0.05 | 43 | flex_no_extra | 0.8007 | 0.5714 | 0.0965 | 0.7209 |
| 0.05 | 44 | fedavg_sgd | 0.1523 | 0.0000 | 0.2023 | 0.0000 |
| 0.05 | 44 | flex_no_extra | 0.9172 | 0.7325 | 0.0939 | 0.7752 |
| 0.1 | 42 | fedavg_sgd | 0.4159 | 0.1475 | 0.1719 | 0.1722 |
| 0.1 | 42 | flex_no_extra | 0.7867 | 0.6111 | 0.1001 | 0.6169 |
| 0.1 | 43 | fedavg_sgd | 0.4086 | 0.1850 | 0.1649 | 0.1867 |
| 0.1 | 43 | flex_no_extra | 0.7691 | 0.5938 | 0.1065 | 0.6421 |
| 0.1 | 44 | fedavg_sgd | 0.4530 | 0.2375 | 0.1578 | 0.2600 |
| 0.1 | 44 | flex_no_extra | 0.8002 | 0.6525 | 0.1094 | 0.6638 |
| 0.5 | 42 | fedavg_sgd | 0.5089 | 0.2750 | 0.1302 | 0.3133 |
| 0.5 | 42 | flex_no_extra | 0.6562 | 0.5475 | 0.0500 | 0.6015 |
| 0.5 | 43 | fedavg_sgd | 0.4715 | 0.3500 | 0.0952 | 0.3680 |
| 0.5 | 43 | flex_no_extra | 0.6710 | 0.5550 | 0.1004 | 0.5595 |
| 0.5 | 44 | fedavg_sgd | 0.5024 | 0.4000 | 0.0596 | 0.4158 |
| 0.5 | 44 | flex_no_extra | 0.5996 | 0.4725 | 0.0681 | 0.5040 |
| 1.0 | 42 | fedavg_sgd | 0.5077 | 0.4450 | 0.0370 | 0.4788 |
| 1.0 | 42 | flex_no_extra | 0.5410 | 0.4450 | 0.0723 | 0.4450 |
| 1.0 | 43 | fedavg_sgd | 0.5248 | 0.4175 | 0.0544 | 0.4333 |
| 1.0 | 43 | flex_no_extra | 0.5856 | 0.5050 | 0.0549 | 0.5343 |
| 1.0 | 44 | fedavg_sgd | 0.5160 | 0.4425 | 0.0423 | 0.4515 |
| 1.0 | 44 | flex_no_extra | 0.5285 | 0.4575 | 0.0445 | 0.4733 |
| 10.0 | 42 | fedavg_sgd | 0.5342 | 0.5000 | 0.0219 | 0.5180 |
| 10.0 | 42 | flex_no_extra | 0.4442 | 0.4025 | 0.0281 | 0.4093 |
| 10.0 | 43 | fedavg_sgd | 0.5475 | 0.5250 | 0.0132 | 0.5317 |
| 10.0 | 43 | flex_no_extra | 0.4597 | 0.4150 | 0.0299 | 0.4263 |
| 10.0 | 44 | fedavg_sgd | 0.5300 | 0.4675 | 0.0256 | 0.5080 |
| 10.0 | 44 | flex_no_extra | 0.4520 | 0.4200 | 0.0198 | 0.4313 |
