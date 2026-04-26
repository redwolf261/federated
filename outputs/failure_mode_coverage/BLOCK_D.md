# Block D: Heterogeneity Sweep Execution Log

**Objective:** Determine whether FLEX-Persona's performance advantage is caused by its ability to handle non-IID data (heterogeneity).

**Configuration:**
- Dataset: CIFAR-10
- Clients: 10
- Seeds: [42, 43, 44]
- Rounds: 20
- Local epochs: 5
- Cluster-aware epochs: 0
- Batch size: 64
- LR: 0.001
- Samples per client: 2000
- Alpha values: [0.05, 0.1, 0.5, 1.0, 10.0]
- Methods: flex_no_extra, fedavg_sgd

**Total planned runs:** 30 (5 alphas × 3 seeds × 2 methods)

---
# Block D Execution Log

**Started:** 2026-04-25 23:20:48
**Total planned runs:** 30
**Already completed:** 0
**Remaining:** 30

## Planned Runs vs Skipped Runs

| Alpha | Seed | Method | Status |
|-------|------|--------|--------|
| 0.05 | 42 | flex_no_extra | RUN |
| 0.05 | 42 | fedavg_sgd | RUN |
| 0.05 | 43 | flex_no_extra | RUN |
| 0.05 | 43 | fedavg_sgd | RUN |
| 0.05 | 44 | flex_no_extra | RUN |
| 0.05 | 44 | fedavg_sgd | RUN |
| 0.1 | 42 | flex_no_extra | RUN |
| 0.1 | 42 | fedavg_sgd | RUN |
| 0.1 | 43 | flex_no_extra | RUN |
| 0.1 | 43 | fedavg_sgd | RUN |
| 0.1 | 44 | flex_no_extra | RUN |
| 0.1 | 44 | fedavg_sgd | RUN |
| 0.5 | 42 | flex_no_extra | RUN |
| 0.5 | 42 | fedavg_sgd | RUN |
| 0.5 | 43 | flex_no_extra | RUN |
| 0.5 | 43 | fedavg_sgd | RUN |
| 0.5 | 44 | flex_no_extra | RUN |
| 0.5 | 44 | fedavg_sgd | RUN |
| 1.0 | 42 | flex_no_extra | RUN |
| 1.0 | 42 | fedavg_sgd | RUN |
| 1.0 | 43 | flex_no_extra | RUN |
| 1.0 | 43 | fedavg_sgd | RUN |
| 1.0 | 44 | flex_no_extra | RUN |
| 1.0 | 44 | fedavg_sgd | RUN |
| 10.0 | 42 | flex_no_extra | RUN |
| 10.0 | 42 | fedavg_sgd | RUN |
| 10.0 | 43 | flex_no_extra | RUN |
| 10.0 | 43 | fedavg_sgd | RUN |
| 10.0 | 44 | flex_no_extra | RUN |
| 10.0 | 44 | fedavg_sgd | RUN |

**Runs to execute:** 30

## Per-Run Execution Logs

### Run 1/30
- **Alpha:** 0.05
- **Seed:** 42
- **Method:** flex_no_extra
- **Started:** 23:20:48
- **Completed:** 23:23:24
- **Duration:** 155.8s
- **Mean accuracy:** 0.8759
- **Worst accuracy:** 0.7425
- **Std:** 0.0880
- **P10:** 0.7448

### Run 2/30
- **Alpha:** 0.05
- **Seed:** 42
- **Method:** fedavg_sgd
- **Started:** 23:23:24
- **Completed:** 23:25:00
- **Duration:** 96.7s
- **Mean accuracy:** 0.4248
- **Worst accuracy:** 0.0625
- **Std:** 0.2914
- **P10:** 0.1030

### Run 3/30
- **Alpha:** 0.05
- **Seed:** 43
- **Method:** flex_no_extra
- **Started:** 23:25:00
- **Completed:** 23:27:33
- **Duration:** 152.4s
- **Mean accuracy:** 0.8007
- **Worst accuracy:** 0.5714
- **Std:** 0.0965
- **P10:** 0.7209

### Run 4/30
- **Alpha:** 0.05
- **Seed:** 43
- **Method:** fedavg_sgd
- **Started:** 23:27:33
- **Completed:** 23:29:05
- **Duration:** 92.7s
- **Mean accuracy:** 0.3979
- **Worst accuracy:** 0.0700
- **Std:** 0.2108
- **P10:** 0.1356

### Run 5/30
- **Alpha:** 0.05
- **Seed:** 44
- **Method:** flex_no_extra
- **Started:** 23:29:05
- **Completed:** 23:31:52
- **Duration:** 166.5s
- **Mean accuracy:** 0.9172
- **Worst accuracy:** 0.7325
- **Std:** 0.0939
- **P10:** 0.7752

### Run 6/30
- **Alpha:** 0.05
- **Seed:** 44
- **Method:** fedavg_sgd
- **Started:** 23:31:52
- **Completed:** 23:33:52
- **Duration:** 119.8s
- **Mean accuracy:** 0.1523
- **Worst accuracy:** 0.0000
- **Std:** 0.2023
- **P10:** 0.0000

### Run 7/30
- **Alpha:** 0.1
- **Seed:** 42
- **Method:** flex_no_extra
- **Started:** 23:33:52
- **Completed:** 23:36:41
- **Duration:** 169.4s
- **Mean accuracy:** 0.7867
- **Worst accuracy:** 0.6111
- **Std:** 0.1001
- **P10:** 0.6169

### Run 8/30
- **Alpha:** 0.1
- **Seed:** 42
- **Method:** fedavg_sgd
- **Started:** 23:36:41
- **Completed:** 23:38:27
- **Duration:** 106.2s
- **Mean accuracy:** 0.4159
- **Worst accuracy:** 0.1475
- **Std:** 0.1719
- **P10:** 0.1722

### Run 9/30
- **Alpha:** 0.1
- **Seed:** 43
- **Method:** flex_no_extra
- **Started:** 23:38:27
- **Completed:** 23:41:39
- **Duration:** 191.4s
- **Mean accuracy:** 0.7691
- **Worst accuracy:** 0.5938
- **Std:** 0.1065
- **P10:** 0.6421

### Run 10/30
- **Alpha:** 0.1
- **Seed:** 43
- **Method:** fedavg_sgd
- **Started:** 23:41:39
- **Completed:** 23:43:33
- **Duration:** 114.6s
- **Mean accuracy:** 0.4086
- **Worst accuracy:** 0.1850
- **Std:** 0.1649
- **P10:** 0.1867

### Run 11/30
- **Alpha:** 0.1
- **Seed:** 44
- **Method:** flex_no_extra
- **Started:** 23:43:33
- **Completed:** 23:46:44
- **Duration:** 190.8s
- **Mean accuracy:** 0.8002
- **Worst accuracy:** 0.6525
- **Std:** 0.1094
- **P10:** 0.6638

### Run 12/30
- **Alpha:** 0.1
- **Seed:** 44
- **Method:** fedavg_sgd
- **Started:** 23:46:44
- **Completed:** 23:48:37
- **Duration:** 112.9s
- **Mean accuracy:** 0.4530
- **Worst accuracy:** 0.2375
- **Std:** 0.1578
- **P10:** 0.2600

### Run 13/30
- **Alpha:** 0.5
- **Seed:** 42
- **Method:** flex_no_extra
- **Started:** 23:48:37
- **Completed:** 23:51:57
- **Duration:** 199.7s
- **Mean accuracy:** 0.6562
- **Worst accuracy:** 0.5475
- **Std:** 0.0500
- **P10:** 0.6015

### Run 14/30
- **Alpha:** 0.5
- **Seed:** 42
- **Method:** fedavg_sgd
- **Started:** 23:51:57
- **Completed:** 23:53:55
- **Duration:** 118.6s
- **Mean accuracy:** 0.5089
- **Worst accuracy:** 0.2750
- **Std:** 0.1302
- **P10:** 0.3133

### Run 15/30
- **Alpha:** 0.5
- **Seed:** 43
- **Method:** flex_no_extra
- **Started:** 23:53:55
- **Completed:** 23:57:11
- **Duration:** 195.6s
- **Mean accuracy:** 0.6710
- **Worst accuracy:** 0.5550
- **Std:** 0.1004
- **P10:** 0.5595

### Run 16/30
- **Alpha:** 0.5
- **Seed:** 43
- **Method:** fedavg_sgd
- **Started:** 23:57:11
- **Completed:** 23:59:30
- **Duration:** 139.4s
- **Mean accuracy:** 0.4715
- **Worst accuracy:** 0.3500
- **Std:** 0.0952
- **P10:** 0.3680

### Run 17/30
- **Alpha:** 0.5
- **Seed:** 44
- **Method:** flex_no_extra
- **Started:** 23:59:30
- **Completed:** 00:02:48
- **Duration:** 198.1s
- **Mean accuracy:** 0.5996
- **Worst accuracy:** 0.4725
- **Std:** 0.0681
- **P10:** 0.5040

### Run 18/30
- **Alpha:** 0.5
- **Seed:** 44
- **Method:** fedavg_sgd
- **Started:** 00:02:48
- **Completed:** 00:05:04
- **Duration:** 135.2s
- **Mean accuracy:** 0.5024
- **Worst accuracy:** 0.4000
- **Std:** 0.0596
- **P10:** 0.4158

### Run 19/30
- **Alpha:** 1.0
- **Seed:** 42
- **Method:** flex_no_extra
- **Started:** 00:05:04
- **Completed:** 00:08:54
- **Duration:** 230.2s
- **Mean accuracy:** 0.5410
- **Worst accuracy:** 0.4450
- **Std:** 0.0723
- **P10:** 0.4450

### Run 20/30
- **Alpha:** 1.0
- **Seed:** 42
- **Method:** fedavg_sgd
- **Started:** 00:08:54
- **Completed:** 00:11:15
- **Duration:** 141.2s
- **Mean accuracy:** 0.5077
- **Worst accuracy:** 0.4450
- **Std:** 0.0370
- **P10:** 0.4788

### Run 21/30
- **Alpha:** 1.0
- **Seed:** 43
- **Method:** flex_no_extra
- **Started:** 00:11:15
- **Completed:** 00:14:43
- **Duration:** 208.1s
- **Mean accuracy:** 0.5856
- **Worst accuracy:** 0.5050
- **Std:** 0.0549
- **P10:** 0.5343

### Run 22/30
- **Alpha:** 1.0
- **Seed:** 43
- **Method:** fedavg_sgd
- **Started:** 00:14:43
- **Completed:** 00:16:54
- **Duration:** 130.6s
- **Mean accuracy:** 0.5248
- **Worst accuracy:** 0.4175
- **Std:** 0.0544
- **P10:** 0.4333

### Run 23/30
- **Alpha:** 1.0
- **Seed:** 44
- **Method:** flex_no_extra
- **Started:** 00:16:54
- **Completed:** 00:20:23
- **Duration:** 209.7s
- **Mean accuracy:** 0.5285
- **Worst accuracy:** 0.4575
- **Std:** 0.0445
- **P10:** 0.4733

### Run 24/30
- **Alpha:** 1.0
- **Seed:** 44
- **Method:** fedavg_sgd
- **Started:** 00:20:23
- **Completed:** 00:22:34
- **Duration:** 130.2s
- **Mean accuracy:** 0.5160
- **Worst accuracy:** 0.4425
- **Std:** 0.0423
- **P10:** 0.4515

### Run 25/30
- **Alpha:** 10.0
- **Seed:** 42
- **Method:** flex_no_extra
- **Started:** 00:22:34
- **Completed:** 00:26:04
- **Duration:** 210.0s
- **Mean accuracy:** 0.4442
- **Worst accuracy:** 0.4025
- **Std:** 0.0281
- **P10:** 0.4093

### Run 26/30
- **Alpha:** 10.0
- **Seed:** 42
- **Method:** fedavg_sgd
- **Started:** 00:26:04
- **Completed:** 00:28:13
- **Duration:** 129.6s
- **Mean accuracy:** 0.5342
- **Worst accuracy:** 0.5000
- **Std:** 0.0219
- **P10:** 0.5180

### Run 27/30
- **Alpha:** 10.0
- **Seed:** 43
- **Method:** flex_no_extra
- **Started:** 00:28:13
- **Completed:** 00:31:42
- **Duration:** 209.0s
- **Mean accuracy:** 0.4597
- **Worst accuracy:** 0.4150
- **Std:** 0.0299
- **P10:** 0.4263

### Run 28/30
- **Alpha:** 10.0
- **Seed:** 43
- **Method:** fedavg_sgd
- **Started:** 00:31:42
- **Completed:** 00:33:48
- **Duration:** 126.2s
- **Mean accuracy:** 0.5475
- **Worst accuracy:** 0.5250
- **Std:** 0.0132
- **P10:** 0.5317

### Run 29/30
- **Alpha:** 10.0
- **Seed:** 44
- **Method:** flex_no_extra
- **Started:** 00:33:48
- **Completed:** 00:37:32
- **Duration:** 223.9s
- **Mean accuracy:** 0.4520
- **Worst accuracy:** 0.4200
- **Std:** 0.0198
- **P10:** 0.4313

### Run 30/30
- **Alpha:** 10.0
- **Seed:** 44
- **Method:** fedavg_sgd
- **Started:** 00:37:32
- **Completed:** 00:39:55
- **Duration:** 143.1s
- **Mean accuracy:** 0.5300
- **Worst accuracy:** 0.4675
- **Std:** 0.0256
- **P10:** 0.5080

## Execution Summary

- **Total runs executed:** 30
- **Total elapsed time:** 4747.5s (79.1 min)
- **Finished:** 2026-04-26 00:39:55

---

## Aggregated Results

### Performance by Alpha (Mean Across Seeds)

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

---

## Pattern Classification

**Pattern detected:** `heterogeneity_dependent`

**Interpretation:** FLEX advantage decreases as heterogeneity decreases. This confirms that FLEX's benefit is specifically tied to its ability to handle non-IID data.

---

## Critical Validation Checks

1. **Alpha = 10 (near-IID):** FedAvg = 0.5372, FLEX = 0.4520, gap = -0.0852
   - ✅ FedAvg improves monotonically with alpha (0.3250 → 0.5372), as expected.
   - ✅ The gap flipped because FLEX became worse under homogeneity, not because FedAvg failed.

2. **Gap shrinkage:** Low-alpha gap = +0.5396, High-alpha gap = -0.0852
   - ✅ Gap decreased monotonically as alpha increased.

3. **Worst-case improvement at low alpha:** +0.6380
   - ✅ FLEX massively improves worst-client fairness in high-heterogeneity regimes.

---

## Mechanism-Level Explanation

| Regime | Winner | Why |
|---|---|---|
| High heterogeneity (α ≤ 0.5) | **FLEX** | Cross-client distributions differ; clustering + alignment resolves misalignment |
| Moderate (α ≈ 1.0) | ~Tie | Some heterogeneity remains, but FedAvg can still cope |
| Near-IID (α = 10) | **FedAvg** | All clients approximate the same distribution; global model = optimal. FLEX's clustering constraints become unnecessary interference |

**Analogy:** Non-IID → students studying different chapters → coordination helps. IID → everyone studying the same chapter → coordination becomes interference.

---

## Conclusion

FLEX-Persona is **conditionally optimal**: it provides substantial gains under heterogeneous (non-IID) data by mitigating cross-client distribution misalignment, but introduces unnecessary representation constraints under near-IID conditions, where standard parameter averaging (FedAvg) becomes optimal and outperforms FLEX.

**Key contribution:** FLEX does not just improve accuracy — it solves fairness imbalance in federated learning. At α = 0.05, worst-client accuracy improves by **+0.6380**, demonstrating that representation-based collaboration specifically protects disadvantaged clients.

**One-line research claim:** *FLEX-Persona is a heterogeneity-aware federated learning method that significantly improves performance and worst-case client outcomes under non-IID conditions, but is outperformed by standard parameter averaging in near-IID regimes due to unnecessary representation constraints.*
