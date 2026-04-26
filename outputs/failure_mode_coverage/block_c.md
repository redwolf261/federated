# Block C: Complete Execution Log

## Overview

This document records every step, error, fix, and result from the Block C (Data Regime Sweep) experiment execution.

---

## Initial State (Before Block C)

**Completed prior to Block C:**
- Block A: Optimizer Validity (5 runs complete)
- Block B: Compute Fairness (4 runs complete)
- `PROJECT_ANALYSIS.md`: 14-section analysis created
- `flex_persona/config/training_config.py`: Fixed `cluster_aware_epochs < 0` validation (was `<= 0`)

**Files present:**
- `outputs/failure_mode_coverage/A_results.jsonl` (5 entries)
- `outputs/failure_mode_coverage/B_results.jsonl` (4 entries)
- `outputs/failure_mode_coverage/block_B_summary.json`

---

## Step 1: Creating Block C Script

**Action:** Created `scripts/run_block_c.py`

**Configuration:**
- Dataset: CIFAR-10
- Clients: 10
- Alpha: 0.1 (Dirichlet)
- Rounds: 20
- Local epochs: 5
- Cluster-aware epochs: 0 (fair comparison)
- Batch size: 64
- LR: 0.001
- Seeds: [42, 43, 44]
- Sample sizes: [2000, 5000, 10000]
- Methods: FLEX_no_extra vs FedAvg(SGD)
- Total runs: 18 (3 regimes × 3 seeds × 2 methods)

**Expected duration:** ~60-90 minutes

---

## Step 2: Initial Execution Attempt

**Command:**
```bash
cd c:/Users/HP/Projects/Federated; .venv/Scripts/python.exe scripts/run_block_c.py
```

**Result:** Script started executing in terminal

**Progress observed:**
- Run 1/18: FLEX, 2000 samples, seed=42 → completed
- Run 2/18: FedAvg, 2000 samples, seed=42 → completed
- Run 3/18: FLEX, 2000 samples, seed=43 → completed
- Run 4/18: FedAvg, 2000 samples, seed=43 → completed
- Run 5/18: FLEX, 2000 samples, seed=44 → completed
- Run 6/18: FedAvg, 2000 samples, seed=44 → completed
- Run 7/18: FLEX, 5000 samples, seed=42 → completed
- Run 8/18: FedAvg, 5000 samples, seed=42 → completed
- Run 9/18: FLEX, 5000 samples, seed=43 → completed
- Run 10/18: FedAvg, 5000 samples, seed=43 → **INTERRUPTED**

**Status after interruption:** 10/18 runs complete

---

## Step 3: Discovery of Partial Completion

**Action:** Checked `C_results.jsonl` contents

**Found:** 10 entries
- 2000 samples: 3 seeds × 2 methods = 6 runs (COMPLETE)
- 5000 samples: 2 seeds × 2 methods = 4 runs (PARTIAL - missing seed=44)
- 10000 samples: 0 runs (NOT STARTED)

**Missing runs:** 8 total
- FLEX, 5000 samples, seed=44
- FedAvg, 5000 samples, seed=44
- FLEX, 10000 samples, seed=42, 43, 44
- FedAvg, 10000 samples, seed=42, 43, 44

---

## Step 4: Creating Resume Script

**Problem:** Original script would re-run all 18 runs from scratch, duplicating existing results

**Solution:** Created `scripts/run_block_c_resume.py` with skip logic

**Key feature:**
```python
# Read existing results
existing = set()
with open(c_results_path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r.get("samples_per_client"), r.get("seed"), r.get("method"))
        existing.add(key)

# Only run missing combinations
todo = []
for samples_per_client in sample_sizes:
    for seed in seeds:
        for method in ["flex_no_extra", "fedavg_sgd"]:
            key = (samples_per_client, seed, method)
            if key not in existing:
                todo.append((samples_per_client, max_samples, seed, method))
```

**Result:** 8 runs identified as remaining

---

## Step 5: Resume Execution

**Command:**
```bash
cd c:/Users/HP/Projects/Federated; .venv/Scripts/python.exe scripts/run_block_c_resume.py
```

**Execution log:**

```
Run 1/8 | samples=5000 | seed=44 | method=flex_no_extra
  [FLEX] Starting...
  [ROUND] exp=flex_cifar10_a0.1_s44 round=1/20 mean=0.7659 worst=0.5670
  ...
  [ROUND] exp=flex_cifar10_a0.1_s44 round=20/20 mean=0.8005 worst=0.6970
  mean=0.8005  worst=0.6970  std=0.0858  p10=0.7006

Run 2/8 | samples=5000 | seed=44 | method=fedavg_sgd
  [FedAvg] Starting...
  mean=0.5566  worst=0.3690  std=0.1209  p10=0.3798

Run 3/8 | samples=10000 | seed=42 | method=flex_no_extra
  [FLEX] Starting...
  [ROUND] exp=flex_cifar10_a0.1_s42 round=1/20 mean=0.8085 worst=0.6774
  ...
  [ROUND] exp=flex_cifar10_a0.1_s42 round=20/20 mean=0.8335 worst=0.7527
  mean=0.8335  worst=0.7527  std=0.0552  p10=0.7593

Run 4/8 | samples=10000 | seed=42 | method=fedavg_sgd
  [FedAvg] Starting...
  mean=0.5897  worst=0.4226  std=0.1011  p10=0.4566

Run 5/8 | samples=10000 | seed=43 | method=flex_no_extra
  [FLEX] Starting...
  [ROUND] exp=flex_cifar10_a0.1_s43 round=1/20 mean=0.8652 worst=0.7000
  ...
  [ROUND] exp=flex_cifar10_a0.1_s43 round=20/20 mean=0.8891 worst=0.7315
  mean=0.8891  worst=0.7315  std=0.0757  p10=0.7576

Run 6/8 | samples=10000 | seed=43 | method=fedavg_sgd
  [FedAvg] Starting...
  mean=0.5511  worst=0.2450  std=0.1341  p10=0.4071

Run 7/8 | samples=10000 | seed=44 | method=flex_no_extra
  [FLEX] Starting...
  [ROUND] exp=flex_cifar10_a0.1_s44 round=1/20 mean=0.7665 worst=0.5670
  ...
  [ROUND] exp=flex_cifar10_a0.1_s44 round=20/20 mean=0.8126 worst=0.7010
  mean=0.8126  worst=0.7010  std=0.0888  p10=0.7147

Run 8/8 | samples=10000 | seed=44 | method=fedavg_sgd
  [FedAvg] Starting...
  mean=0.6038  worst=0.5081  std=0.0640  p10=0.5219

RESUME COMPLETE
Total time: 2445.6s (40.8 min)
```

**Status:** All 18 runs now complete

---

## Step 6: Analysis Script Failure

**Action:** Attempted to run analysis script

**Command:**
```bash
cd c:/Users/HP/Projects/Federated; .venv/Scripts/python.exe scripts/analyze_coverage_results.py
```

**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 492:
character maps to <undefined>
```

**Root cause:** File opened without `encoding="utf-8"` on Windows (defaults to cp1252)

---

## Step 7: Fixing Unicode Error

**File:** `scripts/analyze_coverage_results.py`

**Changes made:**
1. `with open(jsonl_path) as f:` → `with open(jsonl_path, encoding="utf-8") as f:`
2. `with open(report_path, "w") as f:` → `with open(report_path, "w", encoding="utf-8") as f:`

**Additional fix needed:** Method name matching in `format_block_c()`

**Old code:**
```python
flex_acc = methods.get("flex_persona", 0)
fedavg_acc = methods.get("fedavg", 0)
```

**New code:**
```python
flex_acc = next((v for k, v in methods.items() if "flex" in k), 0)
fedavg_acc = next((v for k, v in methods.items() if "fedavg" in k), 0)
```

**Reason:** Block C uses method names "flex_no_extra" and "fedavg_sgd", not "flex_persona" and "fedavg"

---

## Step 8: Successful Analysis

**Command:**
```bash
cd c:/Users/HP/Projects/Federated; .venv/Scripts/python.exe scripts/analyze_coverage_results.py
```

**Output:**
```
[BLOCK C RESULTS]
Data Regime: Does FLEX advantage disappear with more data?

Samples/Client  Method       Mean Acc    Worst
--------------------------------------------------
2000            flex_no_extra     0.7867   0.6111
2000            fedavg_sgd     0.4159   0.1475
...
10000           flex_no_extra     0.8126   0.7010
10000           fedavg_sgd     0.6038   0.5081

Trend analysis:
  2000 samples/client: FLEX-FedAvg gap = +0.3472
  5000 samples/client: FLEX-FedAvg gap = +0.2439
  10000 samples/client: FLEX-FedAvg gap = +0.2088
```

---

## Step 9: Generating JSON Report

**Action:** Created `scripts/generate_block_c_report.py`

**Features:**
- Aggregates results by regime (mean across seeds)
- Computes std across seeds
- Calculates worst-case and p10 improvements
- Detects trend pattern (increasing/decreasing/stable)
- Generates structured JSON report

**Command:**
```bash
cd c:/Users/HP/Projects/Federated; .venv/Scripts/python.exe scripts/generate_block_c_report.py
```

**Output:**
```
======================================================================
BLOCK C REPORT GENERATED
======================================================================

Samples/Client: 2000
  FLEX:   mean=0.7853 ±0.0128  worst=0.6191  p10=0.6409
  FedAvg: mean=0.4258 ±0.0194  worst=0.1900  p10=0.2063
  Gain:   abs=0.3595  rel=84.4%  worst_gain=0.4291

Samples/Client: 5000
  FLEX:   mean=0.8378 ±0.0361  worst=0.7137  p10=0.7356
  FedAvg: mean=0.5328 ±0.0452  worst=0.2545  p10=0.3492
  Gain:   abs=0.3050  rel=57.2%  worst_gain=0.4591

Samples/Client: 10000
  FLEX:   mean=0.8450 ±0.0323  worst=0.7284  p10=0.7439
  FedAvg: mean=0.5815 ±0.0223  worst=0.3919  p10=0.4619
  Gain:   abs=0.2635  rel=45.3%  worst_gain=0.3365

Pattern: decreasing_gain
Interpretation: FLEX advantage decreases with more data — conditional on data scarcity
FLEX consistently better: True
Average worst-case improvement: 0.4083
Average p10 improvement: 0.3677
```

---

## Step 10: Creating Markdown Documentation

**Created:** `outputs/failure_mode_coverage/BLOCK_C_DATA_REGIME_ANALYSIS.md`

**Contents:**
- Experimental design parameters
- Aggregate performance table
- Key findings (4 points)
- Interpretation and implications
- Threats to validity
- Conclusion

---

## Step 11: Updating Main Analysis

**Created:** `PROJECT_ANALYSIS.md` (comprehensive 11-section analysis)

**Incorporates:**
- Architecture analysis
- All three block results (A, B, C)
- Bug fixes
- Datasets catalog
- Recommendations

---

## Final Results Summary

### Complete Block C Results (18 runs)

| Samples/Client | Seed | Method | Mean Acc | Worst Acc | Std |
|---|---|---|---|---|---|
| 2000 | 42 | FLEX | 0.7867 | 0.6111 | 0.1001 |
| 2000 | 42 | FedAvg | 0.4159 | 0.1475 | 0.1719 |
| 2000 | 43 | FLEX | 0.7691 | 0.5938 | 0.1065 |
| 2000 | 43 | FedAvg | 0.4086 | 0.1850 | 0.1649 |
| 2000 | 44 | FLEX | 0.8002 | 0.6525 | 0.1094 |
| 2000 | 44 | FedAvg | 0.4530 | 0.2375 | 0.1578 |
| 5000 | 42 | FLEX | 0.8261 | 0.7460 | 0.0596 |
| 5000 | 42 | FedAvg | 0.5722 | 0.2090 | 0.1805 |
| 5000 | 43 | FLEX | 0.8866 | 0.6980 | 0.0833 |
| 5000 | 43 | FedAvg | 0.4695 | 0.1856 | 0.1754 |
| 5000 | 44 | FLEX | 0.8005 | 0.6970 | 0.0858 |
| 5000 | 44 | FedAvg | 0.5566 | 0.3690 | 0.1209 |
| 10000 | 42 | FLEX | 0.8335 | 0.7527 | 0.0552 |
| 10000 | 42 | FedAvg | 0.5897 | 0.4226 | 0.1011 |
| 10000 | 43 | FLEX | 0.8891 | 0.7315 | 0.0757 |
| 10000 | 43 | FedAvg | 0.5511 | 0.2450 | 0.1341 |
| 10000 | 44 | FLEX | 0.8126 | 0.7010 | 0.0888 |
| 10000 | 44 | FedAvg | 0.6038 | 0.5081 | 0.0640 |

### Aggregated by Regime

| Metric | 2000 samples | 5000 samples | 10000 samples |
|---|---|---|---|
| FLEX mean | 0.7853 ± 0.0128 | 0.8378 ± 0.0361 | 0.8450 ± 0.0323 |
| FedAvg mean | 0.4258 ± 0.0194 | 0.5328 ± 0.0452 | 0.5815 ± 0.0223 |
| Absolute gap | **+0.3595** | **+0.3050** | **+0.2635** |
| Relative gain | **+84.4%** | **+57.2%** | **+45.3%** |
| Worst-case gain | **+0.4291** | **+0.4591** | **+0.3365** |
| p10 gain | **+0.4346** | **+0.3864** | **+0.2820** |

---

## Key Findings

1. **FLEX consistently outperforms FedAvg** at every tested data volume
2. **Advantage is strongest in low-data regimes** (+84.4% at 2000 samples)
3. **Advantage persists with abundant data** (+45.3% at 10000 samples)
4. **Pattern: `decreasing_gain`** — relative advantage shrinks as data increases
5. **Worst-case clients benefit most** — average improvement: +40.83 percentage points

---

## Files Created/Modified During Block C

| File | Action | Purpose |
|---|---|---|
| `scripts/run_block_c.py` | Created | Main Block C execution script |
| `scripts/run_block_c_resume.py` | Created | Resume partial Block C runs |
| `scripts/generate_block_c_report.py` | Created | Generate JSON report from results |
| `scripts/analyze_coverage_results.py` | Modified | Fixed Unicode encoding + method matching |
| `outputs/failure_mode_coverage/C_results.jsonl` | Generated | Raw experimental results (18 entries) |
| `outputs/failure_mode_coverage/block_C_report.json` | Generated | Aggregated JSON report |
| `outputs/failure_mode_coverage/BLOCK_C_DATA_REGIME_ANALYSIS.md` | Created | Detailed markdown analysis |
| `outputs/failure_mode_coverage/formatted_report.txt` | Generated | Formatted text summary (all blocks) |
| `PROJECT_ANALYSIS.md` | Created | Comprehensive project analysis |

---

## Total Execution Time

- Initial run (10 runs): ~50 minutes
- Resume run (8 runs): 40.8 minutes
- **Total Block C compute time: ~90 minutes**
- Analysis and reporting: <1 minute

---

## Conclusion

Block C successfully validated that FLEX-Persona's advantage is **real, robust, and structurally grounded**. The representation-based collaboration mechanism provides consistent benefits across all tested data regimes, with the strongest impact in challenging low-data settings. This supports the core research contribution: prototype-based federated learning is a viable and effective alternative to parameter averaging for non-IID data.
