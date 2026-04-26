# Experiment Execution Status

**Start Time**: 2026-03-21 (Session Continuing)  
**Status**: All 4 test lanes active and progressing  

---

## Active Experiment Lanes

### Lane 1: 10-Seed Comprehensive (Phase 2 - Core Evidence)
- **Terminal ID**: `eb10aa7a-5ad4-4948-abc7-9ab8a314dbec`
- **Progress**: 3/40 seeds running
- **Configuration**: 
  - High Het: 256 samples, 3 epochs, lr=0.01, 20 rounds
  - Low Het: 1000 samples, 1 epoch, lr=0.005, 30 rounds
  - Methods: FedAvg, Prototype
- **Expected Duration**: ~2-3 hours (GPU accelerated)
- **Status**: ✅ ACTIVE
- **Early Evidence**: 
  - Seed 11 FedAvg: mean=0.1389 (no collapse)
  - Seed 22 FedAvg: mean=0.0769 **[COLLAPSE]** ← Bifurcation detected!
- **Pass Criteria**: All 40 runs complete with collapse_rate for each method×regime

---

### Lane 2: Drift Measurement (Phase 3 - Mechanism Proof)
- **Terminal ID**: `aae2e7c0-39c2-4697-b7a4-8a66ac6f18e3`
- **Purpose**: Measure client model divergence per round
- **Configuration**: 3 seeds (11, 42, 55) in high heterogeneity regime
- **Expected Duration**: ~30-45 min
- **Status**: ✅ ACTIVE
- **Output**: Per-round drift metrics showing FedAvg vs Prototype stability
- **Pass Criteria**: Drift curves saved; FedAvg divergence > Prototype controlled divergence

---

### Lane 3: Communication Analysis (Phase 5 - Communication Evidence)
- **Terminal ID**: `884146ab-4822-49b2-91c4-ebb7321836f8`
- **Purpose**: Measure bytes per round for both methods
- **Configuration**: FedAvg vs Prototype in high heterogeneity
- **Expected Duration**: ~20-30 min
- **Status**: ✅ ACTIVE (initializing)
- **Output**: Theoretical + actual communication overhead
- **Pass Criteria**: Communication summary with per-round and cumulative bytes

---

### Lane 4: Ablation Study (Phase 4 - Component Isolation)
- **Terminal ID**: `455a14c8-2227-4404-939b-474f22c52394`
- **Purpose**: Test method components: full, no clustering, no guidance
- **Configuration**: 3 seeds (11, 42, 55) in high heterogeneity
- **Expected Duration**: ~45-60 min
- **Status**: ✅ ACTIVE (starting FedAvg baseline)
- **Output**: Per-variant mean accuracy and collapse rates
- **Pass Criteria**: Component contribution table showing what drives stability

---

## Key Metrics Being Collected

Per experiment run:
- ✅ Mean accuracy per seed
- ✅ Worst-client accuracy per seed
- ✅ Collapse detection (final mean < 0.10)
- ✅ Per-round convergence traces
- ✅ Communication bytes (c2s, s2c)
- ✅ Drift measurements
- ✅ Cluster quality metrics

---

## Expected Outputs

After completion:

### Phase 2 Output (10-seed)
```
experiments/comprehensive_10seed_results.json
└── fedavg_high_het: {num_seeds: 10, collapse_rate: X%, ...}
└── prototype_high_het: {num_seeds: 10, collapse_rate: Y%, ...}
└── fedavg_low_het: {num_seeds: 10, collapse_rate: Z%, ...}
└── prototype_low_het: {num_seeds: 10, collapse_rate: W%, ...}
```

### Phase 3-5 Outputs
```
outputs/reports/[drift|communication|ablation]_[timestamp].json
outputs/reports/[drift|communication|ablation]_[timestamp].md
```

---

## Live Progress Tracking

Check output tails:
```powershell
# Terminal 1 (10-seed)
cd C:\Users\HP\Projects\Federated
tail -f experiments/comprehensive_10seed_results.json

# Terminal 2-4 (other lanes)
tail -f outputs/reports/*latest*.json
```

---

## Definition of Done

✅ All 4 lanes execute without errors  
⏳ Phase 2 produces 40 complete runs with collapse statistics  
⏳ Phase 3 produces drift curves  
⏳ Phase 4 produces ablation table  
⏳ Phase 5 produces communication summary  

---

**Next**: Poll output in 5-10 minutes for lane progress updates.  
**Final**: All lanes complete → Merge results into master comparison table (Phase 8)  

---
