# Phase 2 - Training Setup Fix & Validation Plan

## Problem Statement
Current experimental results show:
- **Centralized accuracy**: 6.7% (random chance baseline: 1.6%)
- **FedAvg**: 3.7% | **FLEX**: 9.4% (2.5× improvement)

But the training regime is too weak to validate method effectiveness. Need to fix training before claiming breakthrough results.

## Root Causes Fixed

### 1. Insufficient Local Training Duration
- **Before**: local_epochs = 1-5 (~5 total epochs per client)
- **After**: local_epochs = 20 (100-200 total per client)
- **Impact**: Sufficient time for model to learn meaningful features

### 2. Delayed Alignment Constraints
- **Before**: alignment_warmup_epochs = 8 (kicks in by epoch 3)
- **After**: alignment_warmup_epochs = 15 (delays until model has features)
- **Impact**: Allow feature learning before alignment constraints apply

### 3. Increased Training Rounds
- **Before**: 10-50 rounds (too few with longer local training)
- **After**: 30-100 rounds (sufficient FL communication rounds)
- **Impact**: Enough federated iterations for model convergence

## Config Changes Made

### File: `flex_persona/config/training_config.py`
```python
rounds: int = 100            # was 50
local_epochs: int = 20       # was 1
```

### File: `flex_persona/training/alignment_aware_trainer.py`
```python
alignment_warmup_epochs: int = 15  # was 8
# alignment_weight: float = 0.01   # unchanged (already correct)
```

### File: `scripts/phase2_q1_validation.py`
```python
rounds = 30                  # was 10
local_epochs = 20            # was 5
```

## Validation Protocol

### Phase 1: Centralized Sanity Check
Run centralized training to verify baseline:
```bash
python scripts/test_alignment_training.py --centralized --epochs 30
```
**Expected**: ≥70-80% accuracy on FEMNIST

### Phase 2: Federated Training with Fixed Setup
```bash
python scripts/phase2_q1_validation.py --alpha 0.1 --rounds 30 --local-epochs 20
```
**Expected metrics**:
- Mean accuracy: ≥50% (significant improvement)
- Worst-client: ≥30% (FLEX focus)
- Fairness gap: Reduced vs FedAvg

### Phase 3: FLEX vs FedAvg Comparison
Compare the following configurations:
- **Config A**: FedAvg (aggregation_mode="fedavg", no alignment)
- **Config B**: FLEX (aggregation_mode="prototype", alignment enabled)

**Key Metrics**:
1. Mean accuracy (should be similar)
2. Worst-client accuracy (FLEX should win by 5-10%)
3. Fairness gap (FLEX should have smaller gap)
4. Standard deviation across clients

### Phase 4: Sensitivity Analysis
Test alignment warmup impact:
- alignment_warmup_epochs: 15, 20, 30 (how early is too early?)
- alignment_weight: 0.005, 0.01, 0.02 (strength of constraint)

**Goal**: Find sweet spot where alignment helps without hindering features

## Expected Progress Timeline

1. **Immediate** (fixes applied): Config updates
2. **Phase 1** (1-2 hours): Centralized training verification
3. **Phase 2** (2-4 hours): Federated training with new config
4. **Phase 3** (4-6 hours): Full comparison across all seeds
5. **Phase 4** (optional): Sensitivity analysis for publication

## Success Criteria

### Minimum (broken → acceptable)
- Centralized accuracy: 6.7% → 70%+
- FedAvg mean: 3.7% → 40%+
- FLEX mean: 9.4% → 50%+

### Target (acceptable → strong result)
- Centralized accuracy: 80%+
- FLEX worst-client: FedAvg worst + 10-15%
- FLEX fairness gap: Reduced by 20% vs FedAvg
- Consistent across all seeds (std < 3%)

### Breakthrough (strong → publishable)
All target metrics met + sensitivity analysis shows alignment strategy is robust

## Notes
- If centralized baseline fails, training/data pipeline is broken
- If FedAvg still low, may need learning rate tuning
- If FLEX still underperforms, method may need adjustment
- Worst-client metric is most important for FLEX claims
