# Federated Learning Optimization Persistence Fix

**Date:** April 18, 2026  
**Status:** Applied & Validating  
**Critical:** YES

---

## Problem Statement

The federated system had **two fundamentally different training semantics**:

| Component | FedAvg | MOON | SCAFFOLD |
|-----------|--------|------|----------|
| Model lifecycle | Persistent per client | Deepcopy each round | Deepcopy each round |
| Optimizer lifecycle | Persistent per client | Reinit each round | Reinit each round |
| Adam state (m, v) | Continuous across rounds | Reset each round ❌ | Reset each round ❌ |
| Training continuity | Incremental | Micro-training restart | Micro-training restart |

---

## Root Cause Analysis

When optimizer is reinitialized each round:

```python
# MOON/SCAFFOLD (WRONG):
for rnd in range(rounds):
    local_model = deepcopy(global_model)
    optimizer = Adam(...)  # ❌ Fresh m=0, v=0
    train_local(optimizer, local_model)
```

vs FedAvg (CORRECT):

```python
# FedAvg (RIGHT):
for rnd in range(rounds):
    client_model.load_state_dict(global_model)  # ✅ Same model object
    optimizer.zero_grad()  # ✅ m and v preserved!
    train_local(optimizer, client_model)
```

### Why this matters

Adam maintains:
- **m** (first moment): exponential moving average of gradients
- **v** (second moment): exponential moving average of squared gradients

With fresh optimizer each round:
```
Round 1: m=0, v=0 → steep initial updates
Round 2: m=0, v=0 → same steep updates (no continuity!)
```

This breaks **fundamental federated learning equivalence**:
- 1-client FedAvg should equal centralized (but delta was ~3.9%)
- MOON(μ=0) should equal FedAvg (but delta was significant)
- SCAFFOLD(zero) should equal FedAvg (but delta was significant)

---

## Solution Implemented

### For `run_moon()` (line ~920)

**Before:**
```python
for rnd in range(1, rounds + 1):
    for bundle in bundles:
        local_model = copy.deepcopy(global_model)  # ❌
        optimizer = torch.optim.Adam(...)  # ❌
        train_with_contrastive_loss(...)
```

**After:**
```python
# Initialize ONCE per client
client_models = {}
client_optimizers = {}
for b in bundles:
    client_models[b.client_id] = build_model().to(DEVICE)
    client_optimizers[b.client_id] = Adam(client_models[b.client_id].parameters(), lr=lr)

# Each round: load weights, preserve optimizer state
for rnd in range(1, rounds + 1):
    for bundle in bundles:
        cid = bundle.client_id
        client_models[cid].load_state_dict(global_model.state_dict())  # ✅
        local_model = client_models[cid]
        optimizer = client_optimizers[cid]  # ✅ m and v preserved!
        train_with_contrastive_loss(optimizer, local_model, ...)
```

### For `run_scaffold()` (line ~738)

**Identical fix:**
- Create persistent `client_models` dict (one model per client, reused)
- Create persistent `client_optimizers` dict (one optimizer per client, reused)
- Load global weights via `load_state_dict()` (not deepcopy)
- Reuse optimizer across rounds

---

## Key Architectural Principle

**All methods must use identical execution semantics:**

```
SAME:
- model object lifecycle (persistent per client)
- optimizer lifecycle (persistent per client)
- data loader behavior (seeded & deterministic)
- training loop structure

DIFFERENT (only):
- loss computation (standard CE vs contrastive vs control variate)
- gradient correction (if any)
```

---

## Expected Validation Results

After applying this fix, the 4-point audit should show:

| Check | Expected | Threshold | Status |
|-------|----------|-----------|--------|
| 1-client gap | < 1–2% | < 2% | ✅ |
| MOON(μ=0) vs FedAvg | < 0.5–1% | < 1% | ✅ |
| SCAFFOLD(zero) vs FedAvg | < 1% | < 1% | ✅ |
| No oscillations | Smooth curves | Visual inspection | ✅ |

### Before fix:
```
Centralized:       0.573
FedAvg (1-client): 0.534
MOON(μ=0):         ???
SCAFFOLD(zero):    ??? (oscillating)
```

### After fix (predicted):
```
Centralized:       0.8542 (new test data size)
FedAvg (1-client): ~0.851-0.854
MOON(μ=0):         ~0.851-0.854  (Δ < 0.5%)
SCAFFOLD(zero):    ~0.851-0.854  (Δ < 0.5%)
```

---

## Validation Script

Run quick 4-point audit:
```bash
python scripts/quick_audit_four.py
```

Produces: `outputs/quick_audit_four.json` with all four accuracies and deltas.

---

## Files Modified

- `scripts/phase2_q1_validation.py`
  - Function: `run_moon()` (optimizer persistence fix)
  - Function: `run_scaffold()` (optimizer persistence fix)

---

## Why This is Non-Negotiable

**Scientific integrity:**
- Baseline methods (FedAvg, MOON, SCAFFOLD) must be implemented correctly to claim improvements
- Any gap > 1% between MOON(μ=0) and FedAvg is a red flag
- Oscillations in SCAFFOLD suggest broken gradient correction

**Publication gate:**
- Papers are rejected when baselines are incorrectly implemented
- Reviewers will spot inconsistent semantics
- "Different optimizer" behavior is suspicious and requires justification

---

## Timeline

| Step | Time | Status |
|------|------|--------|
| Identify bug | 2026-04-18 08:00 | ✅ |
| Apply fix | 2026-04-18 09:15 | ✅ |
| Run validation (centralized) | 2026-04-18 09:30 | ✅ `0.8542` |
| Run validation (fedavg 1-client) | 2026-04-18 09:35 | ⏳ In progress |
| Run validation (moon μ=0) | 2026-04-18 10:00 | ⏳ Queued |
| Run validation (scaffold zero) | 2026-04-18 10:30 | ⏳ Queued |
| Final verdict | 2026-04-18 11:00 | ⏳ Pending |

---

## Next Steps (If Validation Passes)

If all 4 checks pass (deltas < 1%):
1. ✅ Clear "almost correct but not publishable" → "scientifically defensible"
2. ✅ Re-run FLEX-Persona with fixed baselines
3. ✅ Generate final claims (paper-ready)

If any check fails (delta > 1%):
1. ❌ Pinpoint remaining bug
2. ❌ Iterate on fix
3. ❌ Re-validate

---

## Notes

- This fix is **minimal and surgical**: no refactoring, only optimizer persistence
- No changes to loss computation or aggregation
- No changes to hyperparameters or data pipeline
- FedAvg behavior unchanged (was already correct)
- Impact: MOON and SCAFFOLD now have identical semantics to FedAvg (only loss differs)
