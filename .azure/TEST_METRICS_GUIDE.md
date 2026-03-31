# Centralized Baseline Test - Metrics Being Collected

## What We're Testing
After updating the config:
- `local_epochs`: 1 → 20
- `alignment_warmup_epochs`: 8 → 15
- More data: 5000 samples (70/30 train/val split)
- More epochs: 30 (vs 5 in old test)

## Critical Metrics to Analyze

### 1. Learning Progress (Train vs Val)
```
✓ GOOD: train ↑ steadily, val ↑ steadily (slower)
✗ BAD: train ↑, val flat (overfitting)
✗ BAD: both flat (no learning)
```

### 2. Alignment Behavior
```
✓ GOOD: align_score improves gradually, doesn't saturate at 0.99
✗ BAD: align_score saturates by epoch 10 (too early, too aggressive)
```

### 3. Loss Scale
```
✓ GOOD: alignment_loss ≪ task_loss (ratio < 0.01)
✗ BAD: |alignment_loss| ≈ task_loss (alignment dominating again)
```

### 4. Curve Smoothness
```
✓ GOOD: no spikes, gradual changes
✗ BAD: validation bounces up/down (instability)
```

## Success Thresholds
1. **Final val accuracy ≥ 50%** (was 6.7%, that's our fix working)
2. **Val curve smooth** (no wild swings)
3. **Alignment score < 0.95** (not saturated)
4. **Loss behaves smoothly** (not erratic)

**All 4 must pass** to proceed.

## If It Fails
- **Case A** (acc < 30%): Learning rate too low
- **Case B** (train high, val low): Overfitting
- **Case C** (spiky val): Alignment still too strong
