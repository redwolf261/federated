# Experiment Summary: fedavg_high_het_10seed

## Configuration
- **Experiment ID**: fedavg_high_het_10seed_20260322_182527_2995475b
- **Method**: fedavg
- **Regime**: high_het
- **Dataset**: femnist
- **Git Commit**: 343ff206f49de94b6217d9c09b927822579e927b
- **Start Time**: 2026-03-22T18:25:27.569285
- **End Time**: 2026-03-22T18:30:51.693483

## Hyperparameters
- **Seeds**: [11] (1 total)
- **Clients**: 10
- **Rounds**: 20
- **Local Epochs**: 3
- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Max Samples per Client**: 256

## Results Summary

### Primary Metrics
- **Mean Accuracy**: 0.1422 ± 0.0000
  - Range: [0.1422, 0.1422]
  - 95% CI: [0.1422, 0.1422]

- **Worst-Client Accuracy**: 0.1765 ± 0.0000
  - Range: [0.1765, 0.1765]

### Stability Metrics
- **Collapse Rate (< 0.1)**: 0.0% (0/1)
- **Collapse Rate Sensitive (< 0.15)**: 0.0% (0/1)
- **Stability Variance**: 0.0747 ± 0.0000

## Reproducibility
All results can be reproduced using:
- Config file: `config.json`
- Git commit: `343ff206f49de94b6217d9c09b927822579e927b`
- Seed list: [11]

## Artifacts
- **Configuration**: config.json
- **Per-seed results**: per_seed_results.json
- **Aggregate metrics**: aggregate_metrics.json
- **Plots**: plots/

---
*Generated on 2026-03-22T18:30:51.698487*
