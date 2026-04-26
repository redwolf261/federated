# Experiment Summary: test_experiment_demo

## Configuration
- **Experiment ID**: test_experiment_20260322_181041_f04f1be0
- **Method**: prototype
- **Regime**: high_het
- **Dataset**: femnist
- **Git Commit**: 343ff206f49de94b6217d9c09b927822579e927b
- **Start Time**: 2026-03-22T18:10:41.418992
- **End Time**: 2026-03-22T18:10:41.418992

## Hyperparameters
- **Seeds**: [11, 22, 33] (3 total)
- **Clients**: 10
- **Rounds**: 20
- **Local Epochs**: 3
- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Max Samples per Client**: 256

## Results Summary

### Primary Metrics
- **Mean Accuracy**: 0.8234 ± 0.0222
  - Range: [0.8012, 0.8456]
  - 95% CI: [0.8023, 0.8445]

- **Worst-Client Accuracy**: 0.7119 ± 0.0228
  - Range: [0.6890, 0.7345]

### Stability Metrics
- **Collapse Rate (< 0.1)**: 0.0% (0/3)
- **Collapse Rate Sensitive (< 0.15)**: 0.0% (0/3)
- **Stability Variance**: 0.0126 ± 0.0029

## Reproducibility
All results can be reproduced using:
- Config file: `config.json`
- Git commit: `343ff206f49de94b6217d9c09b927822579e927b`
- Seed list: [11, 22, 33]

## Artifacts
- **Configuration**: config.json
- **Per-seed results**: per_seed_results.json
- **Aggregate metrics**: aggregate_metrics.json
- **Plots**: plots/

---
*Generated on 2026-03-22T18:10:41.636081*
