# Experiment Summary: fedavg_high_het_10seed

## Configuration
- **Experiment ID**: fedavg_high_het_10seed_20260322_200801_1ba6f569
- **Method**: fedavg
- **Regime**: high_het
- **Dataset**: femnist
- **Git Commit**: 343ff206f49de94b6217d9c09b927822579e927b
- **Start Time**: 2026-03-22T20:08:01.300710
- **End Time**: 2026-03-22T20:12:47.106024

## Hyperparameters
- **Seeds**: [11, 22, 33, 42, 55, 66, 77, 88, 99, 100] (10 total)
- **Clients**: 10
- **Rounds**: 20
- **Local Epochs**: 3
- **Batch Size**: 128
- **Learning Rate**: 0.01
- **Max Samples per Client**: 256

## Results Summary

### Primary Metrics
- **Mean Accuracy**: 0.0931 ± 0.0148
  - Range: [0.0748, 0.1266]
  - 95% CI: [0.0767, 0.1211]

- **Worst-Client Accuracy**: 0.0510 ± 0.0405
  - Range: [0.0000, 0.1569]

### Stability Metrics
- **Collapse Rate (< 0.1)**: 20.0% (2/10)
- **Collapse Rate Sensitive (< 0.15)**: 60.0% (6/10)
- **Stability Variance**: 0.0282 ± 0.0157

## Reproducibility
All results can be reproduced using:
- Config file: `config.json`
- Git commit: `343ff206f49de94b6217d9c09b927822579e927b`
- Seed list: [11, 22, 33, 42, 55, 66, 77, 88, 99, 100]

## Artifacts
- **Configuration**: config.json
- **Per-seed results**: per_seed_results.json
- **Aggregate metrics**: aggregate_metrics.json
- **Plots**: plots/

---
*Generated on 2026-03-22T20:12:47.111023*
