# Experiment Summary: fedavg_high_het_10seed

## Configuration
- **Experiment ID**: fedavg_high_het_10seed_20260322_183707_5938ced4
- **Method**: fedavg
- **Regime**: high_het
- **Dataset**: femnist
- **Git Commit**: 343ff206f49de94b6217d9c09b927822579e927b
- **Start Time**: 2026-03-22T18:37:07.734858
- **End Time**: 2026-03-22T18:58:16.851845

## Hyperparameters
- **Seeds**: [11, 22, 33, 42, 55, 66, 77, 88, 99, 100] (10 total)
- **Clients**: 10
- **Rounds**: 20
- **Local Epochs**: 3
- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Max Samples per Client**: 256

## Results Summary

### Primary Metrics
- **Mean Accuracy**: 0.1442 ± 0.0815
  - Range: [0.0786, 0.3267]
  - 95% CI: [0.0787, 0.3087]

- **Worst-Client Accuracy**: 0.1529 ± 0.1554
  - Range: [0.0000, 0.4706]

### Stability Metrics
- **Collapse Rate (< 0.1)**: 20.0% (2/10)
- **Collapse Rate Sensitive (< 0.15)**: 30.0% (3/10)
- **Stability Variance**: 0.0678 ± 0.0650

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
*Generated on 2026-03-22T18:58:16.852856*
