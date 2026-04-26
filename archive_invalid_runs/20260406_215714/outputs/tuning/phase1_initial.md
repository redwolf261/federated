# FLEX Hyperparameter Tuning Summary

## Protocol

- dataset: femnist
- num_clients: 10
- alpha: 0.1
- rounds: 10
- max_samples_per_client: 2000
- seeds: [42, 123, 456]

## Objective

`J = worst + 0.25*mean - 0.1*(p90-p10)`

## Top Configurations

| Rank | Phase | Score | Worst | Mean | Gap |
|---:|---|---:|---:|---:|---:|
| 1 | phase1 | 1.0370 | 0.8242 | 0.8820 | 0.0767 |
| 2 | phase1 | 1.0348 | 0.8225 | 0.8781 | 0.0722 |
| 3 | phase1 | 1.0321 | 0.8208 | 0.8747 | 0.0740 |

## Best Configuration

```json
{
  "lambda_cluster": 0.02,
  "lambda_cluster_center": 0.0,
  "cluster_center_warmup_rounds": 5,
  "local_epochs": 5,
  "cluster_aware_epochs": 3,
  "learning_rate": 0.005,
  "weight_decay": 0.0,
  "batch_size": 32
}
```
