# CIFAR-10 Locked Grid: 18-Run Report (MOON Excluded)
Generated: 2026-04-25 14:41:39
---

## 1. Executive Summary
- **Total runs**: 18 (target grid: 24, completed: 18 without MOON)
- **Configuration**: CIFAR-10, 10 clients, 20 rounds
- **Alphas**: [0.1, 1.0] (Dirichlet non-IID)
- **Seeds**: [42, 123, 456]
- **Methods**: FedAvg, SCAFFOLD, FLEX (MOON excluded)
- **Hyperparameters**: local_epochs=5, lr=0.003, batch_size=64

## 2. Cross-Method Comparison (Mean Accuracy)
| Method | α=0.1 | α=1.0 | Overall Mean |
|--------|-------|-------|-------------|
| FedAvg   | 0.4484 | 0.5600 | 0.5042 |
| SCAFFOLD | 0.1319 | 0.2640 | 0.1979 |
| FLEX     | 0.7977 | 0.5463 | 0.6720 |

## 3. Worst-Client Analysis
| Method | α=0.1 Worst | α=1.0 Worst | Overall Worst |
|--------|-------------|-------------|---------------|
| FedAvg   | 0.2197 | 0.4833 | 0.3515 |
| SCAFFOLD | 0.0000 | 0.1583 | 0.0792 |
| FLEX     | 0.6282 | 0.4450 | 0.5366 |

## 4. Per-Seed Stability (Std Dev Across Seeds)
| Method | α=0.1 Std | α=1.0 Std |
|--------|-----------|-----------|
| FedAvg   | 0.0189 | 0.0126 |
| SCAFFOLD | 0.0164 | 0.0801 |
| FLEX     | 0.0102 | 0.0156 |

## 5. Communication Efficiency
| Method | α | Avg Bytes/Round | Total Bytes |
|--------|---|-----------------|-------------|
| SCAFFOLD | 0.1 | 49,460,000 | 989,200,000 |
| SCAFFOLD | 1.0 | 49,460,000 | 989,200,000 |

## 6. Convergence Summary
- **FedAvg α=0.1**: final mean=0.4484, std=0.0189
- **FedAvg α=1.0**: final mean=0.5600, std=0.0126
- **SCAFFOLD α=0.1**: final mean=0.1319, std=0.0164
- **SCAFFOLD α=1.0**: final mean=0.2640, std=0.0801
- **FLEX α=0.1**: final mean=0.7977, std=0.0102
- **FLEX α=1.0**: final mean=0.5463, std=0.0156

## 7. Limitations & Notes
1. **MOON exclusion**: MOON was excluded from this grid due to impractical execution time (~8+ minutes per round on available hardware). A single 20-round MOON run would exceed 2.5 hours.
2. **SCAFFOLD implementation**: Uses control variates with default settings.
3. **Hardware**: NVIDIA GeForce RTX 2050 with limited VRAM (4GB).
4. **Dataset**: CIFAR-10 with Dirichlet partitioning (α=0.1 for high non-IID, α=1.0 for moderate non-IID).
5. **Preservation**: Three existing 20-round FedAvg α=0.1 runs were preserved.
6. **Determinism**: Fixed seeds (42, 123, 456) with torch deterministic mode enabled.

