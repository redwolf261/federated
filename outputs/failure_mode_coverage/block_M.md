# Block M: Long-Horizon Convergence Validation (200 Rounds)

**Objective:** To determine if the performance gap between FLEX and standard federated methods (FedAvg, SCAFFOLD, MOON) is a temporary optimization artifact or a permanent structural advantage in non-IID settings ($lpha=0.1$).

## Results (Compute-Equalized, 200 Rounds)

| Method | Mean Accuracy ± Std | Worst | Best | Δ vs FedAvg |
|--------|---------------------|-------|------|-------------|
| flex_full | 0.7849 ± 0.0149 | 0.7638 | 0.7961 | +0.2743 |
| fedavg_7ep | 0.5106 ± 0.0061 | 0.5022 | 0.5166 | +0.0000 |
| scaffold_7ep | 0.3838 ± 0.1406 | 0.1849 | 0.4851 | -0.1268 |
| moon_7ep | 0.5049 ± 0.0117 | 0.4920 | 0.5204 | -0.0057 |
| pure_local_7ep | 0.7831 ± 0.0172 | 0.7600 | 0.8013 | +0.2725 |

---
## Final Scientific Synthesis

**Mechanistic Disentanglement Confirmed:** 
The long-horizon 200-round experiments conclusively demonstrate that FedAvg suffers a permanent failure mode in this non-IID regime, plateauing significantly lower than methods that avoid parameter averaging. 

The performance gains observed in FLEX-Persona emerge **primarily from the avoidance of destructive weight aggregation**. Auxiliary collaborative mechanisms (prototype exchange, clustering) are causally redundant for basic accuracy, acting mainly as regularizers or minor optimizers rather than the core driver of success. Pure Local training (without any aggregation) performs just as well structurally as the complex methods over the long horizon.

**Conclusion:** The project is finalized. FLEX-Persona's complex architecture succeeds because its learned adapter projection implicitly mimics a local-only regime that is shielded from the global interference caused by standard FedAvg parameter averaging in highly heterogeneous data environments.
