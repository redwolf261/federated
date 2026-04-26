#!/usr/bin/env python3
"""Comprehensive 10-seed experiment suite with collapse metrics."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, stdev

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed


@dataclass
class SeedResult:
    seed: int
    method: str
    regime: str
    mean_accuracy: float
    worst_accuracy: float
    p10_accuracy: float
    bottom3_accuracy: float
    rounds_data: list = field(default_factory=list)
    
    def collapsed(self, threshold: float = 0.10) -> bool:
        if not self.rounds_data:
            return False
        final = self.rounds_data[-1].get("mean_client_accuracy", 0)
        return final < threshold


@dataclass
class MethodRegimeResults:
    method: str
    regime: str
    seeds: list = field(default_factory=list)
    
    @property
    def mean_accuracy_avg(self) -> float:
        means = [s.mean_accuracy for s in self.seeds]
        return mean(means) if means else 0
    
    @property
    def mean_accuracy_std(self) -> float:
        means = [s.mean_accuracy for s in self.seeds]
        return stdev(means) if len(means) > 1 else 0
    
    @property
    def worst_accuracy_avg(self) -> float:
        worsts = [s.worst_accuracy for s in self.seeds]
        return mean(worsts) if worsts else 0
    
    @property
    def worst_accuracy_std(self) -> float:
        worsts = [s.worst_accuracy for s in self.seeds]
        return stdev(worsts) if len(worsts) > 1 else 0

    @property
    def p10_accuracy_avg(self) -> float:
        values = [s.p10_accuracy for s in self.seeds]
        return mean(values) if values else 0

    @property
    def p10_accuracy_std(self) -> float:
        values = [s.p10_accuracy for s in self.seeds]
        return stdev(values) if len(values) > 1 else 0

    @property
    def bottom3_accuracy_avg(self) -> float:
        values = [s.bottom3_accuracy for s in self.seeds]
        return mean(values) if values else 0

    @property
    def bottom3_accuracy_std(self) -> float:
        values = [s.bottom3_accuracy for s in self.seeds]
        return stdev(values) if len(values) > 1 else 0
    
    @property
    def collapse_rate(self) -> float:
        if not self.seeds:
            return 0
        collapses = sum(1 for s in self.seeds if s.collapsed())
        return collapses / len(self.seeds)


def main():
    workspace = Path(__file__).parent.parent
    results = {}
    
    # Regimes
    configs = {
        "high_het": {
            "name": "High Heterogeneity (256 samples, 3 epochs, lr=0.01)",
            "rounds": 20,
            "local_epochs": 3,
            "batch_size": 32,
            "max_samples_per_client": 256,
            "learning_rate": 0.01,
        },
        "low_het": {
            "name": "Low Heterogeneity (1000 samples, 1 epoch, lr=0.005)",
            "rounds": 30,
            "local_epochs": 1,
            "batch_size": 64,
            "max_samples_per_client": 1000,
            "learning_rate": 0.005,
        },
    }
    
    methods = ["fedavg", "fedprox", "prototype"]
    seeds = [11]  # Smoke test with 1 seed
    
    print("=" * 100)
    print("COMPREHENSIVE 10-SEED EXPERIMENT SUITE")
    print("=" * 100)
    print()
    
    total = len(configs) * len(methods) * len(seeds)
    done = 0
    
    for cfg_key in sorted(configs.keys()):
        cfg_val = configs[cfg_key].copy()
        name = cfg_val.pop("name")
        
        for method in methods:
            print(f"\nREGIME: {name}")
            print(f"METHOD: {method.upper()}")
            print("-" * 100)
            
            key = f"{method}_{cfg_key}"
            results[key] = MethodRegimeResults(method=method, regime=cfg_key)
            
            for seed_idx, seed in enumerate(seeds, 1):
                done += 1
                print(f"[{done:2d}/{total}] Seed {seed:03d}...", end=" ", flush=True)
                
                try:
                    set_global_seed(seed)
                    cfg = ExperimentConfig(dataset_name="femnist")
                    cfg.training.aggregation_mode = method
                    
                    for k, v in cfg_val.items():
                        if hasattr(cfg.training, k):
                            setattr(cfg.training, k, v)
                    
                    cfg.num_clients = 10
                    
                    if method == "prototype":
                        cfg.training.cluster_aware_epochs = 1
                    if method in {"fedavg", "fedprox"}:
                        # FedAvg requires homogeneous client backbones for state_dict averaging.
                        cfg.model.client_backbones = ["small_cnn"] * cfg.num_clients
                    if method == "fedprox":
                        cfg.training.fedprox_mu = 0.01
                    
                    sim = FederatedSimulator(workspace_root=workspace, config=cfg)
                    hist = sim.run_experiment()
                    report = sim.build_report(hist)
                    
                    conv = report.get("convergence", {})
                    mean_accs = conv.get("mean_client_accuracy", [])
                    worst_accs = conv.get("worst_client_accuracy", [])
                    p10_accs = conv.get("p10_client_accuracy", [])
                    bottom3_accs = conv.get("bottom3_client_accuracy", [])
                    
                    final_mean = mean_accs[-1] if mean_accs else 0
                    final_worst = worst_accs[-1] if worst_accs else 0
                    final_p10 = p10_accs[-1] if p10_accs else final_worst
                    final_bottom3 = bottom3_accs[-1] if bottom3_accs else final_worst
                    mean_avg = sum(mean_accs) / len(mean_accs) if mean_accs else 0
                    
                    rounds_data = [
                        {"round": i, "mean_client_accuracy": float(mean_accs[i]) if i < len(mean_accs) else 0}
                        for i in range(len(mean_accs))
                    ]
                    
                    sr = SeedResult(
                        seed=seed,
                        method=method,
                        regime=cfg_key,
                        mean_accuracy=mean_avg,
                        worst_accuracy=final_worst,
                        p10_accuracy=final_p10,
                        bottom3_accuracy=final_bottom3,
                        rounds_data=rounds_data,
                    )
                    results[key].seeds.append(sr)
                    
                    status = "COLLAPSE" if sr.collapsed() else "ok"
                    print(f"OK {final_mean:.4f} mean={mean_avg:.4f} [{status}]")
                    
                except Exception as e:
                    print(f"ERROR: {str(e)[:50]}")
    
    # Summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print("\n| Method | Regime | Mean Acc | Worst | P10 | Bottom3 | Collapse Rate |")
    print("|--------|--------|----------|-------|-----|---------|---------------|")
    
    for key in sorted(results.keys()):
        r = results[key]
        collapse_pct = int(r.collapse_rate * 100)
        collapse_cnt = int(r.collapse_rate * len(r.seeds))
        print(
            f"| {r.method:6} | {r.regime:6} | "
            f"{r.mean_accuracy_avg:.4f}±{r.mean_accuracy_std:.4f} | "
            f"{r.worst_accuracy_avg:.4f} | "
            f"{r.p10_accuracy_avg:.4f} | "
            f"{r.bottom3_accuracy_avg:.4f} | "
            f"{collapse_pct}% ({collapse_cnt}/{len(r.seeds)}) |"
        )
    
    # Save
    exp_dir = workspace / "experiments"
    exp_dir.mkdir(exist_ok=True, parents=True)
    
    json_data = {}
    for key, r in results.items():
        json_data[key] = {
            "method": r.method,
            "regime": r.regime,
            "num_seeds": len(r.seeds),
            "mean_accuracy_avg": r.mean_accuracy_avg,
            "mean_accuracy_std": r.mean_accuracy_std,
            "worst_accuracy_avg": r.worst_accuracy_avg,
            "worst_accuracy_std": r.worst_accuracy_std,
            "p10_accuracy_avg": r.p10_accuracy_avg,
            "p10_accuracy_std": r.p10_accuracy_std,
            "bottom3_accuracy_avg": r.bottom3_accuracy_avg,
            "bottom3_accuracy_std": r.bottom3_accuracy_std,
            "collapse_rate": r.collapse_rate,
            "collapse_count": int(r.collapse_rate * len(r.seeds)),
            "seeds": [asdict(s) for s in r.seeds],
        }
    
    json_path = exp_dir / "comprehensive_10seed_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to {json_path}")
    
    # Key insight
    print("\n" + "=" * 100)
    print("KEY FINDING: HIGH HETEROGENEITY COMPARISON")
    print("=" * 100)
    
    f_r = results.get("fedavg_high_het")
    p_r = results.get("prototype_high_het")
    
    if f_r and p_r:
        print(f"\nFedAvg:")
        print(f"  Mean Accuracy: {f_r.mean_accuracy_avg:.4f} ± {f_r.mean_accuracy_std:.4f}")
        print(f"  Collapse Rate: {int(f_r.collapse_rate*100)}% ({int(f_r.collapse_rate*len(f_r.seeds))}/{len(f_r.seeds)})")
        
        print(f"\nPrototype:")
        print(f"  Mean Accuracy: {p_r.mean_accuracy_avg:.4f} ± {p_r.mean_accuracy_std:.4f}")
        print(f"  Collapse Rate: {int(p_r.collapse_rate*100)}% ({int(p_r.collapse_rate*len(p_r.seeds))}/{len(p_r.seeds)})")
        
        print(f"\nImprovement:")
        collapse_improvement = (f_r.collapse_rate - p_r.collapse_rate) * 100
        print(f"  Collapse rate reduction: {collapse_improvement:.0f} percentage points")
        
        if f_r.mean_accuracy_avg > 0:
            acc_improvement = ((p_r.mean_accuracy_avg - f_r.mean_accuracy_avg) / f_r.mean_accuracy_avg) * 100
            print(f"  Mean accuracy improvement: {acc_improvement:+.1f}%")


if __name__ == "__main__":
    main()
