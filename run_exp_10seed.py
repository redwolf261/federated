import json
from pathlib import Path
from statistics import mean, stdev

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed

workspace = Path.cwd()
all_results = {}

configs = {
    "high_het": {
        "name": "High Heterogeneity",
        "rounds": 20,
        "local_epochs": 3,
        "batch_size": 32,
        "max_samples_per_client": 256,
        "learning_rate": 0.01,
    },
    "low_het": {
        "name": "Low Heterogeneity",
        "rounds": 30,
        "local_epochs": 1,
        "batch_size": 64,
        "max_samples_per_client": 1000,
        "learning_rate": 0.005,
    },
}

methods = ["fedavg", "prototype"]
seeds = [11, 22, 33, 42, 55, 66, 77, 88, 99, 100]

print("=" * 100)
print("10-SEED COMPREHENSIVE EXPERIMENT")
print("=" * 100)

total = len(configs) * len(methods) * len(seeds)
done = 0

for cfg_key, cfg_dict in sorted(configs.items()):
    cfg_name = cfg_dict["name"]
    cfg_params = {k: v for k, v in cfg_dict.items() if k != "name"}
    
    for method in methods:
        print(f"\n{cfg_name} | {method.upper()}")
        print("─" * 100)
        
        key = f"{method}_{cfg_key}"
        all_results[key] = {
            "seeds": [],
            "method": method,
            "regime": cfg_key,
        }
        
        for seed in seeds:
            done += 1
            print(f"[{done:2d}/{total}] Seed {seed:03d}...", end=" ", flush=True)
            
            try:
                set_global_seed(seed)
                cfg = ExperimentConfig(dataset_name="femnist")
                cfg.training.aggregation_mode = method
                
                for param_name, param_val in cfg_params.items():
                    if hasattr(cfg.training, param_name):
                        setattr(cfg.training, param_name, param_val)
                
                cfg.training.num_clients = 10
                if method == "prototype":
                    cfg.training.cluster_aware_epochs = 1
                
                sim = FederatedSimulator(workspace_root=workspace, config=cfg)
                hist = sim.run_experiment()
                report = sim.build_report(hist)
                
                conv = report.get("convergence", {})
                mean_accs = conv.get("mean_client_accuracy", [])
                worst_accs = conv.get("worst_client_accuracy", [])
                
                final_mean = mean_accs[-1] if mean_accs else 0
                final_worst = worst_accs[-1] if worst_accs else 0
                mean_avg = sum(mean_accs) / len(mean_accs) if mean_accs else 0
                
                all_results[key]["seeds"].append({
                    "seed": seed,
                    "final_mean": float(final_mean),
                    "mean_avg": float(mean_avg),
                    "worst_final": float(final_worst),
                    "collapsed": final_mean < 0.10,
                })
                
                status = "COLLAPSE" if final_mean < 0.10 else "ok"
                print(f"✓ {final_mean:.4f} [{status}]")
                
            except Exception as e:
                print(f"✗ {str(e)[:40]}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("\n| Method | Regime | Mean Acc | Worst | Collapse |")
print("|--------|--------|----------|-------|----------|")

for key in sorted(all_results.keys()):
    data = all_results[key]
    seeds_data = data["seeds"]
    
    if seeds_data:
        mean_accs_list = [s["mean_avg"] for s in seeds_data]
        worst_accs_list = [s["worst_final"] for s in seeds_data]
        collapses = sum(1 for s in seeds_data if s["collapsed"])
        
        m_avg = mean(mean_accs_list)
        m_std = stdev(mean_accs_list) if len(mean_accs_list) > 1 else 0
        w_avg = mean(worst_accs_list)
        
        print(
            f"| {data['method']:6} | {data['regime']:6} | "
            f"{m_avg:.4f}±{m_std:.4f} | {w_avg:.4f} | {collapses}/10 |"
        )

exp_dir = workspace / "experiments"
exp_dir.mkdir(exist_ok=True, parents=True)

json_path = exp_dir / "comprehensive_10seed_results.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Saved to {json_path}")

print("\n" + "=" * 100)
print("KEY FINDING: HIGH HETEROGENEITY")
print("=" * 100)

f_data = all_results["fedavg_high_het"]["seeds"]
p_data = all_results["prototype_high_het"]["seeds"]

f_mean_val = mean([s["mean_avg"] for s in f_data])
f_collapse_pct = sum(1 for s in f_data if s["collapsed"]) / len(f_data) * 100

p_mean_val = mean([s["mean_avg"] for s in p_data])
p_collapse_pct = sum(1 for s in p_data if s["collapsed"]) / len(p_data) * 100

print(f"\nFedAvg:    collapse={f_collapse_pct:.0f}% mean={f_mean_val:.4f}")
print(f"Prototype: collapse={p_collapse_pct:.0f}% mean={p_mean_val:.4f}")

collapse_improvement = f_collapse_pct - p_collapse_pct
acc_improvement = (p_mean_val - f_mean_val) / f_mean_val * 100 if f_mean_val > 0 else 0

print(f"\nImprovement:")
print(f"  Collapse rate: {collapse_improvement:+.0f} percentage points")
print(f"  Mean accuracy: {acc_improvement:+.1f}%")
