#!/usr/bin/env python3
"""
Ablation study: disable components to isolate what drives stability.

Variants:
1. FedAvg (baseline)
2. Prototype (full method)
3. Prototype without clustering (just averaging representations)
4. Prototype without cluster-aware guidance (clustering only)
"""

from pathlib import Path
from statistics import mean, stdev

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed


def run_ablation_high_heterogeneity():
    """
    Run ablation study comparing method variants.
    
    High heterogeneity is where differences matter most.
    """
    
    workspace = Path(__file__).parent.parent
    
    variants = {
        "fedavg_baseline": {
            "description": "Standard FedAvg (parameter averaging)",
            "aggregation_mode": "fedavg",
            "use_clustering": False,
            "use_guidance": False,
        },
        "prototype_full": {
            "description": "FLEX-Persona (clustering + guidance)",
            "aggregation_mode": "prototype",
            "use_clustering": True,
            "use_guidance": True,
        },
        "prototype_no_clustering": {
            "description": "Prototype without clustering (just representation averaging)",
            "aggregation_mode": "prototype",
            "use_clustering": False,
            "use_guidance": True,
        },
        "prototype_no_guidance": {
            "description": "Prototype without cluster guidance (clustering only)",
            "aggregation_mode": "prototype",
            "use_clustering": True,
            "use_guidance": False,
        },
    }
    
    # High-heterogeneity config
    base_config = {
        "rounds": 20,
        "local_epochs": 3,
        "batch_size": 32,
        "max_samples_per_client": 256,
        "learning_rate": 0.01,
        "num_clients": 10,
    }
    
    seeds = [11, 42, 55]  # Sample 3 seeds for quick validation
    
    print("=" * 120)
    print("ABLATION STUDY: HIGH HETEROGENEITY REGIME")
    print("=" * 120)
    print()
    print(f"Configuration: {base_config}")
    print(f"Seeds: {seeds}")
    print()
    
    results = {}
    
    for var_name, var_config in variants.items():
        print(f"\n{'='*120}")
        print(f"VARIANT: {var_config['description']}")
        print(f"{'='*120}")
        
        variant_results = {"accuracy": [], "worst": [], "collapsed": 0}
        
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"  Seed {seed:02d} ({seed_idx}/{len(seeds)})...", end=" ", flush=True)
            
            try:
                set_global_seed(seed)
                cfg = ExperimentConfig(dataset_name="femnist")
                
                # Apply base config
                for key, val in base_config.items():
                    if hasattr(cfg.training, key):
                        setattr(cfg.training, key, val)
                cfg.num_clients = base_config["num_clients"]
                
                # Apply variant config
                cfg.training.aggregation_mode = var_config["aggregation_mode"]
                cfg.training.use_clustering = var_config["use_clustering"]
                cfg.training.use_guidance = var_config["use_guidance"]

                if var_config["aggregation_mode"] == "fedavg":
                    cfg.model.client_backbones = ["small_cnn"] * cfg.num_clients
                
                sim = FederatedSimulator(workspace_root=workspace, config=cfg)
                hist = sim.run_experiment()
                report = sim.build_report(hist)
                
                convergence = report.get("convergence", {})
                mean_accs = convergence.get("mean_client_accuracy", [])
                worst_accs = convergence.get("worst_client_accuracy", [])
                
                final_mean = mean_accs[-1] if mean_accs else 0
                final_worst = worst_accs[-1] if worst_accs else 0
                mean_avg = sum(mean_accs) / len(mean_accs) if mean_accs else 0
                
                variant_results["accuracy"].append(mean_avg)
                variant_results["worst"].append(final_worst)
                
                if final_mean < 0.10:
                    variant_results["collapsed"] += 1
                
                print(f"[PASS] mean={mean_avg:.4f} final={final_mean:.4f}")
                
            except Exception as e:
                print(f"[ERROR] {e}")
        
        results[var_name] = variant_results
    
    # Summary table
    print()
    print(f"\n{'='*120}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*120}")
    print()
    print("| Variant                              | Mean Accuracy | Worst-Client | Collapses |")
    print("|--------------------------------------|---------------|--------------|-----------|")
    
    for var_name, var_config in variants.items():
        if var_name in results:
            res = results[var_name]
            mean_acc = mean(res["accuracy"]) if res["accuracy"] else 0
            std_acc = stdev(res["accuracy"]) if len(res["accuracy"]) > 1 else 0
            worst_acc = mean(res["worst"]) if res["worst"] else 0
            collapses = res["collapsed"]
            
            print(f"| {var_config['description']:36} | "
                  f"{mean_acc:.4f}±{std_acc:.4f}  | {worst_acc:.4f}       | {collapses}/3       |")
    
    print()
    print("INTERPRETATION:")
    print()
    print("If Full Method performs well but ablations degrade:")
    print("  → Clustering AND guidance both necessary")
    print()
    print("If Full Method ≈ No Clustering:")
    print("  → Guidance doing the work; clustering optional")
    print()
    print("If Full Method ≈ No Guidance:")
    print("  → Clustering doing the work; guidance is overhead")
    print()


if __name__ == "__main__":
    run_ablation_high_heterogeneity()
