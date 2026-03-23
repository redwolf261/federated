#!/usr/bin/env python3
"""GPU-optimized comprehensive experiment runner with larger batch sizes."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import json
from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.utils.seed import set_global_seed

def run_gpu_optimized_experiments():
    """Run experiments with GPU-optimized settings."""

    workspace = Path(__file__).parent.parent

    # GPU-optimized configurations
    configs = {
        "high_het": {
            "name": "High Heterogeneity (GPU Optimized: 256 samples, 3 epochs, batch=128)",
            "rounds": 20,
            "local_epochs": 3,
            "batch_size": 128,  # 4x larger for better GPU utilization
            "max_samples_per_client": 256,
            "learning_rate": 0.01,
        },
        "low_het": {
            "name": "Low Heterogeneity (GPU Optimized: 1000 samples, 1 epoch, batch=256)",
            "rounds": 30,
            "local_epochs": 1,
            "batch_size": 256,  # 4x larger for better GPU utilization
            "max_samples_per_client": 1000,
            "learning_rate": 0.005,
        },
    }

    methods = ["fedavg", "prototype"]
    seeds = [11, 22, 33]  # Quick test with 3 seeds

    print("🚀 GPU-OPTIMIZED EXPERIMENT RUNNER")
    print("=" * 60)
    print("⚡ Larger batch sizes for better GPU utilization")
    print("📊 Monitoring GPU usage during runs")
    print()

    for cfg_key in configs.keys():
        cfg_val = configs[cfg_key].copy()
        name = cfg_val.pop("name")

        for method in methods:
            print(f"\n🎯 {method.upper()} × {cfg_key}")
            print(f"Config: {name}")

            for seed_idx, seed in enumerate(seeds, 1):
                print(f"  Seed {seed} ({seed_idx}/{len(seeds)})...", end=" ", flush=True)

                start_time = time.time()

                try:
                    set_global_seed(seed)
                    cfg = ExperimentConfig(dataset_name="femnist")
                    cfg.training.aggregation_mode = method

                    # Apply GPU-optimized config
                    for k, v in cfg_val.items():
                        if hasattr(cfg.training, k):
                            setattr(cfg.training, k, v)

                    cfg.num_clients = 10

                    if method == "prototype":
                        cfg.training.cluster_aware_epochs = 1
                    if method == "fedavg":
                        cfg.model.client_backbones = ["small_cnn"] * cfg.num_clients

                    sim = FederatedSimulator(workspace_root=workspace, config=cfg)

                    # Check GPU utilization before
                    import subprocess
                    gpu_before = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True
                    ).stdout.strip()

                    hist = sim.run_experiment()
                    report = sim.build_report(hist)

                    # Check GPU utilization after
                    gpu_after = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True
                    ).stdout.strip()

                    elapsed = time.time() - start_time

                    conv = report.get("convergence", {})
                    mean_accs = conv.get("mean_client_accuracy", [])
                    final_mean = mean_accs[-1] if mean_accs else 0

                    print(f"✅ {final_mean:.4f} | {elapsed:.1f}s | GPU: {gpu_before}%→{gpu_after}%")

                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"❌ ERROR: {str(e)[:40]} | {elapsed:.1f}s")

if __name__ == "__main__":
    run_gpu_optimized_experiments()