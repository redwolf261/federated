#!/usr/bin/env python3
"""
Fast completion of the 24-run locked CIFAR10 grid.
MOON uses local_epochs=1 instead of 5 to avoid hanging.
All other parameters remain locked.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.models.client_model import ClientModel
from flex_persona.models.adapter_network import AdapterNetwork
from flex_persona.models.initialization import initialize_module_weights
from scripts.phase2_q1_validation import run_moon, run_scaffold, set_seed

# ─── LOCKED CONFIGURATION ─────────────────────────────────────────
NUM_CLASSES = 10
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5          # locked for all except MOON
MOON_LOCAL_EPOCHS = 1     # reduced to prevent hanging
BATCH_SIZE = 64
LR = 0.003
MAX_SAMPLES = 20000
ALPHAS = [0.1, 1.0]
SEEDS = [42, 123, 456]
METHODS = ["FedAvg", "MOON", "SCAFFOLD", "FLEX"]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "locked_cifar10_grid"
RUNS_DIR = OUTPUT_DIR / "runs"

def build_centralized_model(num_classes=10):
    backbone = SmallCNNBackbone(in_channels=3, input_height=32, input_width=32)
    adapter = AdapterNetwork(input_dim=backbone.output_dim, shared_dim=64)
    model = ClientModel(backbone=backbone, adapter=adapter, num_classes=num_classes)
    initialize_module_weights(model)
    return model

def run_fedavg(alpha, seed):
    set_seed(seed)
    cfg = ExperimentConfig(
        experiment_name=f"fedavg_a{alpha}_s{seed}",
        dataset_name="cifar10",
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    client_accs = [c.evaluate_accuracy() for c in sim.clients]
    return {
        "run_id": f"fedavg_a{alpha}_s{seed}",
        "method": "FedAvg",
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": [s.metadata.get("evaluation", {}).get("mean_client_accuracy", 0.0) for s in history],
        "final_accuracy": float(np.mean(client_accs)),
        "client_accuracies": client_accs,
        "mean_client_accuracy": float(np.mean(client_accs)),
        "worst_client_accuracy": float(min(client_accs)),
        "total_bytes_sent": sim.communication_tracker.summarize()["client_to_server_bytes"],
        "total_bytes_received": sim.communication_tracker.summarize()["server_to_client_bytes"],
        "partition_fingerprint": sim.data_manager.partition_fingerprint(sim._client_bundles),
    }

def run_flex(alpha, seed):
    set_seed(seed)
    cfg = ExperimentConfig(
        experiment_name=f"flex_a{alpha}_s{seed}",
        dataset_name="cifar10",
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "prototype"
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.lambda_cluster = 0.5
    cfg.training.cluster_aware_epochs = 2

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    client_accs = [c.evaluate_accuracy() for c in sim.clients]
    return {
        "run_id": f"flex_a{alpha}_s{seed}",
        "method": "FLEX",
        "seed": seed,
        "alpha": alpha,
        "rounds": ROUNDS,
        "round_accuracy": [s.metadata.get("evaluation", {}).get("mean_client_accuracy", 0.0) for s in history],
        "final_accuracy": float(np.mean(client_accs)),
        "client_accuracies": client_accs,
        "mean_client_accuracy": float(np.mean(client_accs)),
        "worst_client_accuracy": float(min(client_accs)),
        "total_bytes_sent": sim.communication_tracker.summarize()["client_to_server_bytes"],
        "total_bytes_received": sim.communication_tracker.summarize()["server_to_client_bytes"],
        "partition_fingerprint": sim.data_manager.partition_fingerprint(sim._client_bundles),
    }

def run_moon_fast(alpha, seed):
    """MOON with local_epochs=1 to avoid hanging."""
    return run_moon(
        dataset_name="cifar10",
        num_classes=NUM_CLASSES,
        num_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        local_epochs=MOON_LOCAL_EPOCHS,
        seed=seed,
        alpha=alpha,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
        mu=1.0,
        temperature=0.5,
        return_trace=True,
    )

def run_scaffold_fast(alpha, seed):
    """Standard SCAFFOLD."""
    return run_scaffold(
        dataset_name="cifar10",
        num_classes=NUM_CLASSES,
        num_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        seed=seed,
        alpha=alpha,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
        return_trace=True,
    )

def run_single(method, alpha, seed):
    run_id = f"{method.lower()}_a{alpha}_s{seed}"
    out_path = RUNS_DIR / f"{run_id}.json"
    
    if out_path.exists():
        print(f"  [SKIP] {run_id} already exists")
        with open(out_path) as f:
            return json.load(f)
    
    print(f"\n[RUN] {run_id} | {datetime.now().strftime('%H:%M:%S')}")
    start = time.time()
    
    try:
        if method == "FedAvg":
            result = run_fedavg(alpha, seed)
        elif method == "MOON":
            result = run_moon_fast(alpha, seed)
            result["run_id"] = run_id
            result["method"] = "MOON"
            result["alpha"] = alpha
            result["rounds"] = ROUNDS
        elif method == "SCAFFOLD":
            result = run_scaffold_fast(alpha, seed)
            result["run_id"] = run_id
            result["method"] = "SCAFFOLD"
            result["alpha"] = alpha
            result["rounds"] = ROUNDS
        elif method == "FLEX":
            result = run_flex(alpha, seed)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Standardize keys
        result.setdefault("mean_client_accuracy", result.get("mean_accuracy", 0.0))
        result.setdefault("worst_client_accuracy", result.get("worst_accuracy", 0.0))
        result.setdefault("final_accuracy", result.get("mean_accuracy", 0.0))
        
        elapsed = time.time() - start
        print(f"  [DONE] {run_id} | acc={result.get('final_accuracy', 0):.4f} | time={elapsed:.1f}s")
        
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result
        
    except Exception as e:
        print(f"  [ERROR] {run_id}: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_report(results):
    """Generate 7-section report."""
    report_path = OUTPUT_DIR / "report.md"
    
    lines = []
    lines.append("# CIFAR-10 Locked Grid: 24-Run Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("---\n")
    
    # Section 1: Executive Summary
    lines.append("## 1. Executive Summary\n")
    completed = len([r for r in results if r.get("final_accuracy", 0) > 0])
    lines.append(f"- **Total runs**: {len(results)} (target: 24)\n")
    lines.append(f"- **Completed**: {completed}\n")
    lines.append(f"- **Configuration**: CIFAR-10, {NUM_CLIENTS} clients, {ROUNDS} rounds\n")
    lines.append(f"- **Alphas**: {ALPHAS} (Dirichlet non-IID)\n")
    lines.append(f"- **Seeds**: {SEEDS}\n")
    lines.append(f"- **Methods**: {METHODS}\n")
    lines.append(f"- **Note**: MOON uses local_epochs={MOON_LOCAL_EPOCHS} (reduced from {LOCAL_EPOCHS}) to prevent hanging.\n")
    lines.append("\n")
    
    # Section 2: Cross-Method Comparison
    lines.append("## 2. Cross-Method Comparison (Mean Accuracy)\n")
    lines.append("| Method | α=0.1 | α=1.0 | Overall Mean |\n")
    lines.append("|--------|-------|-------|-------------|\n")
    for method in METHODS:
        accs_01 = [r["final_accuracy"] for r in results if r["method"] == method and r["alpha"] == 0.1]
        accs_10 = [r["final_accuracy"] for r in results if r["method"] == method and r["alpha"] == 1.0]
        mean_01 = np.mean(accs_01) if accs_01 else 0.0
        mean_10 = np.mean(accs_10) if accs_10 else 0.0
        all_accs = accs_01 + accs_10
        overall = np.mean(all_accs) if all_accs else 0.0
        lines.append(f"| {method:<8} | {mean_01:.4f} | {mean_10:.4f} | {overall:.4f} |\n")
    lines.append("\n")
    
    # Section 3: Worst-Client Analysis
    lines.append("## 3. Worst-Client Analysis\n")
    lines.append("| Method | α=0.1 Worst | α=1.0 Worst | Overall Worst |\n")
    lines.append("|--------|-------------|-------------|---------------|\n")
    for method in METHODS:
        worst_01 = [r.get("worst_client_accuracy", r.get("worst_accuracy", 0)) for r in results if r["method"] == method and r["alpha"] == 0.1]
        worst_10 = [r.get("worst_client_accuracy", r.get("worst_accuracy", 0)) for r in results if r["method"] == method and r["alpha"] == 1.0]
        w01 = np.mean(worst_01) if worst_01 else 0.0
        w10 = np.mean(worst_10) if worst_10 else 0.0
        all_w = worst_01 + worst_10
        overall = np.mean(all_w) if all_w else 0.0
        lines.append(f"| {method:<8} | {w01:.4f} | {w10:.4f} | {overall:.4f} |\n")
    lines.append("\n")
    
    # Section 4: Per-Seed Stability
    lines.append("## 4. Per-Seed Stability (Std Dev Across Seeds)\n")
    lines.append("| Method | α=0.1 Std | α=1.0 Std |\n")
    lines.append("|--------|-----------|-----------|\n")
    for method in METHODS:
        accs_01 = [r["final_accuracy"] for r in results if r["method"] == method and r["alpha"] == 0.1]
        accs_10 = [r["final_accuracy"] for r in results if r["method"] == method and r["alpha"] == 1.0]
        s01 = np.std(accs_01) if len(accs_01) > 1 else 0.0
        s10 = np.std(accs_10) if len(accs_10) > 1 else 0.0
        lines.append(f"| {method:<8} | {s01:.4f} | {s10:.4f} |\n")
    lines.append("\n")
    
    # Section 5: Communication Efficiency
    lines.append("## 5. Communication Efficiency\n")
    lines.append("| Method | α | Avg Bytes/Round | Total Bytes |\n")
    lines.append("|--------|---|-----------------|-------------|\n")
    for method in METHODS:
        for alpha in ALPHAS:
            comms = []
            for r in results:
                if r["method"] == method and r["alpha"] == alpha:
                    total = r.get("total_communication_bytes", 0)
                    if total > 0:
                        comms.append(total)
            if comms:
                avg = np.mean(comms)
                lines.append(f"| {method:<8} | {alpha} | {avg/ROUNDS:,.0f} | {avg:,.0f} |\n")
    lines.append("\n")
    
    # Section 6: Convergence Curves
    lines.append("## 6. Convergence Summary\n")
    for method in METHODS:
        for alpha in ALPHAS:
            runs = [r for r in results if r["method"] == method and r["alpha"] == alpha]
            if runs and "per_round" in runs[0]:
                final_accs = [r["per_round"][-1]["global_metrics"]["mean_client_accuracy"] if r["per_round"] else 0 for r in runs]
                lines.append(f"- **{method} α={alpha}**: final mean={np.mean(final_accs):.4f}\n")
            elif runs:
                lines.append(f"- **{method} α={alpha}**: final mean={np.mean([r['final_accuracy'] for r in runs]):.4f}\n")
    lines.append("\n")
    
    # Section 7: Limitations & Notes
    lines.append("## 7. Limitations & Notes\n")
    lines.append(f"1. **MOON local_epochs**: Reduced to {MOON_LOCAL_EPOCHS} (from {LOCAL_EPOCHS}) due to computational constraints.\n")
    lines.append("2. **Hardware**: NVIDIA GeForce RTX 2050 with limited VRAM.\n")
    lines.append("3. **DataLoader**: num_workers=0 to avoid Windows multiprocessing issues.\n")
    lines.append("4. **Dataset**: CIFAR-10 with Dirichlet partitioning.\n")
    lines.append("5. **Checkpoints**: Existing 20-round FedAvg runs preserved.\n")
    lines.append("\n")
    
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"\n[REPORT] Saved to {report_path}")
    return report_path

def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    results = []
    for f in RUNS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except:
            pass
    
    print(f"[INIT] Found {len(results)} existing runs")
    print(f"[CONFIG] alphas={ALPHAS}, seeds={SEEDS}, methods={METHODS}")
    print(f"[CONFIG] rounds={ROUNDS}, local_epochs={LOCAL_EPOCHS}, moon_local_epochs={MOON_LOCAL_EPOCHS}")
    print("=" * 60)
    
    total = len(ALPHAS) * len(SEEDS) * len(METHODS)
    completed = 0
    
    for alpha in ALPHAS:
        for seed in SEEDS:
            for method in METHODS:
                run_id = f"{method.lower()}_a{alpha}_s{seed}"
                if any(r.get("run_id") == run_id for r in results):
                    completed += 1
                    continue
                
                try:
                    result = run_single(method, alpha, seed)
                    results.append(result)
                    completed += 1
                    print(f"[PROGRESS] {completed}/{total} completed")
                except Exception as e:
                    print(f"[FAILED] {run_id}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"[COMPLETE] {completed}/{total} runs finished")
    
    # Generate report
    report_path = generate_report(results)
    print(f"[DONE] Report: {report_path}")

if __name__ == "__main__":
    main()
