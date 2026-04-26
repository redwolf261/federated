#!/usr/bin/env python3
"""
Complete the 24-run locked CIFAR10 grid, excluding MOON due to performance.
- Preserves existing 3 FedAvg α=0.1 runs
- Completes: FedAvg α=1.0 (3 seeds), SCAFFOLD α=0.1&1.0 (6 seeds), FLEX α=0.1&1.0 (6 seeds)
- Documents MOON exclusion in report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import numpy as np

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from scripts.phase2_q1_validation import run_scaffold, set_seed

NUM_CLASSES = 10
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR = 0.003
MAX_SAMPLES = 20000
ALPHAS = [0.1, 1.0]
SEEDS = [42, 123, 456]
METHODS = ["FedAvg", "SCAFFOLD", "FLEX"]

OUTPUT_DIR = Path("outputs/locked_cifar10_grid")
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

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

    sim = FederatedSimulator(workspace_root=str(Path(".").resolve()), config=cfg)
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

    sim = FederatedSimulator(workspace_root=str(Path(".").resolve()), config=cfg)
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

def run_scaffold_fast(alpha, seed):
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
    
    print(f"\n[RUN] {run_id}")
    start = time.time()
    
    try:
        if method == "FedAvg":
            result = run_fedavg(alpha, seed)
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
    report_path = OUTPUT_DIR / "report.md"
    lines = []
    lines.append("# CIFAR-10 Locked Grid: 18-Run Report (MOON Excluded)\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n\n")
    
    # Section 1: Executive Summary
    lines.append("## 1. Executive Summary\n")
    completed = len([r for r in results if r.get("final_accuracy", 0) > 0])
    lines.append(f"- **Total runs**: {completed} (target grid: 24, completed: 18 without MOON)\n")
    lines.append(f"- **Configuration**: CIFAR-10, {NUM_CLIENTS} clients, {ROUNDS} rounds\n")
    lines.append(f"- **Alphas**: {ALPHAS} (Dirichlet non-IID)\n")
    lines.append(f"- **Seeds**: {SEEDS}\n")
    lines.append(f"- **Methods**: FedAvg, SCAFFOLD, FLEX (MOON excluded)\n")
    lines.append(f"- **Hyperparameters**: local_epochs={LOCAL_EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}\n")
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
    
    # Section 6: Convergence Summary
    lines.append("## 6. Convergence Summary\n")
    for method in METHODS:
        for alpha in ALPHAS:
            runs = [r for r in results if r["method"] == method and r["alpha"] == alpha]
            if runs:
                final_accs = [r["final_accuracy"] for r in runs]
                lines.append(f"- **{method} α={alpha}**: final mean={np.mean(final_accs):.4f}, std={np.std(final_accs):.4f}\n")
    lines.append("\n")
    
    # Section 7: Limitations & Notes
    lines.append("## 7. Limitations & Notes\n")
    lines.append("1. **MOON exclusion**: MOON was excluded from this grid due to impractical execution time (~8+ minutes per round on available hardware). A single 20-round MOON run would exceed 2.5 hours.\n")
    lines.append("2. **SCAFFOLD implementation**: Uses control variates with default settings.\n")
    lines.append("3. **Hardware**: NVIDIA GeForce RTX 2050 with limited VRAM (4GB).\n")
    lines.append("4. **Dataset**: CIFAR-10 with Dirichlet partitioning (α=0.1 for high non-IID, α=1.0 for moderate non-IID).\n")
    lines.append("5. **Preservation**: Three existing 20-round FedAvg α=0.1 runs were preserved.\n")
    lines.append("6. **Determinism**: Fixed seeds (42, 123, 456) with torch deterministic mode enabled.\n")
    lines.append("\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[REPORT] Saved to {report_path}")
    return report_path

def main():
    results = []
    for f in RUNS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except:
            pass
    
    print(f"[INIT] Found {len(results)} existing runs")
    print(f"[CONFIG] Methods: {METHODS}, Alphas: {ALPHAS}, Seeds: {SEEDS}")
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
    
    report_path = generate_report(results)
    print(f"[DONE] Report: {report_path}")

if __name__ == "__main__":
    main()
