"""Phase 2: Multi-Client Federated Validation (Q1-Grade).

Usage:
    python scripts/phase2_q1_validation.py --stage 0   # Centralized baseline
    python scripts/phase2_q1_validation.py --stage 1   # IID 2-client sanity
    python scripts/phase2_q1_validation.py --stage 2   # Scale clients (IID)
    python scripts/phase2_q1_validation.py --stage 3   # Non-IID Dirichlet
    python scripts/phase2_q1_validation.py --stage 4   # Client diagnostics
    python scripts/phase2_q1_validation.py --stage 5   # Communication accounting
    python scripts/phase2_q1_validation.py --stage 6   # SCAFFOLD/MOON baselines
    python scripts/phase2_q1_validation.py --stage 9   # Ablation study
    python scripts/phase2_q1_validation.py --all        # Run stages 0-6,9
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.data.dataset_registry import DatasetRegistry
from flex_persona.evaluation.metrics import Evaluator
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.models.client_model import ClientModel
from flex_persona.models.adapter_network import AdapterNetwork
from flex_persona.models.initialization import initialize_module_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2_q1"
FEMNIST_NUM_CLASSES = 62


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_centralized_model(num_classes: int = FEMNIST_NUM_CLASSES) -> ClientModel:
    """Build a single SmallCNN model for centralized training."""
    backbone = SmallCNNBackbone(in_channels=1)
    adapter = AdapterNetwork(input_dim=backbone.output_dim, shared_dim=64)
    model = ClientModel(backbone=backbone, adapter=adapter, num_classes=num_classes)
    initialize_module_weights(model)
    return model


def load_femnist_data(max_samples: int | None = None):
    """Load FEMNIST dataset as tensors."""
    registry = DatasetRegistry(PROJECT_ROOT)
    artifact = registry.load("femnist", max_rows=max_samples)
    images = artifact.payload["images"]
    labels = artifact.payload["labels"]
    return images, labels


# ═══════════════════════════════════════════════════════════════════
# STAGE 0: Centralized Baseline
# ═══════════════════════════════════════════════════════════════════

def train_centralized(seed: int, epochs: int = 10, lr: float = 0.003,
                      batch_size: int = 64, max_samples: int = 20000) -> float:
    """Train centralized model on FEMNIST. Returns test accuracy."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  Centralized Training | Seed={seed} | Epochs={epochs}")
    print(f"{'='*60}")

    images, labels = load_femnist_data(max_samples=max_samples)

    # 80/20 train/test split
    n = images.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(images[train_idx], labels[train_idx])
    test_ds = TensorDataset(images[test_idx], labels[test_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_centralized_model()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_updates = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_task(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_updates += 1
            epoch_loss += loss.item() * y_batch.size(0)
            epoch_samples += y_batch.size(0)

        avg_loss = epoch_loss / max(epoch_samples, 1)
        # Eval every epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                preds = model.forward_task(x_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
        acc = correct / max(total, 1)
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.4f} ({correct}/{total})")

    print(f"  Total gradient updates: {total_updates}")
    print(f"  Final accuracy: {acc:.4f}")
    return acc


def run_stage0():
    """Stage 0: Lock the reference system with centralized baseline."""
    print("\n" + "█"*60)
    print("  STAGE 0: CENTRALIZED BASELINE")
    print("█"*60)

    seeds = [42, 123, 456]
    results = {}
    accuracies = []

    for seed in seeds:
        acc = train_centralized(seed)
        results[f"seed_{seed}"] = acc
        accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    results["mean"] = mean_acc
    results["std"] = std_acc
    results["seeds"] = seeds

    print(f"\n{'='*60}")
    print(f"  STAGE 0 RESULTS: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*60}")

    if std_acc > 0.02:
        print("  ⚠️  INSTABILITY DETECTED: std > 2%")
        print("  Fix before continuing to Stage 1.")
    else:
        print("  ✅ Variance stable (< 2%)")

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage0_centralized.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: IID FedAvg (2 Clients) — Sanity Check
# ═══════════════════════════════════════════════════════════════════

def run_fedavg_iid(num_clients: int, rounds: int, local_epochs: int,
                   seed: int, lr: float = 0.003, batch_size: int = 64,
                   max_samples: int = 20000, label: str = "") -> dict:
    """Run FedAvg with IID split. Returns result dict."""
    set_seed(seed)
    print(f"\n  FedAvg IID | Clients={num_clients} | Rounds={rounds} | "
          f"LocalEp={local_epochs} | Seed={seed} {label}")

    cfg = ExperimentConfig(
        experiment_name=f"fedavg_iid_{num_clients}c_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="iid",
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]  # Homogeneous for FedAvg
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = rounds
    cfg.training.local_epochs = local_epochs
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    # Evaluate final accuracy per client
    client_accs = {}
    for client in sim.clients:
        acc = client.evaluate_accuracy()
        client_accs[client.client_id] = acc

    mean_acc = Evaluator.mean_client_accuracy(client_accs)
    worst_acc = Evaluator.worst_client_accuracy(client_accs)

    # Log weight norms from last round
    last_state = history[-1] if history else None
    fedavg_meta = last_state.metadata.get("fedavg", {}) if last_state else {}

    result = {
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "seed": seed,
        "mean_accuracy": mean_acc,
        "worst_accuracy": worst_acc,
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
        "global_weight_norm": fedavg_meta.get("global_parameter_norm", None),
        "client_weight_norms": fedavg_meta.get("client_parameter_norms", None),
    }

    # Print per-round convergence
    for i, state in enumerate(history):
        eval_meta = state.metadata.get("evaluation", {})
        r_mean = eval_meta.get("mean_client_accuracy", "N/A")
        fm = state.metadata.get("fedavg", {})
        gnorm = fm.get("global_parameter_norm", "N/A")
        if isinstance(r_mean, float):
            print(f"    Round {i+1:3d}: mean_acc={r_mean:.4f}, "
                  f"global_norm={gnorm}")

    print(f"  Final: mean={mean_acc:.4f}, worst={worst_acc:.4f}")
    return result


def run_stage1(centralized_acc: float | None = None):
    """Stage 1: IID 2-client FedAvg sanity check."""
    print("\n" + "█"*60)
    print("  STAGE 1: IID FedAvg (2 Clients) SANITY CHECK")
    print("█"*60)

    # Budget matching: centralized does 10 epochs
    # With 2 clients, rounds=10, local_epochs=1 matches budget
    seeds = [42, 123, 456]
    results = {"experiments": [], "seeds": seeds}
    accuracies = []

    for seed in seeds:
        r = run_fedavg_iid(num_clients=2, rounds=10, local_epochs=1, seed=seed)
        results["experiments"].append(r)
        accuracies.append(r["mean_accuracy"])

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    results["mean"] = mean_acc
    results["std"] = std_acc

    print(f"\n{'='*60}")
    print(f"  STAGE 1 RESULTS: {mean_acc:.4f} ± {std_acc:.4f}")
    if centralized_acc is not None:
        gap = abs(mean_acc - centralized_acc)
        results["gap_to_centralized"] = gap
        print(f"  Gap to centralized: {gap:.4f} "
              f"({'✅ PASS' if gap < 0.02 else '⚠️  GAP > 2%'})")
    print(f"{'='*60}")

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage1_iid_2client.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: Scale Clients (Still IID)
# ═══════════════════════════════════════════════════════════════════

def run_stage2(centralized_acc: float | None = None):
    """Stage 2: 5 and 10 client IID scaling."""
    print("\n" + "█"*60)
    print("  STAGE 2: SCALE CLIENTS (IID)")
    print("█"*60)

    configs = [
        {"num_clients": 5, "rounds": 10, "local_epochs": 1},
        {"num_clients": 10, "rounds": 10, "local_epochs": 1},
    ]
    seeds = [42, 123, 456]
    all_results = {}

    for cfg in configs:
        nc = cfg["num_clients"]
        key = f"{nc}_clients"
        accs = []
        exps = []
        print(f"\n  --- {nc} Clients ---")
        for seed in seeds:
            r = run_fedavg_iid(
                num_clients=nc, rounds=cfg["rounds"],
                local_epochs=cfg["local_epochs"], seed=seed,
            )
            exps.append(r)
            accs.append(r["mean_accuracy"])

        mean_a = float(np.mean(accs))
        std_a = float(np.std(accs))
        all_results[key] = {
            "mean": mean_a, "std": std_a,
            "experiments": exps, "seeds": seeds,
        }
        drop = (centralized_acc - mean_a) if centralized_acc else None
        print(f"  {nc} clients: {mean_a:.4f} ± {std_a:.4f}"
              + (f" (drop={drop:.4f})" if drop is not None else ""))

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage2_scale_clients.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: Non-IID Dirichlet
# ═══════════════════════════════════════════════════════════════════

def run_fedavg_dirichlet(alpha: float, num_clients: int, rounds: int,
                         local_epochs: int, seed: int, lr: float = 0.003,
                         batch_size: int = 64, max_samples: int = 20000) -> dict:
    """Run FedAvg with Dirichlet non-IID split."""
    set_seed(seed)
    print(f"    Dirichlet α={alpha} | Clients={num_clients} | Seed={seed}")

    cfg = ExperimentConfig(
        experiment_name=f"fedavg_dir_a{alpha}_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = rounds
    cfg.training.local_epochs = local_epochs
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    client_accs = {}
    for client in sim.clients:
        client_accs[client.client_id] = client.evaluate_accuracy()

    mean_acc = Evaluator.mean_client_accuracy(client_accs)
    worst_acc = Evaluator.worst_client_accuracy(client_accs)
    acc_values = list(client_accs.values())
    std_clients = float(np.std(acc_values)) if acc_values else 0.0

    print(f"      mean={mean_acc:.4f}, worst={worst_acc:.4f}, "
          f"std_clients={std_clients:.4f}")

    return {
        "alpha": alpha, "num_clients": num_clients, "seed": seed,
        "mean_accuracy": mean_acc, "worst_accuracy": worst_acc,
        "std_across_clients": std_clients,
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
    }


def run_stage3(centralized_acc: float | None = None):
    """Stage 3: Non-IID with Dirichlet α = 1.0, 0.5, 0.1."""
    print("\n" + "█"*60)
    print("  STAGE 3: NON-IID DIRICHLET")
    print("█"*60)

    alphas = [1.0, 0.5, 0.1]
    seeds = [42, 123, 456]
    all_results = {}

    for alpha in alphas:
        key = f"alpha_{alpha}"
        exps = []
        accs, worsts, stds = [], [], []
        print(f"\n  --- α = {alpha} ---")
        for seed in seeds:
            r = run_fedavg_dirichlet(
                alpha=alpha, num_clients=10, rounds=20,
                local_epochs=3, seed=seed,
            )
            exps.append(r)
            accs.append(r["mean_accuracy"])
            worsts.append(r["worst_accuracy"])
            stds.append(r["std_across_clients"])

        summary = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "mean_worst": float(np.mean(worsts)),
            "mean_client_std": float(np.mean(stds)),
            "experiments": exps,
        }
        drop = (centralized_acc - summary["mean"]) if centralized_acc else None
        print(f"  α={alpha}: {summary['mean']:.4f} ± {summary['std']:.4f}, "
              f"worst={summary['mean_worst']:.4f}"
              + (f", drop={drop:.4f}" if drop is not None else ""))
        all_results[key] = summary

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage3_noniid.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ═══════════════════════════════════════════════════════════════════
# STAGE 4: Client-Level Diagnostics
# ═══════════════════════════════════════════════════════════════════

def run_stage4():
    """Stage 4: Detailed client-level diagnostics from Stage 3 results."""
    print("\n" + "█"*60)
    print("  STAGE 4: CLIENT-LEVEL DIAGNOSTICS")
    print("█"*60)

    # Load stage3 results
    stage3_path = OUTPUT_DIR / "stage3_noniid.json"
    if not stage3_path.exists():
        print("  ⚠️  Stage 3 results not found. Running Stage 3 first...")
        run_stage3()

    with open(stage3_path) as f:
        stage3 = json.load(f)

    diagnostics = {}
    for alpha_key, data in stage3.items():
        print(f"\n  --- {alpha_key} ---")
        for exp in data.get("experiments", []):
            accs = exp.get("client_accuracies", {})
            if not accs:
                continue
            vals = list(accs.values())
            stats = {
                "min": float(min(vals)),
                "max": float(max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
            }
            seed = exp.get("seed", "?")
            print(f"    Seed {seed}: min={stats['min']:.4f}, "
                  f"max={stats['max']:.4f}, mean={stats['mean']:.4f}, "
                  f"std={stats['std']:.4f}")
            diagnostics[f"{alpha_key}_seed{seed}"] = stats

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage4_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    return diagnostics


# ═══════════════════════════════════════════════════════════════════
# STAGE 5: Communication Accounting
# ═══════════════════════════════════════════════════════════════════

def run_stage5():
    """Stage 5: Communication cost comparison FedAvg vs Prototype."""
    print("\n" + "█"*60)
    print("  STAGE 5: COMMUNICATION ACCOUNTING")
    print("█"*60)

    seed = 42
    num_clients = 10
    rounds = 20
    local_epochs = 3

    def run_and_measure(agg_mode: str) -> dict:
        set_seed(seed)
        cfg = ExperimentConfig(
            experiment_name=f"comm_{agg_mode}",
            dataset_name="femnist",
            num_clients=num_clients,
            random_seed=seed,
            partition_mode="dirichlet",
            dirichlet_alpha=0.5,
            output_dir=str(OUTPUT_DIR),
        )
        cfg.model.num_classes = FEMNIST_NUM_CLASSES
        cfg.model.client_backbones = ["small_cnn"]
        cfg.training.aggregation_mode = agg_mode
        cfg.training.rounds = rounds
        cfg.training.local_epochs = local_epochs
        cfg.training.learning_rate = 0.003
        cfg.training.batch_size = 64
        cfg.training.max_samples_per_client = 2000

        sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
        sim.run_experiment()
        comm = sim.communication_tracker.summarize()
        comm["per_round"] = {
            str(k): v for k, v in sim.communication_tracker.per_round.items()
        }
        return comm

    print("\n  Measuring FedAvg communication...")
    fedavg_comm = run_and_measure("fedavg")
    print(f"  FedAvg total: {fedavg_comm['total_bytes']:,} bytes")

    print("\n  Measuring Prototype communication...")
    proto_comm = run_and_measure("prototype")
    print(f"  Prototype total: {proto_comm['total_bytes']:,} bytes")

    ratio = proto_comm["total_bytes"] / max(fedavg_comm["total_bytes"], 1)
    reduction = (1.0 - ratio) * 100

    results = {
        "fedavg": fedavg_comm,
        "prototype": proto_comm,
        "compression_ratio": ratio,
        "reduction_percent": reduction,
        "config": {
            "num_clients": num_clients, "rounds": rounds,
            "local_epochs": local_epochs, "seed": seed,
        },
    }

    print(f"\n  Compression ratio: {ratio:.4f}")
    print(f"  Communication reduction: {reduction:.1f}%")

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage5_communication.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════
# STAGE 6: Baseline Comparison (FedAvg, SCAFFOLD, MOON, Ours)
# ═══════════════════════════════════════════════════════════════════

class ScaffoldState:
    """Tracks SCAFFOLD control variates."""
    def __init__(self, model: nn.Module):
        self.c_global = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                         if p.requires_grad}
        self.c_locals = {}

    def init_client(self, client_id: int, model: nn.Module):
        if client_id not in self.c_locals:
            self.c_locals[client_id] = {
                n: torch.zeros_like(p) for n, p in model.named_parameters()
                if p.requires_grad
            }


def run_scaffold(num_clients: int, rounds: int, local_epochs: int,
                 seed: int, alpha: float = 0.5, lr: float = 0.003,
                 batch_size: int = 64, max_samples: int = 20000) -> dict:
    """SCAFFOLD with control variates."""
    set_seed(seed)
    print(f"    SCAFFOLD | α={alpha} | Clients={num_clients} | Seed={seed}")

    # Build data
    cfg = ExperimentConfig(
        experiment_name=f"scaffold_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()

    # Build global model
    global_model = build_centralized_model()
    global_model.to(DEVICE)
    scaffold = ScaffoldState(global_model)
    criterion = nn.CrossEntropyLoss()

    for b in bundles:
        scaffold.init_client(b.client_id, global_model)

    round_accs = []
    for rnd in range(1, rounds + 1):
        client_deltas = []
        sample_counts = []

        for bundle in bundles:
            cid = bundle.client_id
            # Clone global model for this client
            local_model = copy.deepcopy(global_model)
            local_model.to(DEVICE)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

            c_local = scaffold.c_locals[cid]
            c_global = scaffold.c_global

            # Local training with SCAFFOLD correction
            local_model.train()
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    logits = local_model.forward_task(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    # Apply SCAFFOLD correction: subtract c_local, add c_global
                    with torch.no_grad():
                        for n, p in local_model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                p.grad.add_(
                                    c_global[n].to(DEVICE) - c_local[n].to(DEVICE)
                                )
                    optimizer.step()

            # Compute delta and update control variates
            delta = {}
            K = local_epochs * len(bundle.train_loader)  # total steps
            with torch.no_grad():
                for n, p_global in global_model.named_parameters():
                    if not p_global.requires_grad:
                        continue
                    p_local = dict(local_model.named_parameters())[n]
                    delta[n] = (p_local.data.cpu() - p_global.data.cpu())
                    # Update c_local
                    new_c = (c_local[n]
                             - c_global[n]
                             + (p_global.data.cpu() - p_local.data.cpu()) / (K * lr))
                    scaffold.c_locals[cid][n] = new_c

            client_deltas.append(delta)
            sample_counts.append(bundle.num_samples)

        # Aggregate deltas (weighted)
        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg_delta = torch.zeros_like(p.data.cpu())
                agg_c_delta = torch.zeros_like(p.data.cpu())
                for i, (delta, ns) in enumerate(zip(client_deltas, sample_counts)):
                    w = ns / total_n
                    agg_delta += w * delta[n]
                    agg_c_delta += (scaffold.c_locals[bundles[i].client_id][n]
                                    / num_clients)
                p.data.add_(agg_delta.to(DEVICE))
                scaffold.c_global[n] = agg_c_delta

        # Evaluate
        global_model.eval()
        client_accs = {}
        for bundle in bundles:
            correct, total = 0, 0
            with torch.no_grad():
                for x_b, y_b in bundle.eval_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    preds = global_model.forward_task(x_b).argmax(dim=1)
                    correct += (preds == y_b).sum().item()
                    total += y_b.size(0)
            client_accs[bundle.client_id] = correct / max(total, 1)

        mean_a = float(np.mean(list(client_accs.values())))
        round_accs.append(mean_a)

    vals = list(client_accs.values())
    return {
        "method": "scaffold", "seed": seed,
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std_across_clients": float(np.std(vals)),
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
    }


def run_moon(num_clients: int, rounds: int, local_epochs: int,
             seed: int, alpha: float = 0.5, lr: float = 0.003,
             batch_size: int = 64, max_samples: int = 20000,
             mu: float = 1.0, temperature: float = 0.5) -> dict:
    """MOON with contrastive loss on representations."""
    set_seed(seed)
    print(f"    MOON | α={alpha} | Clients={num_clients} | Seed={seed}")

    cfg = ExperimentConfig(
        experiment_name=f"moon_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()

    global_model = build_centralized_model()
    global_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    # Track previous local models for contrastive loss
    prev_local_models = {b.client_id: copy.deepcopy(global_model) for b in bundles}

    for rnd in range(1, rounds + 1):
        client_states = []
        sample_counts = []

        for bundle in bundles:
            cid = bundle.client_id
            local_model = copy.deepcopy(global_model)
            local_model.to(DEVICE)
            prev_model = prev_local_models[cid]
            prev_model.to(DEVICE)
            prev_model.eval()
            global_model.eval()  # For computing representations

            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

            local_model.train()
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()

                    # Task loss
                    logits = local_model.forward_task(x_b)
                    task_loss = criterion(logits, y_b)

                    # MOON contrastive loss
                    z_local = local_model.forward_shared(x_b)
                    with torch.no_grad():
                        z_global = global_model.forward_shared(x_b)
                        z_prev = prev_model.forward_shared(x_b)

                    # Positive: similarity to global, Negative: similarity to prev
                    pos = cos_sim(z_local, z_global) / temperature
                    neg = cos_sim(z_local, z_prev) / temperature
                    logits_con = torch.stack([pos, neg], dim=1)
                    labels_con = torch.zeros(x_b.size(0), dtype=torch.long,
                                             device=DEVICE)
                    con_loss = nn.CrossEntropyLoss()(logits_con, labels_con)

                    loss = task_loss + mu * con_loss
                    loss.backward()
                    optimizer.step()

            # Save local model as previous for next round
            prev_local_models[cid] = copy.deepcopy(local_model).cpu()

            state = {n: p.data.cpu().clone() for n, p in local_model.named_parameters()}
            client_states.append(state)
            sample_counts.append(bundle.num_samples)

        # FedAvg aggregation
        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                agg = torch.zeros_like(p.data.cpu())
                for state, ns in zip(client_states, sample_counts):
                    if n in state:
                        agg += (ns / total_n) * state[n]
                p.data.copy_(agg.to(DEVICE))

    # Final evaluation
    global_model.eval()
    client_accs = {}
    for bundle in bundles:
        correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in bundle.eval_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                preds = global_model.forward_task(x_b).argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
        client_accs[bundle.client_id] = correct / max(total, 1)

    vals = list(client_accs.values())
    return {
        "method": "moon", "seed": seed,
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std_across_clients": float(np.std(vals)),
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
    }


def run_stage6():
    """Stage 6: Reproduce baselines — FedAvg, SCAFFOLD, MOON, Ours."""
    print("\n" + "█"*60)
    print("  STAGE 6: BASELINE COMPARISON")
    print("█"*60)

    seeds = [42, 123, 456]
    num_clients = 10
    rounds = 20
    local_epochs = 3
    alpha = 0.5

    methods_results = {}

    # --- FedAvg ---
    print("\n  === FedAvg ===")
    fedavg_accs, fedavg_worsts = [], []
    for seed in seeds:
        r = run_fedavg_dirichlet(alpha=alpha, num_clients=num_clients,
                                 rounds=rounds, local_epochs=local_epochs, seed=seed)
        fedavg_accs.append(r["mean_accuracy"])
        fedavg_worsts.append(r["worst_accuracy"])
    methods_results["fedavg"] = {
        "mean": float(np.mean(fedavg_accs)),
        "std": float(np.std(fedavg_accs)),
        "worst_mean": float(np.mean(fedavg_worsts)),
    }

    # --- SCAFFOLD ---
    print("\n  === SCAFFOLD ===")
    scaffold_accs, scaffold_worsts = [], []
    for seed in seeds:
        r = run_scaffold(num_clients=num_clients, rounds=rounds,
                         local_epochs=local_epochs, seed=seed, alpha=alpha)
        scaffold_accs.append(r["mean_accuracy"])
        scaffold_worsts.append(r["worst_accuracy"])
    methods_results["scaffold"] = {
        "mean": float(np.mean(scaffold_accs)),
        "std": float(np.std(scaffold_accs)),
        "worst_mean": float(np.mean(scaffold_worsts)),
    }

    # --- MOON ---
    print("\n  === MOON ===")
    moon_accs, moon_worsts = [], []
    for seed in seeds:
        r = run_moon(num_clients=num_clients, rounds=rounds,
                     local_epochs=local_epochs, seed=seed, alpha=alpha)
        moon_accs.append(r["mean_accuracy"])
        moon_worsts.append(r["worst_accuracy"])
    methods_results["moon"] = {
        "mean": float(np.mean(moon_accs)),
        "std": float(np.std(moon_accs)),
        "worst_mean": float(np.mean(moon_worsts)),
    }

    # --- Ours (FLEX-Persona Prototype) ---
    print("\n  === FLEX-Persona (Prototype) ===")
    ours_accs, ours_worsts = [], []
    for seed in seeds:
        set_seed(seed)
        cfg = ExperimentConfig(
            experiment_name=f"prototype_s{seed}",
            dataset_name="femnist",
            num_clients=num_clients,
            random_seed=seed,
            partition_mode="dirichlet",
            dirichlet_alpha=alpha,
            output_dir=str(OUTPUT_DIR),
        )
        cfg.model.num_classes = FEMNIST_NUM_CLASSES
        cfg.model.client_backbones = ["small_cnn"]
        cfg.training.aggregation_mode = "prototype"
        cfg.training.rounds = rounds
        cfg.training.local_epochs = local_epochs
        cfg.training.learning_rate = 0.003
        cfg.training.batch_size = 64
        cfg.training.max_samples_per_client = 2000

        sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
        sim.run_experiment()

        client_accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
        vals = list(client_accs.values())
        mean_a = float(np.mean(vals))
        worst_a = float(min(vals))
        ours_accs.append(mean_a)
        ours_worsts.append(worst_a)
        print(f"    Prototype | Seed={seed} | mean={mean_a:.4f}, worst={worst_a:.4f}")

    methods_results["flex_persona"] = {
        "mean": float(np.mean(ours_accs)),
        "std": float(np.std(ours_accs)),
        "worst_mean": float(np.mean(ours_worsts)),
    }

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  {'Method':<20} {'Mean':>8} {'Std':>8} {'Worst':>8}")
    print(f"  {'-'*50}")
    for method, data in methods_results.items():
        print(f"  {method:<20} {data['mean']:>8.4f} {data['std']:>8.4f} "
              f"{data['worst_mean']:>8.4f}")
    print(f"{'='*70}")

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage6_baselines.json", "w") as f:
        json.dump(methods_results, f, indent=2)
    return methods_results


# ═══════════════════════════════════════════════════════════════════
# STAGE 9: Ablation Study
# ═══════════════════════════════════════════════════════════════════

def run_stage9():
    """Stage 9: Ablation — Full vs -clustering vs -guidance."""
    print("\n" + "█"*60)
    print("  STAGE 9: ABLATION STUDY")
    print("█"*60)

    seeds = [42, 123, 456]
    num_clients = 10
    rounds = 20
    local_epochs = 3
    alpha = 0.5

    variants = {
        "full": {"use_clustering": True, "use_guidance": True},
        "no_clustering": {"use_clustering": False, "use_guidance": True},
        "no_guidance": {"use_clustering": True, "use_guidance": False},
    }

    ablation_results = {}
    for vname, toggles in variants.items():
        print(f"\n  --- Variant: {vname} ---")
        accs, worsts = [], []
        for seed in seeds:
            set_seed(seed)
            cfg = ExperimentConfig(
                experiment_name=f"ablation_{vname}_s{seed}",
                dataset_name="femnist",
                num_clients=num_clients,
                random_seed=seed,
                partition_mode="dirichlet",
                dirichlet_alpha=alpha,
                output_dir=str(OUTPUT_DIR),
            )
            cfg.model.num_classes = FEMNIST_NUM_CLASSES
            cfg.model.client_backbones = ["small_cnn"]
            cfg.training.aggregation_mode = "prototype"
            cfg.training.rounds = rounds
            cfg.training.local_epochs = local_epochs
            cfg.training.learning_rate = 0.003
            cfg.training.batch_size = 64
            cfg.training.max_samples_per_client = 2000
            cfg.training.use_clustering = toggles["use_clustering"]
            cfg.training.use_guidance = toggles["use_guidance"]

            sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
            sim.run_experiment()

            client_accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
            vals = list(client_accs.values())
            mean_a = float(np.mean(vals))
            worst_a = float(min(vals))
            accs.append(mean_a)
            worsts.append(worst_a)
            print(f"    {vname} | Seed={seed} | mean={mean_a:.4f}")

        ablation_results[vname] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "worst_mean": float(np.mean(worsts)),
        }

    # Print ablation table
    print(f"\n{'='*60}")
    print(f"  {'Variant':<20} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for vname, data in ablation_results.items():
        print(f"  {vname:<20} {data['mean']:.4f} ± {data['std']:.4f}")
    print(f"{'='*60}")

    ensure_output_dir()
    with open(OUTPUT_DIR / "stage9_ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    return ablation_results


# ═══════════════════════════════════════════════════════════════════
# MAIN: CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Q1-Grade Validation")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run specific stage (0-6, 9)")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages sequentially")
    args = parser.parse_args()

    ensure_output_dir()
    start = time.time()

    if args.stage is not None:
        stage = args.stage
        if stage == 0:
            run_stage0()
        elif stage == 1:
            # Try to load centralized baseline
            s0_path = OUTPUT_DIR / "stage0_centralized.json"
            cent_acc = None
            if s0_path.exists():
                with open(s0_path) as f:
                    cent_acc = json.load(f).get("mean")
            run_stage1(centralized_acc=cent_acc)
        elif stage == 2:
            s0_path = OUTPUT_DIR / "stage0_centralized.json"
            cent_acc = None
            if s0_path.exists():
                with open(s0_path) as f:
                    cent_acc = json.load(f).get("mean")
            run_stage2(centralized_acc=cent_acc)
        elif stage == 3:
            s0_path = OUTPUT_DIR / "stage0_centralized.json"
            cent_acc = None
            if s0_path.exists():
                with open(s0_path) as f:
                    cent_acc = json.load(f).get("mean")
            run_stage3(centralized_acc=cent_acc)
        elif stage == 4:
            run_stage4()
        elif stage == 5:
            run_stage5()
        elif stage == 6:
            run_stage6()
        elif stage == 9:
            run_stage9()
        else:
            print(f"Unknown stage: {stage}")
            sys.exit(1)
    elif args.all:
        print("Running ALL stages sequentially...\n")
        s0 = run_stage0()
        cent_acc = s0.get("mean")
        run_stage1(centralized_acc=cent_acc)
        run_stage2(centralized_acc=cent_acc)
        run_stage3(centralized_acc=cent_acc)
        run_stage4()
        run_stage5()
        run_stage6()
        run_stage9()

        # Final summary
        print("\n" + "█"*60)
        print("  ALL STAGES COMPLETE")
        print(f"  Total time: {time.time() - start:.1f}s")
        print("█"*60)
    else:
        parser.print_help()

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
