
def run_fedprox_sweep():
    """Run FedProx for all (alpha, mu, seed) combinations and print results in strict format."""
    alphas = [0.3, 0.1]
    mus = [0.001, 0.01, 0.1]
    seeds = [42, 123, 456]
    num_clients = 10
    rounds = 30
    local_epochs = 20
    lr = 0.003
    batch_size = 64
    max_samples = 20000

    # Load FedAvg baselines for comparison
    with open(OUTPUT_DIR / "stage3_noniid.json", "r") as f:
        fedavg_results = json.load(f)

    for alpha in alphas:
        for mu in mus:
            all_accs, all_worst, all_stds, all_client_stds, all_p10, all_p90, all_gap = [], [], [], [], [], [], []
            print(f"\nα = {alpha}, μ = {mu}")
            exps = []
            for seed in seeds:
                r = run_fedprox_dirichlet(
                    alpha=alpha, num_clients=num_clients, rounds=rounds,
                    local_epochs=local_epochs, seed=seed, mu=mu, lr=lr, batch_size=batch_size, max_samples=max_samples
                )
                exps.append(r)
                all_accs.append(r["mean_accuracy"])
                all_worst.append(r["worst_accuracy"])
                all_stds.append(r["std_across_clients"])
                # Fairness quantification
                client_accs = list(r["client_accuracies"].values())
                all_client_stds.append(float(np.std(client_accs)))
                all_p10.append(float(np.percentile(client_accs, 10)))
                all_p90.append(float(np.percentile(client_accs, 90)))
                all_gap.append(all_p90[-1] - all_p10[-1])
            mean = float(np.mean(all_accs))
            std = float(np.std(all_accs))
            worst = float(np.mean(all_worst))
            client_std = float(np.mean(all_client_stds))
            p10 = float(np.mean(all_p10))
            p90 = float(np.mean(all_p90))
            gap = float(np.mean(all_gap))
            print(f"* Mean: {mean:.4f}")
            print(f"* Std: {std:.4f}")
            print(f"* Worst client: {worst:.4f}")
            print(f"* Client std: {client_std:.4f}")
            print(f"* p10 / p90 / gap: {p10:.4f} / {p90:.4f} / {gap:.4f}")

            # Comparison vs FedAvg
            fedavg_key = f"alpha_{alpha}"
            if fedavg_key in fedavg_results:
                fedavg = fedavg_results[fedavg_key]
                fedavg_mean = fedavg["mean"]
                fedavg_worst = fedavg["mean_worst"]
                fedavg_gap = fedavg["gap"] if "gap" in fedavg else None
                print("\nComparison vs FedAvg:")
                print(f"* Mean Δ: {mean - fedavg_mean:+.4f}")
                print(f"* Worst-client Δ: {worst - fedavg_worst:+.4f}")
                if fedavg_gap is not None:
                    print(f"* Gap Δ: {gap - fedavg_gap:+.4f}")
            else:
                print("No FedAvg baseline found for comparison.")
def run_fedprox_dirichlet(alpha: float, num_clients: int, rounds: int,
                         local_epochs: int, seed: int, mu: float,
                         lr: float = 0.003, batch_size: int = 64, max_samples: int = 20000) -> dict:
    """Run FedProx with Dirichlet non-IID split."""
    set_seed(seed)
    print(f"    FedProx alpha={alpha} | mu={mu} | Clients={num_clients} | Seed={seed}")

    cfg = ExperimentConfig(
        experiment_name=f"fedprox_dir_a{alpha}_mu{mu}_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "fedprox"
    cfg.training.fedprox_mu = mu
    cfg.training.rounds = rounds
    cfg.training.local_epochs = local_epochs
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    client_accs = {}
    client_histograms = {}
    for i, client in enumerate(sim.clients):
        client_accs[client.client_id] = client.evaluate_accuracy()
        if hasattr(client, 'train_loader') and hasattr(client.train_loader.dataset, 'tensors'):
            labels = client.train_loader.dataset.tensors[1]
            unique, counts = torch.unique(labels, return_counts=True)
            class_hist = {int(k): int(v) for k, v in zip(unique, counts)}
            client_histograms[client.client_id] = class_hist
        elif hasattr(client, 'class_histogram'):
            client_histograms[client.client_id] = client.class_histogram
        else:
            client_histograms[client.client_id] = {}

    mean_acc = Evaluator.mean_client_accuracy(client_accs)
    worst_acc = Evaluator.worst_client_accuracy(client_accs)
    acc_values = list(client_accs.values())
    std_clients = float(np.std(acc_values)) if acc_values else 0.0

    acc_items = sorted(client_accs.items(), key=lambda x: x[1])
    low_id, low_acc = acc_items[0]
    high_id, high_acc = acc_items[-1]
    print(f"      mean={mean_acc:.4f}, worst={worst_acc:.4f}, std_clients={std_clients:.4f}")
    print(f"      LOW client {low_id}: acc={low_acc:.4f}, samples={sum(client_histograms[low_id].values())}, class_dist={client_histograms[low_id]}")
    print(f"      HIGH client {high_id}: acc={high_acc:.4f}, samples={sum(client_histograms[high_id].values())}, class_dist={client_histograms[high_id]}")


    return {
        "alpha": alpha, "num_clients": num_clients, "seed": seed, "mu": mu,
        "mean_accuracy": mean_acc, "worst_accuracy": worst_acc,
        "std_across_clients": std_clients,
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
        "client_histograms": client_histograms,
    }

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _dataset_geometry(dataset_name: str) -> tuple[int, int, int, int]:
    normalized = dataset_name.lower().strip()
    if normalized == "femnist":
        return 1, 28, 28, 62
    if normalized == "cifar10":
        return 3, 32, 32, 10
    if normalized == "cifar100":
        return 3, 32, 32, 100
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_centralized_model(dataset_name: str = "femnist", num_classes: int | None = None) -> ClientModel:
    """Build a single SmallCNN model for centralized training."""
    in_channels, height, width, default_classes = _dataset_geometry(dataset_name)
    if num_classes is None:
        num_classes = default_classes
    backbone = SmallCNNBackbone(in_channels=in_channels, input_height=height, input_width=width)
    adapter = AdapterNetwork(input_dim=backbone.output_dim, shared_dim=64)
    model = ClientModel(backbone=backbone, adapter=adapter, num_classes=num_classes)
    initialize_module_weights(model)
    return model


def load_dataset_data(dataset_name: str, max_samples: int | None = None):
    """Load the requested dataset as tensors."""
    registry = DatasetRegistry(PROJECT_ROOT)
    artifact = registry.load(dataset_name, max_rows=max_samples)
    if artifact.name == "femnist":
        images = artifact.payload["images"]
        labels = artifact.payload["labels"]
    else:
        images = artifact.payload["train_images"]
        labels = artifact.payload["train_labels"]
        # CIFAR loaders use max_train_samples/max_test_samples, not max_rows.
        # Apply an explicit cap here so centralized and federated calls can match sample budgets.
        if max_samples is not None:
            cap = int(max_samples)
            images = images[:cap]
            labels = labels[:cap]
    return images, labels


# ═══════════════════════════════════════════════════════════════════
# STAGE 0: Centralized Baseline
# ═══════════════════════════════════════════════════════════════════

def train_centralized(dataset_name: str, seed: int, epochs: int = 10, lr: float = 0.003,
                      batch_size: int = 64, max_samples: int = 20000) -> float:
    """Train centralized model on the requested dataset. Returns test accuracy."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  Centralized Training | Dataset={dataset_name} | Seed={seed} | Epochs={epochs}")
    print(f"{'='*60}")

    images, labels = load_dataset_data(dataset_name, max_samples=max_samples)

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

    model = build_centralized_model(dataset_name=dataset_name)
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
        acc = train_centralized("femnist", seed)
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

def run_fedavg_iid(dataset_name: str, num_clients: int, rounds: int, local_epochs: int,
                   seed: int, lr: float = 0.003, batch_size: int = 64,
                   max_samples: int = 20000, label: str = "") -> dict:
    """Run FedAvg with IID split. Returns result dict."""
    set_seed(seed)
    print(f"\n  FedAvg IID | Clients={num_clients} | Rounds={rounds} | "
          f"LocalEp={local_epochs} | Seed={seed} {label}")

    cfg = ExperimentConfig(
        experiment_name=f"fedavg_iid_{num_clients}c_s{seed}",
        dataset_name=dataset_name,
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="iid",
        output_dir=str(OUTPUT_DIR),
    )
    _, _, _, num_classes = _dataset_geometry(dataset_name)
    cfg.model.num_classes = num_classes
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
        r = run_fedavg_iid("femnist", num_clients=2, rounds=10, local_epochs=1, seed=seed)
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
        {"num_clients": 10, "rounds": 10, "local_epochs": 5},  # Robustness check: E=5 for 10-client scaling
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
    print(f"    Dirichlet alpha={alpha} | Clients={num_clients} | Seed={seed}")

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
    client_histograms = {}
    for i, client in enumerate(sim.clients):
        client_accs[client.client_id] = client.evaluate_accuracy()
        # Access class_histogram from data manager
        if hasattr(client, 'train_loader') and hasattr(client.train_loader.dataset, 'tensors'):
            # Try to infer class histogram from training labels
            labels = client.train_loader.dataset.tensors[1]
            unique, counts = torch.unique(labels, return_counts=True)
            class_hist = {int(k): int(v) for k, v in zip(unique, counts)}
            client_histograms[client.client_id] = class_hist
        elif hasattr(client, 'class_histogram'):
            client_histograms[client.client_id] = client.class_histogram
        else:
            client_histograms[client.client_id] = {}

    mean_acc = Evaluator.mean_client_accuracy(client_accs)
    worst_acc = Evaluator.worst_client_accuracy(client_accs)
    acc_values = list(client_accs.values())
    std_clients = float(np.std(acc_values)) if acc_values else 0.0

    # Identify lowest and highest accuracy clients
    acc_items = sorted(client_accs.items(), key=lambda x: x[1])
    low_id, low_acc = acc_items[0]
    high_id, high_acc = acc_items[-1]
    print(f"      mean={mean_acc:.4f}, worst={worst_acc:.4f}, std_clients={std_clients:.4f}")
    print(f"      LOW client {low_id}: acc={low_acc:.4f}, samples={sum(client_histograms[low_id].values())}, class_dist={client_histograms[low_id]}")
    print(f"      HIGH client {high_id}: acc={high_acc:.4f}, samples={sum(client_histograms[high_id].values())}, class_dist={client_histograms[high_id]}")

    return {
        "alpha": alpha, "num_clients": num_clients, "seed": seed,
        "mean_accuracy": mean_acc, "worst_accuracy": worst_acc,
        "std_across_clients": std_clients,
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
        "client_histograms": client_histograms,
    }


def run_stage3(centralized_acc: float | None = None):
    """Stage 3: Non-IID with Dirichlet α = 1.0, 0.5, 0.1."""
    print("\n" + "█"*60)
    print("  STAGE 3: NON-IID DIRICHLET")
    print("█"*60)

    alphas = [0.1]  # Now run only α=0.1 for severe Non-IID
    seeds = [42, 123, 456]
    all_results = {}

    for alpha in alphas:
        key = f"alpha_{alpha}"
        exps = []
        accs, worsts, stds = [], [], []
        print(f"\n  --- α = {alpha} ---")
        for seed in seeds:
            r = run_fedavg_dirichlet(
                alpha=alpha, num_clients=10, rounds=10,
                local_epochs=5, seed=seed, lr=0.003
            )
            exps.append(r)
            accs.append(r["mean_accuracy"])
            worsts.append(r["worst_accuracy"])
            stds.append(r["std_across_clients"])

        # Fairness quantification: p10, p90, gap
        all_client_accs = []
        for exp in exps:
            all_client_accs.extend(list(exp["client_accuracies"].values()))
        all_client_accs = np.array(all_client_accs)
        p10 = float(np.percentile(all_client_accs, 10))
        p90 = float(np.percentile(all_client_accs, 90))
        gap = p90 - p10

        summary = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "mean_worst": float(np.mean(worsts)),
            "mean_client_std": float(np.mean(stds)),
            "experiments": exps,
            "p10": p10,
            "p90": p90,
            "gap": gap,
        }
        drop = (centralized_acc - summary["mean"]) if centralized_acc else None
        print(f"  alpha={alpha}: {summary['mean']:.4f} +/- {summary['std']:.4f}, "
              f"worst={summary['mean_worst']:.4f}, p10={p10:.4f}, p90={p90:.4f}, gap={gap:.4f}"
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


def run_scaffold(dataset_name: str, num_classes: int, num_clients: int, rounds: int, local_epochs: int,
                 seed: int, alpha: float = 0.5, lr: float = 0.003,
                 batch_size: int = 64, max_samples: int = 20000,
                 return_trace: bool = False,
                 zero_control: bool = False,
                 control_strength: float = 1.0,
                 apply_control: bool = True,
                 use_control_scaling: bool = False,
                 optimizer_name: str = "adam",
                 control_in_parameter_space: bool = False) -> dict:
    """SCAFFOLD with control variates.
    
    KEY FIX: Use persistent optimizer and model per client (no deepcopy/reinit each round).
    Only gradient correction differs from FedAvg.
    """
    set_seed(seed)
    print(f"    SCAFFOLD | dataset={dataset_name} | alpha={alpha} | Clients={num_clients} | Seed={seed}")

    # Build data
    cfg = ExperimentConfig(
        experiment_name=f"scaffold_s{seed}",
        dataset_name=dataset_name,
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = num_classes
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()
    partition_fingerprint = dm.partition_fingerprint(bundles)

    # Build global model
    global_model = build_centralized_model(dataset_name=dataset_name, num_classes=num_classes)
    global_model.to(DEVICE)

    # Hard reset per invocation: fresh control variates, fresh client models, fresh optimizer states.
    scaffold = ScaffoldState(global_model)
    criterion = nn.CrossEntropyLoss()

    # Persistent models and optimizers per client (initialized once, reused each round)
    client_models = {}
    client_optimizers = {}
    for b in bundles:
        scaffold.init_client(b.client_id, global_model)
        client_models[b.client_id] = build_centralized_model(dataset_name=dataset_name, num_classes=num_classes).to(DEVICE)
        # Optimizer selection for controlled SCAFFOLD compatibility tests.
        if optimizer_name.lower() == "sgd":
            client_optimizers[b.client_id] = torch.optim.SGD(client_models[b.client_id].parameters(), lr=lr)
        elif optimizer_name.lower() == "adam":
            client_optimizers[b.client_id] = torch.optim.Adam(client_models[b.client_id].parameters(), lr=lr, weight_decay=0.0)
        else:
            raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
        # Move all c_local tensors to DEVICE
        for n in scaffold.c_locals[b.client_id]:
            scaffold.c_locals[b.client_id][n] = scaffold.c_locals[b.client_id][n].to(DEVICE)
            if zero_control:
                scaffold.c_locals[b.client_id][n].zero_()
    # Move all c_global tensors to DEVICE
    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)
        if zero_control:
            scaffold.c_global[n].zero_()

    # Canonical alignment checks: every trainable parameter must map to c_global and every client c_i.
    trainable_names = [n for n, p in global_model.named_parameters() if p.requires_grad]
    for n, p in global_model.named_parameters():
        if not p.requires_grad:
            continue
        assert n in scaffold.c_global, f"missing c_global entry for {n}"
        assert scaffold.c_global[n].shape == p.shape, f"shape mismatch c_global[{n}]"
    for b in bundles:
        for n in trainable_names:
            assert n in scaffold.c_locals[b.client_id], f"missing c_i entry for client={b.client_id}, param={n}"
            assert scaffold.c_locals[b.client_id][n].shape == dict(global_model.named_parameters())[n].shape, (
                f"shape mismatch c_i client={b.client_id}, param={n}"
            )

    round_accs = []
    per_round = []
    round_debug = []
    num_params = sum(p.numel() for p in global_model.parameters())
    round_bytes = int((2 * num_clients) * num_params * 4)
    for rnd in range(1, rounds + 1):
        # Canonical SCAFFOLD requirement: freeze global snapshot at round start and
        # use this exact snapshot for all client control updates in the round.
        with torch.no_grad():
            global_before = {
                n: p.detach().clone().to(DEVICE)
                for n, p in global_model.named_parameters()
                if p.requires_grad
            }
            c_global_snapshot = {
                n: t.detach().clone().to(DEVICE)
                for n, t in scaffold.c_global.items()
            }
            c_locals_snapshot = {
                b.client_id: {
                    n: t.detach().clone().to(DEVICE)
                    for n, t in scaffold.c_locals[b.client_id].items()
                }
                for b in bundles
            }

        client_deltas = []
        sample_counts = []
        client_grad_norms = []
        client_cos_sims = []
        new_c_locals = {}

        for bundle in bundles:
            cid = bundle.client_id
            # Load global weights into persistent client model (don't deepcopy)
            client_models[cid].load_state_dict(global_model.state_dict())
            local_model = client_models[cid]
            optimizer = client_optimizers[cid]

            # One-line snapshot sanity check: global_before must equal current global
            # while clients are being processed (before aggregation).
            with torch.no_grad():
                snapshot_equal = True
                for n, p in global_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if not torch.allclose(p.detach().to(DEVICE), global_before[n], atol=0.0, rtol=0.0):
                        snapshot_equal = False
                        break
            assert snapshot_equal, "global_before snapshot drifted during client updates"

            # Freeze controls within round: use only round-start snapshots.
            c_local = c_locals_snapshot[cid]
            c_global = c_global_snapshot
            assert c_local is c_locals_snapshot[cid], "c_i snapshot changed unexpectedly"

            # Local training with SCAFFOLD correction
            local_model.train()
            grad_norm_sum = 0.0
            grad_norm_steps = 0
            cos_sim_values = []
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    logits = local_model.forward_task(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()

                    # Capture raw gradients before correction for directional diagnostics.
                    raw_grads = {}
                    with torch.no_grad():
                        for n, p in local_model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                raw_grads[n] = p.grad.detach().clone()

                    # Apply SCAFFOLD correction in gradient space when requested.
                    if (not zero_control) and apply_control and (not control_in_parameter_space):
                        with torch.no_grad():
                            for n, p in local_model.named_parameters():
                                if p.requires_grad and p.grad is not None:
                                    control_delta = c_global[n].to(DEVICE) - c_local[n].to(DEVICE)
                                    if use_control_scaling:
                                        grad_norm = float(torch.norm(p.grad).item())
                                        delta_norm = float(torch.norm(control_delta).item())
                                        if grad_norm > 0.0 and delta_norm > 0.0:
                                            ratio = delta_norm / (grad_norm + 1e-12)
                                            if ratio > 1.0:
                                                control_delta = control_delta / ratio
                                    corrected_grad = p.grad + float(control_strength) * control_delta
                                    p.grad.copy_(corrected_grad)
                    else:
                        # In zero-control mode gradients must remain unchanged.
                        with torch.no_grad():
                            for n, p in local_model.named_parameters():
                                if n in raw_grads and p.grad is not None:
                                    if zero_control:
                                        assert torch.allclose(raw_grads[n], p.grad, atol=1e-8, rtol=0.0), (
                                            f"zero_control gradient changed for {n}"
                                        )

                    # Directional check: cosine(raw_grad, corrected_grad).
                    with torch.no_grad():
                        for n, p in local_model.named_parameters():
                            if n in raw_grads and p.grad is not None:
                                raw = raw_grads[n].flatten()
                                corrected = p.grad.flatten()
                                raw_norm = float(torch.norm(raw).item())
                                corr_norm = float(torch.norm(corrected).item())
                                if raw_norm > 0.0 and corr_norm > 0.0:
                                    cos_val = float(torch.dot(raw, corrected).item() / (raw_norm * corr_norm + 1e-8))
                                    cos_sim_values.append(cos_val)

                    # Track corrected gradient norm for debugging control-variance behavior.
                    with torch.no_grad():
                        grad_sq = 0.0
                        for p in local_model.parameters():
                            if p.grad is not None:
                                grad_sq += float(torch.sum(p.grad * p.grad).item())
                        grad_norm_sum += float(grad_sq ** 0.5)
                        grad_norm_steps += 1

                    optimizer.step()

                    # Apply SCAFFOLD correction in parameter-update space when requested.
                    if (not zero_control) and apply_control and control_in_parameter_space:
                        step_lr = float(optimizer.param_groups[0].get("lr", lr))
                        with torch.no_grad():
                            for n, p in local_model.named_parameters():
                                if not p.requires_grad:
                                    continue
                                control_delta = c_global[n].to(DEVICE) - c_local[n].to(DEVICE)
                                if use_control_scaling:
                                    grad_ref = raw_grads.get(n, None)
                                    if grad_ref is not None:
                                        grad_norm = float(torch.norm(grad_ref).item())
                                        delta_norm = float(torch.norm(control_delta).item())
                                        if grad_norm > 0.0 and delta_norm > 0.0:
                                            ratio = delta_norm / (grad_norm + 1e-12)
                                            if ratio > 1.0:
                                                control_delta = control_delta / ratio
                                p.data.add_(-step_lr * float(control_strength) * control_delta)

            # Compute delta and update control variates
            delta = {}
            num_batches = len(bundle.train_loader)  # number of batches seen
            K = local_epochs * num_batches  # total gradient steps
            # Canonical SCAFFOLD local control update scale: K * lr
            scale_factor = max(float(K * lr), 1e-12)
            new_c_locals[cid] = {}
            with torch.no_grad():
                for n, p_global in global_model.named_parameters():
                    if not p_global.requires_grad:
                        continue
                    p_local = dict(local_model.named_parameters())[n]
                    # Ensure all are on DEVICE for arithmetic
                    p_local_data = p_local.data.to(DEVICE)
                    p_global_before = global_before[n]
                    # Ensure c_local and c_global are on DEVICE
                    c_local_n = c_local[n].to(DEVICE)
                    c_global_n = c_global[n].to(DEVICE)
                    delta[n] = (p_local_data - p_global_before).to('cpu')
                    # Compute new c_local from round-start snapshots.
                    if not zero_control:
                        new_c = c_local_n - c_global_n + (p_global_before - p_local_data) / scale_factor
                        new_c_locals[cid][n] = new_c.detach().clone().to(DEVICE)
                    else:
                        new_c_locals[cid][n] = torch.zeros_like(c_local_n)

            client_deltas.append(delta)
            sample_counts.append(bundle.num_samples)
            if grad_norm_steps > 0:
                client_grad_norms.append(grad_norm_sum / grad_norm_steps)
            else:
                client_grad_norms.append(0.0)
            if cos_sim_values:
                client_cos_sims.append(float(np.mean(cos_sim_values)))
            else:
                client_cos_sims.append(1.0)

        # Apply control updates only after all clients complete (frozen-within-round invariant).
        with torch.no_grad():
            for b in bundles:
                scaffold.c_locals[b.client_id] = {
                    n: t.detach().clone().to(DEVICE)
                    for n, t in new_c_locals[b.client_id].items()
                }
            if not zero_control:
                for n in scaffold.c_global:
                    agg_c = torch.zeros_like(scaffold.c_global[n])
                    for b in bundles:
                        agg_c += scaffold.c_locals[b.client_id][n] / num_clients
                    scaffold.c_global[n] = agg_c.detach().clone().to(DEVICE)
            else:
                for n in scaffold.c_global:
                    scaffold.c_global[n].zero_()

        # Aggregate model deltas (weighted)
        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg_delta = torch.zeros_like(p.data)
                for i, (delta, ns) in enumerate(zip(client_deltas, sample_counts)):
                    w = ns / total_n
                    # Ensure delta and c_local are on DEVICE
                    agg_delta += w * delta[n].to(DEVICE)
                p.data.add_(agg_delta)

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
        worst_a = float(min(client_accs.values())) if client_accs else 0.0
        p10_a = float(np.percentile(list(client_accs.values()), 10)) if client_accs else 0.0

        # Round diagnostics for micro-tests: ||c_i|| (mean across clients), ||c||, ||gradient||.
        with torch.no_grad():
            c_i_norms = []
            for bundle in bundles:
                cid = bundle.client_id
                c_i_sq = 0.0
                for t in scaffold.c_locals[cid].values():
                    c_i_sq += float(torch.sum(t * t).item())
                c_i_norms.append(float(c_i_sq ** 0.5))

            c_sq = 0.0
            for t in scaffold.c_global.values():
                c_sq += float(torch.sum(t * t).item())
            c_norm = float(c_sq ** 0.5)

        mean_c_i_norm = float(np.mean(c_i_norms)) if c_i_norms else 0.0
        mean_grad_norm = float(np.mean(client_grad_norms)) if client_grad_norms else 0.0
        mean_cos_sim = float(np.mean(client_cos_sims)) if client_cos_sims else 1.0

        round_debug.append(
            {
                "round": rnd,
                "mean_accuracy": mean_a,
                "c_i_norm": mean_c_i_norm,
                "c_norm": c_norm,
                "gradient_norm": mean_grad_norm,
                "cos_sim_raw_vs_corrected": mean_cos_sim,
            }
        )
        round_accs.append(mean_a)
        per_round.append(
            {
                "round": rnd,
                "global_metrics": {
                    "mean_client_accuracy": mean_a,
                    "worst_client_accuracy": worst_a,
                    "p10_client_accuracy": p10_a,
                    "client_accuracies": {str(k): float(v) for k, v in client_accs.items()},
                },
                "communication": {
                    "round_client_to_server_bytes": round_bytes // 2,
                    "round_server_to_client_bytes": round_bytes // 2,
                    "round_total_bytes": round_bytes,
                },
            }
        )
        print(
            f"      [SCAFFOLD ROUND] {rnd}/{rounds} mean={mean_a:.4f} "
            f"worst={worst_a:.4f} p10={p10_a:.4f}",
            flush=True,
        )
        print(
            f"      [SCAFFOLD DEBUG] {rnd}/{rounds} "
            f"c_i_norm={mean_c_i_norm:.4f} c_norm={c_norm:.4f} "
            f"grad_norm={mean_grad_norm:.4f} cos={mean_cos_sim:.4f}",
            flush=True,
        )

    vals = list(client_accs.values())
    result = {
        "method": "scaffold", "seed": seed,
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std_across_clients": float(np.std(vals)),
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
        "partition_fingerprint": partition_fingerprint,
    }
    if return_trace:
        result["per_round"] = per_round
        result["round_debug"] = round_debug
        result["total_communication_bytes"] = int(rounds * round_bytes)
    return result


def run_moon(dataset_name: str, num_classes: int, num_clients: int, rounds: int, local_epochs: int,
             seed: int, alpha: float = 0.5, lr: float = 0.003,
             batch_size: int = 64, max_samples: int = 20000,
             mu: float = 1.0, temperature: float = 0.5,
             return_trace: bool = False) -> dict:
    """MOON with contrastive loss on representations.
    
    KEY FIX: Use persistent optimizer and model per client (no deepcopy/reinit each round).
    Only loss computation differs from FedAvg.
    """
    set_seed(seed)
    print(f"    MOON | dataset={dataset_name} | alpha={alpha} | Clients={num_clients} | Seed={seed}")

    cfg = ExperimentConfig(
        experiment_name=f"moon_s{seed}",
        dataset_name=dataset_name,
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = num_classes
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()
    partition_fingerprint = dm.partition_fingerprint(bundles)

    class _MoonProjector(nn.Module):
        """Projection head for stable contrastive space."""

        def __init__(self, in_dim: int, out_dim: int | None = None):
            super().__init__()
            out_dim = out_dim or in_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    global_model = build_centralized_model(dataset_name=dataset_name, num_classes=num_classes)
    global_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    projector_in_dim = int(global_model.backbone.output_dim)
    global_projector = _MoonProjector(projector_in_dim).to(DEVICE)

    # Persistent models and optimizers per client (initialized once, reused each round)
    client_models = {}
    client_projectors = {}
    client_optimizers = {}
    prev_local_models = {}
    prev_local_projectors = {}
    for b in bundles:
        client_models[b.client_id] = build_centralized_model(dataset_name=dataset_name, num_classes=num_classes).to(DEVICE)
        client_projectors[b.client_id] = _MoonProjector(projector_in_dim).to(DEVICE)
        client_optimizers[b.client_id] = torch.optim.Adam(
            list(client_models[b.client_id].parameters()) + list(client_projectors[b.client_id].parameters()),
            lr=lr,
        )
        prev_local_models[b.client_id] = copy.deepcopy(global_model).to(DEVICE)
        prev_local_projectors[b.client_id] = copy.deepcopy(global_projector).to(DEVICE)

    per_round = []
    num_params = sum(p.numel() for p in global_model.parameters())
    round_bytes = int((2 * num_clients) * num_params * 4)

    for rnd in range(1, rounds + 1):
        client_states = []
        client_projector_states = []
        sample_counts = []

        for bundle in bundles:
            cid = bundle.client_id
            # Load global weights into persistent client model (don't deepcopy)
            client_models[cid].load_state_dict(global_model.state_dict())
            local_model = client_models[cid]
            client_projectors[cid].load_state_dict(global_projector.state_dict())
            local_projector = client_projectors[cid]
            optimizer = client_optimizers[cid]
            
            prev_model = prev_local_models[cid]
            prev_projector = prev_local_projectors[cid]
            prev_model.to(DEVICE)
            prev_projector.to(DEVICE)
            prev_model.eval()
            prev_projector.eval()
            global_model.eval()  # For computing representations
            global_projector.eval()

            local_model.train()
            local_projector.train()
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()

                    # Task loss
                    logits = local_model.forward_task(x_b)
                    task_loss = criterion(logits, y_b)

                    if mu > 0.0:
                        # MOON contrastive branch is only active when mu > 0.
                        f_local = local_model.extract_features(x_b)
                        z_local = local_projector(f_local)
                        with torch.no_grad():
                            f_global = global_model.extract_features(x_b)
                            f_prev = prev_model.extract_features(x_b)
                            z_global = global_projector(f_global).detach()
                            z_prev = prev_projector(f_prev).detach()

                        z_local = F.normalize(z_local, p=2, dim=1)
                        z_global = F.normalize(z_global, p=2, dim=1)
                        z_prev = F.normalize(z_prev, p=2, dim=1)

                        # Positive: similarity to global, Negative: similarity to prev
                        pos = cos_sim(z_local, z_global) / temperature
                        neg = cos_sim(z_local, z_prev) / temperature
                        logits_con = torch.stack([pos, neg], dim=1)
                        labels_con = torch.zeros(x_b.size(0), dtype=torch.long,
                                                 device=DEVICE)
                        con_loss = nn.CrossEntropyLoss()(logits_con, labels_con)
                    else:
                        con_loss = torch.tensor(0.0, device=DEVICE)

                    loss = task_loss + mu * con_loss
                    loss.backward()
                    optimizer.step()

            # Save local model as previous for next round
            prev_local_models[cid] = copy.deepcopy(local_model).cpu()
            prev_local_projectors[cid] = copy.deepcopy(local_projector).cpu()

            state = {n: p.data.cpu().clone() for n, p in local_model.named_parameters()}
            proj_state = {n: p.data.cpu().clone() for n, p in local_projector.named_parameters()}
            client_states.append(state)
            client_projector_states.append(proj_state)
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

            for n, p in global_projector.named_parameters():
                agg = torch.zeros_like(p.data.cpu())
                for state, ns in zip(client_projector_states, sample_counts):
                    if n in state:
                        agg += (ns / total_n) * state[n]
                p.data.copy_(agg.to(DEVICE))

        # Evaluate each round for strict per-round schema.
        global_model.eval()
        round_client_accs = {}
        for bundle in bundles:
            correct, total = 0, 0
            with torch.no_grad():
                for x_b, y_b in bundle.eval_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    preds = global_model.forward_task(x_b).argmax(dim=1)
                    correct += (preds == y_b).sum().item()
                    total += y_b.size(0)
            round_client_accs[bundle.client_id] = correct / max(total, 1)

        mean_a = float(np.mean(list(round_client_accs.values()))) if round_client_accs else 0.0
        worst_a = float(min(round_client_accs.values())) if round_client_accs else 0.0
        p10_a = float(np.percentile(list(round_client_accs.values()), 10)) if round_client_accs else 0.0
        per_round.append(
            {
                "round": rnd,
                "global_metrics": {
                    "mean_client_accuracy": mean_a,
                    "worst_client_accuracy": worst_a,
                    "p10_client_accuracy": p10_a,
                    "client_accuracies": {str(k): float(v) for k, v in round_client_accs.items()},
                },
                "communication": {
                    "round_client_to_server_bytes": round_bytes // 2,
                    "round_server_to_client_bytes": round_bytes // 2,
                    "round_total_bytes": round_bytes,
                },
            }
        )
        print(
            f"      [MOON ROUND] {rnd}/{rounds} mean={mean_a:.4f} "
            f"worst={worst_a:.4f} p10={p10_a:.4f}",
            flush=True,
        )

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
    result = {
        "method": "moon", "seed": seed,
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std_across_clients": float(np.std(vals)),
        "client_accuracies": {str(k): v for k, v in client_accs.items()},
        "partition_fingerprint": partition_fingerprint,
    }
    if return_trace:
        result["per_round"] = per_round
        result["total_communication_bytes"] = int(rounds * round_bytes)
    return result


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
        r = run_scaffold(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES, num_clients=num_clients, rounds=rounds,
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
        r = run_moon(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES, num_clients=num_clients, rounds=rounds,
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
