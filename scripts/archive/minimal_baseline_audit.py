import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (
    DEVICE,
    FEMNIST_NUM_CLASSES,
    PROJECT_ROOT,
    ExperimentConfig,
    ClientDataManager,
    ScaffoldState,
    build_centralized_model,
    run_fedavg_dirichlet,
    run_moon,
    run_scaffold,
    set_seed,
)


class MoonProjector(nn.Module):
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


def train_accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            pred = model.forward_task(x_b).argmax(dim=1)
            correct += (pred == y_b).sum().item()
            total += y_b.numel()
    return float(correct / max(total, 1))


def build_single_client_bundle(seed: int, max_samples: int = 500, batch_size: int = 64):
    cfg = ExperimentConfig(
        experiment_name=f"audit_overfit_s{seed}",
        dataset_name="femnist",
        num_clients=1,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=1.0,
        output_dir=str(Path(PROJECT_ROOT) / "outputs"),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()
    return bundles[0]


def overfit_fedavg(seed: int, epochs: int = 50) -> float:
    set_seed(seed)
    bundle = build_single_client_bundle(seed)
    model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    ce = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x_b, y_b in bundle.train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = ce(model.forward_task(x_b), y_b)
            loss.backward()
            opt.step()
    return train_accuracy(model, bundle.train_loader)


def overfit_moon(seed: int, rounds: int = 50) -> float:
    set_seed(seed)
    bundle = build_single_client_bundle(seed)

    global_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    local_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    prev_model = copy.deepcopy(global_model).to(DEVICE)

    feat_dim = int(global_model.backbone.output_dim)
    global_proj = MoonProjector(feat_dim).to(DEVICE)
    local_proj = MoonProjector(feat_dim).to(DEVICE)
    prev_proj = copy.deepcopy(global_proj).to(DEVICE)

    opt = torch.optim.Adam(list(local_model.parameters()) + list(local_proj.parameters()), lr=0.003)
    ce = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    for _ in range(rounds):
        local_model.load_state_dict(global_model.state_dict())
        local_proj.load_state_dict(global_proj.state_dict())
        local_model.train()
        local_proj.train()
        prev_model.eval()
        prev_proj.eval()
        global_model.eval()
        global_proj.eval()

        for x_b, y_b in bundle.train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()

            task_loss = ce(local_model.forward_task(x_b), y_b)
            z_local = local_proj(local_model.extract_features(x_b))
            with torch.no_grad():
                z_global = global_proj(global_model.extract_features(x_b)).detach()
                z_prev = prev_proj(prev_model.extract_features(x_b)).detach()

            z_local = F.normalize(z_local, p=2, dim=1)
            z_global = F.normalize(z_global, p=2, dim=1)
            z_prev = F.normalize(z_prev, p=2, dim=1)

            pos = cos_sim(z_local, z_global) / 0.5
            neg = cos_sim(z_local, z_prev) / 0.5
            logits_con = torch.stack([pos, neg], dim=1)
            labels_con = torch.zeros(x_b.size(0), dtype=torch.long, device=DEVICE)
            con_loss = ce(logits_con, labels_con)

            loss = task_loss + 1.0 * con_loss
            loss.backward()
            opt.step()

        prev_model.load_state_dict(local_model.state_dict())
        prev_proj.load_state_dict(local_proj.state_dict())
        global_model.load_state_dict(local_model.state_dict())
        global_proj.load_state_dict(local_proj.state_dict())

    return train_accuracy(global_model, bundle.train_loader)


def overfit_scaffold(seed: int, rounds: int = 50) -> float:
    set_seed(seed)
    bundle = build_single_client_bundle(seed)

    global_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    local_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    scaffold = ScaffoldState(global_model)
    scaffold.init_client(bundle.client_id, global_model)

    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)
    for n in scaffold.c_locals[bundle.client_id]:
        scaffold.c_locals[bundle.client_id][n] = scaffold.c_locals[bundle.client_id][n].to(DEVICE)

    opt = torch.optim.Adam(local_model.parameters(), lr=0.003)
    ce = nn.CrossEntropyLoss()

    for _ in range(rounds):
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        for x_b, y_b in bundle.train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = ce(local_model.forward_task(x_b), y_b)
            loss.backward()
            with torch.no_grad():
                for n, p in local_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.add_(scaffold.c_global[n] - scaffold.c_locals[bundle.client_id][n])
            opt.step()

        K = len(bundle.train_loader)
        scale = max(float(K * 0.003), 1e-12)
        with torch.no_grad():
            named_local = dict(local_model.named_parameters())
            for n, p_global in global_model.named_parameters():
                if not p_global.requires_grad:
                    continue
                p_local = named_local[n]
                c_i = scaffold.c_locals[bundle.client_id][n]
                c = scaffold.c_global[n]
                scaffold.c_locals[bundle.client_id][n] = c_i - c + (p_global.data - p_local.data) / scale

            # one-client average => c == c_i
            for n in scaffold.c_global:
                scaffold.c_global[n] = scaffold.c_locals[bundle.client_id][n].clone()

            global_model.load_state_dict(local_model.state_dict())

    return train_accuracy(global_model, bundle.train_loader)


def main():
    """INVARIANT AUDIT (locked config per user spec).
    
    Config:
    - clients: 2
    - alpha: 1.0 (IID)
    - rounds: 10
    - local_epochs: 1
    - seeds: [42, 123, 456]
    - max_samples: 20000
    
    Tests:
    1. MOON vs FedAvg in IID: |delta| < 0.05 => PASS
    2. SCAFFOLD vs FedAvg in IID: value >= FedAvg - 0.05 => PASS
    3. Overfit (1 client, 50 epochs, ~500 samples): all > 0.95 => PASS
    """
    SEEDS = [42, 123, 456]
    NUM_CLIENTS = 2
    ALPHA = 1.0
    ROUNDS = 10
    LOCAL_EPOCHS = 1
    LR = 0.003
    BATCH_SIZE = 64
    MAX_SAMPLES = 20000
    
    # Test 1 + Test 2: IID 2-client, 10 rounds
    fed_accs = []
    moon_accs = []
    scaffold_accs = []
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Running seed={seed}")
        print(f"{'='*60}")
        
        fed = run_fedavg_dirichlet(
            alpha=ALPHA, num_clients=NUM_CLIENTS, rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS, seed=seed, lr=LR, batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
        )
        moon = run_moon(
            dataset_name="femnist",
            num_classes=FEMNIST_NUM_CLASSES,
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            seed=seed,
            alpha=ALPHA,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
            mu=1.0,
            return_trace=False,
        )
        scaffold = run_scaffold(
            dataset_name="femnist",
            num_classes=FEMNIST_NUM_CLASSES,
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            seed=seed,
            alpha=ALPHA,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
            return_trace=False,
        )
        
        fed_accs.append(float(fed["mean_accuracy"]))
        moon_accs.append(float(moon["mean_accuracy"]))
        scaffold_accs.append(float(scaffold["mean_accuracy"]))
        
        print(f"\n  Seed {seed} results:")
        print(f"    FedAvg:   {fed_accs[-1]:.4f}")
        print(f"    MOON:     {moon_accs[-1]:.4f}")
        print(f"    SCAFFOLD: {scaffold_accs[-1]:.4f}")
    
    # Aggregate across seeds
    fed_mean = float(np.mean(fed_accs))
    fed_std = float(np.std(fed_accs))
    moon_mean = float(np.mean(moon_accs))
    moon_std = float(np.std(moon_accs))
    scaffold_mean = float(np.mean(scaffold_accs))
    scaffold_std = float(np.std(scaffold_accs))
    
    moon_delta = abs(moon_mean - fed_mean)
    scaffold_delta = fed_mean - scaffold_mean  # FedAvg - SCAFFOLD
    
    # Test 3: Overfit on tiny one-client split (seeds don't matter much for tiny overfit)
    overfit_seed = 42
    overfit = {
        "fedavg_train_acc": overfit_fedavg(seed=overfit_seed, epochs=50),
        "moon_train_acc": overfit_moon(seed=overfit_seed, rounds=50),
        "scaffold_train_acc": overfit_scaffold(seed=overfit_seed, rounds=50),
    }
    
    # Format results (STRICT OUTPUT)
    results = {
        "config": {
            "seeds": SEEDS,
            "iid_alpha": ALPHA,
            "num_clients": NUM_CLIENTS,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_samples": MAX_SAMPLES,
        },
        "fedavg_anchor": {
            "mean_accuracy": fed_mean,
            "std": fed_std,
            "per_seed": {str(s): float(a) for s, a in zip(SEEDS, fed_accs)},
        },
        "moon_invariant": {
            "mean_accuracy": moon_mean,
            "std": moon_std,
            "per_seed": {str(s): float(a) for s, a in zip(SEEDS, moon_accs)},
            "delta_vs_fedavg": float(moon_delta),
            "threshold": 0.05,
            "pass": bool(moon_delta <= 0.05),
        },
        "scaffold_invariant": {
            "mean_accuracy": scaffold_mean,
            "std": scaffold_std,
            "per_seed": {str(s): float(a) for s, a in zip(SEEDS, scaffold_accs)},
            "delta_vs_fedavg": float(scaffold_delta),
            "threshold": 0.05,
            "pass": bool(scaffold_mean >= fed_mean - 0.05),
        },
        "overfit_test": {
            "fedavg_train_acc": float(overfit["fedavg_train_acc"]),
            "moon_train_acc": float(overfit["moon_train_acc"]),
            "scaffold_train_acc": float(overfit["scaffold_train_acc"]),
            "threshold": 0.95,
            "pass": bool(
                overfit["fedavg_train_acc"] >= 0.95
                and overfit["moon_train_acc"] >= 0.95
                and overfit["scaffold_train_acc"] >= 0.95
            ),
        },
    }
    
    output_path = Path(PROJECT_ROOT) / "outputs" / "minimal_baseline_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print("FINAL AUDIT RESULTS")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
