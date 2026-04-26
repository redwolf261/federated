import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (  # noqa: E402
    DEVICE,
    FEMNIST_NUM_CLASSES,
    ClientDataManager,
    ExperimentConfig,
    ScaffoldState,
    build_centralized_model,
    run_fedavg_dirichlet,
    set_seed,
)


def run_scaffold_with_sign(alpha: float, seed: int, rounds: int, lr: float, sign_mode: str) -> float:
    num_clients = 2
    local_epochs = 1
    batch_size = 64
    max_samples = 20000

    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name=f"scaffold_sign_{sign_mode}_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(PROJECT_ROOT / "outputs"),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()

    global_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    scaffold = ScaffoldState(global_model)
    criterion = nn.CrossEntropyLoss()

    client_models = {}
    client_opts = {}
    for b in bundles:
        scaffold.init_client(b.client_id, global_model)
        client_models[b.client_id] = build_centralized_model(
            dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES
        ).to(DEVICE)
        client_opts[b.client_id] = torch.optim.SGD(client_models[b.client_id].parameters(), lr=lr)
        for n in scaffold.c_locals[b.client_id]:
            scaffold.c_locals[b.client_id][n] = scaffold.c_locals[b.client_id][n].to(DEVICE)
    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)

    for _ in range(rounds):
        client_states = []
        sample_counts = []

        for b in bundles:
            cid = b.client_id
            local = client_models[cid]
            local.load_state_dict(global_model.state_dict())
            opt = client_opts[cid]

            c_i = scaffold.c_locals[cid]
            c = scaffold.c_global

            local.train()
            for _ in range(local_epochs):
                for x_b, y_b in b.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(local.forward_task(x_b), y_b)
                    loss.backward()

                    with torch.no_grad():
                        for n, p in local.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                if sign_mode == "paper":
                                    # grad - c_i + c
                                    p.grad.copy_(p.grad - c_i[n] + c[n])
                                elif sign_mode == "flipped":
                                    # grad + c_i - c
                                    p.grad.copy_(p.grad + c_i[n] - c[n])
                                else:
                                    raise ValueError(sign_mode)

                    opt.step()

            K = local_epochs * len(b.train_loader)
            scale = max(float(K * lr), 1e-12)
            with torch.no_grad():
                local_named = dict(local.named_parameters())
                global_named = dict(global_model.named_parameters())
                for n, p_global in global_named.items():
                    if not p_global.requires_grad:
                        continue
                    p_local = local_named[n]
                    if sign_mode == "paper":
                        c_i[n] = (c_i[n] - c[n] + (p_global.data - p_local.data) / scale).detach().clone()
                    else:
                        # keep mathematically consistent opposite-sign convention
                        c_i[n] = (c_i[n] + c[n] - (p_global.data - p_local.data) / scale).detach().clone()

            client_states.append({n: p.data.detach().clone() for n, p in local.named_parameters() if p.requires_grad})
            sample_counts.append(b.num_samples)

        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg = torch.zeros_like(p.data)
                for state, ns in zip(client_states, sample_counts):
                    agg += (ns / total_n) * state[n]
                p.data.copy_(agg)

            for n in scaffold.c_global:
                agg_c = torch.zeros_like(scaffold.c_global[n])
                for b in bundles:
                    agg_c += scaffold.c_locals[b.client_id][n] / num_clients
                scaffold.c_global[n] = agg_c.detach().clone()

    global_model.eval()
    accs = []
    with torch.no_grad():
        for b in bundles:
            correct, total = 0, 0
            for x_b, y_b in b.eval_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                pred = global_model.forward_task(x_b).argmax(dim=1)
                correct += (pred == y_b).sum().item()
                total += y_b.size(0)
            accs.append(correct / max(total, 1))

    return float(np.mean(accs))


def main() -> None:
    seed = 42
    rounds = 10
    lr = 0.003

    fed_iid = run_fedavg_dirichlet(
        alpha=1.0, num_clients=2, rounds=rounds, local_epochs=1, seed=seed,
        lr=lr, batch_size=64, max_samples=20000,
    )["mean_accuracy"]
    sc_paper_iid = run_scaffold_with_sign(alpha=1.0, seed=seed, rounds=rounds, lr=lr, sign_mode="paper")
    sc_flip_iid = run_scaffold_with_sign(alpha=1.0, seed=seed, rounds=rounds, lr=lr, sign_mode="flipped")

    fed_noniid = run_fedavg_dirichlet(
        alpha=0.1, num_clients=2, rounds=rounds, local_epochs=1, seed=seed,
        lr=lr, batch_size=64, max_samples=20000,
    )["mean_accuracy"]
    sc_paper_noniid = run_scaffold_with_sign(alpha=0.1, seed=seed, rounds=rounds, lr=lr, sign_mode="paper")
    sc_flip_noniid = run_scaffold_with_sign(alpha=0.1, seed=seed, rounds=rounds, lr=lr, sign_mode="flipped")

    out = {
        "iid": {
            "fedavg": fed_iid,
            "scaffold_paper_sign": sc_paper_iid,
            "scaffold_flipped_sign": sc_flip_iid,
        },
        "noniid_alpha_0_1": {
            "fedavg": fed_noniid,
            "scaffold_paper_sign": sc_paper_noniid,
            "scaffold_flipped_sign": sc_flip_noniid,
        },
    }

    p = PROJECT_ROOT / "outputs" / "scaffold_sign_ablation.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
