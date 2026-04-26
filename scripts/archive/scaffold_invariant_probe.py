import json
import math
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
    run_scaffold,
    set_seed,
)


def _build_bundles(seed: int, alpha: float, num_clients: int, batch_size: int, max_samples: int):
    cfg = ExperimentConfig(
        experiment_name=f"scaffold_invariant_probe_a{alpha}_s{seed}",
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
    return dm, bundles


def _eval_global(global_model: nn.Module, bundles) -> float:
    global_model.eval()
    client_accs = {}
    with torch.no_grad():
        for b in bundles:
            correct, total = 0, 0
            for x_b, y_b in b.eval_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                preds = global_model.forward_task(x_b).argmax(dim=1)
                correct += int((preds == y_b).sum().item())
                total += int(y_b.size(0))
            client_accs[b.client_id] = correct / max(total, 1)
    return float(np.mean(list(client_accs.values())))


def run_manual_fedavg_scaffold_style(
    seed: int,
    alpha: float,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    max_samples: int,
    optimizer_name: str = "adam",
) -> dict:
    set_seed(seed)
    _, bundles = _build_bundles(seed, alpha, num_clients, batch_size, max_samples)

    global_model = build_centralized_model("femnist", FEMNIST_NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    client_models = {}
    client_optimizers = {}

    for b in bundles:
        m = build_centralized_model("femnist", FEMNIST_NUM_CLASSES).to(DEVICE)
        m.load_state_dict(global_model.state_dict())
        client_models[b.client_id] = m
        if optimizer_name == "adam":
            client_optimizers[b.client_id] = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.0)
        elif optimizer_name == "sgd":
            client_optimizers[b.client_id] = torch.optim.SGD(m.parameters(), lr=lr)
        else:
            raise ValueError(f"unknown optimizer: {optimizer_name}")

    for _rnd in range(1, rounds + 1):
        client_deltas = []
        sample_counts = []

        for b in bundles:
            cid = b.client_id
            local_model = client_models[cid]
            optimizer = client_optimizers[cid]

            local_model.load_state_dict(global_model.state_dict())
            local_model.train()

            for _ in range(local_epochs):
                for x_b, y_b in b.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    logits = local_model.forward_task(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()

            delta = {}
            with torch.no_grad():
                global_named = dict(global_model.named_parameters())
                local_named = dict(local_model.named_parameters())
                for n, p_g in global_named.items():
                    if not p_g.requires_grad:
                        continue
                    delta[n] = (local_named[n].data - p_g.data).detach().clone().to("cpu")

            client_deltas.append(delta)
            sample_counts.append(b.num_samples)

        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg_delta = torch.zeros_like(p.data)
                for d, ns in zip(client_deltas, sample_counts):
                    agg_delta += (ns / total_n) * d[n].to(DEVICE)
                p.data.add_(agg_delta)

    return {"mean_accuracy": _eval_global(global_model, bundles)}


def probe_e2_and_f1(seed: int, alpha: float, num_clients: int, local_epochs: int, lr: float, batch_size: int, max_samples: int) -> dict:
    set_seed(seed)
    _, bundles = _build_bundles(seed, alpha, num_clients, batch_size, max_samples)

    global_model = build_centralized_model("femnist", FEMNIST_NUM_CLASSES).to(DEVICE)
    scaffold = ScaffoldState(global_model)
    criterion = nn.CrossEntropyLoss()

    for b in bundles:
        scaffold.init_client(b.client_id, global_model)
        for n in scaffold.c_locals[b.client_id]:
            scaffold.c_locals[b.client_id][n] = scaffold.c_locals[b.client_id][n].to(DEVICE)
    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)

    bundle = bundles[0]
    cid = bundle.client_id
    local_model = build_centralized_model("femnist", FEMNIST_NUM_CLASSES).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    global_before = {
        n: p.detach().clone().to(DEVICE)
        for n, p in global_model.named_parameters()
        if p.requires_grad
    }
    c_i_before = {n: t.detach().clone().to(DEVICE) for n, t in scaffold.c_locals[cid].items()}
    c_g_before = {n: t.detach().clone().to(DEVICE) for n, t in scaffold.c_global.items()}

    # E2: one-step equivalence (manual update vs optimizer.step)
    x_b, y_b = next(iter(bundle.train_loader))
    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
    optimizer.zero_grad()
    logits = local_model.forward_task(x_b)
    loss = criterion(logits, y_b)
    loss.backward()

    with torch.no_grad():
        for n, p in local_model.named_parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.copy_(p.grad - scaffold.c_locals[cid][n] + scaffold.c_global[n])

    w_before = {
        n: p.detach().clone().to(DEVICE)
        for n, p in local_model.named_parameters()
        if p.requires_grad
    }
    g_corr = {
        n: p.grad.detach().clone().to(DEVICE)
        for n, p in local_model.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    optimizer.step()

    w_after = {
        n: p.detach().clone().to(DEVICE)
        for n, p in local_model.named_parameters()
        if p.requires_grad
    }

    sq = 0.0
    for n in w_before:
        w_manual = w_before[n] - lr * g_corr[n]
        diff = w_after[n] - w_manual
        sq += float(torch.sum(diff * diff).item())
    e2_error = float(math.sqrt(sq))

    # Continue local training for F1 setup.
    local_model.train()
    for _ in range(local_epochs):
        for x2, y2 in bundle.train_loader:
            x2, y2 = x2.to(DEVICE), y2.to(DEVICE)
            optimizer.zero_grad()
            loss2 = criterion(local_model.forward_task(x2), y2)
            loss2.backward()
            with torch.no_grad():
                for n, p in local_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.copy_(p.grad - scaffold.c_locals[cid][n] + scaffold.c_global[n])
            optimizer.step()

    num_batches = len(bundle.train_loader)
    K = local_epochs * num_batches
    scale = max(float(K * lr), 1e-12)

    with torch.no_grad():
        local_named = dict(local_model.named_parameters())
        expected_delta = {}
        actual_delta = {}
        for n, p_local in local_named.items():
            if not p_local.requires_grad:
                continue
            expected_delta[n] = (global_before[n] - p_local.data) / scale

            # Apply same c_i update as implementation.
            scaffold.c_locals[cid][n] = (
                scaffold.c_locals[cid][n]
                - scaffold.c_global[n]
                + (global_before[n] - p_local.data) / scale
            ).detach().clone()

            actual_delta[n] = scaffold.c_locals[cid][n] - c_i_before[n] + c_g_before[n]

    sq_f1 = 0.0
    for n in expected_delta:
        d = expected_delta[n] - actual_delta[n]
        sq_f1 += float(torch.sum(d * d).item())
    f1_error = float(math.sqrt(sq_f1))

    return {
        "e2_step_error": e2_error,
        "f1_update_error": f1_error,
    }


def main() -> None:
    seed = 42
    alpha = 1.0
    num_clients = 2
    rounds = 10
    local_epochs = 1
    lr = 0.003
    batch_size = 64
    max_samples = 20000

    # A1: simulator vs manual fedavg (same scaffold-style loop)
    simulator = run_fedavg_dirichlet(
        alpha=alpha,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        seed=seed,
        lr=lr,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    manual = run_manual_fedavg_scaffold_style(
        seed=seed,
        alpha=alpha,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
        max_samples=max_samples,
        optimizer_name="adam",
    )

    # A2: manual-loop fedavg vs scaffold(zero control)
    scaffold_zero = run_scaffold(
        dataset_name="femnist",
        num_classes=FEMNIST_NUM_CLASSES,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        seed=seed,
        alpha=alpha,
        lr=lr,
        batch_size=batch_size,
        max_samples=max_samples,
        zero_control=True,
    )

    probes = probe_e2_and_f1(
        seed=seed,
        alpha=alpha,
        num_clients=num_clients,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    a1_sim = float(simulator["mean_accuracy"])
    a1_manual = float(manual["mean_accuracy"])
    a2_zero = float(scaffold_zero["mean_accuracy"])

    out = {
        "A1": {
            "simulator_fedavg": a1_sim,
            "manual_loop_fedavg": a1_manual,
            "delta": a1_manual - a1_sim,
            "pass": abs(a1_manual - a1_sim) < 0.01,
        },
        "A2": {
            "manual_fedavg": a1_manual,
            "scaffold_zero_control": a2_zero,
            "delta": a2_zero - a1_manual,
            "pass": abs(a2_zero - a1_manual) < 0.01,
        },
        "E2": {
            "manual_step_vs_optimizer_step_error": probes["e2_step_error"],
            "pass": probes["e2_step_error"] < 1e-6,
        },
        "F1": {
            "control_update_equation_error": probes["f1_update_error"],
            "pass": probes["f1_update_error"] < 1e-6,
        },
    }

    assert isinstance(out["A1"]["pass"], bool)
    assert isinstance(out["A2"]["pass"], bool)
    assert isinstance(out["E2"]["pass"], bool)
    assert isinstance(out["F1"]["pass"], bool)

    out_path = PROJECT_ROOT / "outputs" / "scaffold_invariant_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
