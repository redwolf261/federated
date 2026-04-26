import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (  # noqa: E402
    DEVICE,
    FEMNIST_NUM_CLASSES,
    ExperimentConfig,
    ClientDataManager,
    ScaffoldState,
    build_centralized_model,
    run_moon,
    run_scaffold,
    set_seed,
)


def grad_norm(grads):
    total = 0.0
    for g in grads:
        if g is None:
            continue
        total += float(torch.sum(g.detach() * g.detach()).item())
    return math.sqrt(max(total, 0.0))


def tensor_dict_norm(d):
    total = 0.0
    for v in d.values():
        total += float(torch.sum(v.detach() * v.detach()).item())
    return math.sqrt(max(total, 0.0))


def debug_moon(seed=42, alpha=0.1, rounds=10, local_epochs=1, lr=0.003, batch_size=64, max_samples=4000, mu=1.0, temperature=0.5):
    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name=f"moon_debug_s{seed}",
        dataset_name="femnist",
        num_clients=10,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(PROJECT_ROOT / "outputs"),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // cfg.num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()

    global_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    client_models = {}
    client_optimizers = {}
    prev_local_models = {}
    for b in bundles:
        client_models[b.client_id] = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
        client_optimizers[b.client_id] = torch.optim.Adam(client_models[b.client_id].parameters(), lr=lr)
        prev_local_models[b.client_id] = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
        prev_local_models[b.client_id].load_state_dict(global_model.state_dict())

    per_round = []

    for _rnd in range(1, rounds + 1):
        client_states = []
        sample_counts = []

        ce_norms = []
        con_norms = []

        for bundle in bundles:
            cid = bundle.client_id
            local_model = client_models[cid]
            local_model.load_state_dict(global_model.state_dict())
            optimizer = client_optimizers[cid]
            prev_model = prev_local_models[cid]
            prev_model.eval()
            global_model.eval()

            params = [p for p in local_model.parameters() if p.requires_grad]

            local_model.train()
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()

                    logits = local_model.forward_task(x_b)
                    task_loss = criterion(logits, y_b)

                    z_local = local_model.forward_shared(x_b)
                    with torch.no_grad():
                        z_global = global_model.forward_shared(x_b)
                        z_prev = prev_model.forward_shared(x_b)

                    pos = cos_sim(z_local, z_global) / temperature
                    neg = cos_sim(z_local, z_prev) / temperature
                    logits_con = torch.stack([pos, neg], dim=1)
                    labels_con = torch.zeros(x_b.size(0), dtype=torch.long, device=DEVICE)
                    con_loss = nn.CrossEntropyLoss()(logits_con, labels_con)

                    ce_grads = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
                    con_grads = torch.autograd.grad(mu * con_loss, params, retain_graph=True, allow_unused=True)
                    ce_norms.append(grad_norm(ce_grads))
                    con_norms.append(grad_norm(con_grads))

                    loss = task_loss + mu * con_loss
                    loss.backward()
                    optimizer.step()

            prev_local_models[cid].load_state_dict(local_model.state_dict())
            state = {n: p.data.cpu().clone() for n, p in local_model.named_parameters()}
            client_states.append(state)
            sample_counts.append(bundle.num_samples)

        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                agg = torch.zeros_like(p.data.cpu())
                for state, ns in zip(client_states, sample_counts):
                    agg += (ns / total_n) * state[n]
                p.data.copy_(agg.to(DEVICE))

        ratio = (float(np.mean(con_norms)) / max(float(np.mean(ce_norms)), 1e-12)) if ce_norms else 0.0
        per_round.append(
            {
                "round": _rnd,
                "ce_grad_norm_mean": float(np.mean(ce_norms)) if ce_norms else 0.0,
                "ce_grad_norm_max": float(np.max(ce_norms)) if ce_norms else 0.0,
                "contrastive_grad_norm_mean": float(np.mean(con_norms)) if con_norms else 0.0,
                "contrastive_grad_norm_max": float(np.max(con_norms)) if con_norms else 0.0,
                "contrastive_to_ce_ratio": float(ratio),
            }
        )

    src = inspect.getsource(run_moon)
    uses_feature_normalize = "normalize(" in src or "F.normalize" in src

    return {
        "mu": mu,
        "temperature": temperature,
        "feature_normalization_present": uses_feature_normalize,
        "round_debug": per_round,
        "aggregate": {
            "ce_grad_norm_mean": float(np.mean([r["ce_grad_norm_mean"] for r in per_round])),
            "contrastive_grad_norm_mean": float(np.mean([r["contrastive_grad_norm_mean"] for r in per_round])),
            "contrastive_to_ce_ratio_mean": float(np.mean([r["contrastive_to_ce_ratio"] for r in per_round])),
            "contrastive_to_ce_ratio_max": float(np.max([r["contrastive_to_ce_ratio"] for r in per_round])),
        },
    }


def debug_scaffold(seed=42, alpha=0.1, rounds=10, local_epochs=1, lr=0.003, batch_size=64, max_samples=4000):
    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name=f"scaffold_debug_s{seed}",
        dataset_name="femnist",
        num_clients=10,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(PROJECT_ROOT / "outputs"),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // cfg.num_clients

    dm = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()

    global_model = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
    scaffold = ScaffoldState(global_model)
    criterion = nn.CrossEntropyLoss()

    client_models = {}
    client_optimizers = {}
    for b in bundles:
        scaffold.init_client(b.client_id, global_model)
        client_models[b.client_id] = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
        client_optimizers[b.client_id] = torch.optim.Adam(client_models[b.client_id].parameters(), lr=lr)
        for n in scaffold.c_locals[b.client_id]:
            scaffold.c_locals[b.client_id][n] = scaffold.c_locals[b.client_id][n].to(DEVICE)
    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)

    per_round = []

    for _rnd in range(1, rounds + 1):
        client_deltas = []
        sample_counts = []

        grad_norms = []
        control_norms = []
        corrected_grad_norms = []

        for bundle in bundles:
            cid = bundle.client_id
            local_model = client_models[cid]
            local_model.load_state_dict(global_model.state_dict())
            optimizer = client_optimizers[cid]

            c_local = scaffold.c_locals[cid]
            c_global = scaffold.c_global

            local_model.train()
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    logits = local_model.forward_task(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()

                    g_sq = 0.0
                    c_sq = 0.0
                    gc_sq = 0.0
                    with torch.no_grad():
                        for n, p in local_model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                g_sq += float(torch.sum(p.grad * p.grad).item())
                                corr = c_global[n] - c_local[n]
                                c_sq += float(torch.sum(corr * corr).item())
                                p.grad.add_(corr)
                                gc_sq += float(torch.sum(p.grad * p.grad).item())
                    grad_norms.append(math.sqrt(max(g_sq, 0.0)))
                    control_norms.append(math.sqrt(max(c_sq, 0.0)))
                    corrected_grad_norms.append(math.sqrt(max(gc_sq, 0.0)))

                    optimizer.step()

            delta = {}
            num_batches = len(bundle.train_loader)
            K = local_epochs * num_batches
            # Scale control update by num_clients*K (not including lr)
            num_clients = len(bundles)
            scale_factor = num_clients * K
            with torch.no_grad():
                named_local = dict(local_model.named_parameters())
                for n, p_global in global_model.named_parameters():
                    if not p_global.requires_grad:
                        continue
                    p_local = named_local[n]
                    p_local_data = p_local.data
                    p_global_data = p_global.data
                    c_local_n = c_local[n]
                    c_global_n = c_global[n]
                    delta[n] = (p_local_data - p_global_data).detach().cpu()
                    new_c = c_local_n - c_global_n + (p_global_data - p_local_data) / scale_factor
                    scaffold.c_locals[cid][n] = new_c.detach().clone()

            client_deltas.append(delta)
            sample_counts.append(bundle.num_samples)

        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg_delta = torch.zeros_like(p.data)
                agg_c_delta = torch.zeros_like(p.data)
                for i, (delta, ns) in enumerate(zip(client_deltas, sample_counts)):
                    w = ns / total_n
                    agg_delta += w * delta[n].to(DEVICE)
                    agg_c_delta += scaffold.c_locals[bundles[i].client_id][n] / len(bundles)
                p.data.add_(agg_delta)
                scaffold.c_global[n] = agg_c_delta.detach().clone()

        ratio = (float(np.mean(control_norms)) / max(float(np.mean(grad_norms)), 1e-12)) if grad_norms else 0.0
        per_round.append(
            {
                "round": _rnd,
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "control_norm_mean": float(np.mean(control_norms)) if control_norms else 0.0,
                "corrected_grad_norm_mean": float(np.mean(corrected_grad_norms)) if corrected_grad_norms else 0.0,
                "control_to_grad_ratio": float(ratio),
                "c_global_norm": float(tensor_dict_norm(scaffold.c_global)),
            }
        )

    eq = "w_i <- w_i - eta * (grad - c_i + c); c_i <- c_i - c + (w_global - w_local)/(K*eta); c <- average_i(c_i)"

    return {
        "implemented_equation": eq,
        "round_debug": per_round,
        "aggregate": {
            "grad_norm_mean": float(np.mean([r["grad_norm_mean"] for r in per_round])),
            "control_norm_mean": float(np.mean([r["control_norm_mean"] for r in per_round])),
            "control_to_grad_ratio_mean": float(np.mean([r["control_to_grad_ratio"] for r in per_round])),
            "control_to_grad_ratio_max": float(np.max([r["control_to_grad_ratio"] for r in per_round])),
            "c_global_norm_final": float(per_round[-1]["c_global_norm"] if per_round else 0.0),
        },
    }


def main():
    moon = debug_moon()
    scaffold = debug_scaffold()
    out = {
        "probe_config": {
            "dataset": "femnist",
            "seed": 42,
            "alpha": 0.1,
            "rounds": 10,
            "local_epochs": 1,
            "num_clients": 10,
            "batch_size": 64,
            "max_samples": 4000,
            "lr": 0.003,
        },
        "moon_debug": moon,
        "scaffold_debug": scaffold,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
