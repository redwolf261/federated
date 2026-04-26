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
    set_seed,
)


def tensor_dict_l2_norm(tensors: dict[str, torch.Tensor]) -> float:
    s = 0.0
    for t in tensors.values():
        s += float(torch.sum(t * t).item())
    return float(s ** 0.5)


def param_l2_diff_norm(model_a: nn.Module, model_b: nn.Module) -> float:
    s = 0.0
    a_named = dict(model_a.named_parameters())
    b_named = dict(model_b.named_parameters())
    for n, p in a_named.items():
        if not p.requires_grad:
            continue
        d = p.data - b_named[n].data
        s += float(torch.sum(d * d).item())
    return float(s ** 0.5)


def grad_l2_norm(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(torch.sum(p.grad * p.grad).item())
    return float(s ** 0.5)


def main() -> None:
    seed = 42
    alpha = 1.0
    num_clients = 2
    rounds = 5
    local_epochs = 1
    lr = 0.003
    batch_size = 64
    max_samples = 20000

    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name="scaffold_control_signal_dump",
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

    client_models: dict[int, nn.Module] = {}
    client_opts: dict[int, torch.optim.Optimizer] = {}
    for b in bundles:
        scaffold.init_client(b.client_id, global_model)
        client_models[b.client_id] = build_centralized_model(
            dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES
        ).to(DEVICE)
        client_opts[b.client_id] = torch.optim.SGD(client_models[b.client_id].parameters(), lr=lr)

    for cid in scaffold.c_locals:
        for n in scaffold.c_locals[cid]:
            scaffold.c_locals[cid][n] = scaffold.c_locals[cid][n].to(DEVICE)
    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)

    logs: dict[str, object] = {
        "config": {
            "seed": seed,
            "alpha": alpha,
            "num_clients": num_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "max_samples": max_samples,
        },
        "per_round_core": [],
        "per_round_client_updates": [],
        "persistence": {},
        "update_order": {
            "followed": "YES",
            "evidence_trace": [],
            "code_snippet": [
                "for each client: load global weights/state, train with grad + (c - c_i), compute w_local, update c_i using (w_global - w_local)/(K*lr)",
                "after all clients: aggregate model deltas into w_global, then update c = average(c_i)",
            ],
        },
        "round2_one_batch": {},
    }

    client_ids = [b.client_id for b in bundles]
    c_i_norm_series = {str(cid): [] for cid in client_ids}

    for rnd in range(1, rounds + 1):
        client_deltas = []
        sample_counts = []
        client_grad_norm_means = []
        round_client_rows = []

        for bundle in bundles:
            cid = bundle.client_id
            logs["update_order"]["evidence_trace"].append(f"round {rnd}, client {cid}: load global model + c")

            client_models[cid].load_state_dict(global_model.state_dict())
            local_model = client_models[cid]
            opt = client_opts[cid]

            # Snapshot global model before this client's local training (required for c_i update).
            global_before = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
            global_before.load_state_dict(global_model.state_dict())

            c_i_before = {n: t.detach().clone() for n, t in scaffold.c_locals[cid].items()}
            c_i_before_norm = tensor_dict_l2_norm(c_i_before)

            grad_norm_sum = 0.0
            grad_steps = 0
            one_batch_captured = False

            local_model.train()
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    opt.zero_grad()
                    logits = local_model.forward_task(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()

                    raw_grad_norm = grad_l2_norm(local_model)

                    with torch.no_grad():
                        for n, p in local_model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                p.grad.add_(scaffold.c_global[n] - scaffold.c_locals[cid][n])

                    applied_grad_norm = grad_l2_norm(local_model)
                    diff_norm = abs(applied_grad_norm - raw_grad_norm)

                    if rnd == 2 and (not one_batch_captured) and (not logs["round2_one_batch"]):
                        logs["round2_one_batch"] = {
                            "round": 2,
                            "client": cid,
                            "batch_index": 0,
                            "raw_gradient_norm": raw_grad_norm,
                            "applied_update_norm": applied_grad_norm,
                            "difference_norm": diff_norm,
                        }
                        one_batch_captured = True

                    grad_norm_sum += applied_grad_norm
                    grad_steps += 1
                    opt.step()

            logs["update_order"]["evidence_trace"].append(f"round {rnd}, client {cid}: train with correction")

            K = local_epochs * len(bundle.train_loader)
            scale = max(float(K * lr), 1e-12)

            w_diff_norm = param_l2_diff_norm(global_before, local_model)

            delta = {}
            with torch.no_grad():
                local_named = dict(local_model.named_parameters())
                global_named = dict(global_model.named_parameters())
                for n, p_global in global_named.items():
                    if not p_global.requires_grad:
                        continue
                    p_local = local_named[n]
                    delta[n] = (p_local.data - p_global.data).detach().clone().to("cpu")

                    scaffold.c_locals[cid][n] = (
                        scaffold.c_locals[cid][n]
                        - scaffold.c_global[n]
                        + (p_global.data - p_local.data) / scale
                    ).detach().clone()

            logs["update_order"]["evidence_trace"].append(f"round {rnd}, client {cid}: update c_i")

            c_i_after_norm = tensor_dict_l2_norm(scaffold.c_locals[cid])
            c_i_norm_series[str(cid)].append(c_i_after_norm)

            computed_delta_norm = w_diff_norm / scale

            row = {
                "round": rnd,
                "client": cid,
                "w_global_before_minus_w_local_after_norm": w_diff_norm,
                "K": int(K),
                "lr": float(lr),
                "computed_delta_norm": computed_delta_norm,
                "c_i_before_norm": c_i_before_norm,
                "c_i_after_norm": c_i_after_norm,
            }
            round_client_rows.append(row)

            client_deltas.append(delta)
            sample_counts.append(bundle.num_samples)
            client_grad_norm_means.append(grad_norm_sum / max(grad_steps, 1))

        # Aggregate model after all clients.
        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg_delta = torch.zeros_like(p.data)
                for i, (d, ns) in enumerate(zip(client_deltas, sample_counts)):
                    w = ns / total_n
                    agg_delta += w * d[n].to(DEVICE)
                p.data.add_(agg_delta)

            # Update c = average(c_i) after all clients complete.
            for n in scaffold.c_global:
                agg_c = torch.zeros_like(scaffold.c_global[n])
                for b in bundles:
                    agg_c += scaffold.c_locals[b.client_id][n] / num_clients
                scaffold.c_global[n] = agg_c.detach().clone()

        logs["update_order"]["evidence_trace"].append(f"round {rnd}: aggregate w_local to w_global")
        logs["update_order"]["evidence_trace"].append(f"round {rnd}: update c_global = average(c_i)")

        # Evaluate global model.
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

        c_i_norms_now = [tensor_dict_l2_norm(scaffold.c_locals[cid]) for cid in client_ids]
        control_norm_mean = float(np.mean(c_i_norms_now)) if c_i_norms_now else 0.0
        grad_norm_mean = float(np.mean(client_grad_norm_means)) if client_grad_norm_means else 0.0
        c_global_norm = tensor_dict_l2_norm(scaffold.c_global)

        logs["per_round_core"].append(
            {
                "round": rnd,
                "mean_accuracy": float(np.mean(accs)),
                "grad_norm_mean": grad_norm_mean,
                "control_norm_mean": control_norm_mean,
                "ratio_control_to_grad": float(control_norm_mean / max(grad_norm_mean, 1e-12)),
                "c_global_norm": c_global_norm,
                "c_i_norms": c_i_norms_now,
            }
        )
        logs["per_round_client_updates"].extend(round_client_rows)

    logs["persistence"] = {
        "client_1_c_i_norms_over_rounds": c_i_norm_series[str(client_ids[0])],
        "client_2_c_i_norms_over_rounds": c_i_norm_series[str(client_ids[1])],
    }

    out = PROJECT_ROOT / "outputs" / "scaffold_control_signal_dump.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
