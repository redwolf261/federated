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
    build_centralized_model,
    run_fedavg_dirichlet,
    set_seed,
)


def run_manual_fedavg_scaffold_loop(
    alpha: float,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    seed: int,
    lr: float,
    batch_size: int,
    max_samples: int,
) -> dict:
    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name=f"manual_fedavg_scaffold_loop_a{alpha}_s{seed}",
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
    criterion = nn.CrossEntropyLoss()

    client_models = {}
    client_optimizers = {}
    for b in bundles:
        m = build_centralized_model(dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES).to(DEVICE)
        m.load_state_dict(global_model.state_dict())
        client_models[b.client_id] = m
        client_optimizers[b.client_id] = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.0)

    optimizer_ids_by_round = []
    model_ids_by_round = []
    first_batch_weight_change = None

    for rnd in range(1, rounds + 1):
        client_deltas = []
        sample_counts = []

        round_opt_ids = {}
        round_model_ids = {}

        for bundle in bundles:
            cid = bundle.client_id
            local_model = client_models[cid]
            optimizer = client_optimizers[cid]

            round_opt_ids[str(cid)] = int(id(optimizer))
            round_model_ids[str(cid)] = int(id(local_model))

            local_model.load_state_dict(global_model.state_dict())
            local_model.train()

            first_step_recorded = False
            for _ in range(local_epochs):
                for x_b, y_b in bundle.train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    logits = local_model.forward_task(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()

                    if first_batch_weight_change is None and not first_step_recorded:
                        # Probe one trainable tensor for non-zero update.
                        probe_name = next(n for n, p in local_model.named_parameters() if p.requires_grad)
                        p_before = dict(local_model.named_parameters())[probe_name].detach().clone()
                        optimizer.step()
                        p_after = dict(local_model.named_parameters())[probe_name].detach().clone()
                        first_batch_weight_change = float(torch.norm(p_after - p_before).item())
                        first_step_recorded = True
                        continue

                    optimizer.step()

            delta = {}
            with torch.no_grad():
                global_named = dict(global_model.named_parameters())
                local_named = dict(local_model.named_parameters())
                for n, p_global in global_named.items():
                    if not p_global.requires_grad:
                        continue
                    delta[n] = (local_named[n].data - p_global.data).detach().clone().to("cpu")

            client_deltas.append(delta)
            sample_counts.append(bundle.num_samples)

        optimizer_ids_by_round.append(round_opt_ids)
        model_ids_by_round.append(round_model_ids)

        total_n = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg_delta = torch.zeros_like(p.data)
                for d, ns in zip(client_deltas, sample_counts):
                    w = ns / total_n
                    agg_delta += w * d[n].to(DEVICE)
                p.data.add_(agg_delta)

        for b in bundles:
            client_models[b.client_id].load_state_dict(global_model.state_dict())

    global_model.eval()
    client_accs = {}
    with torch.no_grad():
        for b in bundles:
            correct, total = 0, 0
            for x_b, y_b in b.eval_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                preds = global_model.forward_task(x_b).argmax(dim=1)
                correct += int((preds == y_b).sum().item())
                total += int(y_b.shape[0])
            client_accs[b.client_id] = correct / max(total, 1)

    mean_acc = float(np.mean(list(client_accs.values())))

    return {
        "mean_accuracy": mean_acc,
        "optimizer_ids_by_round": optimizer_ids_by_round,
        "model_ids_by_round": model_ids_by_round,
        "first_batch_weight_change_norm": first_batch_weight_change,
    }


def main() -> None:
    alpha = 1.0
    num_clients = 2
    rounds = 10
    local_epochs = 1
    seed = 42
    lr = 0.003
    batch_size = 64
    max_samples = 20000

    manual = run_manual_fedavg_scaffold_loop(
        alpha=alpha,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        seed=seed,
        lr=lr,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    reference = run_fedavg_dirichlet(
        alpha=alpha,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        seed=seed,
        lr=lr,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    out = {
        "manual_fedavg_same_loop": manual["mean_accuracy"],
        "reference_fedavg": reference["mean_accuracy"],
        "diagnostics": {
            "first_batch_weight_change_norm": manual["first_batch_weight_change_norm"],
            "optimizer_ids_by_round": manual["optimizer_ids_by_round"],
            "model_ids_by_round": manual["model_ids_by_round"],
        },
    }

    out_path = PROJECT_ROOT / "outputs" / "scaffold_path_equivalence_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
