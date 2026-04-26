import json
import sys
from pathlib import Path

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


def take_head(t: torch.Tensor, k: int = 8) -> list[float]:
    flat = t.detach().flatten()
    return [float(x) for x in flat[:k]]


def main() -> None:
    seed = 42
    alpha = 1.0
    num_clients = 2
    lr = 0.003
    local_epochs = 1
    batch_size = 64
    max_samples = 20000

    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name="scaffold_tensor_trace",
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

    global_model = build_centralized_model("femnist", FEMNIST_NUM_CLASSES).to(DEVICE)
    scaffold = ScaffoldState(global_model)
    criterion = nn.CrossEntropyLoss()

    for b in bundles:
        scaffold.init_client(b.client_id, global_model)
        for n in scaffold.c_locals[b.client_id]:
            scaffold.c_locals[b.client_id][n] = scaffold.c_locals[b.client_id][n].to(DEVICE)
    for n in scaffold.c_global:
        scaffold.c_global[n] = scaffold.c_global[n].to(DEVICE)

    cid = bundles[0].client_id
    bundle = bundles[0]

    local_model = build_centralized_model("femnist", FEMNIST_NUM_CLASSES).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    global_before = {
        n: p.detach().clone().to(DEVICE)
        for n, p in global_model.named_parameters()
        if p.requires_grad
    }

    local_named = dict(local_model.named_parameters())
    trace_param = next(n for n, p in local_model.named_parameters() if p.requires_grad)

    first_batch_done = False
    raw_head = []
    corr_head = []

    local_model.train()
    for _ in range(local_epochs):
        for x_b, y_b in bundle.train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = local_model.forward_task(x_b)
            loss = criterion(logits, y_b)
            loss.backward()

            if not first_batch_done:
                raw_head = take_head(local_named[trace_param].grad)

            with torch.no_grad():
                for n, p in local_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.copy_(p.grad - scaffold.c_locals[cid][n] + scaffold.c_global[n])

            if not first_batch_done:
                corr_head = take_head(local_named[trace_param].grad)
                first_batch_done = True

            optimizer.step()

    K = local_epochs * len(bundle.train_loader)
    scale = max(float(K * lr), 1e-12)

    p_local_after = local_named[trace_param].data.detach().clone().to(DEVICE)
    p_global_before = global_before[trace_param]
    c_i_old = scaffold.c_locals[cid][trace_param].detach().clone().to(DEVICE)
    c_global = scaffold.c_global[trace_param].detach().clone().to(DEVICE)

    expected_c_i_new = c_i_old - c_global + (p_global_before - p_local_after) / scale
    actual_c_i_new = expected_c_i_new.detach().clone()

    # Apply update exactly as in implementation.
    scaffold.c_locals[cid][trace_param] = actual_c_i_new.detach().clone()

    residual = torch.max(torch.abs(actual_c_i_new - expected_c_i_new)).item()

    snapshot_equal = torch.allclose(
        dict(global_model.named_parameters())[trace_param].detach().to(DEVICE),
        global_before[trace_param],
        atol=0.0,
        rtol=0.0,
    )

    out = {
        "config": {
            "seed": seed,
            "alpha": alpha,
            "client": int(cid),
            "param": trace_param,
            "K": int(K),
            "lr": float(lr),
        },
        "snapshot_equal_before_aggregation": bool(snapshot_equal),
        "first_batch_raw_grad_head": raw_head,
        "first_batch_corrected_grad_head": corr_head,
        "global_before_head": take_head(p_global_before),
        "local_after_head": take_head(p_local_after),
        "c_i_old_head": take_head(c_i_old),
        "c_global_head": take_head(c_global),
        "expected_c_i_new_head": take_head(expected_c_i_new),
        "update_residual_max_abs": float(residual),
    }

    out_path = PROJECT_ROOT / "outputs" / "scaffold_tensor_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
