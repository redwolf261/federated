"""Block M method implementations: SCAFFOLD, MOON, PureLocal."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SHARED_DIM = 64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _eval_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += int((preds == yb).sum())
            total += int(yb.shape[0])
    return float(correct / max(total, 1))


def _build_model(num_classes: int = 10) -> nn.Module:
    from flex_persona.models.backbones import SmallCNNBackbone
    from flex_persona.models.adapter_network import AdapterNetwork
    from flex_persona.models.client_model import ClientModel
    bb = SmallCNNBackbone(in_channels=3, input_height=32, input_width=32)
    ad = AdapterNetwork(bb.output_dim, SHARED_DIM)
    return ClientModel(bb, ad, num_classes)


def _get_data(workspace_root, seed: int, num_clients: int, alpha: float,
              batch_size: int, max_per_client: int, dataset: str):
    """Reuse FederatedSimulator data pipeline to get loaders."""
    import sys; sys.path.insert(0, str(workspace_root))
    from flex_persona.config.experiment_config import ExperimentConfig
    from flex_persona.federated.simulator import FederatedSimulator
    cfg = ExperimentConfig(
        experiment_name=f"block_m_data_s{seed}",
        dataset_name=dataset,
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir="/tmp",
    )
    cfg.model.num_classes = 10
    cfg.model.client_backbones = ["small_cnn"]
    cfg.model.shared_dim = SHARED_DIM
    cfg.training.rounds = 1
    cfg.training.local_epochs = 1
    cfg.training.cluster_aware_epochs = 0
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_per_client
    cfg.training.aggregation_mode = "prototype"
    cfg.training.ablation_mode = "no_prototype_sharing"
    sim = FederatedSimulator(workspace_root=str(workspace_root), config=cfg)
    return [(c.train_loader, c.eval_loader) for c in sim.clients]


# ---------------------------------------------------------------------------
# Pure Local Training
# ---------------------------------------------------------------------------

def run_pure_local(loaders, rounds: int, local_epochs: int,
                   lr: float, device: str, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    models = [_build_model().to(device) for _ in loaders]
    opts   = [torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
              for m in models]
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    ce = nn.CrossEntropyLoss()
    history = []

    for rnd in range(rounds):
        for m, opt, (tl, _) in zip(models, opts, loaders):
            m.train()
            for _ in range(local_epochs):
                for xb, yb in tl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                        loss = ce(m(xb), yb)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        accs = [_eval_accuracy(m, dl, device) for m, (_, dl) in zip(models, loaders)]
        entry = {"round": rnd + 1, "mean": float(np.mean(accs)),
                 "worst": float(min(accs)), "std": float(np.std(accs)),
                 "p10": float(np.percentile(accs, 10)),
                 "per_client": [float(a) for a in accs]}
        history.append(entry)
        if (rnd + 1) % 10 == 0:
            print(f"[pure_local s{seed}] r={rnd+1}/{rounds} mean={entry['mean']:.4f}")
    return history


# ---------------------------------------------------------------------------
# SCAFFOLD
# ---------------------------------------------------------------------------

def run_scaffold(loaders, rounds: int, local_epochs: int,
                 lr: float, device: str, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    n = len(loaders)
    global_model = _build_model().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n)]

    # Control variates stored on device
    server_cv = {k: torch.zeros_like(v, device=device)
                 for k, v in global_model.state_dict().items() if v.is_floating_point()}
    client_cvs = [{k: torch.zeros_like(v, device=device)
                   for k, v in global_model.state_dict().items() if v.is_floating_point()}
                  for _ in range(n)]

    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    history = []

    for rnd in range(rounds):
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        delta_states = []

        for i, (cm, (tl, _)) in enumerate(zip(client_models, loaders)):
            cm.load_state_dict(global_state, strict=True)
            opt = torch.optim.SGD(cm.parameters(), lr=lr)
            ci = client_cvs[i]
            steps = 0

            cm.train()
            for _ in range(local_epochs):
                for xb, yb in tl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                        loss = ce(cm(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    # SCAFFOLD correction on device
                    with torch.no_grad():
                        for name, param in cm.named_parameters():
                            if param.grad is not None and name in ci:
                                param.grad.add_(server_cv[name] - ci[name])
                    scaler.step(opt); scaler.update()
                    steps += 1

            # Update client control variate (all on device)
            new_ci = {}
            new_delta = {}
            for name, param in cm.named_parameters():
                g_p = global_state[name].to(device)
                correction = (g_p - param.data) / max(steps * lr, 1e-8)
                new_ci[name] = ci[name] - server_cv[name] + correction
                new_delta[name] = new_ci[name] - ci[name]
            client_cvs[i] = new_ci

            # Delta state (on CPU to save VRAM)
            delta_states.append(
                {k: (cm.state_dict()[k] - global_state[k]).cpu()
                 for k in global_state})
            setattr(cm, '_cv_delta', {k: v.detach() for k, v in new_delta.items()})

        # Aggregate global model on CPU then move back
        new_global = {k: global_state[k].cpu().float() for k in global_state}
        for ds in delta_states:
            for k in new_global:
                new_global[k].add_(ds[k].float() / n)
        global_model.load_state_dict(
            {k: v.to(device=device, dtype=global_state[k].dtype)
             for k, v in new_global.items()}, strict=True)

        # Update server control variate (on device)
        for k in server_cv:
            delta_sum = sum(
                getattr(cm, '_cv_delta', {}).get(k, torch.zeros_like(server_cv[k]))
                for cm in client_models)
            server_cv[k] = server_cv[k] + delta_sum / n

        for cm in client_models:
            cm.load_state_dict(global_model.state_dict(), strict=True)

        accs = [_eval_accuracy(cm, dl, device)
                for cm, (_, dl) in zip(client_models, loaders)]
        entry = {"round": rnd + 1, "mean": float(np.mean(accs)),
                 "worst": float(min(accs)), "std": float(np.std(accs)),
                 "p10": float(np.percentile(accs, 10)),
                 "per_client": [float(a) for a in accs]}
        history.append(entry)
        if (rnd + 1) % 10 == 0:
            print(f"[scaffold s{seed}] r={rnd+1}/{rounds} mean={entry['mean']:.4f}")
    return history


# ---------------------------------------------------------------------------
# MOON
# ---------------------------------------------------------------------------

def run_moon(loaders, rounds: int, local_epochs: int,
             lr: float, device: str, seed: int,
             mu: float = 5.0, temperature: float = 0.5) -> list[dict]:
    torch.manual_seed(seed)
    n = len(loaders)
    global_model = _build_model().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n)]
    prev_models   = [copy.deepcopy(global_model) for _ in range(n)]
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    history = []

    for rnd in range(rounds):
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        client_states = []

        for i, (cm, pm, (tl, _)) in enumerate(zip(client_models, prev_models, loaders)):
            cm.load_state_dict(global_state, strict=True)
            opt = torch.optim.Adam(cm.parameters(), lr=lr, weight_decay=1e-5)
            cm.train(); pm.eval(); global_model.eval()

            for _ in range(local_epochs):
                for xb, yb in tl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()

                    with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                        # Task loss
                        logits = cm(xb)
                        task_loss = ce(logits, yb)

                        # MOON contrastive loss on shared features
                        with torch.no_grad():
                            z_g = global_model.forward_shared(xb)  # global
                            z_p = pm.forward_shared(xb)            # previous
                        z_c = cm.forward_shared(xb)                # current

                        sim_pos = F.cosine_similarity(z_c, z_g, dim=1) / temperature
                        sim_neg = F.cosine_similarity(z_c, z_p, dim=1) / temperature
                        moon_loss = -torch.log(
                            torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg))
                        ).mean()

                        loss = task_loss + mu * moon_loss

                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

            prev_models[i] = copy.deepcopy(cm)
            client_states.append({k: v.cpu() for k, v in cm.state_dict().items()})

        # FedAvg aggregation on CPU
        new_global = {k: global_state[k].cpu().clone().float() for k in global_state}
        for k in new_global: new_global[k].zero_()
        for cs in client_states:
            for k in new_global:
                new_global[k].add_(cs[k].float() / n)
        global_model.load_state_dict(
            {k: v.to(device=device, dtype=global_state[k].dtype) for k, v in new_global.items()},
            strict=True)
        for cm in client_models:
            cm.load_state_dict(global_model.state_dict(), strict=True)

        global_model.eval()
        accs = [_eval_accuracy(cm, dl, device)
                for cm, (_, dl) in zip(client_models, loaders)]
        entry = {"round": rnd + 1, "mean": float(np.mean(accs)),
                 "worst": float(min(accs)), "std": float(np.std(accs)),
                 "p10": float(np.percentile(accs, 10)),
                 "per_client": [float(a) for a in accs]}
        history.append(entry)
        if (rnd + 1) % 10 == 0:
            print(f"[moon s{seed}] r={rnd+1}/{rounds} mean={entry['mean']:.4f}")
    return history
