from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.evaluation.metrics import Evaluator
from flex_persona.federated.simulator import FederatedSimulator
from phase2_q1_validation import build_centralized_model, load_dataset_data, run_moon, run_scaffold

OUT_DIR = WORKSPACE / "artifacts" / "stats"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _state_l2_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for k in a.keys():
        ta = a[k].detach().cpu().float()
        tb = b[k].detach().cpu().float()
        total += float(torch.sum((ta - tb) ** 2).item())
    return float(math.sqrt(total))


def train_centralized(dataset_name: str, seed: int, epochs: int, lr: float, batch_size: int, max_samples: int) -> dict[str, Any]:
    _set_seed(seed)
    images, labels = load_dataset_data(dataset_name, max_samples=max_samples)

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

    losses = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_task(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = int(yb.shape[0])
            epoch_loss += float(loss.item()) * bs
            epoch_samples += bs
        losses.append(epoch_loss / max(epoch_samples, 1))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model.forward_task(xb).argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.shape[0])

    return {
        "accuracy": float(correct / max(total, 1)),
        "losses": losses,
        "split_sizes": {"train": int(len(train_ds)), "test": int(len(test_ds))},
    }


def run_fedavg(dataset_name: str, seed: int, num_clients: int, rounds: int, local_epochs: int, lr: float, batch_size: int,
               partition_mode: str, dirichlet_alpha: float | None, max_samples_per_client: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name=f"audit_fedavg_{dataset_name}_{partition_mode}_{seed}_{num_clients}",
        dataset_name=dataset_name,
        num_clients=num_clients,
        random_seed=seed,
        partition_mode=partition_mode,
        dirichlet_alpha=0.5 if dirichlet_alpha is None else dirichlet_alpha,
        output_dir=str(OUT_DIR),
    )
    _, _, _, num_classes = (3, 32, 32, 10) if dataset_name == "cifar10" else (1, 28, 28, 62)
    cfg.model.num_classes = num_classes
    cfg.model.client_backbones = ["small_cnn"] * num_clients
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = rounds
    cfg.training.local_epochs = local_epochs
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples_per_client

    sim = FederatedSimulator(workspace_root=WORKSPACE, config=cfg)
    history = sim.run_experiment()
    schema = sim.build_run_schema(history)
    final = schema["rounds"][-1]["global_metrics"] if schema["rounds"] else {}

    return {
        "sim": sim,
        "history": history,
        "schema": schema,
        "final_mean": float(final.get("mean_client_accuracy", 0.0)),
        "final_worst": float(final.get("worst_client_accuracy", 0.0)),
        "partition_fingerprint": schema.get("partition_fingerprint"),
        "rounds": int(len(schema.get("rounds", []))),
        "status": schema.get("final_summary", {}).get("run_summary", {}).get("run_status", "UNKNOWN"),
    }


def aggregate_identity_test(seed: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name="audit_b1_agg_identity",
        dataset_name="cifar10",
        num_clients=2,
        random_seed=seed,
        partition_mode="iid",
        output_dir=str(OUT_DIR),
    )
    cfg.model.num_classes = 10
    cfg.model.client_backbones = ["small_cnn", "small_cnn"]
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = 1
    cfg.training.local_epochs = 1
    cfg.training.learning_rate = 0.003
    cfg.training.batch_size = 64
    cfg.training.max_samples_per_client = 128

    sim = FederatedSimulator(workspace_root=WORKSPACE, config=cfg)

    captured: dict[str, Any] = {}
    original = sim._aggregate_fedavg_states

    def wrapped(client_states: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
        out = original(client_states, sample_counts)
        captured["client_states"] = client_states
        captured["sample_counts"] = sample_counts
        captured["global_state"] = out
        return out

    sim._aggregate_fedavg_states = wrapped  # type: ignore[assignment]
    sim.run_round(1)

    cs = captured["client_states"]
    counts = captured["sample_counts"]
    gs = captured["global_state"]

    n1, n2 = float(counts[0]), float(counts[1])
    manual: dict[str, torch.Tensor] = {}
    for k in cs[0].keys():
        t1 = cs[0][k].detach().cpu().float()
        t2 = cs[1][k].detach().cpu().float()
        if torch.is_floating_point(cs[0][k]):
            manual[k] = ((n1 * t1 + n2 * t2) / (n1 + n2)).to(dtype=cs[0][k].dtype)
        else:
            manual[k] = cs[0][k].detach().cpu().clone()

    diff = _state_l2_diff(gs, manual)
    return {
        "n1": n1,
        "n2": n2,
        "l2_diff": diff,
        "pass_threshold": 1e-6,
        "pass": bool(diff < 1e-6),
    }


def no_training_sanity(seed: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name=f"audit_b2_no_training_{seed}",
        dataset_name="cifar10",
        num_clients=2,
        random_seed=seed,
        partition_mode="iid",
        output_dir=str(OUT_DIR),
    )
    cfg.model.num_classes = 10
    cfg.model.client_backbones = ["small_cnn", "small_cnn"]
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = 5
    cfg.training.local_epochs = 1
    cfg.training.learning_rate = 0.003
    cfg.training.batch_size = 64
    cfg.training.max_samples_per_client = 256

    sim = FederatedSimulator(workspace_root=WORKSPACE, config=cfg)
    for client in sim.clients:
        client.train_local = lambda *args, **kwargs: {"local_loss": 0.0}  # type: ignore[method-assign]

    history = sim.run_experiment()
    schema = sim.build_run_schema(history)
    rounds = schema["rounds"]
    means = [float(r["global_metrics"].get("mean_client_accuracy", 0.0)) for r in rounds]
    totals = [
        float(r["communication"].get("round_total_bytes", 0.0))
        for r in rounds
    ]
    span = max(means) - min(means) if means else 0.0
    return {
        "round_means": means,
        "round_total_bytes": totals,
        "accuracy_span": float(span),
        "pass": bool(span < 1e-9),
    }


def moon_loss_scale_probe(seed: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name="audit_c3_moon_loss_scale",
        dataset_name="cifar10",
        num_clients=2,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=0.1,
        output_dir=str(OUT_DIR),
    )
    cfg.model.num_classes = 10
    cfg.model.client_backbones = ["small_cnn", "small_cnn"]
    cfg.training.batch_size = 64
    cfg.training.max_samples_per_client = 256

    dm = ClientDataManager(str(WORKSPACE), cfg)
    bundles = dm.build_client_bundles()

    global_model = build_centralized_model(dataset_name="cifar10", num_classes=10).to(DEVICE)
    local_model = copy.deepcopy(global_model).to(DEVICE)
    prev_model = copy.deepcopy(global_model).to(DEVICE)
    prev_model.eval()
    global_model.eval()

    criterion = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    ce_values = []
    con_values = []

    for xb, yb in bundles[0].train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = local_model.forward_task(xb)
        task_loss = criterion(logits, yb)

        z_local = local_model.forward_shared(xb)
        with torch.no_grad():
            z_global = global_model.forward_shared(xb)
            z_prev = prev_model.forward_shared(xb)

        pos = cos_sim(z_local, z_global) / 0.5
        neg = cos_sim(z_local, z_prev) / 0.5
        logits_con = torch.stack([pos, neg], dim=1)
        labels_con = torch.zeros(xb.size(0), dtype=torch.long, device=DEVICE)
        con_loss = nn.CrossEntropyLoss()(logits_con, labels_con)

        ce_values.append(float(task_loss.item()))
        con_values.append(float(con_loss.item()))
        if len(ce_values) >= 10:
            break

    ce_mean = float(np.mean(ce_values)) if ce_values else 0.0
    con_mean = float(np.mean(con_values)) if con_values else 0.0
    ratio = float(con_mean / max(ce_mean, 1e-12))
    return {
        "ce_losses": ce_values,
        "contrastive_losses": con_values,
        "ce_mean": ce_mean,
        "contrastive_mean": con_mean,
        "contrastive_to_ce_ratio": ratio,
    }


def zero_control_scaffold_equivalence(seed: int) -> dict[str, Any]:
    fed = run_fedavg(
        dataset_name="cifar10",
        seed=seed,
        num_clients=2,
        rounds=10,
        local_epochs=1,
        lr=0.003,
        batch_size=64,
        partition_mode="dirichlet",
        dirichlet_alpha=0.1,
        max_samples_per_client=256,
    )

    sca = run_scaffold(
        dataset_name="cifar10",
        num_classes=10,
        num_clients=2,
        rounds=10,
        local_epochs=1,
        seed=seed,
        alpha=0.1,
        lr=0.003,
        batch_size=64,
        max_samples=512,
        return_trace=True,
    )

    fed_mean = float(fed["final_mean"])
    sca_mean = float(sca["mean_accuracy"])
    return {
        "fedavg_mean": fed_mean,
        "scaffold_mean": sca_mean,
        "abs_gap": float(abs(sca_mean - fed_mean)),
    }


def paired_t(a: list[float], b: list[float]) -> dict[str, float]:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = float(sum(d) / n)
    var = float(sum((x - mean) ** 2 for x in d) / (n - 1))
    sd = float(math.sqrt(var))
    t_stat = float(mean / (sd / math.sqrt(n)))
    p = float(2 * student_t.sf(abs(t_stat), df=n - 1))
    return {"n": float(n), "t_stat": t_stat, "p_value": p}


def run_audit() -> dict[str, Any]:
    _set_seed(42)
    results: dict[str, Any] = {"checks": {}}

    # A1
    a1 = train_centralized(
        dataset_name="cifar10",
        seed=42,
        epochs=20,
        lr=0.003,
        batch_size=64,
        max_samples=5000,
    )
    results["checks"]["A1_centralized"] = a1

    # A2
    a2 = run_fedavg(
        dataset_name="cifar10",
        seed=42,
        num_clients=1,
        rounds=20,
        local_epochs=1,
        lr=0.003,
        batch_size=64,
        partition_mode="iid",
        dirichlet_alpha=None,
        max_samples_per_client=5000,
    )
    results["checks"]["A2_one_client_fedavg"] = {
        "centralized_accuracy": float(a1["accuracy"]),
        "fedavg_1client_accuracy": float(a2["final_mean"]),
        "abs_gap": float(abs(a1["accuracy"] - a2["final_mean"])),
    }

    # B1
    results["checks"]["B1_weight_aggregation_identity"] = aggregate_identity_test(seed=42)

    # B2
    results["checks"]["B2_no_training_sanity"] = no_training_sanity(seed=42)

    # B3
    b3 = run_fedavg(
        dataset_name="cifar10",
        seed=42,
        num_clients=2,
        rounds=20,
        local_epochs=1,
        lr=0.003,
        batch_size=64,
        partition_mode="iid",
        dirichlet_alpha=None,
        max_samples_per_client=2500,
    )
    results["checks"]["B3_iid_2client_equivalence"] = {
        "centralized_accuracy": float(a1["accuracy"]),
        "fedavg_2client_accuracy": float(b3["final_mean"]),
        "abs_gap": float(abs(a1["accuracy"] - b3["final_mean"])),
    }

    # C1 (mu=0 vs fedavg with matched non-iid setup)
    fed_c1 = run_fedavg(
        dataset_name="cifar10",
        seed=42,
        num_clients=2,
        rounds=10,
        local_epochs=1,
        lr=0.003,
        batch_size=64,
        partition_mode="dirichlet",
        dirichlet_alpha=0.1,
        max_samples_per_client=256,
    )
    moon_mu0 = run_moon(
        dataset_name="cifar10",
        num_classes=10,
        num_clients=2,
        rounds=10,
        local_epochs=1,
        seed=42,
        alpha=0.1,
        lr=0.003,
        batch_size=64,
        max_samples=512,
        mu=0.0,
        return_trace=True,
    )
    results["checks"]["C1_moon_mu0_equals_fedavg"] = {
        "fedavg_mean": float(fed_c1["final_mean"]),
        "moon_mu0_mean": float(moon_mu0["mean_accuracy"]),
        "abs_gap": float(abs(fed_c1["final_mean"] - moon_mu0["mean_accuracy"])),
    }

    # C2 representation usage check from source
    src = inspect.getsource(run_moon)
    results["checks"]["C2_representation_consistency"] = {
        "uses_forward_shared": bool("forward_shared" in src),
        "uses_forward_task_for_contrastive": bool("z_local = local_model.forward_task" in src),
    }

    # C3 loss scale
    results["checks"]["C3_loss_scale"] = moon_loss_scale_probe(seed=42)

    # D1 source check for scaffold equation signs
    scaffold_src = inspect.getsource(run_scaffold)
    results["checks"]["D1_scaffold_source_equation"] = {
        "has_grad_add_cglobal_minus_clocal": bool("p.grad.add_(" in scaffold_src and "c_global" in scaffold_src and "c_local" in scaffold_src),
        "has_clocal_update": bool("new_c = (c_local_n - c_global_n" in scaffold_src),
        "has_cglobal_update": bool("scaffold.c_global[n] = agg_c_delta" in scaffold_src),
    }

    # D2 zero-control proxy equivalence check (current scaffold implementation vs fedavg)
    results["checks"]["D2_scaffold_fedavg_equivalence_proxy"] = zero_control_scaffold_equivalence(seed=42)

    # D3 non-iid drift effect check with 3 seeds
    seeds = [42, 43, 44]
    fed_vals = []
    sca_vals = []
    for seed in seeds:
        fed = run_fedavg(
            dataset_name="cifar10",
            seed=seed,
            num_clients=10,
            rounds=6,
            local_epochs=3,
            lr=0.003,
            batch_size=64,
            partition_mode="dirichlet",
            dirichlet_alpha=0.1,
            max_samples_per_client=256,
        )
        sca = run_scaffold(
            dataset_name="cifar10",
            num_classes=10,
            num_clients=10,
            rounds=6,
            local_epochs=3,
            seed=seed,
            alpha=0.1,
            lr=0.003,
            batch_size=64,
            max_samples=2560,
            return_trace=False,
        )
        fed_vals.append(float(fed["final_mean"]))
        sca_vals.append(float(sca["mean_accuracy"]))
    results["checks"]["D3_scaffold_vs_fedavg_non_iid"] = {
        "fedavg": fed_vals,
        "scaffold": sca_vals,
        "mean_fedavg": float(np.mean(fed_vals)),
        "mean_scaffold": float(np.mean(sca_vals)),
    }

    # E1 exact initialization hash
    _set_seed(42)
    m1 = build_centralized_model(dataset_name="cifar10", num_classes=10)
    _set_seed(42)
    m2 = build_centralized_model(dataset_name="cifar10", num_classes=10)
    results["checks"]["E1_initialization_hash"] = {
        "hash_model_1": _state_hash(m1),
        "hash_model_2": _state_hash(m2),
        "identical": _state_hash(m1) == _state_hash(m2),
    }

    # E2 same partition fingerprint across methods (same config/seed)
    fed_e2 = run_fedavg(
        dataset_name="cifar10",
        seed=42,
        num_clients=10,
        rounds=1,
        local_epochs=1,
        lr=0.003,
        batch_size=64,
        partition_mode="dirichlet",
        dirichlet_alpha=0.1,
        max_samples_per_client=256,
    )
    moon_e2 = run_moon(
        dataset_name="cifar10",
        num_classes=10,
        num_clients=10,
        rounds=1,
        local_epochs=1,
        seed=42,
        alpha=0.1,
        lr=0.003,
        batch_size=64,
        max_samples=2560,
        return_trace=False,
    )
    results["checks"]["E2_same_data_per_client"] = {
        "fedavg_partition_fingerprint": fed_e2["partition_fingerprint"],
        "moon_partition_fingerprint": moon_e2.get("partition_fingerprint"),
        "identical": fed_e2["partition_fingerprint"] == moon_e2.get("partition_fingerprint"),
    }

    # E3 budget parity
    rounds = 20
    local_epochs = 1
    batches_per_client = math.ceil(2500 / 64)
    fed_updates = rounds * local_epochs * batches_per_client
    centralized_updates = 20 * math.ceil((5000 * 0.8) / 64)
    results["checks"]["E3_training_budget"] = {
        "fedavg_total_updates_per_client": int(fed_updates),
        "centralized_total_updates": int(centralized_updates),
    }

    # E4 optimizer persistence
    cfg_opt = ExperimentConfig(
        experiment_name="audit_e4_optimizer",
        dataset_name="cifar10",
        num_clients=2,
        random_seed=42,
        partition_mode="iid",
        output_dir=str(OUT_DIR),
    )
    cfg_opt.model.num_classes = 10
    cfg_opt.model.client_backbones = ["small_cnn", "small_cnn"]
    cfg_opt.training.aggregation_mode = "fedavg"
    cfg_opt.training.rounds = 2
    cfg_opt.training.local_epochs = 1
    cfg_opt.training.learning_rate = 0.003
    cfg_opt.training.batch_size = 64
    cfg_opt.training.max_samples_per_client = 128
    sim_opt = FederatedSimulator(workspace_root=WORKSPACE, config=cfg_opt)
    sim_opt.run_round(1)
    id1 = id(sim_opt.clients[0].local_trainer._optimizer)
    sim_opt.run_round(2)
    id2 = id(sim_opt.clients[0].local_trainer._optimizer)
    results["checks"]["E4_optimizer_state_persistence"] = {
        "optimizer_id_round1": id1,
        "optimizer_id_round2": id2,
        "persisted": bool(id1 == id2),
    }

    # F1 expected ranking in IID and non-IID
    iid_fed = run_fedavg("cifar10", 42, 2, 20, 1, 0.003, 64, "iid", None, 2500)
    iid_moon = run_moon("cifar10", 10, 2, 20, 1, 42, 1.0, 0.003, 64, 5000, 1.0, 0.5, False)
    iid_sca = run_scaffold("cifar10", 10, 2, 20, 1, 42, 1.0, 0.003, 64, 5000, False)

    niid_fed = run_fedavg("cifar10", 42, 10, 6, 3, 0.003, 64, "dirichlet", 0.1, 256)
    niid_moon = run_moon("cifar10", 10, 10, 6, 3, 42, 0.1, 0.003, 64, 2560, 1.0, 0.5, False)
    niid_sca = run_scaffold("cifar10", 10, 10, 6, 3, 42, 0.1, 0.003, 64, 2560, False)

    results["checks"]["F1_expected_ranking"] = {
        "iid": {
            "fedavg": float(iid_fed["final_mean"]),
            "moon": float(iid_moon["mean_accuracy"]),
            "scaffold": float(iid_sca["mean_accuracy"]),
        },
        "non_iid": {
            "fedavg": float(niid_fed["final_mean"]),
            "moon": float(niid_moon["mean_accuracy"]),
            "scaffold": float(niid_sca["mean_accuracy"]),
        },
    }

    # F2 learning curve shape
    fed_series = [float(r["global_metrics"].get("mean_client_accuracy", 0.0)) for r in niid_fed["schema"]["rounds"]]
    diffs = [fed_series[i] - fed_series[i - 1] for i in range(1, len(fed_series))]
    sign_flips = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
    results["checks"]["F2_learning_curve_shape"] = {
        "fedavg_non_iid_series": fed_series,
        "sign_flips": int(sign_flips),
    }

    # G minimal gold test (2 clients IID, 20 rounds, 5k subset)
    gold_fed = run_fedavg("cifar10", 42, 2, 20, 1, 0.003, 64, "iid", None, 2500)
    gold_moon = run_moon("cifar10", 10, 2, 20, 1, 42, 1.0, 0.003, 64, 5000, 1.0, 0.5, False)
    gold_sca = run_scaffold("cifar10", 10, 2, 20, 1, 42, 1.0, 0.003, 64, 5000, False)
    results["checks"]["G_minimal_gold_test"] = {
        "fedavg": float(gold_fed["final_mean"]),
        "moon": float(gold_moon["mean_accuracy"]),
        "scaffold": float(gold_sca["mean_accuracy"]),
        "centralized": float(a1["accuracy"]),
        "fedavg_gap_to_centralized": float(abs(gold_fed["final_mean"] - a1["accuracy"])),
    }

    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_audit()
    out_path = OUT_DIR / "strict_baseline_audit.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path).replace("\\", "/"),
        "checks": sorted(list(results.get("checks", {}).keys())),
    }, indent=2))


if __name__ == "__main__":
    main()
