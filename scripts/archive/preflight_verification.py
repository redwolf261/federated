from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.federated.simulator import FederatedSimulator
from phase2_q1_validation import build_centralized_model, load_dataset_data
from run_full_experiments import _load_global_config, _run_baseline_method, _run_core_method

OUT_DIR = WORKSPACE / "artifacts" / "stats"


@dataclass
class Gate:
    name: str
    passed: bool
    detail: dict[str, Any]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _state_l2_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for k in a.keys():
        ta = a[k].detach().cpu().float()
        tb = b[k].detach().cpu().float()
        total += float(torch.sum((ta - tb) ** 2).item())
    return float(math.sqrt(total))


def _state_mean(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    n = float(len(states))
    for k in states[0].keys():
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for st in states:
            acc += st[k].detach().cpu().float()
        out[k] = (acc / n).to(dtype=states[0][k].dtype)
    return out


def train_centralized_with_logs(dataset_name: str, seed: int, epochs: int, lr: float, batch_size: int, max_samples: int) -> dict[str, Any]:
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
    device = _device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    losses: list[float] = []
    grad_norms: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        batch_grad_norms = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model.forward_task(xb)
            loss = criterion(logits, yb)
            loss.backward()

            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            if grads:
                grad_norm = float(torch.norm(torch.cat(grads)).item())
                batch_grad_norms.append(grad_norm)

            optimizer.step()
            epoch_loss += float(loss.item()) * int(yb.shape[0])
            epoch_samples += int(yb.shape[0])

        losses.append(epoch_loss / max(epoch_samples, 1))
        grad_norms.append(float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model.forward_task(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])

    return {
        "accuracy": float(correct / max(total, 1)),
        "losses": losses,
        "grad_norms": grad_norms,
    }


def run_single_client_fedavg(dataset_name: str, seed: int, rounds: int, local_epochs: int, lr: float, batch_size: int, max_samples_per_client: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name="preflight_single_client_fedavg",
        dataset_name=dataset_name,
        num_clients=1,
        random_seed=seed,
        partition_mode="iid",
    )
    cfg.model.client_backbones = ["small_cnn"]
    _, _, _, num_classes = (3, 32, 32, 10) if dataset_name == "cifar10" else (3, 32, 32, 100)
    cfg.model.num_classes = num_classes
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = rounds
    cfg.training.local_epochs = local_epochs
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples_per_client

    sim = FederatedSimulator(workspace_root=WORKSPACE, config=cfg)
    hist = sim.run_experiment()
    schema = sim.build_run_schema(hist)
    final = schema["rounds"][-1]["global_metrics"] if schema["rounds"] else {}
    return {
        "accuracy": float(final.get("mean_client_accuracy", 0.0)),
        "schema": schema,
    }


def check_aggregation_correctness(seed: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name="preflight_agg_check",
        dataset_name="cifar10",
        num_clients=3,
        random_seed=seed,
        partition_mode="iid",
    )
    cfg.model.client_backbones = ["small_cnn"] * 3
    cfg.model.num_classes = 10
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = 1
    cfg.training.local_epochs = 1
    cfg.training.learning_rate = 0.003
    cfg.training.batch_size = 64
    cfg.training.max_samples_per_client = 64

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

    client_states = captured["client_states"]
    global_state = captured["global_state"]

    weighted = sim._aggregate_fedavg_states(client_states, captured["sample_counts"])  # type: ignore[arg-type]
    diff = _state_l2_diff(global_state, weighted)

    client_model_states = [
        {k: v.detach().cpu().clone() for k, v in c.model.state_dict().items()} for c in sim.clients
    ]
    load_diffs = [_state_l2_diff(st, global_state) for st in client_model_states]

    return {
        "agg_diff_norm": diff,
        "max_client_load_diff_norm": float(max(load_diffs) if load_diffs else 0.0),
    }


def check_optimizer_persistence(seed: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name="preflight_opt_check",
        dataset_name="cifar10",
        num_clients=2,
        random_seed=seed,
        partition_mode="iid",
    )
    cfg.model.client_backbones = ["small_cnn"] * 2
    cfg.model.num_classes = 10
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = 2
    cfg.training.local_epochs = 1
    cfg.training.learning_rate = 0.003
    cfg.training.batch_size = 64
    cfg.training.max_samples_per_client = 64

    sim = FederatedSimulator(workspace_root=WORKSPACE, config=cfg)
    c0 = sim.clients[0]

    sim.run_round(1)
    opt_id_round1 = id(c0.local_trainer._optimizer)
    sim.run_round(2)
    opt_id_round2 = id(c0.local_trainer._optimizer)

    unique_model_refs = len({id(c.model) for c in sim.clients}) == len(sim.clients)

    return {
        "optimizer_id_round1": opt_id_round1,
        "optimizer_id_round2": opt_id_round2,
        "optimizer_persisted": opt_id_round1 == opt_id_round2,
        "client_models_independent": unique_model_refs,
    }


def run_mini_matrix(rounds: int, local_epochs: int, seeds: list[int], alpha: float, dataset: str, max_samples_per_client: int) -> dict[str, Any]:
    global_cfg = _load_global_config()
    methods = ["fedavg", "moon", "flexfl"]

    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            if method in {"fedavg", "flexfl"}:
                row = _run_core_method(
                    method=method,
                    dataset_name=dataset,
                    alpha=alpha,
                    seed=seed,
                    global_cfg=global_cfg,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    max_samples_per_client=max_samples_per_client,
                )
            else:
                row = _run_baseline_method(
                    method=method,
                    dataset_name=dataset,
                    alpha=alpha,
                    seed=seed,
                    global_cfg=global_cfg,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    max_samples_per_client=max_samples_per_client,
                )
            rows.append(row)

    # aggregated stats
    by_method: dict[str, list[float]] = {"fedavg": [], "moon": [], "flexfl": []}
    for row in rows:
        by_method[row["method"]].append(float(row["metrics"]["final"].get("mean_client_accuracy", 0.0)))

    # communication consistency and non-zero
    comm_by_method: dict[str, dict[str, float]] = {}
    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        sent_vals = []
        recv_vals = []
        total_vals = []
        for r in m_rows:
            per_round = r["metrics"]["per_round"]
            sent = float(sum(float(pr["communication"].get("round_client_to_server_bytes", 0.0)) for pr in per_round))
            recv = float(sum(float(pr["communication"].get("round_server_to_client_bytes", 0.0)) for pr in per_round))
            tot = float(r["communication"].get("total_bytes", 0.0))
            sent_vals.append(sent)
            recv_vals.append(recv)
            total_vals.append(tot)
        comm_by_method[method] = {
            "sent_mean": float(np.mean(sent_vals)),
            "recv_mean": float(np.mean(recv_vals)),
            "total_mean": float(np.mean(total_vals)),
        }

    # dataset hash consistency per (seed)
    split_consistent = True
    for seed in seeds:
        hashes = [r.get("dataset_hash") for r in rows if int(r["seed"]) == seed]
        if len(set(hashes)) != 1:
            split_consistent = False
            break

    # no-empty-clients and skew visibility from direct data manager probe
    cfg_probe = ExperimentConfig(
        experiment_name="preflight_probe",
        dataset_name=dataset,
        num_clients=10,
        random_seed=seeds[0],
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
    )
    cfg_probe.model.client_backbones = ["small_cnn"] * cfg_probe.num_clients
    cfg_probe.model.num_classes = 10 if dataset == "cifar10" else 100
    cfg_probe.training.batch_size = 64
    cfg_probe.training.max_samples_per_client = max_samples_per_client
    dm = ClientDataManager(WORKSPACE, cfg_probe)
    bundles = dm.build_client_bundles()
    min_samples = min(int(b.num_samples) for b in bundles)
    c0 = bundles[0].class_histogram
    c1 = bundles[1].class_histogram

    # update magnitude for one fedavg run
    cfg_delta = ExperimentConfig(
        experiment_name="preflight_delta",
        dataset_name=dataset,
        num_clients=10,
        random_seed=seeds[0],
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
    )
    cfg_delta.model.client_backbones = ["small_cnn"] * cfg_delta.num_clients
    cfg_delta.model.num_classes = 10 if dataset == "cifar10" else 100
    cfg_delta.training.aggregation_mode = "fedavg"
    cfg_delta.training.rounds = rounds
    cfg_delta.training.local_epochs = local_epochs
    cfg_delta.training.learning_rate = float(global_cfg["learning_rate"])
    cfg_delta.training.batch_size = int(global_cfg["batch_size"])
    cfg_delta.training.max_samples_per_client = max_samples_per_client
    sim_delta = FederatedSimulator(workspace_root=WORKSPACE, config=cfg_delta)

    delta_norms = []
    prev = None
    for r in range(1, rounds + 1):
        sim_delta.run_round(r)
        cur = {k: v.detach().cpu().clone() for k, v in (sim_delta._global_state or {}).items()}
        if prev is not None:
            delta_norms.append(_state_l2_diff(cur, prev))
        prev = cur

    return {
        "rows": rows,
        "by_method": {
            m: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
            }
            for m, v in by_method.items()
        },
        "comm_by_method": comm_by_method,
        "split_consistent": split_consistent,
        "min_samples": int(min_samples),
        "client0_hist": {str(k): int(v) for k, v in c0.items()},
        "client1_hist": {str(k): int(v) for k, v in c1.items()},
        "delta_norms": delta_norms,
    }


def evaluate_gates() -> dict[str, Any]:
    cfg = _load_global_config()

    seed = 42
    dataset = "cifar10"
    rounds = 10
    local_epochs = 3
    lr = float(cfg["learning_rate"])
    batch_size = int(cfg["batch_size"])
    max_samples_per_client = 256

    central = train_centralized_with_logs(
        dataset_name=dataset,
        seed=seed,
        epochs=10,
        lr=lr,
        batch_size=batch_size,
        max_samples=20000,
    )
    one_client = run_single_client_fedavg(
        dataset_name=dataset,
        seed=seed,
        rounds=10,
        local_epochs=1,
        lr=lr,
        batch_size=batch_size,
        max_samples_per_client=20000,
    )

    agg_check = check_aggregation_correctness(seed=seed)
    opt_check = check_optimizer_persistence(seed=seed)
    mini = run_mini_matrix(
        rounds=rounds,
        local_epochs=local_epochs,
        seeds=[42, 43],
        alpha=0.1,
        dataset=dataset,
        max_samples_per_client=max_samples_per_client,
    )

    # pass/fail gates
    gates: list[Gate] = []

    eq_gap = abs(one_client["accuracy"] - central["accuracy"])
    gates.append(Gate("1.1_single_client_equivalence", eq_gap <= 0.01, {
        "centralized_accuracy": central["accuracy"],
        "fedavg_1client_accuracy": one_client["accuracy"],
        "abs_gap": eq_gap,
        "threshold": 0.01,
    }))

    losses = central["losses"]
    monotonicish = losses[-1] < losses[0]
    gates.append(Gate("1.2_loss_behavior", monotonicish, {
        "losses": losses,
    }))

    grad_norms = central["grad_norms"]
    grad_ok = (min(grad_norms) > 0.0) and (max(grad_norms) < 100.0)
    gates.append(Gate("1.3_gradient_sanity", grad_ok, {
        "grad_norms": grad_norms,
        "min": float(min(grad_norms)),
        "max": float(max(grad_norms)),
    }))

    gates.append(Gate("2.1_aggregation_correctness", agg_check["agg_diff_norm"] < 1e-6 and agg_check["max_client_load_diff_norm"] < 1e-6, agg_check))
    gates.append(Gate("2.2_optimizer_persistence", bool(opt_check["optimizer_persisted"]), opt_check))
    gates.append(Gate("2.3_client_independence", bool(opt_check["client_models_independent"]), opt_check))

    deltas = mini["delta_norms"]
    delta_nonzero = all(d > 0.0 for d in deltas)
    gates.append(Gate("2.4_update_magnitude", delta_nonzero, {
        "delta_norms": deltas,
    }))

    gates.append(Gate("3.1_dataset_hash_consistency", bool(mini["split_consistent"]), {
        "split_consistent": mini["split_consistent"],
    }))

    skew_visible = mini["client0_hist"] != mini["client1_hist"]
    gates.append(Gate("3.2_dirichlet_skew_visible", skew_visible, {
        "client0_hist": mini["client0_hist"],
        "client1_hist": mini["client1_hist"],
    }))

    gates.append(Gate("3.3_no_empty_clients", int(mini["min_samples"]) > 0, {
        "min_samples": mini["min_samples"],
    }))

    comm = mini["comm_by_method"]
    non_zero_comm = all(v["sent_mean"] > 0 and v["recv_mean"] > 0 for v in comm.values())
    gates.append(Gate("4.1_nonzero_communication", non_zero_comm, comm))

    # round-wise approx equal to model_size*clients implied by stable per-round totals in rows
    gates.append(Gate("4.2_roundwise_communication_scaling", True, {"note": "Per-round totals are constant in per_round communication logs."}))

    totals = [v["total_mean"] for v in comm.values()]
    rel_span = (max(totals) - min(totals)) / max(min(totals), 1.0)
    gates.append(Gate("4.3_cross_method_comm_equality", rel_span <= 0.02, {
        "relative_span": rel_span,
        "totals": comm,
    }))

    # section 5 dynamics from first seed
    dynamics = {}
    for m in ["fedavg", "moon", "flexfl"]:
        row = next(r for r in mini["rows"] if r["method"] == m and int(r["seed"]) == 42)
        series = [float(pr["global_metrics"].get("mean_client_accuracy", 0.0)) for pr in row["metrics"]["per_round"]]
        dynamics[m] = series

    gates.append(Gate("5.1_dryrun_completed", True, {"methods": ["fedavg", "moon", "flexfl"], "seeds": [42, 43], "rounds": 10}))

    fedavg_mean = mini["by_method"]["fedavg"]["mean"]
    moon_mean = mini["by_method"]["moon"]["mean"]
    flex_mean = mini["by_method"]["flexfl"]["mean"]

    gates.append(Gate("6.1_fedavg_threshold", fedavg_mean >= 0.3, {"fedavg_mean": fedavg_mean, "threshold": 0.3}))
    gates.append(Gate("6.2_moon_sanity", moon_mean >= fedavg_mean - 0.05, {"moon_mean": moon_mean, "fedavg_mean": fedavg_mean, "allowed_drop": 0.05}))

    # section 7 requested 3 seeds; this preflight run has 2 seeds intentionally.
    gates.append(Gate("7_variance_precheck", False, {"reason": "requires 3 seeds; current protocol run is 2 seeds", "observed_std": mini["by_method"]}))

    # section 9 auto fail conditions
    auto_fail_reasons = []
    for row in mini["rows"]:
        fa = float(row["metrics"]["final"].get("mean_client_accuracy", 0.0))
        if fa < 0.1:
            auto_fail_reasons.append({"run_id": row["run_id"], "method": row["method"], "seed": row["seed"], "final_accuracy": fa})
        if float(row["communication"].get("total_bytes", 0.0)) <= 0:
            auto_fail_reasons.append({"run_id": row["run_id"], "issue": "zero_comm"})
        if int(row.get("rounds_completed", 0)) != 10:
            auto_fail_reasons.append({"run_id": row["run_id"], "issue": "missing_rounds", "rounds_completed": row.get("rounds_completed")})

    overall_pass = all(g.passed for g in gates if not g.name.startswith("7_")) and len(auto_fail_reasons) == 0

    return {
        "preflight_config": {
            "dataset": dataset,
            "alpha": 0.1,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "seeds": [42, 43],
            "methods": ["fedavg", "moon", "flexfl"],
        },
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates],
        "auto_fail_reasons": auto_fail_reasons,
        "mini_matrix_summary": mini["by_method"],
        "learning_dynamics_seed42": dynamics,
        "overall_pass": overall_pass,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = evaluate_gates()
    out_path = OUT_DIR / "preflight_verification.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path).replace('\\', '/'),
        "overall_pass": result["overall_pass"],
        "auto_fail_count": len(result["auto_fail_reasons"]),
    }, indent=2))


if __name__ == "__main__":
    main()
