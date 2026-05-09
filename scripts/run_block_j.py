#!/usr/bin/env python3
"""Block J: Final Causal Disentanglement.

Isolates whether FLEX gains come from:
  A) Architecture (adapter)
  B) Auxiliary loss (regularization)
  C) Feature transformation (geometry/random projection)
  D) Interaction of the above

Methods:
  1. fedavg_sgd                    - backbone+classifier, CE only (baseline)
  2. backbone_only                 - FLEX backbone+classifier, CE only (control)
  3. backbone_plus_adapter_no_loss - backbone+adapter+classifier, CE only
  4. backbone_plus_random_proj     - backbone+frozen_random_proj+classifier, CE only
  5. backbone_plus_dummy_loss      - backbone+classifier, CE + ||z||^2
  6. backbone_plus_adapter_dummy   - backbone+adapter+classifier, CE + ||z||^2
  7. flex_full_reference           - full FLEX with cluster-aware epochs

Output: outputs/failure_mode_coverage/block_J_results.json
        outputs/failure_mode_coverage/block_J.md
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.models.adapter_network import AdapterNetwork
from scripts.phase2_q1_validation import set_seed

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = COVERAGE_DIR / "block_J_results.json"
REPORT_MD    = COVERAGE_DIR / "block_J.md"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment hyperparams (FIXED across all methods)
DATASET      = "cifar10"
NUM_CLASSES  = 10
NUM_CLIENTS  = 10
ROUNDS       = 20
LOCAL_EPOCHS = 5
BATCH_SIZE   = 64
LR           = 0.001
MAX_SAMPLES  = 20_000   # 2000/client
ALPHA        = 0.1
SEEDS        = [42, 43, 44]
LAMBDA_DUMMY = 0.1      # same scale as lambda_cluster

# CIFAR-10 SmallCNN geometry
BACKBONE_DIM = 128 * 8 * 8   # = 8192
SHARED_DIM   = 64

ALL_METHODS = [
    "fedavg_sgd",
    "backbone_only",
    "backbone_plus_adapter_no_loss",
    "backbone_plus_random_proj",
    "backbone_plus_dummy_loss",
    "backbone_plus_adapter_dummy",
    "flex_full_reference",
]

LABELS = {
    "fedavg_sgd":                    "FedAvg SGD (Baseline)",
    "backbone_only":                 "Backbone Only (Control)",
    "backbone_plus_adapter_no_loss": "Backbone + Adapter, No Loss",
    "backbone_plus_random_proj":     "Backbone + Random Proj (Frozen)",
    "backbone_plus_dummy_loss":      "Backbone + Dummy Loss",
    "backbone_plus_adapter_dummy":   "Backbone + Adapter + Dummy Loss",
    "flex_full_reference":           "FLEX Full (Reference)",
}

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

class _BackboneClassifier(nn.Module):
    """Backbone → Linear classifier (no adapter)."""
    def __init__(self, backbone_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = SmallCNNBackbone(in_channels=3, input_height=32, input_width=32)
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _AdapterClassifier(nn.Module):
    """Backbone → Adapter (trainable) → classifier."""
    def __init__(self, backbone_dim: int, shared_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone   = SmallCNNBackbone(in_channels=3, input_height=32, input_width=32)
        self.adapter    = AdapterNetwork(backbone_dim, shared_dim)
        self.classifier = nn.Linear(shared_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        h = self.adapter(z)
        return self.classifier(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _RandomProjClassifier(nn.Module):
    """Backbone → FROZEN random linear projection → classifier."""
    def __init__(self, backbone_dim: int, shared_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone    = SmallCNNBackbone(in_channels=3, input_height=32, input_width=32)
        proj             = nn.Linear(backbone_dim, shared_dim, bias=False)
        nn.init.orthogonal_(proj.weight)
        proj.weight.requires_grad_(False)   # FROZEN
        self.proj        = proj
        self.classifier  = nn.Linear(shared_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        h = F.normalize(self.proj(z), p=2, dim=-1)
        return self.classifier(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _build_model(method: str) -> nn.Module:
    if method in ("fedavg_sgd", "backbone_only",
                  "backbone_plus_dummy_loss"):
        return _BackboneClassifier(BACKBONE_DIM, NUM_CLASSES)
    elif method in ("backbone_plus_adapter_no_loss",
                    "backbone_plus_adapter_dummy"):
        return _AdapterClassifier(BACKBONE_DIM, SHARED_DIM, NUM_CLASSES)
    elif method == "backbone_plus_random_proj":
        return _RandomProjClassifier(BACKBONE_DIM, SHARED_DIM, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _loss(method: str, model: nn.Module, x: torch.Tensor,
          y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    if method in ("backbone_plus_dummy_loss", "backbone_plus_adapter_dummy"):
        z = model.features(x)
        dummy = LAMBDA_DUMMY * z.pow(2).mean()
        return ce + dummy
    return ce


# ---------------------------------------------------------------------------
# Custom FedAvg loop (for methods 1-6)
# ---------------------------------------------------------------------------

def _run_custom_fedavg(method: str, seed: int) -> dict:
    set_seed(seed)
    cfg = ExperimentConfig(
        experiment_name=f"block_j_{method}_s{seed}",
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes  = NUM_CLASSES
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS

    dm      = ClientDataManager(str(PROJECT_ROOT), cfg)
    bundles = dm.build_client_bundles()

    # Per-client models + optimisers
    client_models = {}
    client_optims = {}
    for b in bundles:
        m = _build_model(method).to(DEVICE)
        client_models[b.client_id] = m
        client_optims[b.client_id] = torch.optim.Adam(m.parameters(), lr=LR)

    # Global model (same architecture, for aggregation)
    global_model = _build_model(method).to(DEVICE)

    for rnd in range(1, ROUNDS + 1):
        client_states  = []
        sample_counts  = []

        for bundle in bundles:
            cid   = bundle.client_id
            model = client_models[cid]
            model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            opt   = client_optims[cid]
            model.train()

            for _ in range(LOCAL_EPOCHS):
                for xb, yb in bundle.train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss = _loss(method, model, xb, yb)
                    loss.backward()
                    opt.step()

            client_states.append(
                {n: p.data.cpu().clone()
                 for n, p in model.named_parameters()
                 if p.requires_grad}
            )
            sample_counts.append(bundle.num_samples)

        # Weighted FedAvg aggregation (trainable params only)
        total = sum(sample_counts)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if not p.requires_grad:
                    continue
                agg = torch.zeros_like(p.data.cpu())
                for state, ns in zip(client_states, sample_counts):
                    if n in state:
                        agg += (ns / total) * state[n]
                p.data.copy_(agg.to(DEVICE))

        # Broadcast
        gstate = global_model.state_dict()
        for b in bundles:
            client_models[b.client_id].load_state_dict(
                copy.deepcopy(gstate))

    # Evaluate
    client_accs: dict[str, float] = {}
    for bundle in bundles:
        model = client_models[bundle.client_id]
        model.eval()
        correct = total_n = 0
        with torch.no_grad():
            for xb, yb in bundle.eval_loader:
                xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
                preds    = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total_n += yb.size(0)
        client_accs[bundle.client_id] = correct / max(total_n, 1)

    vals = list(client_accs.values())
    return {
        "method": method, "seed": seed, "block": "J",
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std": float(np.std(vals)),
        "p10": float(np.percentile(vals, 10)),
        "client_accuracies": {str(k): float(v)
                              for k, v in client_accs.items()},
    }


# ---------------------------------------------------------------------------
# FLEX full reference (method 7)
# ---------------------------------------------------------------------------

def _run_flex_reference(seed: int) -> dict:
    set_seed(seed)
    cfg = ExperimentConfig(
        experiment_name=f"block_j_flex_full_s{seed}",
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes           = NUM_CLASSES
    cfg.model.client_backbones      = ["small_cnn"]
    cfg.training.aggregation_mode   = "prototype"
    cfg.training.rounds             = ROUNDS
    cfg.training.local_epochs       = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = 2
    cfg.training.learning_rate      = LR
    cfg.training.batch_size         = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.lambda_cluster     = 0.1
    cfg.training.lambda_cluster_center = 0.01
    cfg.training.ablation_mode      = "full"
    cfg.training.alignment_mode     = "cluster_prototype"

    sim     = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    sim.run_experiment()

    client_accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
    vals = list(client_accs.values())
    return {
        "method": "flex_full_reference", "seed": seed, "block": "J",
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std": float(np.std(vals)),
        "p10": float(np.percentile(vals, 10)),
        "client_accuracies": {str(k): float(v)
                              for k, v in client_accs.items()},
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    return json.loads(RESULTS_JSON.read_text())


def _save(results: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(results, indent=2))


def _done(results: list[dict], method: str, seed: int) -> bool:
    return any(r["method"] == method and r["seed"] == seed for r in results)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(results: list[dict]) -> dict[str, dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        groups[r["method"]].append(r)
    out = {}
    for m, runs in groups.items():
        means   = [r["mean_accuracy"]  for r in runs]
        worsts  = [r["worst_accuracy"] for r in runs]
        p10s    = [r["p10"]            for r in runs]
        out[m]  = {
            "n":    len(runs),
            "mean": float(np.mean(means)),
            "std":  float(np.std(means)),
            "worst":float(np.mean(worsts)),
            "p10":  float(np.mean(p10s)),
            "seeds":[r["seed"] for r in runs],
        }
    return out


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def _report(results: list[dict], stats: dict[str, dict]) -> str:
    flex = stats.get("flex_full_reference", {})
    fedavg = stats.get("fedavg_sgd", {})
    flex_mean   = flex.get("mean", 0.0)
    fedavg_mean = fedavg.get("mean", 0.0)
    gap = flex_mean - fedavg_mean

    lines = [
        "# Block J: Final Causal Disentanglement",
        "",
        "**Question:** Do FLEX gains arise from (A) architecture, (B) auxiliary loss, "
        "(C) geometry/random transformation, or (D) their interaction?",
        "",
        f"**Total gap (FLEX vs FedAvg):** {gap:+.4f} ({gap/max(fedavg_mean,1e-9)*100:.1f}%)",
        "",
        "---",
        "## Results Table",
        "",
        "| Method | Mean ± Std | Worst | P10 | Δ vs FLEX | Δ vs FedAvg |",
        "|--------|-----------|-------|-----|-----------|-------------|",
    ]

    for m in ALL_METHODS:
        if m not in stats:
            continue
        s = stats[m]
        d_flex   = s["mean"] - flex_mean
        d_fedavg = s["mean"] - fedavg_mean
        lines.append(
            f"| {LABELS[m]} | {s['mean']:.4f} ± {s['std']:.4f} | "
            f"{s['worst']:.4f} | {s['p10']:.4f} | "
            f"{d_flex:+.4f} | {d_fedavg:+.4f} |"
        )

    lines += ["", "---", "## Causal Classification", ""]

    # Compute key deltas
    def _m(k: str) -> float:
        return stats.get(k, {}).get("mean", float("nan"))

    adapter_no_loss_gain = _m("backbone_plus_adapter_no_loss") - fedavg_mean
    random_proj_gain     = _m("backbone_plus_random_proj")     - fedavg_mean
    dummy_loss_gain      = _m("backbone_plus_dummy_loss")      - fedavg_mean
    adapter_dummy_gain   = _m("backbone_plus_adapter_dummy")   - fedavg_mean
    backbone_only_gain   = _m("backbone_only")                 - fedavg_mean

    THR = 0.02   # 2pp threshold for "meaningful" difference

    # Sanity checks
    lines.append("### Sanity Checks")
    lines.append("")
    backbone_ok = abs(backbone_only_gain) < THR
    lines.append(f"- backbone_only ≈ fedavg_sgd: "
                 f"{'PASS ✅' if backbone_ok else 'FAIL ❌'} "
                 f"(Δ={backbone_only_gain:+.4f})")

    # Case logic
    case_a = (abs(dummy_loss_gain) >= THR and
              abs(adapter_no_loss_gain) < THR)
    case_b = (adapter_no_loss_gain >= THR and
              abs(random_proj_gain - adapter_no_loss_gain) < THR)
    case_c = (adapter_no_loss_gain >= THR and
              (random_proj_gain - adapter_no_loss_gain) < -THR)
    case_d = (abs(adapter_no_loss_gain) < THR and
              abs(dummy_loss_gain) < THR and
              adapter_dummy_gain >= THR)

    if case_b:
        case, primary, secondary, rejected = (
            "B", "Feature Transformation (Adapter Architecture)",
            "N/A", "auxiliary loss, semantic signal, learned representation"
        )
        explanation = (
            "The adapter architecture alone (no aux loss) outperforms FedAvg, "
            "and a frozen random projection achieves similar gains. "
            "The driver is geometric transformation, not learned content."
        )
    elif case_c:
        case, primary, secondary, rejected = (
            "C", "Learned Representation (Adapter Training)",
            "N/A", "auxiliary loss, random geometry, semantic signal"
        )
        explanation = (
            "The trainable adapter outperforms FedAvg but the frozen random "
            "projection does not — the learning in the adapter is the key driver."
        )
    elif case_a:
        case, primary, secondary, rejected = (
            "A", "Auxiliary Regularization Loss",
            "N/A", "adapter architecture, learned representation, geometry"
        )
        explanation = (
            "Dummy loss alone matches FLEX, while adapter-no-loss ≈ FedAvg. "
            "Any auxiliary gradient constraint drives the gains."
        )
    elif case_d:
        case, primary, secondary, rejected = (
            "D", "Interaction (Architecture + Auxiliary Loss)",
            "Neither alone is sufficient", "semantic signal, learned representation"
        )
        explanation = (
            "Neither adapter alone nor dummy loss alone is sufficient. "
            "The combination (adapter + auxiliary constraint) replicates FLEX gains."
        )
    else:
        # Determine by largest gain
        gains = {
            "adapter_no_loss": adapter_no_loss_gain,
            "dummy_loss":      dummy_loss_gain,
            "random_proj":     random_proj_gain,
            "adapter_dummy":   adapter_dummy_gain,
        }
        best = max(gains, key=gains.get)
        case, primary, secondary, rejected = (
            "MIXED", f"Inconclusive — largest: {best} ({gains[best]:+.4f})",
            "See deltas above", "none conclusively rejected"
        )
        explanation = "Results are mixed; no single mechanism dominates cleanly."

    lines += [
        "",
        f"### Case Verdict: **CASE {case}**",
        "",
        explanation,
        "",
        "---",
        "## Final Causal Statement",
        "",
        "```",
        f"Primary driver:   {primary}",
        f"Secondary driver: {secondary}",
        f"Rejected mechanisms: {rejected}",
        "```",
        "",
        "---",
        "## Per-Seed Raw Results",
        "",
        "| Method | Seed 42 | Seed 43 | Seed 44 |",
        "|--------|---------|---------|---------|",
    ]

    for m in ALL_METHODS:
        per = {r["seed"]: r["mean_accuracy"]
               for r in results if r["method"] == m}
        s42 = f"{per.get(42, float('nan')):.4f}"
        s43 = f"{per.get(43, float('nan')):.4f}"
        s44 = f"{per.get(44, float('nan')):.4f}"
        lines.append(f"| {LABELS[m]} | {s42} | {s43} | {s44} |")

    lines += [
        "",
        "---",
        "## Aggregated Statistics (JSON)",
        "",
        "```json",
        json.dumps(
            {m: {k: round(v, 6) if isinstance(v, float) else v
                 for k, v in s.items()}
             for m, s in stats.items()},
            indent=2
        ),
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    stats = _agg(results)
    flex_mean = stats.get("flex_full_reference", {}).get("mean", 0.0)
    print(f"\n{'='*72}")
    print("  BLOCK J SUMMARY")
    print(f"  {'Method':<38} {'Done':>4}  {'Mean':>7}  {'Drop':>7}")
    print(f"  {'-'*60}")
    for m in ALL_METHODS:
        if m not in stats:
            print(f"  {m:<38} {'0/3':>4}  {'---':>7}  {'---':>7}")
            continue
        s = stats[m]
        drop = s["mean"] - flex_mean
        print(f"  {m:<38} {str(s['n'])+'/3':>4}  {s['mean']:>7.4f}  {drop:>+7.4f}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  BLOCK J: FINAL CAUSAL DISENTANGLEMENT")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  Methods: {len(ALL_METHODS)}   Seeds: {SEEDS}   Total: {len(ALL_METHODS)*len(SEEDS)}")
    print("=" * 70)

    results = _load()
    print(f"\n  Loaded {len(results)} existing results\n")

    total_start = time.time()
    completed = skipped = 0

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        for method in ALL_METHODS:
            if _done(results, method, seed):
                print(f"  SKIP {method} s{seed}")
                skipped += 1
                continue

            print(f"\n  RUN  {method} | seed={seed}")
            t0 = time.time()
            try:
                if method == "flex_full_reference":
                    result = _run_flex_reference(seed)
                else:
                    result = _run_custom_fedavg(method, seed)
            except Exception as e:
                print(f"  ERROR: {e}")
                raise

            result["wall_time_s"] = round(time.time() - t0, 1)
            results.append(result)
            _save(results)
            completed += 1
            print(f"  => mean={result['mean_accuracy']:.4f}  "
                  f"worst={result['worst_accuracy']:.4f}  "
                  f"p10={result['p10']:.4f}  "
                  f"({result['wall_time_s']:.0f}s)")

    elapsed = time.time() - total_start
    print(f"\n  Done. {completed} new | {skipped} cached | {elapsed/60:.1f} min")

    _print_summary(results)

    stats  = _agg(results)
    report = _report(results, stats)
    REPORT_MD.write_text(report, encoding="utf-8")

    print(f"  JSON: {RESULTS_JSON}")
    print(f"  MD:   {REPORT_MD}\n")


if __name__ == "__main__":
    main()
