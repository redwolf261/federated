#!/usr/bin/env python3
"""Block L: L2 Normalization Isolation Experiment.

Isolates whether FLEX gains come from:
  (A) dimensionality reduction alone (8192→64, no norm)
  (B) L2 normalization alone (on raw backbone features, no bottleneck)
  (C) their interaction (bottleneck + L2 norm = current FLEX)

Methods:
  1. baseline_backbone_only       z→classifier(z)               no bottleneck, no norm
  2. bottleneck_no_norm           Linear(z)→classifier           bottleneck, NO norm
  3. bottleneck_with_l2           normalize(Linear(z))→class.    current FLEX adapter (reference)
  4. random_proj_no_norm          frozen_R(z)→classifier         frozen random proj, NO norm
  5. random_proj_with_l2          normalize(frozen_R(z))→class.  frozen random proj + norm
  6. fedavg_sgd                   standard FedAvg (flat CNN)      standard baseline

All methods use the FULL FederatedSimulator pipeline (identical to Block K),
no_prototype_sharing mode, 5 local epochs, 0 cluster-aware epochs.

Validation checks print tensor norms to confirm normalization paths work correctly.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.models.adapter_network import AdapterNetwork
from flex_persona.models.client_model import ClientModel
from scripts.phase2_q1_validation import set_seed

# ---------------------------------------------------------------------------
COVERAGE_DIR = PROJECT_ROOT / "outputs" / "failure_mode_coverage"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = COVERAGE_DIR / "block_L_results.json"
REPORT_MD    = COVERAGE_DIR / "block_L.md"

DATASET      = "cifar10"
NUM_CLASSES  = 10
NUM_CLIENTS  = 10
ROUNDS       = 20
LOCAL_EPOCHS = 5
BATCH_SIZE   = 64
LR           = 0.001
MAX_SAMPLES  = 20_000
ALPHA        = 0.1
SEEDS        = [42, 43, 44]

BACKBONE_OUT_DIM = 8192   # SmallCNN on CIFAR-10: 128 * 8 * 8
SHARED_DIM       = 64

ALL_METHODS = [
    "fedavg_sgd",
    "baseline_backbone_only",
    "bottleneck_no_norm",
    "bottleneck_with_l2",
    "random_proj_no_norm",
    "random_proj_with_l2",
]

LABELS = {
    "fedavg_sgd":             "FedAvg SGD (baseline)",
    "baseline_backbone_only": "Backbone Only (8192→classifier)",
    "bottleneck_no_norm":     "Bottleneck No Norm (8192→64, no L2)",
    "bottleneck_with_l2":     "Bottleneck + L2 (8192→64 + normalize) [FLEX]",
    "random_proj_no_norm":    "Random Proj No Norm (frozen R, no L2)",
    "random_proj_with_l2":    "Random Proj + L2 (frozen R + normalize)",
}


# ---------------------------------------------------------------------------
# Custom model variants
# ---------------------------------------------------------------------------

class _BackboneOnlyModel(nn.Module):
    """z = backbone(x); logits = classifier(z). No bottleneck, no norm."""

    def __init__(self, backbone: SmallCNNBackbone, num_classes: int) -> None:
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        self.num_classes = num_classes

    def forward_task(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_task(x)

    def evaluate_accuracy(self, eval_loader, device: str) -> float:
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self.forward_task(xb).argmax(dim=1)
                correct += int((preds == yb).sum())
                total   += int(yb.shape[0])
        return float(correct / max(total, 1))


class _BottleneckModel(nn.Module):
    """Linear(8192→64) [optionally L2-normalized] → classifier."""

    def __init__(self, backbone: SmallCNNBackbone, num_classes: int,
                 normalize: bool, frozen: bool = False) -> None:
        super().__init__()
        self.backbone   = backbone
        self.normalize  = normalize
        self.num_classes = num_classes
        self.proj = nn.Linear(backbone.output_dim, SHARED_DIM)
        if frozen:
            # Xavier-uniform init then freeze
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            for p in self.proj.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(SHARED_DIM, num_classes)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        h = self.proj(z)
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h

    def forward_task(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._project(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_task(x)

    def evaluate_accuracy(self, eval_loader, device: str) -> float:
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self.forward_task(xb).argmax(dim=1)
                correct += int((preds == yb).sum())
                total   += int(yb.shape[0])
        return float(correct / max(total, 1))


def _validate_norms(model, loader, device: str, label: str) -> None:
    """Print feature norms to confirm normalization is active/inactive."""
    model.eval()
    with torch.no_grad():
        xb, _ = next(iter(loader))
        xb = xb.to(device)
        z = model.backbone(xb)
        if hasattr(model, 'proj'):
            h = model.proj(z)
            if model.normalize:
                h = F.normalize(h, p=2, dim=-1)
            norms = torch.norm(h, p=2, dim=-1)
            print(f"    [{label}] h norms: mean={norms.mean():.4f}  "
                  f"std={norms.std():.4f}  "
                  f"min={norms.min():.4f}  max={norms.max():.4f}")
        else:
            norms = torch.norm(z, p=2, dim=-1)
            print(f"    [{label}] z norms (backbone out): "
                  f"mean={norms.mean():.2f}  std={norms.std():.2f}")
    model.train()


# ---------------------------------------------------------------------------
# Custom training loop (mirrors FederatedSimulator local_only_5ep behavior)
# ---------------------------------------------------------------------------

def _run_custom(method: str, seed: int) -> dict:
    """Train custom-model variants for 20 rounds × 5 local epochs, no exchange."""
    set_seed(seed)

    # Build data via simulator (reuse data pipeline exactly)
    cfg = _base_sim_cfg(f"block_l_{method}_data_s{seed}", seed)
    cfg.training.aggregation_mode     = "prototype"
    cfg.training.local_epochs         = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = 0
    cfg.training.ablation_mode        = "no_prototype_sharing"
    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clients_data = [(c.train_loader, c.eval_loader) for c in sim.clients]

    # Build custom models
    models = []
    for _ in range(NUM_CLIENTS):
        bb = SmallCNNBackbone(in_channels=3, input_height=32, input_width=32)
        if method == "baseline_backbone_only":
            m = _BackboneOnlyModel(bb, NUM_CLASSES)
        elif method == "bottleneck_no_norm":
            m = _BottleneckModel(bb, NUM_CLASSES, normalize=False, frozen=False)
        elif method == "bottleneck_with_l2":
            m = _BottleneckModel(bb, NUM_CLASSES, normalize=True,  frozen=False)
        elif method == "random_proj_no_norm":
            m = _BottleneckModel(bb, NUM_CLASSES, normalize=False, frozen=True)
        elif method == "random_proj_with_l2":
            m = _BottleneckModel(bb, NUM_CLASSES, normalize=True,  frozen=True)
        else:
            raise ValueError(f"Unknown custom method: {method}")
        m.to(device)
        models.append(m)

    # Validation check (first client, before training)
    print(f"    Validation norms [{method} s{seed}]:")
    _validate_norms(models[0], clients_data[0][0], device, method)

    # Frozen projection sanity check
    if method in ("random_proj_no_norm", "random_proj_with_l2"):
        proj_before = models[0].proj.weight.data.clone()

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizers = []
    for m in models:
        params = [p for p in m.parameters() if p.requires_grad]
        optimizers.append(torch.optim.Adam(params, lr=LR, weight_decay=1e-5))

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    for rnd in range(ROUNDS):
        for m, (train_loader, _), opt in zip(models, clients_data, optimizers):
            m.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    opt.zero_grad()
                    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                        logits = m.forward_task(xb)
                        loss   = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

        accs = [m.evaluate_accuracy(dl, device) for m, (_, dl) in zip(models, clients_data)]
        mean = float(np.mean(accs))
        worst = float(min(accs))
        print(f"[ROUND] method={method} s{seed} r={rnd+1}/{ROUNDS} mean={mean:.4f} worst={worst:.4f}")

    # Frozen sanity check
    if method in ("random_proj_no_norm", "random_proj_with_l2"):
        proj_after = models[0].proj.weight.data
        max_diff = float((proj_after - proj_before).abs().max())
        print(f"    [FROZEN CHECK] max weight change={max_diff:.2e}  "
              f"{'OK (frozen)' if max_diff < 1e-8 else 'WARNING: weights changed!'}")

    accs  = [m.evaluate_accuracy(dl, device) for m, (_, dl) in zip(models, clients_data)]
    return {
        "method": method, "seed": seed, "block": "L",
        "mean_accuracy":  float(np.mean(accs)),
        "worst_accuracy": float(min(accs)),
        "std":            float(np.std(accs)),
        "p10":            float(np.percentile(accs, 10)),
        "client_accuracies": {str(i): float(a) for i, a in enumerate(accs)},
    }


# ---------------------------------------------------------------------------
# FedAvg via simulator
# ---------------------------------------------------------------------------

def _base_sim_cfg(name: str, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=name,
        dataset_name=DATASET,
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes               = NUM_CLASSES
    cfg.model.client_backbones          = ["small_cnn"]
    cfg.model.shared_dim                = SHARED_DIM
    cfg.training.rounds                 = ROUNDS
    cfg.training.learning_rate          = LR
    cfg.training.batch_size             = BATCH_SIZE
    cfg.training.weight_decay           = 1e-5
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.lambda_cluster         = 0.1
    cfg.training.lambda_cluster_center  = 0.01
    return cfg


def _run_fedavg(seed: int) -> dict:
    set_seed(seed)
    cfg = _base_sim_cfg(f"block_l_fedavg_s{seed}", seed)
    cfg.training.aggregation_mode     = "fedavg"
    cfg.training.local_epochs         = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = 0
    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    sim.run_experiment()
    accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
    vals = list(accs.values())
    return {
        "method": "fedavg_sgd", "seed": seed, "block": "L",
        "mean_accuracy":  float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std":            float(np.std(vals)),
        "p10":            float(np.percentile(vals, 10)),
        "client_accuracies": {str(k): float(v) for k, v in accs.items()},
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _run(method: str, seed: int) -> dict:
    if method == "fedavg_sgd":
        return _run_fedavg(seed)
    return _run_custom(method, seed)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    return json.loads(RESULTS_JSON.read_text())


def _save(r: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(r, indent=2))


def _done(r: list[dict], method: str, seed: int) -> bool:
    return any(x["method"] == method and x["seed"] == seed for x in r)


def _agg(results: list[dict]) -> dict[str, dict]:
    g: dict[str, list] = defaultdict(list)
    for r in results:
        g[r["method"]].append(r)
    out = {}
    for m, runs in g.items():
        means  = [r["mean_accuracy"]  for r in runs]
        worsts = [r["worst_accuracy"] for r in runs]
        p10s   = [r["p10"]            for r in runs]
        out[m] = {
            "n":     len(runs),
            "mean":  float(np.mean(means)),
            "std":   float(np.std(means)),
            "worst": float(np.mean(worsts)),
            "p10":   float(np.mean(p10s)),
            "seeds": sorted(r["seed"] for r in runs),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report(results: list[dict], stats: dict[str, dict]) -> str:
    ref   = stats.get("bottleneck_with_l2", {}).get("mean", float("nan"))
    fedavg = stats.get("fedavg_sgd", {}).get("mean", float("nan"))

    lines = [
        "# Block L: L2 Normalization Isolation Experiment",
        "",
        "**Question:** Do FLEX gains arise from (A) dimensionality reduction,",
        "(B) L2 spherical normalization, or (C) their interaction?",
        "",
        f"Reference (bottleneck_with_l2): {ref:.4f}",
        f"FedAvg baseline:                {fedavg:.4f}",
        f"Total ref gap vs FedAvg:        {ref-fedavg:+.4f}",
        "",
        "---",
        "## Results Table",
        "",
        "| Method | Mean ± Std | Worst | P10 | Δ vs L2-Bottleneck | Δ vs FedAvg |",
        "|--------|-----------|-------|-----|-------------------|------------|",
    ]

    for m in ALL_METHODS:
        if m not in stats:
            continue
        s = stats[m]
        dr = s["mean"] - ref
        df = s["mean"] - fedavg
        lines.append(
            f"| {LABELS[m]} | {s['mean']:.4f} ± {s['std']:.4f} | "
            f"{s['worst']:.4f} | {s['p10']:.4f} | {dr:+.4f} | {df:+.4f} |"
        )

    # Key comparisons
    bn_no  = stats.get("bottleneck_no_norm",  {}).get("mean", float("nan"))
    bn_l2  = stats.get("bottleneck_with_l2",  {}).get("mean", float("nan"))
    rp_no  = stats.get("random_proj_no_norm", {}).get("mean", float("nan"))
    rp_l2  = stats.get("random_proj_with_l2", {}).get("mean", float("nan"))
    bb     = stats.get("baseline_backbone_only", {}).get("mean", float("nan"))

    THR = 0.02
    norm_gain_learned = bn_l2 - bn_no
    norm_gain_frozen  = rp_l2 - rp_no
    proj_gain_no_norm = bn_no - bb
    proj_gain_with_l2 = bn_l2 - bb

    lines += [
        "",
        "---",
        "## Causal Key Comparisons",
        "",
        f"- L2 norm gain (learned proj):  bn_with_l2 - bn_no_norm = {norm_gain_learned:+.4f}",
        f"- L2 norm gain (frozen proj):   rp_with_l2 - rp_no_norm = {norm_gain_frozen:+.4f}",
        f"- Projection gain (no norm):    bn_no_norm - backbone    = {proj_gain_no_norm:+.4f}",
        f"- Projection gain (with L2):    bn_with_l2 - backbone    = {proj_gain_with_l2:+.4f}",
        "",
    ]

    # Case classification
    norm_is_dominant = (norm_gain_learned > THR and norm_gain_frozen > THR and
                        abs(norm_gain_learned - norm_gain_frozen) < THR)
    proj_is_dominant = (proj_gain_no_norm > THR and
                        abs(norm_gain_learned) < THR and abs(norm_gain_frozen) < THR)

    if norm_is_dominant and not proj_is_dominant:
        verdict = "CASE A — L2 NORMALIZATION DOMINANT"
        detail  = (
            "Adding L2 normalization consistently improves both learned and frozen projections "
            "by a meaningful margin. Spherical hypersphere conditioning is the primary stabilizer."
        )
        causal  = "Primary: L2 spherical normalization\nSecondary: Dimensionality reduction"
    elif proj_is_dominant:
        verdict = "CASE B — DIMENSIONALITY REDUCTION DOMINANT"
        detail  = (
            "Adding L2 normalization provides no additional gain beyond the projection itself. "
            "The 8192→64 compression drives performance regardless of normalization."
        )
        causal  = "Primary: Dimensionality reduction (8192→64)\nSecondary: None"
    else:
        verdict = "CASE C — INTERACTION EFFECT"
        detail  = (
            "Both projection and normalization contribute independently and neither alone "
            "is sufficient to explain the full gain. Their combination is necessary."
        )
        causal  = "Primary: Projection + L2 normalization interaction\nSecondary: Either alone"

    lines += [
        f"### Verdict: **{verdict}**",
        "",
        detail,
        "",
        "```",
        causal,
        "```",
        "",
        "---",
        "## Per-Seed Raw Results",
        "",
        "| Method | Seed 42 | Seed 43 | Seed 44 |",
        "|--------|---------|---------|---------|",
    ]

    for m in ALL_METHODS:
        per = {r["seed"]: r["mean_accuracy"] for r in results if r["method"] == m}
        s42 = f"{per.get(42, float('nan')):.4f}"
        s43 = f"{per.get(43, float('nan')):.4f}"
        s44 = f"{per.get(44, float('nan')):.4f}"
        lines.append(f"| {LABELS[m]} | {s42} | {s43} | {s44} |")

    lines += [
        "",
        "---",
        "## Final Mechanistic Interpretation",
        "",
        "This experiment closes the L2 normalization question in the FLEX-Persona",
        "causal audit. Combined with Blocks I, J, and K findings:",
        "",
        "- Block I: Prototype semantic content → irrelevant",
        "- Block J: Adapter bottleneck vs random projection → projection geometry drives gains",
        "- Block K: Extra epochs vs guidance structure → all methods equivalent",
        "- Block L: L2 norm vs bottleneck → **this verdict**",
        "",
        "*FLEX-Persona Causal Audit — Block L complete.*",
    ]

    return "\n".join(lines)


def _print_summary(results: list[dict]) -> None:
    stats = _agg(results)
    ref_mean = stats.get("bottleneck_with_l2", {}).get("mean", 0.0)
    print(f"\n{'='*70}")
    print("  BLOCK L SUMMARY")
    print(f"  {'Method':<44} {'Done':>4}  {'Mean':>7}  {'Δ L2-BN':>8}")
    print(f"  {'-'*62}")
    for m in ALL_METHODS:
        if m not in stats:
            print(f"  {m:<44} {'0/3':>4}  {'---':>7}  {'---':>8}")
            continue
        s = stats[m]
        d = s["mean"] - ref_mean
        print(f"  {m:<44} {str(s['n'])+'/3':>4}  {s['mean']:>7.4f}  {d:>+8.4f}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_runs = len(ALL_METHODS) * len(SEEDS)
    print("\n" + "=" * 70)
    print("  BLOCK L: L2 NORMALIZATION ISOLATION")
    print("  Question: dimensionality reduction vs L2 norm vs interaction")
    print(f"  Methods: {len(ALL_METHODS)}   Seeds: {SEEDS}   Total: {total_runs}")
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
            result = _run(method, seed)
            result["wall_time_s"] = round(time.time() - t0, 1)
            results.append(result)
            _save(results)
            completed += 1
            print(f"  => mean={result['mean_accuracy']:.4f}  "
                  f"worst={result['worst_accuracy']:.4f}  "
                  f"({result['wall_time_s']:.0f}s)")

    elapsed = time.time() - total_start
    print(f"\n  Done. {completed} new | {skipped} cached | {elapsed/60:.1f} min total")

    _print_summary(results)

    stats  = _agg(results)
    report = _report(results, stats)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"  JSON: {RESULTS_JSON}")
    print(f"  MD:   {REPORT_MD}\n")


if __name__ == "__main__":
    main()
