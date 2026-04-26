from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager

WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORKSPACE / "artifacts" / "datasets"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _entropy(class_dist: dict[str, int]) -> float:
    total = float(sum(class_dist.values()))
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in class_dist.values() if v > 0], dtype=np.float64)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _kl_to_global(class_dist: dict[str, int], global_dist: dict[str, int]) -> float:
    keys = sorted(set(class_dist.keys()).union(global_dist.keys()))
    p = np.array([float(class_dist.get(k, 0)) for k in keys], dtype=np.float64)
    q = np.array([float(global_dist.get(k, 0)) for k in keys], dtype=np.float64)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def _dataset_geometry(dataset_name: str) -> tuple[int, int, int, int]:
    normalized = dataset_name.lower().strip()
    if normalized == "femnist":
        return 1, 28, 28, 62
    if normalized == "cifar10":
        return 3, 32, 32, 10
    if normalized == "cifar100":
        return 3, 32, 32, 100
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_manifest(dataset_name: str, alpha: float, seed: int, num_clients: int) -> dict[str, Any]:
    cfg = ExperimentConfig(
        experiment_name=f"dataset_manifest_{dataset_name}_a{alpha}_s{seed}",
        dataset_name=dataset_name,
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
    )
    in_channels, height, width, num_classes = _dataset_geometry(dataset_name)
    cfg.model.client_backbones = ["small_cnn"] * num_clients
    cfg.model.num_classes = num_classes
    cfg.model.shared_dim = 64
    cfg.training.max_samples_per_client = 2000

    manager = ClientDataManager(WORKSPACE, cfg)
    bundles = manager.build_client_bundles()

    global_counter: Counter[int] = Counter()
    for b in bundles:
        global_counter.update(b.class_histogram)
    global_dist = {str(k): int(v) for k, v in sorted(global_counter.items())}

    clients: dict[str, Any] = {}
    min_samples = 10**9
    for b in bundles:
        class_dist = {str(k): int(v) for k, v in sorted(b.class_histogram.items())}
        min_samples = min(min_samples, int(b.num_samples))
        clients[str(b.client_id)] = {
            "num_samples": int(b.num_samples),
            "source_indices": [int(i) for i in b.source_indices],
            "class_distribution": class_dist,
            "entropy": _entropy(class_dist),
            "kl_to_global": _kl_to_global(class_dist, global_dist),
        }

    if min_samples <= 0:
        raise RuntimeError("dataset split invariant violation: empty client detected")

    dataset_files = {
        "femnist": [WORKSPACE / "dataset" / "femnist" / "train-00000-of-00001.parquet"],
        "cifar10": [
            WORKSPACE / "dataset" / "cifar-10-batches-py" / "batches.meta",
            *[WORKSPACE / "dataset" / "cifar-10-batches-py" / f"data_batch_{i}" for i in range(1, 6)],
            WORKSPACE / "dataset" / "cifar-10-batches-py" / "test_batch",
        ],
        "cifar100": [
            WORKSPACE / "dataset" / "cifar-100-python" / "meta",
            WORKSPACE / "dataset" / "cifar-100-python" / "train",
            WORKSPACE / "dataset" / "cifar-100-python" / "test",
        ],
    }.get(dataset_name.lower().strip(), [])
    dataset_hashes = {
        str(p.relative_to(WORKSPACE)): {
            "bytes": int(p.stat().st_size) if p.exists() else 0,
            "sha256": _sha256(p) if p.exists() else None,
            "exists": bool(p.exists()),
        }
        for p in dataset_files
    }

    return {
        "dataset": dataset_name,
        "alpha": alpha,
        "seed": seed,
        "num_clients": num_clients,
        "dataset_hashes": dataset_hashes,
        "partition_fingerprint": manager.partition_fingerprint(bundles),
        "global_class_distribution": global_dist,
        "clients": clients,
        "constraints": {
            "min_samples_per_client": min_samples,
            "no_empty_clients": min_samples > 0,
        },
    }


def main(datasets: list[str] | None = None) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    alphas = [1.0, 0.5, 0.1]
    datasets = datasets or ["cifar10", "cifar100"]

    manifests = [
        build_manifest(dataset_name=dataset_name, alpha=a, seed=0, num_clients=10)
        for dataset_name in datasets
        for a in alphas
    ]
    payload = {
        "datasets": datasets,
        "alphas": alphas,
        "manifests": manifests,
    }

    out = ARTIFACT_ROOT / "manifest.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for manifest in manifests:
        dataset_name = manifest["dataset"]
        alpha = manifest["alpha"]
        alpha_dir = ARTIFACT_ROOT / dataset_name / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        entropy_rows = {
            client_id: float(client_payload["entropy"])
            for client_id, client_payload in manifest["clients"].items()
        }
        kl_rows = {
            client_id: float(client_payload["kl_to_global"])
            for client_id, client_payload in manifest["clients"].items()
        }

        (alpha_dir / "entropy.json").write_text(json.dumps(entropy_rows, indent=2), encoding="utf-8")
        (alpha_dir / "kl_divergence.json").write_text(json.dumps(kl_rows, indent=2), encoding="utf-8")

        plt.figure(figsize=(12, 6))
        clients = sorted(manifest["clients"].keys(), key=lambda x: int(x))
        for client_id in clients:
            class_dist = manifest["clients"][client_id]["class_distribution"]
            xs = [int(k) for k in class_dist.keys()]
            ys = [int(v) for v in class_dist.values()]
            plt.plot(xs, ys, alpha=0.35)
        plt.title(f"Class Distribution Per Client (alpha={alpha})")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(alpha_dir / "histograms.png", dpi=150)
        plt.close()

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
