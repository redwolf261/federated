import json
import io
import contextlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (
    run_fedavg_dirichlet,
    run_moon,
    run_scaffold,
    train_centralized,
    FEMNIST_NUM_CLASSES,
)


def main() -> None:
    seeds = [42, 123, 456]
    # Fast parity config (same for all three methods)
    cfg = {
        "alpha": 0.5,
        "num_clients": 10,
        "rounds": 6,
        "local_epochs": 1,
        "lr": 0.003,
        "batch_size": 64,
        "max_samples": 4000,
    }

    # 1-client sanity config (matches the known stable setup)
    sanity = {
        "alpha": 0.5,
        "num_clients": 1,
        "rounds": 20,
        "local_epochs": 3,
        "lr": 0.003,
        "batch_size": 64,
        "max_samples": 20000,
        "centralized_epochs": 10,
    }

    out = {
        "config": cfg,
        "seeds": seeds,
        "moon_mu0_vs_fedavg": {},
        "scaffold_zero_vs_fedavg": {},
        "one_client_check": {},
    }

    # Keep logs quiet; only emit final JSON.
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        fedavg_acc = {}
        moon_acc = {}
        scaffold_acc = {}

        for seed in seeds:
            fr = run_fedavg_dirichlet(
                alpha=cfg["alpha"],
                num_clients=cfg["num_clients"],
                rounds=cfg["rounds"],
                local_epochs=cfg["local_epochs"],
                seed=seed,
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                max_samples=cfg["max_samples"],
            )
            mr = run_moon(
                dataset_name="femnist",
                num_classes=FEMNIST_NUM_CLASSES,
                num_clients=cfg["num_clients"],
                rounds=cfg["rounds"],
                local_epochs=cfg["local_epochs"],
                seed=seed,
                alpha=cfg["alpha"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                max_samples=cfg["max_samples"],
                mu=0.0,
                return_trace=False,
            )
            sr = run_scaffold(
                dataset_name="femnist",
                num_classes=FEMNIST_NUM_CLASSES,
                num_clients=cfg["num_clients"],
                rounds=cfg["rounds"],
                local_epochs=cfg["local_epochs"],
                seed=seed,
                alpha=cfg["alpha"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                max_samples=cfg["max_samples"],
                return_trace=False,
                zero_control=True,
            )
            fedavg_acc[str(seed)] = float(fr["mean_accuracy"])
            moon_acc[str(seed)] = float(mr["mean_accuracy"])
            scaffold_acc[str(seed)] = float(sr["mean_accuracy"])

        moon_d = {s: abs(moon_acc[s] - fedavg_acc[s]) for s in moon_acc}
        scaf_d = {s: abs(scaffold_acc[s] - fedavg_acc[s]) for s in scaffold_acc}

        out["moon_mu0_vs_fedavg"] = {
            "per_seed_fedavg": fedavg_acc,
            "per_seed_moon_mu0": moon_acc,
            "per_seed_delta": moon_d,
            "mean_delta": float(sum(moon_d.values()) / len(moon_d)),
        }

        out["scaffold_zero_vs_fedavg"] = {
            "per_seed_fedavg": fedavg_acc,
            "per_seed_scaffold_zero": scaffold_acc,
            "per_seed_delta": scaf_d,
            "mean_delta": float(sum(scaf_d.values()) / len(scaf_d)),
        }

        # 1-client check on one fixed seed (same as earlier sanity gate)
        s = 42
        c_acc = float(
            train_centralized(
                dataset_name="femnist",
                seed=s,
                epochs=sanity["centralized_epochs"],
                lr=sanity["lr"],
                batch_size=sanity["batch_size"],
                max_samples=sanity["max_samples"],
            )
        )
        f1 = run_fedavg_dirichlet(
            alpha=sanity["alpha"],
            num_clients=sanity["num_clients"],
            rounds=sanity["rounds"],
            local_epochs=sanity["local_epochs"],
            seed=s,
            lr=sanity["lr"],
            batch_size=sanity["batch_size"],
            max_samples=sanity["max_samples"],
        )
        f1_acc = float(f1["mean_accuracy"])

        out["one_client_check"] = {
            "seed": s,
            "centralized": c_acc,
            "fedavg_1client": f1_acc,
            "delta": abs(c_acc - f1_acc),
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
