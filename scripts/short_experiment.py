import contextlib
import io
import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (
    run_moon,
    run_scaffold,
    ExperimentConfig,
    FederatedSimulator,
    FEMNIST_NUM_CLASSES,
    PROJECT_ROOT as VALIDATION_ROOT,
    OUTPUT_DIR,
)

SEEDS = [42, 123, 456]
ALPHAS = [0.1, 1.0]
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 1
LR = 0.003
BATCH_SIZE = 64
MAX_SAMPLES = 4000


def run_flex(alpha: float, seed: int):
    cfg = ExperimentConfig(
        experiment_name=f"prototype_short_a{alpha}_s{seed}",
        dataset_name="femnist",
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "prototype"
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.use_clustering = True
    cfg.training.use_guidance = True

    sim = FederatedSimulator(workspace_root=str(VALIDATION_ROOT), config=cfg)
    history = sim.run_experiment()
    client_accs = {str(c.client_id): float(c.evaluate_accuracy()) for c in sim.clients}
    round_curve = [float(state.metadata["evaluation"]["mean_client_accuracy"]) for state in history]
    worst_curve = [float(state.metadata["evaluation"]["worst_client_accuracy"]) for state in history]
    return {
        "mean_accuracy": float(sum(client_accs.values()) / len(client_accs)),
        "worst_accuracy": float(min(client_accs.values())),
        "client_accuracies": client_accs,
        "round_curve": round_curve,
        "worst_curve": worst_curve,
    }


def run_fedavg(alpha: float, seed: int):
    cfg = ExperimentConfig(
        experiment_name=f"fedavg_short_a{alpha}_s{seed}",
        dataset_name="femnist",
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS

    sim = FederatedSimulator(workspace_root=str(VALIDATION_ROOT), config=cfg)
    history = sim.run_experiment()
    client_accs = {str(c.client_id): float(c.evaluate_accuracy()) for c in sim.clients}
    round_curve = [float(state.metadata["evaluation"]["mean_client_accuracy"]) for state in history]
    worst_curve = [float(state.metadata["evaluation"]["worst_client_accuracy"]) for state in history]
    return {
        "mean_accuracy": float(sum(client_accs.values()) / len(client_accs)),
        "worst_accuracy": float(min(client_accs.values())),
        "client_accuracies": client_accs,
        "round_curve": round_curve,
        "worst_curve": worst_curve,
    }


def run_method(method_name: str, alpha: float, seed: int):
    if method_name == "fedavg":
        return run_fedavg(alpha, seed)
    if method_name == "moon":
        result = run_moon(
            dataset_name="femnist",
            num_classes=FEMNIST_NUM_CLASSES,
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            seed=seed,
            alpha=alpha,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
            mu=1.0,
            return_trace=True,
        )
        return {
            "mean_accuracy": float(result["mean_accuracy"]),
            "worst_accuracy": float(result["worst_accuracy"]),
            "client_accuracies": result["client_accuracies"],
            "round_curve": [float(r["global_metrics"]["mean_client_accuracy"]) for r in result["per_round"]],
            "worst_curve": [float(r["global_metrics"]["worst_client_accuracy"]) for r in result["per_round"]],
        }
    if method_name == "scaffold":
        result = run_scaffold(
            dataset_name="femnist",
            num_classes=FEMNIST_NUM_CLASSES,
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            seed=seed,
            alpha=alpha,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
            return_trace=True,
        )
        return {
            "mean_accuracy": float(result["mean_accuracy"]),
            "worst_accuracy": float(result["worst_accuracy"]),
            "client_accuracies": result["client_accuracies"],
            "round_curve": [float(r["global_metrics"]["mean_client_accuracy"]) for r in result["per_round"]],
            "worst_curve": [float(r["global_metrics"]["worst_client_accuracy"]) for r in result["per_round"]],
        }
    if method_name == "flex":
        return run_flex(alpha, seed)
    raise ValueError(method_name)


def summarise(vals):
    mean = sum(vals) / len(vals)
    return float(mean)


def main():
    parser = argparse.ArgumentParser(description="Short federated experiment")
    parser.add_argument("--alpha", type=float, default=None, help="Run only one alpha value")
    args = parser.parse_args()

    selected_alphas = [args.alpha] if args.alpha is not None else ALPHAS

    quiet = io.StringIO()
    results = {}
    with contextlib.redirect_stdout(quiet):
        for alpha in selected_alphas:
            results[f"alpha_{alpha}"] = {}
            for method in ["fedavg", "moon", "scaffold", "flex"]:
                per_seed = {}
                mean_values = []
                worst_values = []
                round_curves = []
                worst_curves = []
                for seed in SEEDS:
                    r = run_method(method, alpha, seed)
                    per_seed[str(seed)] = {
                        "mean_accuracy": float(r["mean_accuracy"]),
                        "worst_accuracy": float(r["worst_accuracy"]),
                        "client_accuracies": r["client_accuracies"],
                    }
                    mean_values.append(float(r["mean_accuracy"]))
                    worst_values.append(float(r["worst_accuracy"]))
                    round_curves.append(r["round_curve"])
                    if r.get("worst_curve") is not None:
                        worst_curves.append(r["worst_curve"])
                results[f"alpha_{alpha}"][method] = {
                    "mean_accuracy": summarise(mean_values),
                    "worst_accuracy": summarise(worst_values),
                    "per_seed": per_seed,
                    "round_curves": round_curves,
                    "worst_curves": worst_curves,
                }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
