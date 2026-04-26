import json
import io
import contextlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (
    train_centralized,
    run_moon,
    run_scaffold,
    FEMNIST_NUM_CLASSES,
    ExperimentConfig,
    FederatedSimulator,
    OUTPUT_DIR,
)

seed = 42
num_clients = 1
rounds = 20
local_epochs = 3
alpha = 0.5
lr = 0.003
batch_size = 64
max_samples = 20000

buf = io.StringIO()

print("CENTRALIZED")
with contextlib.redirect_stdout(buf):
    central_acc = train_centralized("femnist", seed=seed)

print("FEDAVG")
with contextlib.redirect_stdout(buf):
    cfg = ExperimentConfig(
        experiment_name=f"fedavg_dir_a{alpha}_s{seed}",
        dataset_name="femnist",
        num_clients=num_clients,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=alpha,
        output_dir=str(OUTPUT_DIR),
    )
    cfg.model.num_classes = FEMNIST_NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "fedavg"
    cfg.training.rounds = rounds
    cfg.training.local_epochs = local_epochs
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.max_samples_per_client = max_samples // num_clients
    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()
    fedavg_acc = float(sim.clients[0].evaluate_accuracy())
    fedavg_rounds = [float(s.metadata["evaluation"]["mean_client_accuracy"]) for s in history]
    fedavg_client_accs = [float(sim.clients[0].evaluate_accuracy())]

print("MOON")
with contextlib.redirect_stdout(buf):
    moon = run_moon(
        "femnist", FEMNIST_NUM_CLASSES, num_clients, rounds, local_epochs, seed,
        alpha=alpha, lr=lr, batch_size=batch_size, max_samples=max_samples,
        mu=0.0, temperature=0.5, return_trace=True,
    )
    moon_acc = float(moon["mean_accuracy"])
    moon_rounds = [float(r["global_metrics"]["mean_client_accuracy"]) for r in moon["per_round"]]
    moon_client_accs = [float(v) for v in moon["client_accuracies"].values()]

print("SCAFFOLD")
with contextlib.redirect_stdout(buf):
    scaffold = run_scaffold(
        "femnist", FEMNIST_NUM_CLASSES, num_clients, rounds, local_epochs,
        seed, alpha=alpha, lr=lr, batch_size=batch_size, max_samples=max_samples,
        return_trace=True,
    )
    scaffold_acc = float(scaffold["mean_accuracy"])
    scaffold_rounds = [float(r["global_metrics"]["mean_client_accuracy"]) for r in scaffold["per_round"]]
    scaffold_client_accs = [float(v) for v in scaffold["client_accuracies"].values()]

result = {
    "Centralized_accuracy": float(central_acc),
    "FedAvg_1client_accuracy": fedavg_acc,
    "MOON_mu0_accuracy": moon_acc,
    "SCAFFOLD_zero_accuracy": scaffold_acc,
    "delta_fedavg": abs(float(central_acc) - fedavg_acc),
    "delta_moon": abs(moon_acc - fedavg_acc),
    "delta_scaf": abs(scaffold_acc - fedavg_acc),
    "FedAvg": fedavg_rounds,
    "MOON": moon_rounds,
    "SCAFFOLD": scaffold_rounds,
    "FedAvg_client_acc": fedavg_client_accs,
    "MOON_client_acc": moon_client_accs,
    "SCAFFOLD_client_acc": scaffold_client_accs,
}
print(json.dumps(result, indent=2))
