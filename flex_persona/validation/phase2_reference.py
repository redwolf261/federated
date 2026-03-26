"""Stage 0/1 reference validation helpers for FEMNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from ..config.experiment_config import ExperimentConfig
from ..federated.simulator import FederatedSimulator
from ..utils.seed import set_global_seed


REFERENCE_DATASET = "femnist"
REFERENCE_BACKBONE = "small_cnn"
REFERENCE_LR = 0.003
REFERENCE_BATCH_SIZE = 64
REFERENCE_CENTRALIZED_EPOCHS = 10
REFERENCE_LOCAL_EPOCHS = 5


@dataclass(frozen=True)
class CentralizedReferenceResult:
    seed: int
    accuracy: float


@dataclass(frozen=True)
class FedAvgValidationResult:
    seed: int
    rounds: int
    final_accuracy: float
    round_accuracies: list[float]
    round_parameter_norms: list[dict[str, object]]
    communication: dict[str, int]


def _base_reference_config(seed: int) -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_name=f"phase2_reference_seed_{seed}",
        dataset_name=REFERENCE_DATASET,
        random_seed=seed,
    )
    config.model.num_classes = 62
    config.model.client_backbones = [REFERENCE_BACKBONE]
    config.training.batch_size = REFERENCE_BATCH_SIZE
    config.training.learning_rate = REFERENCE_LR
    config.training.weight_decay = 1e-4
    return config


def train_centralized(seed: int, workspace_root: str | Path = ".") -> float:
    """Train the locked centralized FEMNIST reference and return accuracy."""
    set_global_seed(seed)
    config = _base_reference_config(seed)
    config.num_clients = 1
    config.partition_mode = "iid"
    config.training.aggregation_mode = "fedavg"
    config.training.rounds = 1
    config.training.local_epochs = REFERENCE_CENTRALIZED_EPOCHS
    config.model.client_backbones = [REFERENCE_BACKBONE]
    config.validate()

    simulator = FederatedSimulator(workspace_root=Path(workspace_root), config=config)
    history = simulator.run_experiment()
    if not history:
        raise RuntimeError("Centralized reference run did not produce any rounds")
    final_eval = history[-1].metadata.get("evaluation", {})
    return float(final_eval.get("mean_client_accuracy", 0.0))


def run_centralized_reference(
    workspace_root: str | Path,
    seeds: list[int] | tuple[int, ...] = (11, 17, 23),
) -> dict[str, object]:
    results = [
        CentralizedReferenceResult(
            seed=seed,
            accuracy=train_centralized(seed=seed, workspace_root=workspace_root),
        )
        for seed in seeds
    ]
    accuracies = [result.accuracy for result in results]
    return {
        "dataset": REFERENCE_DATASET,
        "optimizer": "Adam (persistent)",
        "learning_rate": REFERENCE_LR,
        "batch_size": REFERENCE_BATCH_SIZE,
        "epochs": REFERENCE_CENTRALIZED_EPOCHS,
        "backbone": REFERENCE_BACKBONE,
        "per_seed": [{"seed": result.seed, "accuracy": result.accuracy} for result in results],
        "mean_accuracy": float(mean(accuracies)),
        "std_accuracy": float(pstdev(accuracies)) if len(accuracies) > 1 else 0.0,
    }


def _matching_rounds(centralized_epochs: int, local_epochs: int) -> int:
    if centralized_epochs % local_epochs != 0:
        raise ValueError(
            "Centralized epochs must be divisible by local_epochs to match the training budget exactly"
        )
    return centralized_epochs // local_epochs


def run_iid_fedavg_validation(
    workspace_root: str | Path,
    seed: int,
    num_clients: int = 2,
    local_epochs: int = REFERENCE_LOCAL_EPOCHS,
    centralized_epochs: int = REFERENCE_CENTRALIZED_EPOCHS,
) -> FedAvgValidationResult:
    """Run Stage 1 IID FedAvg with the centralized budget matched exactly."""
    rounds = _matching_rounds(centralized_epochs=centralized_epochs, local_epochs=local_epochs)
    set_global_seed(seed)

    config = _base_reference_config(seed)
    config.num_clients = num_clients
    config.partition_mode = "iid"
    config.training.aggregation_mode = "fedavg"
    config.training.rounds = rounds
    config.training.local_epochs = local_epochs
    config.model.client_backbones = [REFERENCE_BACKBONE] * num_clients
    config.validate()

    simulator = FederatedSimulator(workspace_root=Path(workspace_root), config=config)
    history = simulator.run_experiment()
    report = simulator.build_report(history)

    round_accuracies: list[float] = []
    round_parameter_norms: list[dict[str, object]] = []
    for state in history:
        evaluation = state.metadata.get("evaluation", {})
        fedavg = state.metadata.get("fedavg", {})
        round_accuracies.append(float(evaluation.get("mean_client_accuracy", 0.0)))
        round_parameter_norms.append(
            {
                "round": int(state.round_idx),
                "global_parameter_norm": float(fedavg.get("global_parameter_norm", 0.0)),
                "client_parameter_norms": dict(fedavg.get("client_parameter_norms", {})),
            }
        )

    final_metrics = report.get("final_metrics", {})
    communication = report.get("communication", {})
    return FedAvgValidationResult(
        seed=seed,
        rounds=rounds,
        final_accuracy=float(final_metrics.get("mean_client_accuracy", 0.0)),
        round_accuracies=round_accuracies,
        round_parameter_norms=round_parameter_norms,
        communication={str(k): int(v) for k, v in communication.items()},
    )
