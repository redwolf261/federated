from pathlib import Path
from typing import cast

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator


def test_single_round_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 3
    config.training.rounds = 1
    config.training.local_epochs = 1
    config.training.cluster_aware_epochs = 1
    config.training.max_samples_per_client = 64
    config.training.batch_size = 32
    config.clustering.num_clusters = 2

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    state = simulator.run_round(1)

    assert state.distance_matrix is not None
    assert state.similarity_matrix is not None
    assert state.cluster_assignments is not None
    assert state.metadata.get("aggregation_mode") == "prototype"


def test_single_round_fedavg_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 3
    config.partition_mode = "iid"
    config.training.rounds = 1
    config.training.local_epochs = 1
    config.training.max_samples_per_client = 64
    config.training.batch_size = 32
    config.training.aggregation_mode = "fedavg"
    config.model.client_backbones = ["small_cnn"] * config.num_clients

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    state = simulator.run_round(1)

    assert state.distance_matrix is None
    assert state.similarity_matrix is None
    assert state.cluster_assignments is None
    assert state.metadata.get("aggregation_mode") == "fedavg"
    fedavg_meta = state.metadata.get("fedavg", {})
    assert "global_parameter_norm" in fedavg_meta
    assert "client_parameter_norms" in fedavg_meta


def test_early_stopping_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 3
    config.training.rounds = 5
    config.training.local_epochs = 1
    config.training.cluster_aware_epochs = 1
    config.training.max_samples_per_client = 64
    config.training.batch_size = 32
    config.clustering.num_clusters = 2
    config.training.early_stopping_enabled = True
    config.training.early_stopping_patience = 1
    config.training.early_stopping_min_delta = 2.0

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    history = simulator.run_experiment()
    report = simulator.build_report(history)

    run_summary = report.get("run_summary", {})
    assert isinstance(run_summary, dict)
    assert len(history) < config.training.rounds
    assert "stopped_early" in run_summary
    assert run_summary["stopped_early"] is True
    assert "rounds_executed" in run_summary
    rounds_executed = cast(int, run_summary["rounds_executed"])
    assert isinstance(rounds_executed, int)
    assert rounds_executed == len(history)


def test_unlimited_rounds_terminates_via_early_stopping() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 3
    config.training.rounds = -1
    config.training.local_epochs = 1
    config.training.cluster_aware_epochs = 1
    config.training.max_samples_per_client = 64
    config.training.batch_size = 32
    config.clustering.num_clusters = 2
    config.training.early_stopping_enabled = True
    config.training.early_stopping_patience = 1
    config.training.early_stopping_min_delta = 2.0
    config.training.max_unlimited_rounds = 10

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    history = simulator.run_experiment()
    report = simulator.build_report(history)

    run_summary = report.get("run_summary", {})
    assert isinstance(run_summary, dict)
    assert len(history) < config.training.max_unlimited_rounds
    assert run_summary.get("stopped_early") is True
    assert run_summary.get("termination_reason") == "early_stopping"
