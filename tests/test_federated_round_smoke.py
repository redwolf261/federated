from pathlib import Path

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator


def test_single_round_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    config.training.rounds = 1
    config.training.local_epochs = 1
    config.training.cluster_aware_epochs = 1

    simulator = FederatedSimulator(workspace_root=workspace_root, config=config)
    state = simulator.run_round(1)

    assert state.distance_matrix is not None
    assert state.similarity_matrix is not None
    assert state.cluster_assignments is not None
