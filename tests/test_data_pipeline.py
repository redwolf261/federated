from pathlib import Path

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.data.partition_strategies import PartitionStrategies


def test_build_client_bundles_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    manager = ClientDataManager(workspace_root=workspace_root, config=config)
    bundles = manager.build_client_bundles()
    assert len(bundles) > 0
    assert all(bundle.num_samples > 0 for bundle in bundles)


def test_iid_partition_is_nearly_even() -> None:
    partition = PartitionStrategies.iid_even(num_samples=103, num_clients=4, seed=7)
    sizes = sorted(len(indices) for indices in partition.client_indices.values())
    assert sum(sizes) == 103
    assert sizes[-1] - sizes[0] <= 1


def test_femnist_iid_bundles_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 2
    config.partition_mode = "iid"
    config.training.max_samples_per_client = 32
    manager = ClientDataManager(workspace_root=workspace_root, config=config)
    bundles = manager.build_client_bundles()
    assert len(bundles) == 2
