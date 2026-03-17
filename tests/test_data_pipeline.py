from pathlib import Path

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager


def test_build_client_bundles_smoke() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(dataset_name="femnist")
    manager = ClientDataManager(workspace_root=workspace_root, config=config)
    bundles = manager.build_client_bundles()
    assert len(bundles) > 0
    assert all(bundle.num_samples > 0 for bundle in bundles)
