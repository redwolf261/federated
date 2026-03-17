import torch

from flex_persona.config.model_config import ModelConfig
from flex_persona.models.model_factory import ModelFactory


def test_client_model_interfaces_smoke() -> None:
    cfg = ModelConfig(num_classes=10, shared_dim=16, client_backbones=["small_cnn"])
    model = ModelFactory.build_client_model(client_id=0, model_config=cfg, dataset_name="cifar100")

    x = torch.randn(2, 3, 32, 32)
    logits = model.forward_task(x)
    features = model.extract_features(x)
    shared = model.project_shared(features)

    assert logits.shape == (2, 10)
    assert shared.shape[1] == 16
