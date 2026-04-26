"""Unit tests for Client training and prototype extraction."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from flex_persona.models.adapter_network import AdapterNetwork
from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.models.client_model import ClientModel
from flex_persona.federated.client import Client


def _create_synthetic_client(num_samples: int = 20, num_classes: int = 5) -> Client:
    backbone = SmallCNNBackbone(in_channels=1, input_height=28, input_width=28)
    adapter = AdapterNetwork(input_dim=backbone.output_dim, shared_dim=8)
    model = ClientModel(backbone=backbone, adapter=adapter, num_classes=num_classes)
    
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4)
    
    return Client(
        client_id=0,
        model=model,
        train_loader=loader,
        eval_loader=loader,
        num_classes=num_classes,
        device="cpu",
    )


def test_client_local_training() -> None:
    client = _create_synthetic_client()
    metrics = client.train_local(
        local_epochs=1,
        learning_rate=0.01,
        weight_decay=1e-5,
    )
    assert "local_loss" in metrics
    assert metrics["local_loss"] > 0


def test_client_extract_shared_representations() -> None:
    client = _create_synthetic_client(num_samples=8)
    features, labels = client.extract_shared_representations()
    assert features.shape[0] == 8
    assert labels.shape[0] == 8
    assert features.shape[1] == 8  # shared_dim


def test_client_compute_prototype_distribution() -> None:
    client = _create_synthetic_client(num_samples=20, num_classes=3)
    dist, proto_dict, class_counts = client.compute_prototype_distribution()
    assert dist.num_support > 0
    assert len(proto_dict) > 0
    assert sum(class_counts.values()) == 20


def test_client_evaluate_accuracy() -> None:
    client = _create_synthetic_client(num_samples=16, num_classes=5)
    # Before training, accuracy should be in valid range [0, 1]
    acc = client.evaluate_accuracy()
    assert 0.0 <= acc <= 1.0


def test_client_upload_message_structure() -> None:
    client = _create_synthetic_client(num_samples=12, num_classes=3)
    message = client.build_upload_message(round_idx=1)
    assert message.client_id == 0
    assert message.round_idx == 1
    assert message.prototype_distribution is not None
    assert len(message.class_counts) > 0
