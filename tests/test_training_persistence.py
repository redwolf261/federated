import torch
from torch.utils.data import DataLoader, TensorDataset

from flex_persona.config.model_config import ModelConfig
from flex_persona.models.model_factory import ModelFactory
from flex_persona.training.local_trainer import LocalTrainer


def test_local_trainer_reuses_optimizer_for_same_hyperparameters() -> None:
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=ModelConfig(num_classes=62, client_backbones=["small_cnn"]),
        dataset_name="femnist",
    )
    dataset = TensorDataset(
        torch.zeros((8, 1, 28, 28), dtype=torch.float32),
        torch.zeros((8,), dtype=torch.long),
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    trainer = LocalTrainer()
    trainer.train(
        model=model,
        train_loader=train_loader,
        device="cpu",
        local_epochs=1,
        learning_rate=0.003,
        weight_decay=1e-4,
    )
    optimizer_first = trainer._optimizer

    trainer.train(
        model=model,
        train_loader=train_loader,
        device="cpu",
        local_epochs=1,
        learning_rate=0.003,
        weight_decay=1e-4,
    )

    assert optimizer_first is not None
    assert trainer._optimizer is optimizer_first
