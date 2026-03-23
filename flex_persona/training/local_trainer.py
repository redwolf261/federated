"""Local supervised training routine for a client model."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..models.client_model import ClientModel
from .losses import LossComposer
from .optim_factory import OptimizerFactory


class LocalTrainer:
    """Runs local cross-entropy optimization for client models."""

    def __init__(self) -> None:
        self.loss_composer = LossComposer()

    def train(
        self,
        model: ClientModel,
        train_loader: DataLoader,
        device: str,
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        fedprox_mu: float = 0.0,
        reference_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        model.train()
        optimizer = OptimizerFactory.adam(model, learning_rate=learning_rate, weight_decay=weight_decay)

        total_loss = 0.0
        total_samples = 0

        for _ in range(local_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model.forward_task(x_batch)
                loss = self.loss_composer.local_task_loss(logits, y_batch)

                if fedprox_mu > 0.0 and reference_state is not None:
                    prox_term = torch.tensor(0.0, device=device)
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        ref_param = reference_state[name].to(device=device, dtype=param.dtype)
                        prox_term = prox_term + torch.sum((param - ref_param) ** 2)
                    loss = loss + (0.5 * fedprox_mu * prox_term)

                loss.backward()
                optimizer.step()

                batch_size = int(y_batch.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        return {"local_loss": avg_loss}
