"""Factory for building heterogeneous client models."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.model_config import ModelConfig
from .adapter_network import AdapterNetwork
from .backbones import MLPBackbone, ResNet8Backbone, SmallCNNBackbone
from .client_model import ClientModel
from .initialization import initialize_module_weights


@dataclass(frozen=True)
class InputSpec:
    in_channels: int
    height: int
    width: int


class ModelFactory:
    """Creates client-specific models with heterogeneous backbones."""

    @staticmethod
    def infer_input_spec(dataset_name: str) -> InputSpec:
        normalized = dataset_name.lower().strip()
        if normalized == "femnist":
            return InputSpec(in_channels=1, height=28, width=28)
        if normalized == "cifar100":
            return InputSpec(in_channels=3, height=32, width=32)
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @classmethod
    def build_client_model(
        cls,
        client_id: int,
        model_config: ModelConfig,
        dataset_name: str,
    ) -> ClientModel:
        spec = cls.infer_input_spec(dataset_name)
        backbone_name = cls._select_backbone_name(client_id, model_config)
        backbone = cls._build_backbone(backbone_name, spec)
        adapter = AdapterNetwork(input_dim=backbone.output_dim, shared_dim=model_config.shared_dim)

        model = ClientModel(
            backbone=backbone,
            adapter=adapter,
            num_classes=model_config.num_classes,
        )
        initialize_module_weights(model)
        return model

    @staticmethod
    def _select_backbone_name(client_id: int, model_config: ModelConfig) -> str:
        names = model_config.client_backbones
        if not names:
            raise ValueError("model_config.client_backbones cannot be empty")
        return names[client_id % len(names)].lower().strip()

    @staticmethod
    def _build_backbone(backbone_name: str, spec: InputSpec):
        if backbone_name == "small_cnn":
            return SmallCNNBackbone(in_channels=spec.in_channels)
        if backbone_name == "resnet8":
            return ResNet8Backbone(in_channels=spec.in_channels)
        if backbone_name == "mlp":
            return MLPBackbone(input_shape=(spec.in_channels, spec.height, spec.width))
        raise ValueError(f"Unsupported backbone name: {backbone_name}")
