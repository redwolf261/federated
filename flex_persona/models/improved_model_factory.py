"""Improved Factory for building client models with architectural fixes.

Addresses the architectural issues identified in the technical critique:

1. **Uses ImprovedAdapterNetwork**: Reduced compression (6272->512) with non-linearity
2. **Uses ImprovedClientModel**: Proper alignment between adapter and classifier
3. **Configurable Architecture**: Supports both original and improved components
4. **Alignment Support**: Can build alignment-aware models

Key improvements:
- Default shared_dim increased to 512 (vs 64) for less aggressive compression
- Multi-layer adapters with ReLU/BatchNorm instead of single linear layer
- Proper adapter-classifier alignment in the forward path
- Support for alignment loss computation

This factory enables building research-grade models instead of the broken architecture
identified in the critique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config.model_config import ModelConfig
from .adapter_network import AdapterNetwork  # Original (for backward compatibility)
from .improved_adapter_network import ImprovedAdapterNetwork, AlignmentAwareAdapter
from .backbones import MLPBackbone, ResNet8Backbone, SmallCNNBackbone
from .client_model import ClientModel  # Original (for backward compatibility)
from .improved_client_model import ImprovedClientModel
from .initialization import initialize_module_weights


@dataclass(frozen=True)
class InputSpec:
    in_channels: int
    height: int
    width: int


AdapterType = Literal["original", "improved", "alignment_aware"]
ModelType = Literal["original", "improved"]


class ImprovedModelFactory:
    """Factory for building improved client models with architectural fixes.

    Provides both original components (for backward compatibility) and improved
    components (for research-grade performance) based on configuration.
    """

    # Improved default configuration addressing critique points
    DEFAULT_IMPROVED_CONFIG = {
        "shared_dim": 512,  # Much less aggressive compression (6272->512 vs 6272->64)
        "adapter_hidden_dims": [1536, 768],  # Progressive compression
        "adapter_dropout": 0.2,
        "use_residual": True,
        "alignment_dim": 256,
        "use_alignment_loss": True
    }

    @staticmethod
    def infer_input_spec(dataset_name: str) -> InputSpec:
        """Infer input specification from dataset name."""
        normalized = dataset_name.lower().strip()
        if normalized == "femnist":
            return InputSpec(in_channels=1, height=28, width=28)
        if normalized == "cifar100":
            return InputSpec(in_channels=3, height=32, width=32)
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @classmethod
    def build_improved_client_model(
        cls,
        client_id: int,
        model_config: ModelConfig,
        dataset_name: str,
        adapter_type: AdapterType = "improved",
        model_type: ModelType = "improved",
        custom_config: dict | None = None
    ) -> ClientModel | ImprovedClientModel:
        """Build a client model with improved architecture.

        Args:
            client_id: Client identifier for backbone selection
            model_config: Model configuration (may override with improved defaults)
            dataset_name: Dataset name for input specification
            adapter_type: Type of adapter to use ("original", "improved", "alignment_aware")
            model_type: Type of model to use ("original", "improved")
            custom_config: Custom configuration overrides

        Returns:
            Client model with improved architecture
        """
        spec = cls.infer_input_spec(dataset_name)
        backbone_name = cls._select_backbone_name(client_id, model_config)
        backbone = cls._build_backbone(backbone_name, spec)

        # Use improved defaults unless overridden
        config = cls.DEFAULT_IMPROVED_CONFIG.copy()
        if custom_config:
            config.update(custom_config)

        # Override model_config shared_dim with improved default if not explicitly set
        improved_shared_dim = config.get("shared_dim", model_config.shared_dim)

        # Build adapter based on type
        adapter = cls._build_adapter(
            adapter_type=adapter_type,
            input_dim=backbone.output_dim,
            shared_dim=improved_shared_dim,
            config=config
        )

        # Build model based on type
        if model_type == "improved":
            model = ImprovedClientModel(
                backbone=backbone,
                adapter=adapter,
                num_classes=model_config.num_classes,
                use_alignment_loss=config.get("use_alignment_loss", False)
            )
        else:
            # Original model (for backward compatibility)
            model = ClientModel(
                backbone=backbone,
                adapter=adapter,
                num_classes=model_config.num_classes
            )

        initialize_module_weights(model)
        return model

    @classmethod
    def _build_adapter(
        cls,
        adapter_type: AdapterType,
        input_dim: int,
        shared_dim: int,
        config: dict
    ) -> AdapterNetwork | ImprovedAdapterNetwork | AlignmentAwareAdapter:
        """Build adapter based on specified type."""

        if adapter_type == "original":
            return AdapterNetwork(input_dim=input_dim, shared_dim=shared_dim)

        elif adapter_type == "improved":
            return ImprovedAdapterNetwork(
                input_dim=input_dim,
                shared_dim=shared_dim,
                hidden_dims=config.get("adapter_hidden_dims"),
                use_residual=config.get("use_residual", True),
                dropout_rate=config.get("adapter_dropout", 0.2)
            )

        elif adapter_type == "alignment_aware":
            return AlignmentAwareAdapter(
                input_dim=input_dim,
                shared_dim=shared_dim,
                alignment_dim=config.get("alignment_dim", 256),
                hidden_dims=config.get("adapter_hidden_dims"),
                use_residual=config.get("use_residual", True),
                dropout_rate=config.get("adapter_dropout", 0.2)
            )

        else:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")

    @classmethod
    def build_client_model(
        cls,
        client_id: int,
        model_config: ModelConfig,
        dataset_name: str,
    ) -> ClientModel:
        """Build original client model for backward compatibility."""
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
        """Select backbone name for client based on configuration."""
        names = model_config.client_backbones
        if not names:
            raise ValueError("model_config.client_backbones cannot be empty")
        return names[client_id % len(names)].lower().strip()

    @staticmethod
    def _build_backbone(backbone_name: str, spec: InputSpec):
        """Build backbone network based on name and input specification."""
        if backbone_name == "small_cnn":
            return SmallCNNBackbone(in_channels=spec.in_channels)
        if backbone_name == "resnet8":
            return ResNet8Backbone(in_channels=spec.in_channels)
        if backbone_name == "mlp":
            return MLPBackbone(input_shape=(spec.in_channels, spec.height, spec.width))
        raise ValueError(f"Unsupported backbone name: {backbone_name}")

    @classmethod
    def get_compression_analysis(cls, dataset_name: str) -> dict:
        """Analyze compression ratios for different adapter configurations."""
        spec = cls.infer_input_spec(dataset_name)

        # Use SmallCNN as reference (most common)
        backbone = SmallCNNBackbone(in_channels=spec.in_channels)
        input_dim = backbone.output_dim

        analysis = {
            "backbone_output_dim": input_dim,
            "configurations": {}
        }

        # Original configuration
        original_shared_dim = 64
        analysis["configurations"]["original"] = {
            "shared_dim": original_shared_dim,
            "compression_ratio": input_dim / original_shared_dim,
            "information_preservation": f"{(original_shared_dim / input_dim) * 100:.2f}%"
        }

        # Improved configuration
        improved_shared_dim = cls.DEFAULT_IMPROVED_CONFIG["shared_dim"]
        analysis["configurations"]["improved"] = {
            "shared_dim": improved_shared_dim,
            "compression_ratio": input_dim / improved_shared_dim,
            "information_preservation": f"{(improved_shared_dim / input_dim) * 100:.2f}%"
        }

        return analysis


# For backward compatibility, alias the original factory
class ModelFactory(ImprovedModelFactory):
    """Original ModelFactory - kept for backward compatibility."""
    pass