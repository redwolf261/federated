"""Top-level experiment configuration for FLEX-Persona."""

from __future__ import annotations

from dataclasses import dataclass, field

from .clustering_config import ClusteringConfig
from .eval_config import EvaluationConfig
from .model_config import ModelConfig
from .similarity_config import SimilarityConfig
from .training_config import TrainingConfig
from ..utils.constants import DEFAULT_NUM_CLIENTS, DEFAULT_RANDOM_SEED, MAX_CLIENTS, MIN_CLIENTS


@dataclass
class ExperimentConfig:
    """Main configuration object consumed by all pipeline modules."""

    experiment_name: str = "flex_persona_baseline"
    dataset_name: str = "femnist"
    num_clients: int = DEFAULT_NUM_CLIENTS
    random_seed: int = DEFAULT_RANDOM_SEED
    output_dir: str = "outputs"

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")
        if self.num_clients < MIN_CLIENTS or self.num_clients > MAX_CLIENTS:
            raise ValueError(
                f"num_clients must be between {MIN_CLIENTS} and {MAX_CLIENTS}"
            )

        self.model.validate()
        self.training.validate()
        self.similarity.validate()
        self.clustering.validate()
        self.evaluation.validate()
