#!/usr/bin/env python3
"""Quick test to verify FEMNIST class count fix."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.model_factory import ModelFactory

def quick_test():
    print("Quick FEMNIST class count verification...")

    # Test old (broken) config
    config_old = ExperimentConfig(dataset_name="femnist")
    print(f"Default config num_classes: {config_old.model.num_classes}")

    # Test new (fixed) config
    config_new = ExperimentConfig(dataset_name="femnist")
    config_new.model.num_classes = 62
    print(f"Fixed config num_classes: {config_new.model.num_classes}")

    # Test model creation
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config_new.model,
        dataset_name=config_new.dataset_name,
    )

    print(f"Model classifier output features: {model.classifier.out_features}")
    print(f"Model expects {model.num_classes} classes")

    if model.classifier.out_features == 62:
        print("[SUCCESS] Model correctly configured for 62 FEMNIST classes!")
    else:
        print(f"[ERROR] Model still has {model.classifier.out_features} output features")

if __name__ == "__main__":
    quick_test()