#!/usr/bin/env python3
"""Simple baseline test to verify FEMNIST accuracy improvement after class count fix."""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.models.model_factory import ModelFactory

def simple_baseline_test():
    """Test single client training with corrected FEMNIST configuration."""
    print("="*60)
    print("SIMPLE BASELINE ACCURACY TEST")
    print("="*60)
    print("Testing: Single client training with corrected 62-class FEMNIST setup")
    print()

    # Create corrected config
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 1
    config.model.num_classes = 62  # CRITICAL FIX
    config.training.max_samples_per_client = 1000  # Limit for speed
    config.training.learning_rate = 0.01
    config.training.batch_size = 64

    print(f"Configuration:")
    print(f"  - Dataset: {config.dataset_name}")
    print(f"  - Classes: {config.model.num_classes}")
    print(f"  - Max samples per client: {config.training.max_samples_per_client}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Batch size: {config.training.batch_size}")
    print()

    # Load data
    print("Loading FEMNIST data...")
    manager = ClientDataManager(project_root, config)
    bundles = manager.build_client_bundles()
    bundle = bundles[0]

    print(f"Client data:")
    print(f"  - Training samples: {bundle.num_samples}")
    print(f"  - Classes present: {len(bundle.class_histogram)}")
    print(f"  - Train batches: {len(bundle.train_loader)}")
    print(f"  - Eval batches: {len(bundle.eval_loader)}")
    print()

    # Create model
    print("Creating model...")
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    print(f"Model architecture:")
    print(f"  - Backbone output dim: {model.backbone.output_dim}")
    print(f"  - Classifier output features: {model.classifier.out_features}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training setup
    optimizer = optim.SGD(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Test initial accuracy
    print("Testing initial (random) accuracy...")
    initial_acc = evaluate_model(model, bundle.eval_loader)
    print(f"Initial accuracy: {initial_acc:.4f} ({initial_acc:.1%})")
    print()

    # Train for a few epochs
    print("Training...")
    epochs = 5
    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in bundle.train_loader:
            optimizer.zero_grad()
            logits = model.forward_task(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time

        # Evaluate
        acc = evaluate_model(model, bundle.eval_loader)

        print(f"Epoch {epoch+1}/{epochs}: Acc={acc:.4f} ({acc:.1%}), Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")

    print()
    print("Final results:")
    final_acc = evaluate_model(model, bundle.eval_loader)
    improvement = final_acc - initial_acc

    print(f"  - Final accuracy: {final_acc:.4f} ({final_acc:.1%})")
    print(f"  - Improvement: {improvement:+.4f} ({improvement:+.1%})")
    print()

    # Interpret results
    if final_acc > 0.7:
        print("[EXCELLENT] >70% accuracy - implementation is working correctly!")
    elif final_acc > 0.5:
        print("[GOOD] >50% accuracy - major improvement, likely hyperparameter tuning needed")
    elif final_acc > 0.3:
        print("[IMPROVED] >30% accuracy - significant improvement from class fix")
    elif final_acc > 0.15:
        print("[SOME PROGRESS] >15% accuracy - partial improvement, other issues remain")
    else:
        print("[STILL BROKEN] <15% accuracy - other fundamental issues exist")

    print()
    print("Expected FEMNIST single-client accuracy: 70-90%+")
    print("Previous broken accuracy: 9-14%")
    print(f"Current achieved accuracy: {final_acc:.1%}")


def evaluate_model(model, eval_loader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            logits = model.forward_task(batch_x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    simple_baseline_test()