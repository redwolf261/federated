#!/usr/bin/env python3
"""Detailed debugging test to identify remaining issues."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.models.model_factory import ModelFactory

def detailed_debug_test():
    """Deep dive debugging to find remaining issues."""
    print("="*60)
    print("DETAILED DEBUGGING TEST")
    print("="*60)

    # Create config with more data and lower learning rate
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 1
    config.model.num_classes = 62
    config.training.max_samples_per_client = None  # Use all data
    config.training.learning_rate = 0.001  # Lower learning rate
    config.training.batch_size = 32

    print(f"Configuration:")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Max samples: {config.training.max_samples_per_client}")
    print()

    # Load data
    manager = ClientDataManager(project_root, config)
    bundles = manager.build_client_bundles()
    bundle = bundles[0]

    print(f"Data statistics:")
    print(f"  - Training samples: {bundle.num_samples}")
    print(f"  - Classes: {len(bundle.class_histogram)}")

    # Show class distribution
    sorted_hist = dict(sorted(bundle.class_histogram.items()))
    min_samples = min(sorted_hist.values())
    max_samples = max(sorted_hist.values())
    avg_samples = np.mean(list(sorted_hist.values()))

    print(f"  - Samples per class: min={min_samples}, max={max_samples}, avg={avg_samples:.1f}")
    print()

    # Create model
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    print(f"Model details:")
    print(f"  - Backbone type: {type(model.backbone).__name__}")
    print(f"  - Backbone output: {model.backbone.output_dim}")
    print(f"  - Adapter input→shared: {model.adapter.input_dim}→{model.adapter.shared_dim}")
    print(f"  - Classifier: {model.classifier.in_features}→{model.classifier.out_features}")
    print()

    # Test forward pass in detail
    print("Forward pass analysis:")
    batch_x, batch_y = next(iter(bundle.train_loader))
    print(f"  - Input shape: {batch_x.shape}")
    print(f"  - Input range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
    print(f"  - Label shape: {batch_y.shape}")
    print(f"  - Label range: [{batch_y.min()}, {batch_y.max()}]")
    print(f"  - Unique labels in batch: {sorted(batch_y.unique().tolist())}")
    print()

    with torch.no_grad():
        # Test backbone
        features = model.extract_features(batch_x)
        print(f"  - Backbone features shape: {features.shape}")
        print(f"  - Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  - Features std: {features.std():.3f}")

        # Test adapter
        shared = model.project_shared(features)
        print(f"  - Shared repr shape: {shared.shape}")
        print(f"  - Shared range: [{shared.min():.3f}, {shared.max():.3f}]")
        print(f"  - Shared std: {shared.std():.3f}")

        # Test classifier
        logits = model.classifier(features)
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"  - Logits std: {logits.std():.3f}")

        # Test predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        print(f"  - Predictions: {preds.tolist()[:10]}...")
        print(f"  - True labels: {batch_y.tolist()[:10]}...")

        # Check if predictions are reasonable
        pred_range = [preds.min().item(), preds.max().item()]
        print(f"  - Prediction range: {pred_range}")

        accuracy = (preds == batch_y).float().mean()
        print(f"  - Random batch accuracy: {accuracy:.4f}")
    print()

    # Test loss computation
    print("Loss analysis:")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        loss = criterion(logits, batch_y)
        print(f"  - Cross-entropy loss: {loss:.4f}")

        # Test with random logits for comparison
        random_logits = torch.randn_like(logits)
        random_loss = criterion(random_logits, batch_y)
        print(f"  - Random logits loss: {random_loss:.4f}")

        # Expected loss for random guessing with 62 classes
        expected_random_loss = -np.log(1.0 / 62)
        print(f"  - Expected random loss (62 classes): {expected_random_loss:.4f}")
    print()

    # Quick training test
    print("Quick training test (1 epoch):")
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    model.train()
    epoch_losses = []

    for i, (batch_x, batch_y) in enumerate(bundle.train_loader):
        optimizer.zero_grad()

        logits = model.forward_task(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        if i < 5:  # Show first few batches
            print(f"  - Batch {i+1}: loss={loss.item():.4f}")

    avg_loss = np.mean(epoch_losses)
    print(f"  - Average epoch loss: {avg_loss:.4f}")

    # Check if loss is decreasing
    if len(epoch_losses) > 10:
        first_half = np.mean(epoch_losses[:len(epoch_losses)//2])
        second_half = np.mean(epoch_losses[len(epoch_losses)//2:])
        improvement = first_half - second_half
        print(f"  - Loss improvement during epoch: {improvement:+.4f}")

    print()
    print("Potential issues to investigate:")

    if features.std() < 0.1:
        print("  - Low feature variance - backbone may not be learning")
    if logits.std() < 0.5:
        print("  - Low logit variance - classifier may have poor initialization")
    if avg_loss > 4.5:
        print("  - High training loss - learning rate may be too low or model capacity insufficient")
    if pred_range[1] - pred_range[0] < 20:
        print("  - Limited prediction range - model may be stuck on few classes")

    return config, model, bundle

if __name__ == "__main__":
    detailed_debug_test()