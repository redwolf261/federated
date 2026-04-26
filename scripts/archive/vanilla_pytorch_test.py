#!/usr/bin/env python3
"""Vanilla PyTorch test - bypass federated learning to test basic FEMNIST classification."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.data.dataset_registry import DatasetRegistry

def create_simple_cnn(num_classes=62):
    """Create a simple CNN similar to SmallCNNBackbone + classifier."""
    return nn.Sequential(
        # Backbone (similar to SmallCNNBackbone)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),

        # Classifier
        nn.Linear(128, num_classes)
    )

def vanilla_femnist_test():
    """
    Pure PyTorch test without federated learning complexity.

    This should achieve 70%+ accuracy if the data and model are correct.
    If this fails, the issue is in data loading or basic model setup.
    If this succeeds, the issue is in the federated learning implementation.
    """
    print("="*70)
    print("VANILLA PYTORCH FEMNIST TEST")
    print("="*70)
    print("Testing basic FEMNIST classification without federated learning")
    print()

    # Load raw FEMNIST data
    print("Loading FEMNIST data...")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=50000)  # Limit for speed

    images = artifact.payload["images"]  # Already normalized [0,1] tensors
    labels = artifact.payload["labels"]

    print(f"Data loaded:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    print(f"  Unique labels: {len(labels.unique())}")
    print()

    # Create train/test split
    num_samples = len(images)
    num_train = int(0.8 * num_samples)

    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    print(f"Split:")
    print(f"  Training: {len(train_images)} samples")
    print(f"  Testing: {len(test_images)} samples")
    print()

    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = create_simple_cnn(num_classes=62)

    print(f"Model:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test forward pass
    test_batch_x, test_batch_y = next(iter(train_loader))
    with torch.no_grad():
        test_output = model(test_batch_x)
        print(f"  Forward pass: {test_batch_x.shape} -> {test_output.shape}")
        print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    model.train()

    for epoch in range(10):  # Train for 10 epochs
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # Track metrics
            _, preds = torch.max(outputs, 1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += batch_y.size(0)
            epoch_loss += loss.item()

        # Calculate epoch metrics
        epoch_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / len(train_loader)

        # Test accuracy
        test_acc = evaluate_model(model, test_loader)

        print(f"Epoch {epoch+1:2d}: Train Acc={epoch_acc:.4f}, Test Acc={test_acc:.4f}, Loss={avg_loss:.4f}")

        # Early stopping if we hit good accuracy
        if test_acc > 0.8:
            print("  -> Excellent accuracy reached!")
            break

    print()

    # Final evaluation
    final_test_acc = evaluate_model(model, test_loader)
    print(f"Final Results:")
    print(f"  Test accuracy: {final_test_acc:.4f} ({final_test_acc:.1%})")
    print()

    # Interpret results
    if final_test_acc > 0.7:
        print("[SUCCESS] >70% accuracy - Basic FEMNIST classification works!")
        print("-> The issue is likely in the federated learning implementation")
    elif final_test_acc > 0.4:
        print("[PARTIAL] 40-70% accuracy - Model works but may need tuning")
        print("-> Check hyperparameters and training setup")
    elif final_test_acc > 0.2:
        print("[POOR] 20-40% accuracy - Significant issues with model or data")
        print("-> Check data preprocessing and model architecture")
    else:
        print("[FAILED] <20% accuracy - Fundamental issues")
        print("-> Check data loading, model architecture, or loss function")

    return final_test_acc

def evaluate_model(model, data_loader):
    """Evaluate model on data loader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    model.train()
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    vanilla_femnist_test()