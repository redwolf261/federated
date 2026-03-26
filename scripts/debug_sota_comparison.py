"""
Debug script to identify issues with SOTA comparison implementation.

This script will:
1. Test centralized performance to verify data/model setup works
2. Identify potential issues with federated implementation
3. Provide debugging insights
"""

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

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class DebugModel(nn.Module):
    """Simplified model for debugging"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim
        print(f"DEBUG: Backbone output dim = {backbone_dim}")

        # Simpler architecture for debugging
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def debug_data_loading():
    """Debug data loading and preprocessing"""

    print("DEBUG: Data Loading")
    print("-" * 30)

    config = ExperimentConfig(dataset_name="femnist")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1000)

    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels range: [{labels.min()}, {labels.max()}]")
    print(f"Unique labels: {len(torch.unique(labels))} classes")

    # Check for data issues
    if torch.isnan(images).any():
        print("WARNING: NaN values in images!")
    if torch.isinf(images).any():
        print("WARNING: Inf values in images!")

    return images, labels


def test_centralized_performance(images, labels):
    """Test centralized performance to verify setup works"""

    print("\nDEBUG: Centralized Performance Test")
    print("-" * 40)

    # Create train/test split
    indices = torch.randperm(len(images))
    train_size = int(0.8 * len(images))

    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    print(f"Train: {len(train_images)} samples")
    print(f"Test: {len(test_images)} samples")

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Create model
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model = DebugModel(backbone, 62)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_images, sample_labels = sample_batch
        print(f"Sample batch - Images: {sample_images.shape}, Labels: {sample_labels.shape}")

        try:
            outputs = model(sample_images)
            print(f"Model output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            return False

    # Quick training test
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\nQuick training test (5 epochs):")

    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()

            try:
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

            except Exception as e:
                print(f"ERROR in training step: {e}")
                return False

        train_accuracy = correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={train_accuracy:.3f}")

        # Early success check
        if train_accuracy > 0.1:  # If we get >10% accuracy, things are working
            print("SUCCESS: Training is working, accuracy increasing")
            break

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            outputs = model(batch_images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == batch_labels).sum().item()
            test_total += batch_labels.size(0)

    test_accuracy = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.3f}")

    return test_accuracy > 0.05  # Even 5% is better than random (1/62 ≈ 1.6%)


def debug_heterogeneous_splits():
    """Debug the heterogeneous data splitting logic"""

    print("\nDEBUG: Heterogeneous Data Splits")
    print("-" * 40)

    config = ExperimentConfig(dataset_name="femnist")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1000)

    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]

    # Examine label distribution
    unique_labels = torch.unique(labels)
    print(f"Available classes: {len(unique_labels)} (range: {unique_labels.min()}-{unique_labels.max()})")

    # Count samples per class
    class_counts = {}
    for label in unique_labels:
        count = (labels == label).sum().item()
        class_counts[label.item()] = count

    # Show distribution
    print("Samples per class:")
    for class_id in sorted(class_counts.keys())[:10]:  # Show first 10
        print(f"  Class {class_id}: {class_counts[class_id]} samples")
    print(f"  ... (showing 10/{len(class_counts)} classes)")

    # Test client splits
    sorted_indices = torch.argsort(labels)
    sorted_labels = labels[sorted_indices]

    print(f"\nClient split analysis (4 clients, 150 samples each):")

    samples_per_client = 150
    for client_id in range(4):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client

        if end_idx <= len(sorted_labels):
            client_labels = sorted_labels[start_idx:end_idx]
            unique_client_labels = torch.unique(client_labels)

            print(f"  Client {client_id}: classes {unique_client_labels.min()}-{unique_client_labels.max()} "
                  f"({len(unique_client_labels)} unique classes)")

            # Check if this is too extreme
            if len(unique_client_labels) < 5:
                print(f"    WARNING: Only {len(unique_client_labels)} classes - very heterogeneous!")
        else:
            print(f"  Client {client_id}: Not enough data")

    return True


def main():
    """Main debugging pipeline"""

    print("SOTA COMPARISON DEBUG")
    print("=" * 50)
    print("Purpose: Identify issues with SOTA comparison implementation")
    print()

    # Step 1: Debug data loading
    try:
        images, labels = debug_data_loading()
    except Exception as e:
        print(f"CRITICAL ERROR in data loading: {e}")
        return

    # Step 2: Test centralized performance
    try:
        centralized_works = test_centralized_performance(images, labels)
        if not centralized_works:
            print("CRITICAL ERROR: Centralized training failed")
            return
        else:
            print("SUCCESS: Centralized setup works correctly")
    except Exception as e:
        print(f"CRITICAL ERROR in centralized test: {e}")
        return

    # Step 3: Debug data splits
    try:
        debug_heterogeneous_splits()
    except Exception as e:
        print(f"ERROR in split analysis: {e}")

    print(f"\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    print("If centralized training works but federated fails, likely issues:")
    print("1. Data splits too heterogeneous (each client sees too few classes)")
    print("2. Insufficient local training (need more epochs)")
    print("3. Poor aggregation strategy")
    print("4. Learning rate too high/low for federated setting")
    print()
    print("RECOMMENDATION: Increase local epochs and reduce heterogeneity")


if __name__ == "__main__":
    main()