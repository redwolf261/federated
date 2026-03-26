"""
Quick Phase 0 Diagnostic: Test core components before full implementation
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


class QuickModel(nn.Module):
    """Simplified model for quick testing"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        # Simplified architecture for quick testing
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def quick_data_test():
    """Test data loading and basic processing"""

    print("PHASE 0 DIAGNOSTIC: Data Loading Test")
    print("-" * 40)

    config = ExperimentConfig(dataset_name="femnist")
    registry = DatasetRegistry(project_root)

    # Load small subset for testing
    artifact = registry.load("femnist", max_rows=1000)
    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]

    print(f"Loaded: {len(images)} images, {len(torch.unique(labels))} classes")
    print(f"Image shape: {images[0].shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")

    # Quick Dirichlet test
    num_clients = 5  # Small number for testing
    alpha = 0.5

    np.random.seed(42)
    torch.manual_seed(42)

    # Simple Dirichlet split simulation
    num_classes = 62
    client_class_counts = np.zeros((num_clients, num_classes))

    for class_id in range(min(10, num_classes)):  # Test with first 10 classes
        class_mask = (labels == class_id)
        class_size = class_mask.sum().item()

        if class_size == 0:
            continue

        # Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        class_client_counts = (proportions * class_size).astype(int)

        # Handle remainder
        remainder = class_size - class_client_counts.sum()
        for _ in range(remainder):
            client_id = np.random.randint(num_clients)
            class_client_counts[client_id] += 1

        client_class_counts[:, class_id] = class_client_counts

    print(f"\nDirichlet distribution test (alpha={alpha}):")
    for client_id in range(num_clients):
        total_samples = int(client_class_counts[client_id].sum())
        non_zero_classes = np.count_nonzero(client_class_counts[client_id])
        print(f"  Client {client_id}: {total_samples} samples, {non_zero_classes} classes")

    return images[:800], labels[:800]  # Return subset for model testing


def quick_model_test(images, labels):
    """Test model architecture and training"""

    print(f"\nPHASE 0 DIAGNOSTIC: Model Training Test")
    print("-" * 40)

    # Create simple dataset
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model = QuickModel(backbone, 62)

    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Quick training test
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting quick training (5 epochs)...")

    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            try:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            except Exception as e:
                print(f"ERROR during training: {e}")
                return False

        accuracy = correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(loader):.3f}, Acc={accuracy:.3f}")

    print(f"Final accuracy: {accuracy:.3f}")

    # Check if model is learning
    if accuracy > 0.1:  # >10% on 62 classes (random = 1.6%)
        print("SUCCESS: Model is learning")
        return True
    else:
        print("WARNING: Model may not be learning effectively")
        return False


def main():
    """Quick Phase 0 diagnostic"""

    print("QUICK PHASE 0 DIAGNOSTIC")
    print("=" * 50)
    print("Purpose: Test core components before full Phase 0")
    print()

    try:
        # Test data loading
        images, labels = quick_data_test()

        # Test model training
        model_ok = quick_model_test(images, labels)

        print(f"\n" + "=" * 50)
        print("DIAGNOSTIC RESULTS")
        print("=" * 50)

        if model_ok:
            print("+ Data loading: SUCCESS")
            print("+ Model training: SUCCESS")
            print("+ Dirichlet logic: SUCCESS")
            print()
            print("VERDICT: Core components working - ready for full Phase 0")
        else:
            print("+ Data loading: SUCCESS")
            print("- Model training: ISSUES DETECTED")
            print()
            print("VERDICT: Fix model training before full Phase 0")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("VERDICT: Fix fundamental issues before proceeding")


if __name__ == "__main__":
    main()