#!/usr/bin/env python3
"""Ultra-simple FEMNIST test to isolate the absolute basics."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.data.dataset_registry import DatasetRegistry

def ultra_simple_test():
    """Minimal FEMNIST test with zero complexityto isolate fundamental issues."""
    print("="*60)
    print("ULTRA-SIMPLE FEMNIST TEST")
    print("="*60)

    # Load just a small sample of data
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1000)  # Very small sample

    images = artifact.payload["images"]
    labels = artifact.payload["labels"]

    print(f"Loaded {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Label range: {labels.min()}-{labels.max()}")
    print()

    # Create the simplest possible model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 12 * 12, 120)  # 28->24->12 after conv+pool
            self.fc2 = nn.Linear(120, 62)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv(x)))
            x = x.view(-1, 16 * 12 * 12)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch = images[:32]
    labels_batch = labels[:32]

    with torch.no_grad():
        output = model(batch)
        print(f"Forward pass: {batch.shape} -> {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Test prediction accuracy (should be ~1/62 = 1.6% for random)
        preds = torch.argmax(output, dim=1)
        acc = (preds == labels_batch).float().mean()
        print(f"Random accuracy: {acc:.4f} ({acc:.1%})")
    print()

    # Create simple train/test split
    n = len(images)
    train_size = int(0.8 * n)

    train_images = images[:train_size]
    train_labels = labels[:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training for 10 epochs...")

    for epoch in range(10):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (preds == batch_y).sum().item()

        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                test_total += batch_y.size(0)
                test_correct += (preds == batch_y).sum().item()

        train_acc = correct / total
        test_acc = test_correct / test_total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1:2d}: Train={train_acc:.4f}, Test={test_acc:.4f}, Loss={avg_loss:.4f}")

        # Early exit if we achieve good performance
        if test_acc > 0.6:
            print(f"  -> Good performance reached!")
            break

    print()
    print("Results:")
    print(f"  Final test accuracy: {test_acc:.4f} ({test_acc:.1%})")

    if test_acc > 0.6:
        print("  [SUCCESS] >60% - Basic FEMNIST classification works!")
        print("  -> Issue is likely in federated learning implementation")
    elif test_acc > 0.3:
        print("  [PARTIAL] 30-60% - Model learning but suboptimal")
        print("  -> May need hyperparameter tuning or architecture changes")
    elif test_acc > 0.1:
        print("  [POOR] 10-30% - Limited learning")
        print("  -> Check data preprocessing or model capacity")
    else:
        print("  [FAILED] <10% - Not learning at all")
        print("  -> Fundamental issue with data or model")

if __name__ == "__main__":
    ultra_simple_test()