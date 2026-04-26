#!/usr/bin/env python3
"""Test fixed FLEX backbone that preserves spatial information."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.data.dataset_registry import DatasetRegistry

class FixedSmallCNNBackbone(nn.Module):
    """Fixed version of SmallCNNBackbone that preserves spatial information."""
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # CRITICAL FIX: Replace AdaptiveAvgPool2d(1,1) with Flatten
            # to preserve spatial information
            nn.Flatten(),  # 7*7*128 = 6272 features instead of 128
        )
        # Update output dimension to reflect actual feature size
        self.output_dim = 128 * 7 * 7  # 6272

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class WorkingSimpleCNN(nn.Module):
    """Reference model for comparison."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 62)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FixedFlexModel(nn.Module):
    """Complete model using fixed backbone."""
    def __init__(self):
        super().__init__()
        self.backbone = FixedSmallCNNBackbone(in_channels=1)
        self.classifier = nn.Linear(self.backbone.output_dim, 62)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def test_fixed_backbone():
    """Test the fixed backbone against the broken one."""
    print("="*70)
    print("TESTING FIXED FLEX BACKBONE")
    print("="*70)

    # Load test data
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"]
    labels = artifact.payload["labels"]

    # Create train/test split
    n = len(images)
    train_size = int(0.8 * n)
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test models
    models = {
        'simple_reference': WorkingSimpleCNN(),
        'broken_flex_backbone': None,  # Will create with original backbone
        'fixed_flex_backbone': FixedFlexModel(),
    }

    # Create broken FLEX model for comparison
    class BrokenFlexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SmallCNNBackbone(in_channels=1)  # Original broken version
            self.classifier = nn.Linear(self.backbone.output_dim, 62)

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    models['broken_flex_backbone'] = BrokenFlexModel()

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TESTING: {model_name.upper()}")
        print('='*50)

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        if hasattr(model, 'backbone'):
            print(f"Backbone output dim: {model.backbone.output_dim}")

        # Test forward pass
        test_batch = train_images[:8]
        with torch.no_grad():
            output = model(test_batch)
            print(f"Forward: {test_batch.shape} -> {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output std: {output.std():.3f}")

        # Quick training test
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"\nTraining for 5 epochs...")
        model.train()

        for epoch in range(5):
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
            model.train()

            train_acc = correct / total
            test_acc = test_correct / test_total
            avg_loss = running_loss / len(train_loader)

            print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}, Loss={avg_loss:.4f}")

        results[model_name] = {
            'final_test_acc': test_acc,
            'params': total_params
        }

    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)

    simple_acc = results['simple_reference']['final_test_acc']
    broken_acc = results['broken_flex_backbone']['final_test_acc']
    fixed_acc = results['fixed_flex_backbone']['final_test_acc']

    print(f"Simple Reference Model:  {simple_acc:.1%}")
    print(f"Broken FLEX Backbone:    {broken_acc:.1%}")
    print(f"Fixed FLEX Backbone:     {fixed_acc:.1%}")
    print()

    improvement = fixed_acc - broken_acc
    vs_simple = fixed_acc - simple_acc

    print(f"Fix improvement:  {improvement:+.1%}")
    print(f"vs Simple model:  {vs_simple:+.1%}")
    print()

    if fixed_acc > broken_acc + 0.15:  # >15% improvement
        print("[SUCCESS] Fix significantly improves performance!")
        if fixed_acc > simple_acc * 0.8:  # Within 20% of simple model
            print("[EXCELLENT] Fixed backbone performs comparably to simple model")
        else:
            print("[GOOD] Fixed backbone much better but still room for improvement")
    elif fixed_acc > broken_acc + 0.05:  # >5% improvement
        print("[PARTIAL] Fix helps but may not be the only issue")
    else:
        print("[FAILED] Fix doesn't significantly improve performance")

    print(f"\nNext steps:")
    if fixed_acc > broken_acc + 0.15:
        print("1. Update SmallCNNBackbone in FLEX-Persona codebase with this fix")
        print("2. Re-run federated learning experiments")
        print("3. Should now achieve 70%+ FEMNIST performance")
    else:
        print("1. Investigate additional architectural issues")
        print("2. May need further backbone redesign")

if __name__ == "__main__":
    test_fixed_backbone()