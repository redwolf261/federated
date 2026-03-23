#!/usr/bin/env python3
"""Systematic investigation of FLEX-Persona architectural components."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.model_factory import ModelFactory
from flex_persona.models.backbones import SmallCNNBackbone
from flex_persona.models.adapter_network import AdapterNetwork
from flex_persona.data.dataset_registry import DatasetRegistry

class SimpleWorkingCNN(nn.Module):
    """Reference simple model that achieves 37% accuracy."""
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

class BackboneOnlyModel(nn.Module):
    """FLEX-Persona backbone + direct classifier (bypass adapter)."""
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNNBackbone(in_channels=1)  # 128 output
        self.classifier = nn.Linear(128, 62)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class FixedAdapterModel(nn.Module):
    """FLEX-Persona with fixed adapter (no dimensionality reduction)."""
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNNBackbone(in_channels=1)
        self.adapter = nn.Linear(128, 128)  # Same dim, no reduction
        self.classifier = nn.Linear(128, 62)

    def forward(self, x):
        features = self.backbone(x)
        adapted = self.adapter(features)
        return self.classifier(adapted)

class LargerAdapterModel(nn.Module):
    """FLEX-Persona with larger adapter output."""
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNNBackbone(in_channels=1)
        self.adapter = nn.Linear(128, 128)  # Keep full dimensionality
        self.classifier = nn.Linear(128, 62)

    def forward(self, x):
        features = self.backbone(x)
        adapted = self.adapter(features)
        return self.classifier(adapted)

def investigate_architectural_components():
    """Test different architectural variations to isolate the issue."""
    print("="*80)
    print("FLEX-PERSONA ARCHITECTURAL COMPONENT INVESTIGATION")
    print("="*80)

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

    # Test different model architectures
    models = {
        'baseline_simple': SimpleWorkingCNN(),
        'backbone_only': BackboneOnlyModel(),
        'fixed_adapter': FixedAdapterModel(),
        'original_flex': None,  # Will create using ModelFactory
    }

    # Create original FLEX-Persona model
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62
    models['original_flex'] = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"TESTING: {model_name.upper()}")
        print('='*60)

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        if model_name == 'baseline_simple':
            print("Architecture: Simple CNN (reference)")
        elif model_name == 'backbone_only':
            print("Architecture: FLEX Backbone -> Classifier (no adapter)")
        elif model_name == 'fixed_adapter':
            print("Architecture: FLEX Backbone -> 128->128 Adapter -> Classifier")
        elif model_name == 'original_flex':
            print(f"Architecture: FLEX Backbone -> {model.adapter.input_dim}->{model.adapter.shared_dim} Adapter -> Classifier")

        # Test forward pass
        test_batch = train_images[:32]
        test_labels_batch = train_labels[:32]

        with torch.no_grad():
            if model_name == 'original_flex':
                output = model.forward_task(test_batch)
            else:
                output = model(test_batch)

            print(f"Forward pass: {test_batch.shape} -> {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output std: {output.std():.3f}")

            # Check for any NaN or extreme values
            if torch.isnan(output).any():
                print("WARNING: NaN values in output!")
            if output.abs().max() > 10:
                print("WARNING: Very large output values!")

        # Training test
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining for 5 epochs...")
        model.train()

        epoch_accs = []
        for epoch in range(5):
            running_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                if model_name == 'original_flex':
                    outputs = model.forward_task(batch_x)
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()

                # Check for gradient issues
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                if grad_norm > 100:
                    print(f"    WARNING: Large gradients ({grad_norm:.1f}) in epoch {epoch+1}")

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
                    if model_name == 'original_flex':
                        outputs = model.forward_task(batch_x)
                    else:
                        outputs = model(batch_x)

                    _, preds = torch.max(outputs, 1)
                    test_total += batch_y.size(0)
                    test_correct += (preds == batch_y).sum().item()

            model.train()

            train_acc = correct / total
            test_acc = test_correct / test_total
            avg_loss = running_loss / len(train_loader)

            epoch_accs.append(test_acc)
            print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}, Loss={avg_loss:.4f}")

        results[model_name] = {
            'final_test_acc': test_acc,
            'best_test_acc': max(epoch_accs),
            'improvement': epoch_accs[-1] - epoch_accs[0],
            'params': total_params
        }

    # Analysis
    print(f"\n{'='*80}")
    print("ARCHITECTURAL ANALYSIS")
    print('='*80)

    baseline_acc = results['baseline_simple']['final_test_acc']

    print(f"Performance Comparison (vs baseline {baseline_acc:.1%}):")
    for name, res in results.items():
        acc = res['final_test_acc']
        diff = acc - baseline_acc
        print(f"  {name:20s}: {acc:.1%} ({diff:+.1%})")

    print(f"\nDiagnostic Insights:")

    backbone_acc = results['backbone_only']['final_test_acc']
    adapter_acc = results['fixed_adapter']['final_test_acc']
    flex_acc = results['original_flex']['final_test_acc']

    if backbone_acc > baseline_acc * 0.8:
        print("✓ FLEX backbone architecture is reasonable")
    else:
        print("✗ FLEX backbone has issues")

    if adapter_acc > backbone_acc:
        print("✓ Adapter helps performance")
    elif adapter_acc < backbone_acc * 0.9:
        print("✗ Adapter hurts performance significantly")
    else:
        print("~ Adapter has minimal impact")

    if flex_acc < adapter_acc * 0.8:
        print("✗ Dimensionality reduction (128->64) is the main issue")
    else:
        print("~ Dimensionality reduction has acceptable impact")

    # Recommendation
    print(f"\nRecommendations:")
    best_flex_variant = max(['backbone_only', 'fixed_adapter'],
                           key=lambda x: results[x]['final_test_acc'])
    best_acc = results[best_flex_variant]['final_test_acc']

    print(f"Best FLEX variant: {best_flex_variant} ({best_acc:.1%})")

    if best_acc > baseline_acc * 0.9:
        print("-> Fix: Use this architecture variant in FLEX-Persona")
    else:
        print("-> The FLEX backbone itself needs redesign")

if __name__ == "__main__":
    investigate_architectural_components()