#!/usr/bin/env python3
"""Compare fixed backbone performance in isolation vs integrated FLEX system."""

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
from flex_persona.data.dataset_registry import DatasetRegistry

class FixedBackboneDirect(nn.Module):
    """Fixed backbone with direct classifier (no adapter)."""
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNNBackbone(in_channels=1)
        self.classifier = nn.Linear(self.backbone.output_dim, 62)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def debug_integration_issues():
    """Debug why fixed backbone works alone but not in FLEX system."""
    print("="*70)
    print("DEBUGGING: ISOLATION VS INTEGRATION PERFORMANCE")
    print("="*70)

    # Load test data
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1500)
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
    models = {}

    # 1. Fixed backbone in isolation (direct classifier)
    models['fixed_backbone_direct'] = FixedBackboneDirect()

    # 2. Full FLEX system with fixed backbone
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62
    models['full_flex_system'] = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TESTING: {model_name.upper()}")
        print('='*50)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        if hasattr(model, 'backbone'):
            print(f"Backbone output: {model.backbone.output_dim}")
        if hasattr(model, 'adapter'):
            print(f"Adapter: {model.adapter.input_dim} -> {model.adapter.shared_dim}")
        if hasattr(model, 'classifier'):
            print(f"Classifier: {model.classifier.in_features} -> {model.classifier.out_features}")

        # Test forward pass
        test_batch = train_images[:8]
        with torch.no_grad():
            if model_name == 'full_flex_system':
                output = model.forward_task(test_batch)
            else:
                output = model(test_batch)

            print(f"Forward: {test_batch.shape} -> {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output std: {output.std():.3f}")

        # Training test
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"\nTraining (5 epochs):")
        model.train()

        for epoch in range(5):
            running_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                if model_name == 'full_flex_system':
                    outputs = model.forward_task(batch_x)
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()

                # Check gradient magnitudes
                total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                if total_grad_norm > 1000:
                    print(f"    [WARNING] Large gradients: {total_grad_norm:.1f}")

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
                    if model_name == 'full_flex_system':
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

            print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}, Loss={avg_loss:.4f}")

        results[model_name] = {
            'final_test_acc': test_acc,
            'params': total_params
        }

    # Analysis
    print(f"\n{'='*70}")
    print("INTEGRATION ANALYSIS")
    print('='*70)

    direct_acc = results['fixed_backbone_direct']['final_test_acc']
    flex_acc = results['full_flex_system']['final_test_acc']
    performance_loss = direct_acc - flex_acc

    print(f"Fixed Backbone (Direct):     {direct_acc:.1%}")
    print(f"Full FLEX System:            {flex_acc:.1%}")
    print(f"Performance Loss:            {performance_loss:+.1%}")
    print()

    if performance_loss > 0.15:  # >15% loss
        print("[CRITICAL] Full FLEX system significantly underperforms isolated backbone")
        print("-> Issue is in adapter network or system integration")

        # Check adapter impact
        adapter_reduction = 6272 / 64  # Input to output ratio
        print(f"Adapter dimensionality reduction: {adapter_reduction:.1f}x (6272 -> 64)")

        if adapter_reduction > 50:
            print("-> POTENTIAL CAUSE: Excessive dimensionality reduction in adapter")
            print("-> RECOMMENDATION: Increase adapter output dim from 64 to 512+")

    elif performance_loss > 0.05:  # >5% loss
        print("[MODERATE] Some performance loss in full system")
        print("-> May need hyperparameter tuning or adapter adjustment")
    else:
        print("[GOOD] Minimal performance loss in full system integration")

    print(f"\nNext steps:")
    if performance_loss > 0.15:
        print("1. Test with larger adapter output dimensions (64 -> 256 -> 512)")
        print("2. Investigate gradient flow through adapter network")
        print("3. Consider bypassing adapter for testing")
    else:
        print("1. System integration appears functional")
        print("2. Focus on hyperparameter optimization")

if __name__ == "__main__":
    debug_integration_issues()