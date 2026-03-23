#!/usr/bin/env python3
"""Compare FLEX-Persona ClientModel vs simple working model."""

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
from flex_persona.data.dataset_registry import DatasetRegistry

class SimpleWorkingCNN(nn.Module):
    """The simple CNN that achieved 28.5% accuracy."""
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

def compare_model_architectures():
    """Compare FLEX-Persona ClientModel vs simple working model."""
    print("="*70)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*70)

    # Load same data for both
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

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    models = {}

    # Test 1: Simple working model
    print("1. SIMPLE WORKING MODEL")
    print("-" * 30)
    simple_model = SimpleWorkingCNN()
    total_params = sum(p.numel() for p in simple_model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Architecture: Conv2d(1->16) -> Pool -> FC(2304->120) -> FC(120->62)")

    models['simple'] = simple_model

    # Test 2: FLEX-Persona ClientModel
    print("\n2. FLEX-PERSONA CLIENT MODEL")
    print("-" * 30)
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    flex_model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    total_params = sum(p.numel() for p in flex_model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Architecture:")
    print(f"     - Backbone: {type(flex_model.backbone).__name__} -> {flex_model.backbone.output_dim}")
    print(f"     - Adapter: {flex_model.adapter.input_dim} -> {flex_model.adapter.shared_dim}")
    print(f"     - Classifier: {flex_model.classifier.in_features} -> {flex_model.classifier.out_features}")

    models['flex'] = flex_model

    # Test both models
    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TESTING: {model_name.upper()}")
        print('='*50)

        # Test forward pass
        test_batch = train_images[:32]
        test_labels_batch = train_labels[:32]

        with torch.no_grad():
            if model_name == 'flex':
                # Use FLEX-Persona's forward_task method
                output = model.forward_task(test_batch)
            else:
                output = model(test_batch)

            print(f"Forward pass: {test_batch.shape} -> {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Output std: {output.std():.3f}")

        # Train for a few epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\nTraining for 5 epochs...")
        model.train()

        for epoch in range(5):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                if model_name == 'flex':
                    outputs = model.forward_task(batch_x)
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (preds == batch_y).sum().item()

            # Test accuracy
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    if model_name == 'flex':
                        outputs = model.forward_task(batch_x)
                    else:
                        outputs = model(batch_x)

                    _, preds = torch.max(outputs, 1)
                    test_total += batch_y.size(0)
                    test_correct += (preds == batch_y).sum().item()

            model.train()

            train_acc = correct / total
            test_acc = test_correct / test_total
            avg_loss = epoch_loss / len(train_loader)

            print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}, Loss={avg_loss:.4f}")

        results[model_name] = {
            'final_test_acc': test_acc,
            'final_train_acc': train_acc,
            'final_loss': avg_loss
        }

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    simple_acc = results['simple']['final_test_acc']
    flex_acc = results['flex']['final_test_acc']
    difference = simple_acc - flex_acc

    print(f"Simple Working Model:     {simple_acc:.1%}")
    print(f"FLEX-Persona ClientModel: {flex_acc:.1%}")
    print(f"Difference:               {difference:+.1%}")

    if abs(difference) < 0.05:
        print("\n[SIMILAR] Both models perform similarly")
        print("-> Issue may be in federated training loop or data handling")
    elif simple_acc > flex_acc:
        print(f"\n[ISSUE FOUND] Simple model significantly outperforms FLEX-Persona")
        print("-> Issue is in FLEX-Persona model architecture or forward pass")
    else:
        print(f"\n[UNEXPECTED] FLEX-Persona outperforms simple model")

    print("\nArchitectural differences to investigate:")
    print("1. FLEX-Persona uses Backbone → Adapter → Classifier pipeline")
    print("2. Simple model uses direct Conv → FC pipeline")
    print("3. FLEX-Persona may have initialization issues")
    print("4. FLEX-Persona shared space projection may hurt performance")

if __name__ == "__main__":
    compare_model_architectures()