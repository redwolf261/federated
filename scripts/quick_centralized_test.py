"""Quick centralized performance test to validate overfitting issue.

This is a fast diagnosis of the 91.7% train vs 50% val overfitting problem
identified in the research review. Tests key architectural variants quickly.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


def quick_overfitting_test():
    """Quick test to validate and fix overfitting issue."""

    print("QUICK CENTRALIZED PERFORMANCE TEST")
    print("="*50)
    print("Diagnosing overfitting: 91.7% train vs 50% val")
    print()

    # Load smaller dataset for quick test
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1000)
    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]

    # Proper train/val split
    dataset = TensorDataset(images, labels)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Data: {train_size} train, {val_size} val")

    # Test current architecture
    factory = ImprovedModelFactory()
    model = factory.build_improved_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
        adapter_type="improved",
        model_type="improved"
    )

    # Check architecture
    total_params = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"\nCurrent Architecture:")
    print(f"  Total params: {total_params:,}")
    print(f"  Classifier params: {classifier_params:,} ({classifier_params/total_params:.1%})")
    print(f"  Backbone -> Classifier: {model.backbone.output_dim} -> {config.model.num_classes}")

    # Training test
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining test (10 epochs):")

    for epoch in range(10):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model.forward_task(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model.forward_task(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        overfitting_gap = train_acc - val_acc

        print(f"  Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, Gap={overfitting_gap:+.4f}")

        # Early warning
        if epoch >= 5 and overfitting_gap > 0.3:
            print(f"    WARNING: Gap > 30% - severe overfitting detected")

    print(f"\nFinal Assessment:")
    print(f"  Final gap: {overfitting_gap:+.4f}")

    if overfitting_gap > 0.4:
        print(f"  CRITICAL: Severe overfitting (>40% gap)")
        print(f"  -> Architecture needs major regularization")
    elif overfitting_gap > 0.2:
        print(f"  HIGH: Significant overfitting (>20% gap)")
        print(f"  -> Need regularization improvements")
    elif overfitting_gap > 0.1:
        print(f"  MODERATE: Some overfitting (>10% gap)")
        print(f"  -> Minor regularization needed")
    else:
        print(f"  GOOD: Minimal overfitting (<10% gap)")

    # Architecture assessment
    print(f"\nArchitecture Issues:")

    # Check if classifier is too large
    backbone_to_classes_ratio = model.backbone.output_dim // config.model.num_classes
    print(f"  Backbone-to-classes ratio: {backbone_to_classes_ratio}:1")

    if backbone_to_classes_ratio > 100:
        print(f"    HIGH RISK: Very large feature-to-class ratio")
        print(f"    -> Consider intermediate layers to reduce capacity")

    # Check parameter distribution
    if classifier_params / total_params > 0.7:
        print(f"    HIGH RISK: Classifier dominates parameters")
        print(f"    -> Most capacity in final layer = memorization risk")

    print(f"\nRecommended Fixes:")
    print(f"1. Add intermediate layers with dropout in classifier")
    print(f"2. Increase regularization (weight decay, dropout)")
    print(f"3. Reduce classifier capacity")
    print(f"4. Improve data augmentation")

    return train_acc, val_acc, overfitting_gap


if __name__ == "__main__":
    quick_overfitting_test()