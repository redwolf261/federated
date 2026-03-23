#!/usr/bin/env python3
"""A/B test: Compare 100-class vs 62-class FEMNIST performance directly."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.models.model_factory import ModelFactory

def compare_class_configs():
    """Direct A/B test of 100-class vs 62-class configuration."""
    print("="*70)
    print("A/B TEST: 100-CLASS vs 62-CLASS FEMNIST CONFIGURATION")
    print("="*70)

    results = {}

    for num_classes, label in [(100, "OLD (BROKEN)"), (62, "NEW (FIXED)")]:
        print(f"\n{label} Configuration - {num_classes} classes:")
        print("-" * 50)

        # Create config
        config = ExperimentConfig(dataset_name="femnist")
        config.model.num_classes = num_classes
        config.num_clients = 1
        config.training.max_samples_per_client = 500  # Small for speed
        config.training.learning_rate = 0.005  # Conservative
        config.training.batch_size = 32

        # Load data (same for both)
        manager = ClientDataManager(project_root, config)
        bundles = manager.build_client_bundles()
        bundle = bundles[0]

        # Create model
        model = ModelFactory.build_client_model(
            client_id=0,
            model_config=config.model,
            dataset_name=config.dataset_name,
        )

        print(f"  Model classifier: {model.classifier.in_features} -> {model.classifier.out_features}")
        print(f"  Data classes: {len(bundle.class_histogram)} unique classes (0-{max(bundle.class_histogram.keys())})")

        # Quick training test (3 epochs)
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        accuracies = []

        for epoch in range(3):
            epoch_correct = 0
            epoch_total = 0

            for batch_x, batch_y in bundle.train_loader:
                optimizer.zero_grad()

                try:
                    logits = model.forward_task(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    # Track accuracy
                    preds = torch.argmax(logits, dim=1)
                    epoch_correct += (preds == batch_y).sum().item()
                    epoch_total += batch_y.size(0)

                except Exception as e:
                    print(f"    ERROR in epoch {epoch+1}: {e}")
                    break

            if epoch_total > 0:
                acc = epoch_correct / epoch_total
                accuracies.append(acc)
                print(f"    Epoch {epoch+1}: {acc:.4f} ({acc:.1%})")

        # Final evaluation
        model.eval()
        eval_correct = 0
        eval_total = 0

        with torch.no_grad():
            for batch_x, batch_y in bundle.eval_loader:
                try:
                    logits = model.forward_task(batch_x)
                    preds = torch.argmax(logits, dim=1)
                    eval_correct += (preds == batch_y).sum().item()
                    eval_total += batch_y.size(0)
                except Exception as e:
                    print(f"    ERROR in evaluation: {e}")
                    break

        final_acc = eval_correct / eval_total if eval_total > 0 else 0
        print(f"  Final accuracy: {final_acc:.4f} ({final_acc:.1%})")

        results[num_classes] = {
            'label': label,
            'final_acc': final_acc,
            'training_acc': accuracies[-1] if accuracies else 0,
            'improvement': accuracies[-1] - accuracies[0] if len(accuracies) >= 2 else 0
        }

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    old_acc = results[100]['final_acc']
    new_acc = results[62]['final_acc']
    improvement = new_acc - old_acc

    print(f"100-class (BROKEN):  {old_acc:.1%}")
    print(f"62-class (FIXED):    {new_acc:.1%}")
    print(f"Improvement:         {improvement:+.1%}")

    if improvement > 0.05:  # >5% improvement
        print("\n[SUCCESS] SIGNIFICANT IMPROVEMENT - Class fix is working!")
    elif improvement > 0:
        print("\n[SUCCESS] MINOR IMPROVEMENT - Class fix helps but other issues remain")
    else:
        print("\n[WARNING] NO IMPROVEMENT - Other fundamental issues need fixing")

    print(f"\nNote: Both configs still far below expected 70%+ FEMNIST performance")
    print(f"This suggests additional issues beyond class count mismatch")

if __name__ == "__main__":
    compare_class_configs()