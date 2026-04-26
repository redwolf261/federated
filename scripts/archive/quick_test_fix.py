#!/usr/bin/env python3
"""Quick test of fixed FLEX-Persona (Unicode-safe version)."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.model_factory import ModelFactory
from flex_persona.data.client_data_manager import ClientDataManager

def quick_test_fixed_system():
    """Quick test of the fixed FLEX-Persona system."""
    print("="*60)
    print("QUICK TEST: FIXED FLEX-PERSONA SYSTEM")
    print("="*60)

    # Create config
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62
    config.num_clients = 1
    config.training.max_samples_per_client = 500  # Small for speed
    config.training.learning_rate = 0.001
    config.training.batch_size = 32

    # Create client data
    manager = ClientDataManager(project_root, config)
    bundles = manager.build_client_bundles()
    client_bundle = bundles[0]

    print(f"Client data: {client_bundle.num_samples} samples, {len(client_bundle.class_histogram)} classes")

    # Create model
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Backbone output: {model.backbone.output_dim}")

    # Verify fix worked
    if model.backbone.output_dim == 6272:
        print("[SUCCESS] Backbone fix applied correctly!")
    else:
        print(f"[ERROR] Fix not applied (got {model.backbone.output_dim}, expected 6272)")
        return

    # Quick training test
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining test (3 epochs):")
    model.train()

    for epoch in range(3):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in client_bundle.train_loader:
            optimizer.zero_grad()
            logits = model.forward_task(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        # Evaluate
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch_x, batch_y in client_bundle.eval_loader:
                logits = model.forward_task(batch_x)
                preds = torch.argmax(logits, dim=1)
                eval_correct += (preds == batch_y).sum().item()
                eval_total += batch_y.size(0)
        model.train()

        train_acc = correct / total
        eval_acc = eval_correct / eval_total
        avg_loss = epoch_loss / len(client_bundle.train_loader)

        print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Eval={eval_acc:.4f}, Loss={avg_loss:.4f}")

    print(f"\nFinal Results:")
    print(f"  Evaluation accuracy: {eval_acc:.4f} ({eval_acc:.1%})")

    if eval_acc > 0.3:
        print("  [EXCELLENT] >30% - Fix successful!")
        print("  -> FLEX-Persona architecture issues resolved")
        print("  -> Ready for full federated experiments")
    elif eval_acc > 0.15:
        print("  [GOOD] 15-30% - Major improvement over broken 7%")
        print("  -> Significant progress, may need hyperparameter tuning")
    elif eval_acc > 0.07:
        print("  [SOME PROGRESS] 7-15% - Improvement but issues remain")
    else:
        print("  [STILL ISSUES] <7% - Additional problems need fixing")

    print(f"\nComparison:")
    print(f"  Previous (broken): ~7%")
    print(f"  Fixed system: {eval_acc:.1%}")
    print(f"  Expected federated: 70%+")

if __name__ == "__main__":
    quick_test_fixed_system()