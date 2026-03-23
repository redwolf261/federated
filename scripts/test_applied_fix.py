#!/usr/bin/env python3
"""Test FLEX-Persona with the applied architectural fix."""

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

def test_fixed_flex_persona_system():
    """Test the complete FLEX-Persona system with the architectural fix applied."""
    print("="*70)
    print("TESTING FLEX-PERSONA WITH APPLIED ARCHITECTURAL FIX")
    print("="*70)

    # Create config
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62
    config.num_clients = 1  # Single client test
    config.training.max_samples_per_client = 1000  # Reasonable test size
    config.training.learning_rate = 0.001
    config.training.batch_size = 32

    print(f"Configuration:")
    print(f"  - Dataset: {config.dataset_name}")
    print(f"  - Classes: {config.model.num_classes}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Max samples per client: {config.training.max_samples_per_client}")
    print()

    # Create client data
    print("Loading client data...")
    manager = ClientDataManager(project_root, config)
    bundles = manager.build_client_bundles()
    client_bundle = bundles[0]

    print(f"Client data loaded:")
    print(f"  - Samples: {client_bundle.num_samples}")
    print(f"  - Classes: {len(client_bundle.class_histogram)}")
    print(f"  - Train batches: {len(client_bundle.train_loader)}")
    print(f"  - Eval batches: {len(client_bundle.eval_loader)}")
    print()

    # Create model using ModelFactory (now with fixed backbone)
    print("Creating FLEX-Persona model with fixed backbone...")
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    # Verify the fix was applied
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Backbone type: {type(model.backbone).__name__}")
    print(f"  - Backbone output dim: {model.backbone.output_dim}")
    print(f"  - Adapter: {model.adapter.input_dim} -> {model.adapter.shared_dim}")
    print(f"  - Classifier: {model.classifier.in_features} -> {model.classifier.out_features}")

    # Verify the fix worked
    if model.backbone.output_dim == 6272:
        print("  ✅ Backbone fix successfully applied!")
    else:
        print(f"  ❌ Backbone fix not applied (expected 6272, got {model.backbone.output_dim})")
        return

    print()

    # Test forward pass
    print("Testing forward pass...")
    test_batch_x, test_batch_y = next(iter(client_bundle.train_loader))
    print(f"Input shape: {test_batch_x.shape}")

    with torch.no_grad():
        # Test each component
        features = model.extract_features(test_batch_x)
        shared_repr = model.project_shared(features)
        task_output = model.forward_task(test_batch_x)

        print(f"Backbone features: {features.shape}")
        print(f"  Range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Std: {features.std():.3f}")

        print(f"Shared representation: {shared_repr.shape}")
        print(f"  Range: [{shared_repr.min():.3f}, {shared_repr.max():.3f}]")
        print(f"  Std: {shared_repr.std():.3f}")

        print(f"Task output: {task_output.shape}")
        print(f"  Range: [{task_output.min():.3f}, {task_output.max():.3f}]")
        print(f"  Std: {task_output.std():.3f}")

        # Check for issues
        if torch.isnan(features).any() or torch.isnan(task_output).any():
            print("  ❌ NaN values detected in forward pass!")
        else:
            print("  ✅ Forward pass clean (no NaN values)")

    print()

    # Training test
    print("Training test (5 epochs)...")
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epoch_results = []

    for epoch in range(5):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in client_bundle.train_loader:
            optimizer.zero_grad()

            logits = model.forward_task(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Check gradient norms
            total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        # Evaluation
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

        epoch_results.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'eval_acc': eval_acc,
            'loss': avg_loss,
            'grad_norm': total_grad_norm
        })

        print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Eval={eval_acc:.4f}, Loss={avg_loss:.4f}, GradNorm={total_grad_norm:.2f}")

    # Final analysis
    final_eval_acc = epoch_results[-1]['eval_acc']
    initial_eval_acc = epoch_results[0]['eval_acc']
    improvement = final_eval_acc - initial_eval_acc

    print(f"\n{'='*70}")
    print("PERFORMANCE ANALYSIS")
    print('='*70)

    print(f"Final evaluation accuracy: {final_eval_acc:.4f} ({final_eval_acc:.1%})")
    print(f"Improvement over epochs: {improvement:+.4f} ({improvement:+.1%})")
    print()

    if final_eval_acc > 0.4:
        print("[SUCCESS] >40% accuracy - Architectural fix successful!")
        print("✅ FLEX-Persona now achieves reasonable FEMNIST performance")
        print("✅ Ready for full federated learning experiments")
    elif final_eval_acc > 0.2:
        print("[GOOD] 20-40% accuracy - Major improvement, may need hyperparameter tuning")
    elif final_eval_acc > 0.1:
        print("[IMPROVED] 10-20% accuracy - Some improvement but issues remain")
    else:
        print("[POOR] <10% accuracy - Additional issues need investigation")

    print(f"\nComparison to previous performance:")
    print(f"  - Previous FLEX-Persona: ~7% (broken)")
    print(f"  - Fixed FLEX-Persona: {final_eval_acc:.1%}")
    print(f"  - Expected with proper federated training: 70%+")
    print()

    # Learning diagnostics
    print("Learning diagnostics:")
    is_learning = improvement > 0.05  # >5% improvement
    gradients_healthy = all(r['grad_norm'] > 0.01 and r['grad_norm'] < 100 for r in epoch_results)
    loss_decreasing = epoch_results[-1]['loss'] < epoch_results[0]['loss']

    print(f"  - Learning observed: {'✅' if is_learning else '❌'} ({improvement:+.1%} improvement)")
    print(f"  - Gradient flow: {'✅' if gradients_healthy else '❌'}")
    print(f"  - Loss decreasing: {'✅' if loss_decreasing else '❌'}")

    if final_eval_acc > 0.2 and is_learning:
        print(f"\n🎉 SUCCESS: Architectural fix resolves the FLEX-Persona performance issue!")
        print(f"📈 Ready to proceed with full federated learning experiments")
    elif final_eval_acc > 0.1:
        print(f"\n📈 PROGRESS: Significant improvement but may need additional tuning")
    else:
        print(f"\n🔍 INVESTIGATE: Additional issues may remain in the implementation")

if __name__ == "__main__":
    test_fixed_flex_persona_system()