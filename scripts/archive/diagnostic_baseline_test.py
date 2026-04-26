#!/usr/bin/env python3
"""
Diagnostic script to identify fundamental issues causing low FEMNIST accuracy.

This script tests core components in isolation to pinpoint the root cause:
1. Data loading and preprocessing
2. Model architecture and forward pass
3. Training loop convergence
4. Hyperparameter sensitivity

Expected FEMNIST performance: 70-85%+ for reasonable federated settings
Current performance: 9-14% (indicates serious implementation bug)
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.models.model_factory import ModelFactory


def test_data_loading():
    """Test 1: Verify data loading and basic statistics."""
    print("\n" + "="*60)
    print("TEST 1: DATA LOADING DIAGNOSTICS")
    print("="*60)

    # Simple config for data loading test
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 3  # Small number for quick test
    config.training.max_samples_per_client = None  # Don't limit data!

    # CRITICAL FIX: Set correct number of classes for FEMNIST
    config.model.num_classes = 62  # FEMNIST has 62 classes, not 100!

    manager = ClientDataManager(project_root, config)
    bundles = manager.build_client_bundles()

    print(f"Number of clients: {len(bundles)}")

    total_samples = 0
    total_classes = set()

    for i, bundle in enumerate(bundles):
        print(f"\nClient {bundle.client_id}:")
        print(f"  - Training samples: {bundle.num_samples}")
        print(f"  - Classes present: {len(bundle.class_histogram)}")
        print(f"  - Class histogram: {dict(sorted(bundle.class_histogram.items()))}")

        total_samples += bundle.num_samples
        total_classes.update(bundle.class_histogram.keys())

        # Test a batch
        batch_x, batch_y = next(iter(bundle.train_loader))
        print(f"  - Batch shape: {batch_x.shape}, labels shape: {batch_y.shape}")
        print(f"  - Data range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"  - Label range: [{batch_y.min()}, {batch_y.max()}]")

        if i >= 2:  # Only show first 3 clients
            break

    print(f"\nOverall Statistics:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Total classes: {len(total_classes)} ({min(total_classes)}-{max(total_classes)})")

    # Check if data looks reasonable
    if total_samples < 1000:
        print("[WARNING] Very few samples - potential data loading issue")
    if len(total_classes) < 10:
        print("[WARNING] Few classes - potential FEMNIST loading issue")

    return bundles[0]  # Return first bundle for further testing


def test_model_architecture(sample_bundle):
    """Test 2: Verify model architecture and forward pass."""
    print("\n" + "="*60)
    print("TEST 2: MODEL ARCHITECTURE DIAGNOSTICS")
    print("="*60)

    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 1

    # CRITICAL FIX: Set correct number of classes for FEMNIST
    config.model.num_classes = 62  # FEMNIST has 62 classes, not 100!

    # Test model creation
    model = ModelFactory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
    )

    print(f"Model architecture:")
    print(f"  - Backbone output dim: {model.backbone.output_dim}")
    print(f"  - Adapter input dim: {model.adapter.input_dim}")
    print(f"  - Adapter shared dim: {model.adapter.shared_dim}")
    print(f"  - Classifier input dim: {model.classifier.in_features}")
    print(f"  - Classifier output dim: {model.classifier.out_features}")

    # Test forward pass
    batch_x, batch_y = next(iter(sample_bundle.train_loader))
    print(f"\nTesting forward pass:")
    print(f"  - Input shape: {batch_x.shape}")

    with torch.no_grad():
        # Test task prediction
        logits = model.forward_task(batch_x)
        print(f"  - Task logits shape: {logits.shape}")
        print(f"  - Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

        # Test shared representation
        shared_repr = model.forward_shared(batch_x)
        print(f"  - Shared repr shape: {shared_repr.shape}")
        print(f"  - Shared repr range: [{shared_repr.min():.3f}, {shared_repr.max():.3f}]")

        # Test predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        print(f"  - Predictions: {preds.tolist()}")
        print(f"  - True labels: {batch_y.tolist()}")

    # Check for reasonable initialization
    if torch.isnan(logits).any():
        print("ERROR: NaN values in forward pass!")
    elif logits.std() < 0.01:
        print("[WARNING] Very small logit variance - potential initialization issue")
    elif logits.std() > 10:
        print("[WARNING] Very large logit variance - potential initialization issue")
    else:
        print("[SUCCESS] Forward pass looks reasonable")

    return model


def test_single_client_training(model, sample_bundle):
    """Test 3: Single client training convergence."""
    print("\n" + "="*60)
    print("TEST 3: SINGLE CLIENT TRAINING CONVERGENCE")
    print("="*60)

    # Test various learning rates
    learning_rates = [0.001, 0.01, 0.1]

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")

        # Fresh model copy
        test_config = ExperimentConfig(dataset_name="femnist")
        test_config.model.num_classes = 62  # CRITICAL FIX: FEMNIST has 62 classes
        test_model = ModelFactory.build_client_model(
            client_id=0,
            model_config=test_config.model,
            dataset_name=test_config.dataset_name,
        )
        optimizer = optim.SGD(test_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Track accuracy over epochs
        accuracies = []
        losses = []

        test_model.train()
        for epoch in range(5):  # Just a few epochs for diagnostic
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0

            for batch_x, batch_y in sample_bundle.train_loader:
                optimizer.zero_grad()

                logits = test_model.forward_task(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                # Track metrics
                preds = torch.argmax(logits, dim=1)
                epoch_correct += (preds == batch_y).sum().item()
                epoch_total += batch_y.size(0)
                epoch_loss += loss.item()

            accuracy = epoch_correct / epoch_total
            avg_loss = epoch_loss / len(sample_bundle.train_loader)

            accuracies.append(accuracy)
            losses.append(avg_loss)

            print(f"  Epoch {epoch+1}: Acc={accuracy:.4f} ({epoch_correct}/{epoch_total}), Loss={avg_loss:.4f}")

        # Analyze convergence
        final_acc = accuracies[-1]
        improvement = accuracies[-1] - accuracies[0]

        print(f"  Final accuracy: {final_acc:.4f}")
        print(f"  Improvement: {improvement:+.4f}")

        if final_acc < 0.2:  # Should get at least 20% on single client
            print(f"  [WARNING] Very low final accuracy - potential issue")
        elif improvement < 0.05:  # Should improve by at least 5%
            print(f"  [WARNING] Little improvement - potential learning issue")
        else:
            print(f"  [SUCCESS] Learning appears functional")

        print()


def test_realistic_federated_config():
    """Test 4: Realistic federated config with proper hyperparameters."""
    print("\n" + "="*60)
    print("TEST 4: REALISTIC FEDERATED CONFIGURATION")
    print("="*60)

    # Create config with reasonable hyperparameters
    config = ExperimentConfig(dataset_name="femnist")
    config.num_clients = 5  # Small for testing
    config.training.max_samples_per_client = None  # Don't limit data!
    config.training.learning_rate = 0.01  # Standard FL learning rate
    config.training.local_epochs = 2  # Reasonable local epochs
    config.training.batch_size = 32  # Reasonable batch size
    config.training.rounds = 3  # Few rounds for testing

    # CRITICAL FIX: Set correct number of classes for FEMNIST
    config.model.num_classes = 62  # FEMNIST has 62 classes, not 100!

    print("Configuration:")
    print(f"  - Clients: {config.num_clients}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Local epochs: {config.training.local_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Rounds: {config.training.rounds}")
    print(f"  - Max samples per client: {config.training.max_samples_per_client}")

    # Load data with this config
    manager = ClientDataManager(project_root, config)
    bundles = manager.build_client_bundles()

    print(f"\nData loading results:")
    for bundle in bundles:
        print(f"  - Client {bundle.client_id}: {bundle.num_samples} samples, {len(bundle.class_histogram)} classes")

    # Test one round of federated learning
    print(f"\nTesting FedAvg-style training (1 round):")

    models = []
    for i in range(config.num_clients):
        model = ModelFactory.build_client_model(
            client_id=i,
            model_config=config.model,
            dataset_name=config.dataset_name,
        )
        models.append(model)

    # Train each client locally
    client_accuracies = []

    for i, (model, bundle) in enumerate(zip(models, bundles)):
        optimizer = optim.SGD(model.parameters(), lr=config.training.learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(config.training.local_epochs):
            for batch_x, batch_y in bundle.train_loader:
                optimizer.zero_grad()
                logits = model.forward_task(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in bundle.eval_loader:
                logits = model.forward_task(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total
        client_accuracies.append(accuracy)
        print(f"  - Client {i}: {accuracy:.4f} ({correct}/{total})")

    mean_acc = sum(client_accuracies) / len(client_accuracies)
    print(f"\nMean client accuracy: {mean_acc:.4f}")

    if mean_acc < 0.3:
        print("[CRITICAL] Very low accuracy suggests fundamental implementation issue!")
    elif mean_acc < 0.5:
        print("[WARNING] Low accuracy - hyperparameter or architecture issues likely")
    else:
        print("[SUCCESS] Reasonable accuracy - implementation basics seem functional")


def main():
    """Run comprehensive diagnostics."""
    print("[DIAGNOSTIC] FLEX-PERSONA BASELINE DIAGNOSTIC SUITE")
    print("="*80)
    print("Expected FEMNIST performance: 70-85%+")
    print("Current observed performance: 9-14% (CRITICAL ISSUE)")
    print("="*80)

    start_time = time.time()

    try:
        # Test 1: Data loading
        sample_bundle = test_data_loading()

        # Test 2: Model architecture
        model = test_model_architecture(sample_bundle)

        # Test 3: Single client training
        test_single_client_training(model, sample_bundle)

        # Test 4: Realistic federated config
        test_realistic_federated_config()

    except Exception as e:
        print(f"\n[ERROR] DIAGNOSTIC FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return

    duration = time.time() - start_time

    print("\n" + "="*80)
    print(f"[COMPLETE] DIAGNOSTIC FINISHED (took {duration:.1f}s)")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the diagnostic output above")
    print("2. Identify and fix any flagged issues")
    print("3. Re-run diagnostics until all tests pass")
    print("4. Then proceed with comparative experiments")


if __name__ == "__main__":
    main()