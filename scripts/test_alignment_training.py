"""Test alignment-aware training to validate representation alignment works.

This script validates that the alignment mechanisms properly align adapter and
classifier representations during training, addressing the technical critique
about missing alignment between these components.

Tests:
1. Alignment loss computation and gradients
2. Progressive alignment during training
3. Alignment quality metrics
4. Impact on task performance
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry
from flex_persona.training.alignment_aware_trainer import AlignmentAwareTrainer, AlignmentConfig


def test_alignment_training():
    """Test alignment-aware training functionality."""
    print("="*70)
    print("TESTING ALIGNMENT-AWARE TRAINING")
    print("="*70)

    # Setup
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    # Load test data
    print("Loading test data...")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1000)
    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]

    # Create data loaders
    train_size = 700
    train_dataset = TensorDataset(images[:train_size], labels[:train_size])
    val_dataset = TensorDataset(images[train_size:], labels[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Data: {train_size} train, {len(images) - train_size} validation")
    print()

    # === Test 1: Model Creation and Forward Pass ===
    print("1. MODEL CREATION AND FORWARD PASS")
    print("-" * 40)

    factory = ImprovedModelFactory()

    # Create alignment-aware model
    model = factory.build_improved_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
        adapter_type="alignment_aware",
        model_type="improved"
    )

    print("Model created successfully!")
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"Model info: {info['total_parameters']:,} parameters")
        print(f"Alignment aware: {info['alignment_aware']}")

    # Test forward pass with alignment
    test_batch_x, test_batch_y = next(iter(train_loader))
    print(f"\nTesting forward pass with batch: {test_batch_x.shape}")

    with torch.no_grad():
        logits, alignment_info = model.forward_task_with_alignment(test_batch_x)
        alignment_loss = model.compute_alignment_loss(alignment_info)

        print(f"Forward pass successful:")
        print(f"  Logits: {logits.shape}")
        print(f"  Alignment info keys: {list(alignment_info.keys())}")
        print(f"  Alignment loss: {alignment_loss.item():.6f}")

    print()

    # === Test 2: Alignment Training Setup ===
    print("2. ALIGNMENT TRAINING SETUP")
    print("-" * 40)

    # Create alignment configuration
    alignment_config = AlignmentConfig(
        alignment_weight=0.1,
        alignment_warmup_epochs=3,
        alignment_schedule="linear",
        alignment_target="cosine"
    )

    # Create trainer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = AlignmentAwareTrainer(
        model=model,
        optimizer=optimizer,
        alignment_config=alignment_config
    )

    print(f"Trainer created:")
    print(f"  Supports alignment: {trainer.supports_alignment}")
    print(f"  Alignment config: {alignment_config}")
    print()

    # === Test 3: Training with Alignment ===
    print("3. TRAINING WITH ALIGNMENT (5 epochs)")
    print("-" * 40)

    training_history = []
    validation_history = []

    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5:")

        # Train one epoch
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        training_history.append(train_metrics)
        validation_history.append(val_metrics)

        print(f"  Train: Loss={train_metrics['total_loss']:.4f}, "
              f"Task={train_metrics['task_loss']:.4f}, "
              f"Align={train_metrics['alignment_loss']:.6f}, "
              f"Acc={train_metrics['accuracy']:.4f}")

        print(f"  Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Task={val_metrics['task_loss']:.4f}, "
              f"Align={val_metrics['alignment_loss']:.6f}, "
              f"Acc={val_metrics['accuracy']:.4f}")

        print(f"  Alignment: Weight={train_metrics['alignment_weight']:.4f}, "
              f"Score={train_metrics['alignment_score']:.4f}")

    print()

    # === Test 4: Alignment Analysis ===
    print("4. ALIGNMENT ANALYSIS")
    print("-" * 40)

    alignment_summary = trainer.get_alignment_summary()
    print(f"Alignment Summary:")
    for key, value in alignment_summary.items():
        if key != 'config':
            print(f"  {key}: {value}")

    # Analyze alignment progression
    alignment_losses = [h['alignment_loss'] for h in training_history]
    alignment_scores = [h['alignment_score'] for h in training_history]
    task_accuracies = [h['accuracy'] for h in training_history]

    print(f"\nAlignment Progression:")
    print(f"  Initial alignment loss: {alignment_losses[0]:.6f}")
    print(f"  Final alignment loss: {alignment_losses[-1]:.6f}")
    print(f"  Alignment loss trend: {alignment_summary['alignment_loss_trend']}")

    print(f"\nAlignment Quality:")
    print(f"  Initial alignment score: {alignment_scores[0]:.4f}")
    print(f"  Final alignment score: {alignment_scores[-1]:.4f}")
    improvement = alignment_scores[-1] - alignment_scores[0]
    print(f"  Alignment improvement: {improvement:+.4f}")

    print(f"\nTask Performance:")
    print(f"  Initial accuracy: {task_accuracies[0]:.4f}")
    print(f"  Final accuracy: {task_accuracies[-1]:.4f}")
    acc_improvement = task_accuracies[-1] - task_accuracies[0]
    print(f"  Accuracy improvement: {acc_improvement:+.4f}")

    print()

    # === Test 5: Alignment Evaluation ===
    print("5. ALIGNMENT EVALUATION")
    print("-" * 40)

    # Test final alignment quality
    model.eval()
    with torch.no_grad():
        total_alignment_score = 0
        num_batches = 0

        for batch_x, batch_y in val_loader:
            logits, alignment_info = model.forward_task_with_alignment(batch_x)

            # Compute detailed alignment metrics
            backbone_features = alignment_info['backbone_features']
            shared_repr = alignment_info['shared_repr']

            # Feature similarity analysis
            backbone_mean = backbone_features.mean(dim=0)
            shared_mean = shared_repr.mean(dim=0)

            # Compute alignment score
            if 'adapter_alignment_features' in alignment_info:
                adapter_align = alignment_info['adapter_alignment_features']
                backbone_align = alignment_info['backbone_alignment_features']

                # Normalized cosine similarity
                adapter_norm = torch.nn.functional.normalize(adapter_align, p=2, dim=1)
                backbone_norm = torch.nn.functional.normalize(backbone_align, p=2, dim=1)
                batch_alignment = torch.sum(adapter_norm * backbone_norm, dim=1).mean()
                total_alignment_score += batch_alignment.item()
                num_batches += 1

        avg_alignment_score = total_alignment_score / num_batches if num_batches > 0 else 0

        print(f"Final alignment evaluation:")
        print(f"  Average alignment score: {avg_alignment_score:.4f}")
        print(f"  Backbone feature dim: {backbone_features.shape[1]}")
        print(f"  Shared representation dim: {shared_repr.shape[1]}")

        # Compression analysis
        compression_ratio = backbone_features.shape[1] / shared_repr.shape[1]
        print(f"  Compression ratio: {compression_ratio:.1f}x")

    # === Results Summary ===
    print(f"\n{'='*70}")
    print("ALIGNMENT TRAINING RESULTS")
    print('='*70)

    alignment_working = alignment_losses[-1] < alignment_losses[0]  # Loss should decrease
    alignment_improving = alignment_scores[-1] > alignment_scores[0]  # Score should increase
    task_maintained = task_accuracies[-1] >= task_accuracies[0] * 0.95  # Task performance maintained

    print(f"✅ Alignment Loss Decreasing: {alignment_working}")
    print(f"✅ Alignment Score Improving: {alignment_improving}")
    print(f"✅ Task Performance Maintained: {task_maintained}")

    if alignment_working and alignment_improving and task_maintained:
        print(f"\n🎉 SUCCESS: Alignment training is working correctly!")
        print(f"📈 Adapter and classifier representations are being aligned")
        print(f"🔧 Architectural fix addresses the alignment critique")
    elif alignment_working and task_maintained:
        print(f"\n📈 PARTIAL SUCCESS: Alignment loss decreasing, task maintained")
        print(f"🔍 May need hyperparameter tuning for optimal alignment")
    else:
        print(f"\n🔧 NEEDS TUNING: Alignment training needs adjustment")
        print(f"🔍 Check alignment weights, learning rates, or model architecture")

    print(f"\nKey Metrics:")
    print(f"- Final alignment loss: {alignment_losses[-1]:.6f}")
    print(f"- Final alignment score: {alignment_scores[-1]:.4f}")
    print(f"- Final task accuracy: {task_accuracies[-1]:.4f}")
    print(f"- Compression ratio: {compression_ratio:.1f}x (vs original 98x)")

if __name__ == "__main__":
    test_alignment_training()