"""Centralized training baseline test with detailed metrics.

Validates that the updated training configuration achieves reasonable accuracy.
Tracks the specific metrics needed to diagnose training health:
- Train vs validation curves
- Alignment loss vs task loss
- Alignment score progression
"""


import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

# Force GPU usage for all training
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this experiment. Please run on a machine with a CUDA-capable GPU.")
DEVICE = "cuda"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry
from flex_persona.training.alignment_aware_trainer import AlignmentAwareTrainer, AlignmentConfig


def run_centralized_baseline():
    """Run centralized training with new config and collect diagnostic metrics."""

    print("=" * 80)
    print("CENTRALIZED BASELINE TEST WITH UPDATED CONFIG")
    print("=" * 80)
    print()

    # Configuration
    num_epochs = 30
    batch_size = 32
    alignment_warmup_epochs = 15
    alignment_weight = 0.01

    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Alignment warmup: {alignment_warmup_epochs}")
    print(f"  Alignment weight: {alignment_weight}")
    print()

    # Setup
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    # Load test data - use more samples for better baseline
    print("Loading FEMNIST data...")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=5000)  # More data for better validation
    images = artifact.payload["images"][:5000]
    labels = artifact.payload["labels"][:5000]

    # Split: 70% train, 30% val
    train_size = int(0.7 * len(images))
    train_dataset = TensorDataset(images[:train_size], labels[:train_size])
    val_dataset = TensorDataset(images[train_size:], labels[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {len(images) - train_size}")
    print()

    # Create model
    print("Creating alignment-aware model...")
    factory = ImprovedModelFactory()
    model = factory.build_improved_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
        adapter_type="alignment_aware",
        model_type="improved"
    )

    # Create trainer with updated alignment config
    alignment_config = AlignmentConfig(
        alignment_weight=alignment_weight,
        alignment_warmup_epochs=alignment_warmup_epochs,
        alignment_schedule="linear"
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = AlignmentAwareTrainer(
        model=model,
        optimizer=optimizer,
        alignment_config=alignment_config
    )

    print(f"Model created. Alignment-aware: {trainer.supports_alignment}")
    print()

    # Training loop with detailed logging
    print("=" * 80)
    print("TRAINING PROGRESS")
    print("=" * 80)

    training_history = []
    validation_history = []

    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        # Validate
        val_metrics = trainer.validate(val_loader)

        training_history.append(train_metrics)
        validation_history.append(val_metrics)

        # Print every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch < 5:
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  TRAIN: Loss={train_metrics['total_loss']:.4f}, "
                  f"Task={train_metrics['task_loss']:.4f}, "
                  f"Align={train_metrics['alignment_loss']:.6f}, "
                  f"Acc={train_metrics['accuracy']:.4f}")
            print(f"  VAL:   Loss={val_metrics['total_loss']:.4f}, "
                  f"Task={val_metrics['task_loss']:.4f}, "
                  f"Align={val_metrics['alignment_loss']:.6f}, "
                  f"Acc={val_metrics['accuracy']:.4f}")
            print(f"  Align: Weight={train_metrics['alignment_weight']:.5f}, "
                  f"Score={train_metrics['alignment_score']:.4f}")

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()

    # Extract key metrics
    final_train_acc = training_history[-1]['accuracy']
    final_val_acc = validation_history[-1]['accuracy']
    final_align_loss = training_history[-1]['alignment_loss']
    final_task_loss = training_history[-1]['task_loss']
    final_align_score = training_history[-1]['alignment_score']
    final_align_weight = training_history[-1]['alignment_weight']

    print(f"ACCURACY:")
    print(f"  Final train accuracy: {final_train_acc:.4f}")
    print(f"  Final val accuracy:   {final_val_acc:.4f}")
    print(f"  First val accuracy:   {validation_history[0]['accuracy']:.4f}")
    print()

    # Check overfitting
    overfitting_ratio = final_train_acc / (final_val_acc + 1e-8)
    print(f"OVERFITTING CHECK:")
    print(f"  Train/Val ratio: {overfitting_ratio:.2f}x")
    if overfitting_ratio > 1.3:
        print(f"  [!] Possible overfitting detected")
    else:
        print(f"  [OK] Reasonable gap")
    print()

    # Alignment behavior
    align_losses = [h['alignment_loss'] for h in training_history]
    align_scores = [h['alignment_score'] for h in training_history]
    task_losses = [h['task_loss'] for h in training_history]
    train_accs = [h['accuracy'] for h in training_history]
    val_accs = [h['accuracy'] for h in validation_history]

    print(f"ALIGNMENT BEHAVIOR:")
    print(f"  Initial alignment loss: {align_losses[0]:.6f}")
    print(f"  Final alignment loss:   {align_losses[-1]:.6f}")
    print(f"  Alignment loss trend:   {'[down] Decreasing' if align_losses[-1] < align_losses[0] else '[up] Increasing'}")
    print()

    print(f"ALIGNMENT SCORE PROGRESSION:")
    print(f"  Initial score: {align_scores[0]:.4f}")
    print(f"  Final score:   {align_scores[-1]:.4f}")
    print(f"  Trend: {'[up] Improving' if align_scores[-1] > align_scores[0] else '[down] Declining'}")
    print()

    # Loss scale analysis
    print(f"LOSS SCALE ANALYSIS:")
    final_align_to_task_ratio = final_align_loss / (final_task_loss + 1e-8)
    print(f"  Final alignment loss: {final_align_loss:.6f}")
    print(f"  Final task loss:      {final_task_loss:.4f}")
    print(f"  Ratio (align/task):   {final_align_to_task_ratio:.6f}")
    if final_align_to_task_ratio > 0.1:
        print(f"  [!] Alignment loss is large relative to task loss")
    else:
        print(f"  [OK] Alignment loss appropriately small")
    print()

    # Curve patterns
    print(f"CURVE PATTERNS:")
    train_smooth = np.std(np.diff(train_accs[-5:])) < 0.02
    val_smooth = np.std(np.diff(val_accs[-5:])) < 0.02
    print(f"  Train curve smooth (last 5): {train_smooth}")
    print(f"  Val curve smooth (last 5):   {val_smooth}")

    train_increasing = train_accs[-1] > train_accs[0]
    val_increasing = val_accs[-1] > val_accs[0]
    print(f"  Train accuracy increasing:   {train_increasing}")
    print(f"  Val accuracy increasing:     {val_increasing}")
    print()

    # Success criteria
    print(f"=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    print()

    criteria = {
        "Centralized accuracy >= 50%": final_val_acc >= 0.50,
        "Validation curve stable": val_smooth,
        "Alignment does NOT saturate early": final_align_score < 0.95,
        "Loss behaves smoothly": train_smooth and val_smooth,
    }

    passed = sum(1 for v in criteria.values() if v)
    print(f"Passed: {passed}/{len(criteria)}")
    for criterion, result in criteria.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {criterion}")

    print()

    # Recommendations
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    if passed == len(criteria):
        print(f"[OK] SYSTEM IS HEALTHY - PROCEED TO FEDERATED TRAINING")
    else:
        print(f"[!] SYSTEM NEEDS ADJUSTMENT")
        if final_val_acc < 0.30:
            print(f"  -> Accuracy too low (<30%): Increase learning rate or use Adam")
        elif final_val_acc < 0.50 and overfitting_ratio > 1.3:
            print(f"  -> Overfitting: Add dropout (0.3-0.5) or reduce model size")
        if not val_smooth and val_accs[-1] < val_accs[-5]:
            print(f"  -> Instability: Reduce alignment_weight to 0.005 or increase warmup to 20")

    # Save results
    results = {
        "final_train_accuracy": float(final_train_acc),
        "final_val_accuracy": float(final_val_acc),
        "final_alignment_loss": float(final_align_loss),
        "final_alignment_score": float(final_align_score),
        "alignment_warmup_epochs": alignment_warmup_epochs,
        "criteria_passed": passed,
        "total_criteria": len(criteria),
        "system_healthy": passed == len(criteria),
    }

    output_path = project_root / "outputs" / "centralized_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_centralized_baseline()
