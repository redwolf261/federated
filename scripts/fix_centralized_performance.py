"""Diagnose and fix centralized performance for research-grade validation.

CRITICAL ISSUE: Current 49% FEMNIST performance is 25-30% below research baseline (75-85%).
Major overfitting detected: 91.7% train vs 50% validation.

This script:
1. Diagnoses the overfitting causes
2. Implements architectural fixes for better generalization
3. Validates centralized performance improvements
4. Targets >=75% FEMNIST accuracy for research credibility

Key fixes to implement:
- Reduce classifier overfitting (6272->62 is too large)
- Add proper regularization (dropout, normalization)
- Improve data augmentation and validation
- Test multiple architectural configurations
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Dict, List, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class CentralizedModelVariant(nn.Module):
    """Regularized centralized model variants to fix overfitting."""

    def __init__(self, backbone, num_classes: int, variant: str = "standard"):
        super().__init__()
        self.backbone = backbone
        self.variant = variant

        backbone_dim = backbone.output_dim  # 6272 for fixed SmallCNN

        if variant == "standard":
            # Original problematic architecture
            self.classifier = nn.Linear(backbone_dim, num_classes)

        elif variant == "regularized":
            # Add intermediate layers with dropout
            self.classifier = nn.Sequential(
                nn.Linear(backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        elif variant == "compressed":
            # Strong compression to reduce overfitting
            self.classifier = nn.Sequential(
                nn.Linear(backbone_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        elif variant == "minimal":
            # Minimal parameters to test lower bound
            self.classifier = nn.Sequential(
                nn.Linear(backbone_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


def diagnose_overfitting_sources():
    """Analyze the current architecture to identify overfitting sources."""

    print("="*70)
    print("DIAGNOSING OVERFITTING SOURCES")
    print("="*70)

    # Load model architecture
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    factory = ImprovedModelFactory()

    # Current architecture analysis
    model = factory.build_improved_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
        adapter_type="improved",
        model_type="improved"
    )

    total_params = model.get_model_info()['total_parameters']
    backbone_dim = model.backbone.output_dim
    classifier_params = model.classifier.weight.numel() + model.classifier.bias.numel()

    print(f"Current Architecture Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Backbone output: {backbone_dim} dimensions")
    print(f"  Classifier params: {classifier_params:,}")
    print(f"  Classifier ratio: {classifier_params/total_params:.1%} of total")

    # Problem identification
    print(f"\nOverfitting Risk Analysis:")
    classifier_ratio = classifier_params / total_params

    if classifier_ratio > 0.5:
        print(f"  HIGH RISK: Classifier has {classifier_ratio:.1%} of parameters")
        print(f"  -> Direct 6272->62 mapping creates {backbone_dim * 62:,} parameters")
        print(f"  -> This is prone to memorization, not generalization")

    # Capacity analysis per class
    params_per_class = classifier_params / config.model.num_classes
    print(f"  Parameters per class: {params_per_class:,.0f}")

    if params_per_class > 1000:
        print(f"  HIGH CAPACITY: {params_per_class:,.0f} params/class >> typical samples/class")
        print(f"  -> Model can easily memorize training examples")

    print()
    return backbone_dim, config.model.num_classes


def test_architectural_variants():
    """Test different architectural variants to reduce overfitting."""

    print("TESTING ARCHITECTURAL VARIANTS")
    print("-" * 50)

    # Setup data
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=3000)  # Larger dataset for better validation
    images = artifact.payload["images"][:3000]
    labels = artifact.payload["labels"][:3000]

    # Proper train/val/test split
    dataset = TensorDataset(images, labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")

    # Create backbone
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))

    variants = ["standard", "regularized", "compressed", "minimal"]
    results = {}

    for variant in variants:
        print(f"\nTesting {variant.upper()} variant...")

        # Create model
        model = CentralizedModelVariant(backbone, config.model.num_classes, variant)

        print(f"  Parameters: {model.get_param_count():,}")

        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        max_patience = 8

        train_accuracies = []
        val_accuracies = []

        for epoch in range(30):  # More epochs for convergence
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_y.size(0)

            train_acc = train_correct / train_total
            train_accuracies.append(train_acc)

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_acc = val_correct / val_total
            val_accuracies.append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

            if epoch % 5 == 0 or epoch < 5:
                print(f"    Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}, "
                      f"Gap={train_acc-val_acc:+.4f}")

        # Final test evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

        test_acc = test_correct / test_total
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        overfitting_gap = final_train_acc - final_val_acc

        results[variant] = {
            'train_acc': final_train_acc,
            'val_acc': final_val_acc,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'overfitting_gap': overfitting_gap,
            'parameters': model.get_param_count()
        }

        print(f"    Final: Train={final_train_acc:.4f}, Val={final_val_acc:.4f}, "
              f"Test={test_acc:.4f}")
        print(f"    Best Val: {best_val_acc:.4f}, Gap: {overfitting_gap:+.4f}")

    return results


def analyze_centralized_results(results: Dict):
    """Analyze results and determine best architecture for research baseline."""

    print(f"\n{'='*70}")
    print("CENTRALIZED PERFORMANCE ANALYSIS")
    print('='*70)

    print(f"{'Variant':<12} {'Test Acc':<10} {'Best Val':<10} {'Gap':<8} {'Params':<12} {'Status':<15}")
    print('-'*70)

    research_threshold = 0.75  # 75% target
    acceptable_gap = 0.15      # 15% overfitting tolerance

    best_variant = None
    best_score = 0

    for variant, result in results.items():
        test_acc = result['test_acc']
        gap = result['overfitting_gap']
        params = result['parameters']

        # Research viability assessment
        if test_acc >= research_threshold and gap <= acceptable_gap:
            status = "RESEARCH READY"
            score = test_acc - gap * 0.5  # Penalize overfitting
        elif test_acc >= research_threshold:
            status = "HIGH PERFORMANCE"
            score = test_acc - gap * 0.5
        elif gap <= acceptable_gap:
            status = "LOW OVERFITTING"
            score = test_acc - gap * 0.2
        else:
            status = "NEEDS WORK"
            score = test_acc - gap * 0.7

        if score > best_score:
            best_score = score
            best_variant = variant

        print(f"{variant:<12} {test_acc:<10.4f} {result['best_val_acc']:<10.4f} "
              f"{gap:<8.4f} {params:<12,} {status:<15}")

    print(f"\nRESEARCH ASSESSMENT:")
    print(f"  Target: >={research_threshold:.1%} test accuracy with <{acceptable_gap:.1%} overfitting gap")

    research_ready_variants = [
        v for v, r in results.items()
        if r['test_acc'] >= research_threshold and r['overfitting_gap'] <= acceptable_gap
    ]

    if research_ready_variants:
        print(f"  RESEARCH READY: {', '.join(research_ready_variants)}")
        print(f"  RECOMMENDED: {best_variant} (score: {best_score:.3f})")
    else:
        print(f"  NONE meet research criteria yet")
        print(f"  BEST AVAILABLE: {best_variant} (needs improvement)")

        # Specific recommendations
        best_result = results[best_variant]
        if best_result['test_acc'] < research_threshold:
            deficit = research_threshold - best_result['test_acc']
            print(f"  ACCURACY DEFICIT: {deficit:.1%} - need better architecture/training")

        if best_result['overfitting_gap'] > acceptable_gap:
            excess_gap = best_result['overfitting_gap'] - acceptable_gap
            print(f"  OVERFITTING EXCESS: {excess_gap:.1%} - need more regularization")

    # Next steps
    print(f"\nNEXT STEPS:")
    if not research_ready_variants:
        print(f"  1. Improve best variant ({best_variant}) to meet research criteria")
        print(f"  2. Try stronger regularization (weight decay, dropout)")
        print(f"  3. Experiment with data augmentation")
        print(f"  4. Consider ensemble methods")
    else:
        print(f"  1. Validate {best_variant} architecture in federated setting")
        print(f"  2. Proceed with baseline comparisons")
        print(f"  3. Test clustering contribution")

    return best_variant, research_ready_variants


def main():
    """Main diagnostic and improvement pipeline."""

    print("CENTRALIZED PERFORMANCE IMPROVEMENT PIPELINE")
    print("="*70)
    print("Goal: Achieve >=75% FEMNIST for research credibility")
    print("Current: ~49% (below research baseline)")
    print("Issue: 91.7% train vs 50% val (massive overfitting)")
    print()

    # Step 1: Diagnose current issues
    backbone_dim, num_classes = diagnose_overfitting_sources()

    # Step 2: Test architectural fixes
    results = test_architectural_variants()

    # Step 3: Analyze and recommend best approach
    best_variant, research_ready = analyze_centralized_results(results)

    print(f"\n{'='*70}")
    print("CENTRALIZED IMPROVEMENT COMPLETE")
    print('='*70)

    if research_ready:
        print(f"SUCCESS: Research-grade centralized performance achieved!")
        print(f"Ready to proceed with federated baselines comparison.")
    else:
        print(f"PROGRESS: Best architecture identified, but needs further improvement.")
        print(f"Must reach >=75% before federated experiments are meaningful.")


if __name__ == "__main__":
    main()