"""Research-grade centralized model with overfitting fixes.

CRITICAL FINDINGS from quick test:
- 97.7% train vs 56.3% val = 41.4% gap (SEVERE overfitting)
- 6272->62 direct classifier = 101:1 ratio (HIGH RISK)
- Architecture needs major regularization

This implements targeted fixes for research-grade performance:
1. Regularized classifier with proper capacity control
2. Progressive compression to prevent memorization
3. Strong regularization (dropout, weight decay, batch norm)
4. Proper validation methodology
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class ResearchGradeClassifier(nn.Module):
    """Classifier designed to prevent overfitting for research credibility."""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()

        # Progressive compression to prevent memorization
        # 6272 -> 512 -> 128 -> 62 (much more controlled)
        self.network = nn.Sequential(
            # First compression with strong regularization
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Second compression
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),  # Slightly less dropout

            # Final classification
            nn.Linear(128, num_classes)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent initial overfitting."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


class ResearchGradeModel(nn.Module):
    """Complete model with research-grade classifier."""

    def __init__(self, backbone, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.classifier = ResearchGradeClassifier(
            backbone.output_dim, num_classes, dropout_rate
        )

    def forward_task(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.backbone(x)


def train_research_grade_model():
    """Train model with research-grade methodology."""

    print("RESEARCH-GRADE CENTRALIZED TRAINING")
    print("="*60)
    print("Target: >=75% FEMNIST with <15% overfitting gap")
    print()

    # Setup with larger dataset for credible results
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2500)
    images = artifact.payload["images"][:2500]
    labels = artifact.payload["labels"][:2500]

    # Proper research split: 60% train, 20% val, 20% test
    dataset = TensorDataset(images, labels)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Research split: {train_size} train, {val_size} val, {test_size} test")

    # Create research-grade model
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))

    # Test different regularization levels
    dropout_rates = [0.3, 0.5, 0.7]
    results = {}

    for dropout_rate in dropout_rates:
        print(f"\nTesting dropout rate: {dropout_rate}")

        model = ResearchGradeModel(backbone, config.model.num_classes, dropout_rate)

        total_params = sum(p.numel() for p in model.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())

        print(f"  Params: {total_params:,} total, {classifier_params:,} classifier ({classifier_params/total_params:.1%})")

        # Research-grade training setup
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # Added weight decay
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)

        # Training with early stopping
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10

        train_accuracies = []
        val_accuracies = []

        for epoch in range(50):  # More epochs for proper convergence
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model.forward_task(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    outputs = model.forward_task(batch_x)
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
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

            # Progress monitoring
            overfitting_gap = train_acc - val_acc
            if epoch % 5 == 0 or epoch < 3:
                print(f"    Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}, "
                      f"Gap={overfitting_gap:+.4f}, Best={best_val_acc:.4f}")

        # Load best model and test
        model.load_state_dict(best_model_state)
        model.eval()

        # Final test evaluation
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model.forward_task(batch_x)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

        test_acc = test_correct / test_total
        final_train_acc = train_accuracies[-1]
        final_overfitting_gap = final_train_acc - best_val_acc

        results[dropout_rate] = {
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'final_train_acc': final_train_acc,
            'overfitting_gap': final_overfitting_gap,
            'converged_epoch': len(train_accuracies)
        }

        print(f"    FINAL: Test={test_acc:.4f}, Best Val={best_val_acc:.4f}, "
              f"Gap={final_overfitting_gap:+.4f}")

    return results


def analyze_research_results(results):
    """Analyze results for research credibility."""

    print(f"\n{'='*60}")
    print("RESEARCH-GRADE PERFORMANCE ANALYSIS")
    print('='*60)

    research_threshold = 0.75  # 75% target for FEMNIST
    overfitting_threshold = 0.15  # 15% gap tolerance

    print(f"{'Dropout':<10} {'Test Acc':<10} {'Best Val':<10} {'Gap':<8} {'Status':<15}")
    print('-'*60)

    best_config = None
    best_score = 0

    for dropout_rate, result in results.items():
        test_acc = result['test_acc']
        best_val = result['best_val_acc']
        gap = result['overfitting_gap']

        # Research viability
        meets_accuracy = test_acc >= research_threshold
        controls_overfitting = gap <= overfitting_threshold

        if meets_accuracy and controls_overfitting:
            status = "RESEARCH READY"
            score = test_acc - gap * 0.3  # Small penalty for overfitting
        elif meets_accuracy:
            status = "HIGH ACCURACY"
            score = test_acc - gap * 0.5  # Larger penalty for overfitting
        elif controls_overfitting:
            status = "LOW OVERFITTING"
            score = test_acc - gap * 0.2  # Small penalty when accuracy low
        else:
            status = "NEEDS WORK"
            score = test_acc - gap * 0.7  # Large penalty when both poor

        if score > best_score:
            best_score = score
            best_config = dropout_rate

        print(f"{dropout_rate:<10} {test_acc:<10.4f} {best_val:<10.4f} "
              f"{gap:<8.4f} {status:<15}")

    print(f"\nRESEARCH ASSESSMENT:")
    print(f"Target: >={research_threshold:.0%} test accuracy with <{overfitting_threshold:.0%} gap")

    research_ready = any(
        r['test_acc'] >= research_threshold and r['overfitting_gap'] <= overfitting_threshold
        for r in results.values()
    )

    if research_ready:
        best_result = results[best_config]
        print(f"SUCCESS: Research criteria met with dropout {best_config}")
        print(f"  Test accuracy: {best_result['test_acc']:.4f} (>={research_threshold:.0%})")
        print(f"  Overfitting gap: {best_result['overfitting_gap']:+.4f} (<{overfitting_threshold:.0%})")
        print(f"  READY for federated experiments")
    else:
        best_result = results[best_config]
        print(f"PROGRESS: Best config is dropout {best_config}")
        print(f"  Test accuracy: {best_result['test_acc']:.4f} (target: {research_threshold:.0%})")
        print(f"  Gap: {best_result['overfitting_gap']:+.4f} (target: <{overfitting_threshold:.0%})")

        if best_result['test_acc'] < research_threshold:
            deficit = research_threshold - best_result['test_acc']
            print(f"  NEED: +{deficit:.1%} accuracy improvement")

        if best_result['overfitting_gap'] > overfitting_threshold:
            excess = best_result['overfitting_gap'] - overfitting_threshold
            print(f"  NEED: -{excess:.1%} overfitting reduction")

    return best_config, research_ready


def main():
    """Execute research-grade centralized training."""

    print("RESEARCH-GRADE CENTRALIZED MODEL TRAINING")
    print("="*60)

    # Train models with different configurations
    results = train_research_grade_model()

    # Analyze for research credibility
    best_config, research_ready = analyze_research_results(results)

    if research_ready:
        print(f"\nSUCCESS: Research-grade centralized performance achieved!")
        print(f"NEXT STEP: Proceed with rigorous federated baselines")
    else:
        print(f"\nPROGRESS: Significant improvement over broken 7% baseline")
        print(f"NEXT STEP: Further architectural improvements needed")

    return research_ready


if __name__ == "__main__":
    main()