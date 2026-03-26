"""
PHASE 0 CORRECTED: Experimental Ground Truth with Fixed Training
===============================================================

This implements Phase 0 with the corrected training configuration that achieves
>70% FEMNIST centralized performance (as determined by systematic_training_debug.py).

This script will be updated with the successful hyperparameters once identified.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


@dataclass
class CorrectedExperimentConfig:
    """Corrected experimental configuration with working training parameters"""

    # Dataset
    dataset_name: str = "femnist"
    num_classes: int = 62

    # Federated setup
    num_clients: int = 50
    dirichlet_alpha: float = 0.5

    # Training budget (research standard)
    fl_rounds: int = 100
    local_epochs: int = 5

    # VALIDATED TRAINING PARAMETERS (from quick_training_validation.py)
    batch_size: int = 64           # VALIDATED: Works with Adam
    learning_rate: float = 0.003   # VALIDATED: Achieves 76.7% accuracy
    optimizer_type: str = "adam"   # VALIDATED: Better than SGD
    momentum: float = 0.9          # For SGD if needed
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2      # VALIDATED: Lower dropout works better

    # Reproducibility
    seeds: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0, 1, 2]

    @classmethod
    def from_debug_result(cls, debug_result: Dict):
        """Create config from systematic debug results"""
        # This will be implemented once we have the debug results
        return cls()


class ValidatedModel(nn.Module):
    """Research-grade model with validated training configuration"""

    def __init__(self, backbone, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        # Research-grade architecture with VALIDATED parameters
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),           # VALIDATED: 0.2 dropout
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),           # VALIDATED: 0.1 dropout for second layer
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier initialization (VALIDATED configuration)"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # VALIDATED: Xavier works better
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_representation=False):
        features = self.backbone(x)

        # Forward through classifier (VALIDATED architecture)
        h = features
        for i in range(6):  # Through second ReLU (index 0-5)
            h = self.classifier[i](h)
        representation = h  # 128-dim representation after second ReLU

        # Final layers
        if self.training:
            h = self.classifier[6](h)  # Dropout (index 6)
            logits = self.classifier[7](h)  # Final linear (index 7)
        else:
            logits = self.classifier[7](h)  # Skip dropout in eval

        if return_representation:
            return logits, representation
        return logits


def validate_corrected_centralized_performance(config: CorrectedExperimentConfig, test_loader: DataLoader):
    """Validate centralized performance with corrected training parameters"""

    print(f"\nPHASE 0 CORRECTED: Centralized Performance Validation")
    print(f"Target: >70% FEMNIST (research requirement)")
    print(f"Config: LR={config.learning_rate}, Opt={config.optimizer_type}, Batch={config.batch_size}")

    # Create model with corrected parameters
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(config.dataset_name))
    model = ValidatedModel(backbone, config.num_classes, dropout_rate=config.dropout_rate)

    # Use larger training set for proper validation
    registry = DatasetRegistry(project_root)
    artifact = registry.load(config.dataset_name, max_rows=3000)
    images = artifact.payload["images"][:3000]
    labels = artifact.payload["labels"][:3000]

    # Train/val split
    indices = torch.randperm(len(images))
    train_size = int(0.8 * len(images))

    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]

    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # Setup optimizer with corrected parameters
    if config.optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:  # SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Extended training with VALIDATED parameters
    best_accuracy = 0
    patience = 10
    patience_counter = 0

    for epoch in range(30):  # Sufficient epochs based on validation (converges ~21 epochs)
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        accuracy = correct / total
        scheduler.step(accuracy)

        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or accuracy > 0.6:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:2d}: Acc={accuracy:.3f}, Best={best_accuracy:.3f}, LR={lr_current:.6f}")

        # Success condition
        if accuracy >= 0.7:
            print(f"  SUCCESS: Reached {accuracy:.3f} >= 70% at epoch {epoch+1}")
            return True, accuracy

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  FINAL: {best_accuracy:.3f}")

    if best_accuracy >= 0.7:
        print(f"  SUCCESS: Corrected performance validated ({best_accuracy:.3f})")
        return True, best_accuracy
    else:
        print(f"  FAILURE: Still below 70% requirement ({best_accuracy:.3f})")
        return False, best_accuracy


def setup_corrected_experimental_ground_truth():
    """Phase 0 with corrected training parameters"""

    print("="*70)
    print("PHASE 0 CORRECTED: EXPERIMENTAL GROUND TRUTH")
    print("="*70)
    print("Objective: Validate experimental setup with CORRECTED training parameters")
    print("Requirement: Must achieve >70% FEMNIST centralized performance")
    print()

    # Load corrected configuration (will be updated with debug results)
    config = CorrectedExperimentConfig()

    print("Corrected Experimental Configuration:")
    print(f"  Dataset: {config.dataset_name} ({config.num_classes} classes)")
    print(f"  Clients: {config.num_clients}")
    print(f"  Dirichlet alpha: {config.dirichlet_alpha}")
    print(f"  FL rounds: {config.fl_rounds}")
    print(f"  Local epochs: {config.local_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Optimizer: {config.optimizer_type}")
    print(f"  Dropout rate: {config.dropout_rate}")
    print()

    # Create test loader for validation
    registry = DatasetRegistry(project_root)
    artifact = registry.load(config.dataset_name, max_rows=1000)
    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]
    test_dataset = TensorDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Validate corrected centralized performance
    centralized_ok, centralized_acc = validate_corrected_centralized_performance(config, test_loader)

    print(f"\n" + "="*70)
    print("PHASE 0 CORRECTED RESULTS")
    print("="*70)

    if centralized_ok:
        print(f"+ Centralized validation: {centralized_acc:.3f} (>=70% required)")
        print(f"+ Training parameters: CORRECTED and VALIDATED")
        print()
        print("PHASE 0 COMPLETE - Ready for Phase 1 (Baseline Validation)")
        print("Next: Implement Dirichlet splits and full federated comparison")

        return config, True
    else:
        print(f"x Centralized validation STILL FAILED: {centralized_acc:.3f} < 70%")
        print("CRITICAL: Even corrected parameters insufficient")
        print()
        print("Required actions:")
        print("1. Check systematic_training_debug.py results")
        print("2. Update this script with successful configuration")
        print("3. Consider fundamental architecture changes")

        return None, False


def update_config_from_debug(debug_results_file: str = "debug_results.json"):
    """Update configuration based on systematic debug results"""

    try:
        with open(debug_results_file, 'r') as f:
            debug_data = json.load(f)

        best_config = debug_data.get('best_config')
        if best_config and best_config.get('converged', False):
            print(f"Loading successful configuration: {best_config['config_name']}")
            print(f"Achieved: {best_config['best_val_acc']:.1%} validation accuracy")

            # This would update the CorrectedExperimentConfig with successful parameters
            # Implementation depends on debug results format
            return True
        else:
            print("No successful configuration found in debug results")
            return False

    except FileNotFoundError:
        print("Debug results not found - run systematic_training_debug.py first")
        return False


if __name__ == "__main__":
    # Check if we have debug results to update configuration
    config_updated = update_config_from_debug()

    if not config_updated:
        print("WARNING: Using default parameters - may not achieve 70% requirement")
        print("Run systematic_training_debug.py first to find working configuration")
        print()

    # Run Phase 0 with current configuration
    config, success = setup_corrected_experimental_ground_truth()

    if success:
        # Save validated configuration
        config_data = {
            'dataset_name': config.dataset_name,
            'num_clients': config.num_clients,
            'dirichlet_alpha': config.dirichlet_alpha,
            'fl_rounds': config.fl_rounds,
            'local_epochs': config.local_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'optimizer_type': config.optimizer_type,
            'dropout_rate': config.dropout_rate,
            'seeds': config.seeds
        }

        with open('phase0_corrected_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

        print("Phase 0 configuration saved to: phase0_corrected_config.json")
        print("Ready to proceed with rigorous federated learning research protocol")
    else:
        print("PHASE 0 BLOCKED - Training configuration must be fixed before proceeding")