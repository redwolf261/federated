"""
Quick Training Validation: Test promising configuration while full debug runs
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class QuickValidatedModel(nn.Module):
    """Model with proper initialization"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Lower dropout
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Even lower dropout
            nn.Linear(128, num_classes)
        )

        # Proper initialization
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def quick_training_test():
    """Test a promising configuration quickly"""

    print("QUICK TRAINING VALIDATION")
    print("-" * 40)
    print("Testing: Adam, LR=0.003, Batch=64, Low dropout")

    # Load data
    config = ExperimentConfig(dataset_name="femnist")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)

    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Train/val split
    indices = torch.randperm(len(images))
    train_size = int(0.8 * len(images))

    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    val_images = images[indices[train_size:]]
    val_labels = labels[indices[train_size:]]

    # Data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model = QuickValidatedModel(backbone, 62)

    # Optimizer (promising configuration)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")

    best_val_acc = 0

    for epoch in range(25):
        # Training
        model.train()
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or val_acc > 0.5:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:2d}: Train={train_acc:.3f}, Val={val_acc:.3f}, Best={best_val_acc:.3f}, LR={lr:.6f}")

        if val_acc >= 0.7:
            print(f"  SUCCESS: Reached {val_acc:.3f} >= 70% at epoch {epoch+1}")
            return True

    print(f"  FINAL: Best validation = {best_val_acc:.3f}")

    if best_val_acc >= 0.7:
        print("SUCCESS: Configuration achieves research requirement")
        return True
    else:
        print(f"PARTIAL: Reached {best_val_acc:.1%} but need 70%")
        return False


def main():
    print("Quick validation of promising training configuration")
    print("=" * 50)

    success = quick_training_test()

    print()
    if success:
        print("RESULT: Promising configuration VALIDATED")
        print("ACTION: Update Phase 0 with these parameters:")
        print("  - Adam optimizer")
        print("  - Learning rate: 0.003")
        print("  - Batch size: 64")
        print("  - Dropout: 0.2/0.1")
        print("  - Xavier initialization")
    else:
        print("RESULT: Still need to find working configuration")
        print("ACTION: Wait for systematic debug results")

    return success


if __name__ == "__main__":
    main()