"""
MINIMAL COMPARISON: Centralized vs Federated (Exact Same Setup)
============================================================

Isolates the exact difference between centralized and federated training
using identical models, data, and hyperparameters.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class SimpleDebugModel(nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        backbone_dim = backbone.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def create_identical_data():
    """Create identical data setup for both tests"""

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1500)  # Medium size for speed
    images = artifact.payload["images"][:1500]
    labels = artifact.payload["labels"][:1500]

    # Fixed split
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=64, shuffle=False)

    return train_loader, test_loader, config_data


def quick_evaluate(model, test_loader):
    """Quick evaluation function"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def test_centralized_reference():
    """Reference centralized training"""

    print("CENTRALIZED REFERENCE TEST")

    train_loader, test_loader, config_data = create_identical_data()

    torch.manual_seed(42)  # Fixed seed
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model = SimpleDebugModel(backbone, 62)

    optimizer = optim.Adam(model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Config: LR={config_data['learning_rate']}, BatchSize=64")

    # Train for 25 epochs (5 rounds × 5 epochs equivalent)
    model.train()
    for epoch in range(25):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate every 5 epochs (equivalent to federated rounds)
        if (epoch + 1) % 5 == 0:
            acc = quick_evaluate(model, test_loader)
            print(f"  Epoch {epoch+1:2d}: {acc:.3f}")

    final_acc = quick_evaluate(model, test_loader)
    print(f"Final centralized: {final_acc:.3f}")
    return final_acc


def test_federated_current():
    """Current federated approach (with fixed optimizer)"""

    print("\nFEDERATED CURRENT TEST")

    train_loader, test_loader, config_data = create_identical_data()

    torch.manual_seed(42)  # Identical seed
    factory = ImprovedModelFactory()

    # Global model
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = SimpleDebugModel(global_backbone, 62)

    # Client model
    client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    client_model = SimpleDebugModel(client_backbone, 62)
    client_model.load_state_dict(global_model.state_dict())

    # Fixed optimizer (no recreation bug)
    optimizer = optim.Adam(client_model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Config: LR={config_data['learning_rate']}, BatchSize=64")

    # 5 federated rounds × 5 local epochs = 25 total epochs
    for fl_round in range(5):
        client_model.train()

        # 5 local epochs
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Aggregation (trivial with 1 client)
        global_model.load_state_dict(client_model.state_dict())

        # Evaluate
        acc = quick_evaluate(global_model, test_loader)
        print(f"  Round {fl_round+1:2d}: {acc:.3f}")

    final_acc = quick_evaluate(global_model, test_loader)
    print(f"Final federated: {final_acc:.3f}")
    return final_acc


def main():
    """Minimal comparison test"""

    print("MINIMAL COMPARISON: CENTRALIZED VS FEDERATED")
    print("="*50)
    print("Purpose: Isolate exact difference with identical setup")
    print("Training: 25 epochs total (5×5 for federated)")
    print()

    start_time = time.time()

    # Run both tests
    centralized_acc = test_centralized_reference()
    federated_acc = test_federated_current()

    elapsed = time.time() - start_time

    # Analysis
    print("\n" + "="*50)
    print("MINIMAL COMPARISON RESULTS")
    print("="*50)
    print(f"Centralized performance: {centralized_acc:.1%}")
    print(f"Federated performance:    {federated_acc:.1%}")

    gap = centralized_acc - federated_acc
    print(f"Performance gap:         {gap:+.1%}")
    print(f"Test duration:           {elapsed:.1f}s")

    if abs(gap) < 0.02:
        print("\nCONCLUSION: No significant difference - federated setup works correctly")
        print("NEXT: Original 59.5% result may need longer training or different parameters")
    elif gap > 0.05:
        print("\nCONCLUSION: Significant federated performance loss")
        print("ROOT CAUSE: Federated structure itself degrades learning")
    elif gap > 0.02:
        print("\nCONCLUSION: Moderate federated performance loss")
        print("ROOT CAUSE: Subtle differences in federated training mechanics")
    else:
        print("\nCONCLUSION: Federated performs better")
        print("INVESTIGATION: Why federated structure helps")


if __name__ == "__main__":
    main()