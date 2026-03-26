"""
DEEP DIAGNOSTIC: Model State Synchronization in Federated Training
================================================================

Investigates the remaining 16.5% performance gap:
- 59.5% (fixed optimizer) vs 76% (centralized target)

Potential issues:
1. Model state copying disrupts optimizer parameter tracking
2. BatchNorm statistics not properly synchronized
3. Parameter references broken after load_state_dict()
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class DebugFederatedModel(nn.Module):
    """Model for debugging federation issues"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        backbone_dim = backbone.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def create_test_data():
    """Create test data for diagnostic"""

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    indices = torch.randperm(len(images))
    train_size = int(0.8 * len(images))

    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, config_data


def test_centralized_baseline(train_loader, test_loader, config_data):
    """Test centralized training as reference"""

    print("CENTRALIZED BASELINE (Reference)")

    torch.manual_seed(42)

    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model = DebugFederatedModel(backbone, 62)

    optimizer = optim.Adam(model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train for equivalent epochs (10 rounds * 5 local epochs = 50 epochs)
    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"Centralized accuracy: {accuracy:.3f}")
    return accuracy


def test_federated_no_sync(train_loader, test_loader, config_data):
    """Test federated training WITHOUT model state copying"""

    print("\nFEDERATED WITHOUT STATE SYNCHRONIZATION")

    torch.manual_seed(42)

    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    client_model = DebugFederatedModel(backbone, 62)

    optimizer = optim.Adam(client_model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 10 rounds of 5 epochs each (same total as centralized)
    for fl_round in range(10):
        client_model.train()

        # 5 local epochs
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    # Evaluate
    client_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = client_model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"Federated (no sync): {accuracy:.3f}")
    return accuracy


def test_federated_with_sync(train_loader, test_loader, config_data):
    """Test federated training WITH model state copying (current approach)"""

    print("\nFEDERATED WITH STATE SYNCHRONIZATION (Current)")

    torch.manual_seed(42)

    factory = ImprovedModelFactory()

    # Global model
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = DebugFederatedModel(global_backbone, 62)

    # Client model
    client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    client_model = DebugFederatedModel(client_backbone, 62)
    client_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(client_model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for fl_round in range(10):
        client_model.train()

        # 5 local epochs
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Synchronization
        global_model.load_state_dict(client_model.state_dict())
        # NOTE: No client update needed for 1-client case

    # Evaluate
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = global_model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"Federated (with sync): {accuracy:.3f}")
    return accuracy


def test_federated_proper_sync(train_loader, test_loader, config_data):
    """Test federated training with PROPER synchronization handling"""

    print("\nFEDERATED WITH PROPER SYNC (Potential Fix)")

    torch.manual_seed(42)

    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model = DebugFederatedModel(backbone, 62)

    # Single model - no copying (simulate federated structure without state disruption)
    optimizer = optim.Adam(model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for fl_round in range(10):
        model.train()

        # Simulate "client training" phase
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate after each "round"
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total
        print(f"  Round {fl_round+1:2d}: {accuracy:.3f}")

    print(f"Federated (proper): {accuracy:.3f}")
    return accuracy


def main():
    """Comprehensive federated training diagnostics"""

    print("COMPREHENSIVE FEDERATED TRAINING DIAGNOSTICS")
    print("=" * 60)

    train_loader, test_loader, config_data = create_test_data()

    print(f"Dataset: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print(f"Config: LR={config_data['learning_rate']}, Epochs=50 total")
    print()

    # Test different approaches
    centralized_acc = test_centralized_baseline(train_loader, test_loader, config_data)
    federated_no_sync_acc = test_federated_no_sync(train_loader, test_loader, config_data)
    federated_with_sync_acc = test_federated_with_sync(train_loader, test_loader, config_data)
    proper_sync_acc = test_federated_proper_sync(train_loader, test_loader, config_data)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Centralized baseline:     {centralized_acc:.1%}")
    print(f"Federated (no sync):      {federated_no_sync_acc:.1%}")
    print(f"Federated (current sync): {federated_with_sync_acc:.1%}")
    print(f"Federated (proper sync):  {proper_sync_acc:.1%}")

    print(f"\nPerformance gaps:")
    print(f"No sync vs centralized:     {centralized_acc - federated_no_sync_acc:+.1%}")
    print(f"Current sync vs centralized: {centralized_acc - federated_with_sync_acc:+.1%}")
    print(f"Proper sync vs centralized:  {centralized_acc - proper_sync_acc:+.1%}")

    # Diagnosis
    if abs(proper_sync_acc - centralized_acc) < 0.02:  # Within 2%
        print("\nDIAGNOSIS: Model state synchronization disrupts training")
        print("FIX: Eliminate unnecessary model copying in single-client case")
    elif abs(federated_no_sync_acc - centralized_acc) < 0.02:
        print("\nDIAGNOSIS: Federated structure itself is not the issue")
        print("FIX: Optimize synchronization mechanism")
    else:
        print("\nDIAGNOSIS: Fundamental difference in training regimes")
        print("INVESTIGATE: Batch ordering, random seeds, or architecture issues")


if __name__ == "__main__":
    main()