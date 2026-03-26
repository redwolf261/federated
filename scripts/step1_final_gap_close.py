"""
STEP 1 FINAL: Close the Gap to 76% Target
==========================================

Quick test with extended training to reach centralized performance
in 1-client federated setting.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class FinalModel(nn.Module):
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


def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def final_1client_test():
    """Final 1-client test with extended training"""

    print("STEP 1 FINAL: Closing Gap to 76% Target")
    print("=" * 60)

    # Load config
    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    # Load data (same as centralized success)
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Train/test split
    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    train_loader = DataLoader(TensorDataset(train_images, train_labels),
                            batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                           batch_size=64, shuffle=False)

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test")

    # Create model
    torch.manual_seed(42)
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = FinalModel(backbone, 62)

    client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    client_model = FinalModel(client_backbone, 62)
    client_model.load_state_dict(global_model.state_dict())

    # FIXED: Persistent optimizer (NOT recreated each round)
    optimizer = optim.Adam(client_model.parameters(),
                          lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Config: LR={config_data['learning_rate']}, Local Epochs={config_data['local_epochs']}")
    print("Fix: Persistent optimizer across rounds")

    # Extended training: 20 rounds instead of 10
    num_rounds = 20
    print(f"Training: {num_rounds} rounds x {config_data['local_epochs']} epochs = {num_rounds * config_data['local_epochs']} total epochs")

    for fl_round in range(num_rounds):
        client_model.train()

        # Local training
        for local_epoch in range(config_data['local_epochs']):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Aggregation (trivial with 1 client)
        global_model.load_state_dict(client_model.state_dict())

        # Evaluate
        acc = evaluate(global_model, test_loader)
        print(f"  Round {fl_round+1:2d}: {acc:.3f}")

    final_acc = evaluate(global_model, test_loader)

    print("\n" + "=" * 60)
    print("STEP 1 FINAL RESULTS")
    print("=" * 60)
    print(f"Final 1-Client FedAvg: {final_acc:.1%}")
    print(f"Centralized target:    76.0%")
    print(f"Gap:                   {0.76 - final_acc:+.1%}")

    if final_acc >= 0.70:
        print("\nVERDICT: TRAINING PIPELINE VALIDATED")
        print("1-Client FedAvg >= 70% threshold")
        print("READY FOR STEP 2: Multi-client federation debugging")
        return final_acc, True
    else:
        print("\nVERDICT: Additional training improvements needed")
        print("Continue investigating federated training efficiency")
        return final_acc, False


if __name__ == "__main__":
    accuracy, success = final_1client_test()

    print(f"\nSTEP 1 COMPLETE: {accuracy:.1%}")
    if success:
        print("Status: READY for Step 2 (Multi-client testing)")
    else:
        print("Status: Need more training optimization")