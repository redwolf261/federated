"""
RAPID TEST: Model State Copying Impact on Optimizer
=================================================

Quick test of the hypothesis: load_state_dict() disrupts optimizer tracking
Tests: Federated training with vs without model state copying
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class DebugModel(nn.Module):
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


def rapid_sync_test():
    """Rapid test of model synchronization impact"""

    print("RAPID SYNC TEST: Model State Copying vs No Copying")
    print("="*50)

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1000)  # Smaller for speed
    images = artifact.payload["images"][:1000]
    labels = artifact.payload["labels"][:1000]

    # Quick train/test split
    train_size = int(0.8 * len(images))
    train_images, train_labels = images[:train_size], labels[:train_size]
    test_images, test_labels = images[train_size:], labels[train_size:]

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=64, shuffle=False)

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test")

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

    # TEST 1: No model copying (direct training)
    print("\nTest 1: No Model Copying (Direct Training)")
    torch.manual_seed(42)

    factory = ImprovedModelFactory()
    backbone1 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model1 = DebugModel(backbone1, 62)
    optimizer1 = optim.Adam(model1.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for round_num in range(5):  # 5 rounds for speed
        model1.train()
        for local_epoch in range(3):  # 3 epochs per round
            for batch_x, batch_y in train_loader:
                optimizer1.zero_grad()
                outputs = model1(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer1.step()
        acc = evaluate(model1, test_loader)
        print(f"  Round {round_num+1}: {acc:.3f}")

    final_acc_no_copy = acc

    # TEST 2: With model copying (current federated approach)
    print("\nTest 2: With Model Copying (Current Federated)")
    torch.manual_seed(42)

    backbone2 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = DebugModel(backbone2, 62)

    backbone3 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    client_model = DebugModel(backbone3, 62)
    client_model.load_state_dict(global_model.state_dict())

    optimizer2 = optim.Adam(client_model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)

    for round_num in range(5):
        client_model.train()
        for local_epoch in range(3):
            for batch_x, batch_y in train_loader:
                optimizer2.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer2.step()

        # Model synchronization (the suspected issue)
        global_model.load_state_dict(client_model.state_dict())

        acc = evaluate(global_model, test_loader)
        print(f"  Round {round_num+1}: {acc:.3f}")

    final_acc_with_copy = acc

    # RESULTS
    print("\n" + "="*50)
    print("RAPID SYNC TEST RESULTS")
    print("="*50)
    print(f"No model copying:    {final_acc_no_copy:.1%}")
    print(f"With model copying:  {final_acc_with_copy:.1%}")
    gap = final_acc_no_copy - final_acc_with_copy
    print(f"Performance gap:     {gap:+.1%}")

    if abs(gap) < 0.02:
        print("\nCONCLUSION: Model copying is NOT the issue")
        print("NEXT: Investigate other federated training differences")
    elif gap > 0.02:
        print("\nCONCLUSION: Model copying DEGRADES performance")
        print("FIX: Eliminate unnecessary model synchronization")
    else:
        print("\nCONCLUSION: Model copying IMPROVES performance")
        print("INVESTIGATE: Why synchronization helps")

    return final_acc_no_copy, final_acc_with_copy


if __name__ == "__main__":
    rapid_sync_test()