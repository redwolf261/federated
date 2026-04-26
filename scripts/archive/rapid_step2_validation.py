"""
RAPID STEP 2: Quick Multi-Client Validation
===========================================

Fast test to validate multi-client aggregation works correctly.
Uses small dataset and few rounds for speed.
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


class SimpleModel(nn.Module):
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


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def fedavg_aggregate(global_model, client_models, client_sizes):
    """Proper FedAvg aggregation"""
    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    for client_model, client_size in zip(client_models, client_sizes):
        client_state = client_model.state_dict()
        weight = client_size / total_size

        for key in global_state.keys():
            if not global_state[key].is_floating_point():
                if client_model == client_models[0]:
                    global_state[key] = client_state[key].clone()
            else:
                global_state[key] += client_state[key].float() * weight

    global_model.load_state_dict(global_state)


def rapid_test(num_clients):
    """Rapid multi-client test"""

    # Load config
    with open('phase0_corrected_config.json', 'r') as f:
        config = json.load(f)

    # Small dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=800)
    images = artifact.payload["images"][:800]
    labels = artifact.payload["labels"][:800]

    # Split
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]

    # Create client data
    samples_per_client = len(train_images) // num_clients
    client_loaders = []
    client_sizes = []

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(train_images)
        client_images = train_images[start:end]
        client_labels = train_labels[start:end]
        loader = DataLoader(TensorDataset(client_images, client_labels),
                          batch_size=32, shuffle=True, drop_last=True)
        client_loaders.append(loader)
        client_sizes.append(len(client_images))

    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                            batch_size=32, shuffle=False)

    # Models
    torch.manual_seed(42)
    factory = ImprovedModelFactory()

    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = SimpleModel(global_backbone, 62)

    client_models = []
    client_optimizers = []
    for _ in range(num_clients):
        backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        model = SimpleModel(backbone, 62)
        model.load_state_dict(global_model.state_dict())
        client_models.append(model)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    # Training (5 rounds only for speed)
    for round_num in range(5):
        # Client training
        for client_model, client_loader, optimizer, client_size in zip(
            client_models, client_loaders, client_optimizers, client_sizes):

            client_model.train()
            for epoch in range(3):  # 3 local epochs
                for batch_x, batch_y in client_loader:
                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # Aggregation
        fedavg_aggregate(global_model, client_models, client_sizes)

        # Update clients
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # Evaluate
        acc = evaluate(global_model, test_loader)
        print(f"  Round {round_num+1}: {acc:.3f}")

    return evaluate(global_model, test_loader)


def main():
    print("RAPID STEP 2: Multi-Client Aggregation Validation")
    print("=" * 60)

    results = {}
    for num_clients in [1, 2, 4]:
        print(f"\n{num_clients}-Client Test:")
        acc = rapid_test(num_clients)
        results[num_clients] = acc
        print(f"Final: {acc:.3f}")

    print("\n" + "=" * 60)
    print("RESULTS:")
    for num_clients, acc in results.items():
        print(f"  {num_clients} client(s): {acc:.1%}")

    baseline = results[1]
    print(f"\nDegradation vs 1-client:")
    print(f"  2 clients: {baseline - results[2]:+.1%}")
    print(f"  4 clients: {baseline - results[4]:+.1%}")

    max_degradation = max(abs(baseline - results[2]), abs(baseline - results[4]))
    if max_degradation < 0.05:
        print("\nVERDICT: Multi-client aggregation VALIDATED")
        print("READY FOR PHASE 2")
        return True
    else:
        print("\nVERDICT: Multi-client aggregation needs debugging")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nStatus: {'READY FOR PHASE 2' if success else 'NEEDS DEBUGGING'}")