"""
STEP 2: Multi-Client Federation Debugging
==========================================

With 1-client training validated, test multi-client federation:
- 2 clients: Minimal aggregation test
- 4 clients: Standard federated scenario
- Compare to 1-client baseline

Goal: Identify if bug is in multi-client aggregation logic
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import json
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class FedModel(nn.Module):
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


def create_federated_data(num_clients):
    """Create data split across multiple clients"""

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

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

    # Split training data across clients
    samples_per_client = len(train_images) // num_clients
    client_loaders = []

    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else len(train_images)

        client_images = train_images[start_idx:end_idx]
        client_labels = train_labels[start_idx:end_idx]

        client_dataset = TensorDataset(client_images, client_labels)
        client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True, drop_last=True)
        client_loaders.append(client_loader)

        print(f"  Client {client_id+1}: {len(client_images)} samples")

    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                            batch_size=64, shuffle=False)

    return client_loaders, test_loader, config_data


def fedavg_aggregate(global_model, client_models, client_sizes):
    """FedAvg aggregation: weighted average by client data size"""

    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    # Initialize with zeros
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    # Weighted sum
    for client_model, client_size in zip(client_models, client_sizes):
        client_state = client_model.state_dict()
        weight = client_size / total_size

        for key in global_state.keys():
            # Skip non-float tensors (like num_batches_tracked in BatchNorm)
            if not global_state[key].is_floating_point():
                # For integer tensors, just copy from first client
                if client_model == client_models[0]:
                    global_state[key] = client_state[key].clone()
            else:
                # For float tensors, do weighted average
                global_state[key] += client_state[key].float() * weight

    global_model.load_state_dict(global_state)


def test_multiclient_fedavg(num_clients):
    """Test FedAvg with multiple clients"""

    print(f"\n{'='*60}")
    print(f"TESTING {num_clients}-CLIENT FEDAVG")
    print('='*60)

    # Create federated data
    client_loaders, test_loader, config_data = create_federated_data(num_clients)

    # Initialize models
    torch.manual_seed(42)
    factory = ImprovedModelFactory()

    # Global model
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = FedModel(global_backbone, 62)

    # Client models
    client_models = []
    client_optimizers = []  # FIXED: Persistent optimizers
    for _ in range(num_clients):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = FedModel(client_backbone, 62)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

        # Create persistent optimizer for each client
        optimizer = optim.Adam(client_model.parameters(),
                              lr=config_data['learning_rate'], weight_decay=1e-4)
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: 20 rounds x {config_data['local_epochs']} local epochs")
    print("Fix: Persistent optimizers for all clients")

    # Federated training
    for fl_round in range(20):
        # CLIENT TRAINING (parallel in simulation)
        client_sizes = []

        for client_id, (client_model, client_loader, optimizer) in enumerate(
            zip(client_models, client_loaders, client_optimizers)):

            client_model.train()

            # Local training
            batch_count = 0
            for local_epoch in range(config_data['local_epochs']):
                for batch_x, batch_y in client_loader:
                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    batch_count += len(batch_y)

            client_sizes.append(batch_count)

        # AGGREGATION
        fedavg_aggregate(global_model, client_models, client_sizes)

        # UPDATE CLIENTS with global model
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # EVALUATION
        acc = evaluate(global_model, test_loader)
        print(f"  Round {fl_round+1:2d}: {acc:.3f}")

    final_acc = evaluate(global_model, test_loader)
    print(f"\nFinal {num_clients}-Client FedAvg: {final_acc:.1%}")

    return final_acc


def main():
    """Step 2: Multi-client federation debugging"""

    print("STEP 2: MULTI-CLIENT FEDERATION DEBUGGING")
    print("=" * 60)
    print("Test multi-client aggregation with fixed training pipeline")
    print()

    # Test different client counts
    results = {}

    for num_clients in [1, 2, 4]:
        acc = test_multiclient_fedavg(num_clients)
        results[num_clients] = acc

    # Analysis
    print("\n" + "=" * 60)
    print("STEP 2 ANALYSIS: MULTI-CLIENT RESULTS")
    print("=" * 60)

    for num_clients, acc in results.items():
        print(f"{num_clients} client(s): {acc:.1%}")

    # Check for multi-client degradation
    baseline_1client = results[1]
    degradation_2client = baseline_1client - results[2]
    degradation_4client = baseline_1client - results[4]

    print(f"\nPerformance vs 1-client baseline:")
    print(f"  2 clients: {degradation_2client:+.1%}")
    print(f"  4 clients: {degradation_4client:+.1%}")

    if abs(degradation_2client) < 0.05 and abs(degradation_4client) < 0.05:
        print("\nVERDICT: Multi-client aggregation WORKS")
        print("No significant degradation with multiple clients")
        print("READY FOR PHASE 2: Full research validation")
    else:
        print("\nVERDICT: Multi-client aggregation has issues")
        print("Investigate FedAvg aggregation logic")


if __name__ == "__main__":
    main()