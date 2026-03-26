"""
PHASE 2 FIXED: Multi-Client Federation with Proper Optimizer Handling
=====================================================================

Fixed critical bug: After load_state_dict(), optimizer state must be updated
to track the new parameter objects.

Standard FedAvg approach: Each round, clients start fresh from global model
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FederatedModel(nn.Module):
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


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def fedavg_aggregate(global_model, client_models, client_sizes):
    """FedAvg aggregation with proper handling of all tensor types"""
    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    aggregated = {}
    for key in global_state.keys():
        if global_state[key].is_floating_point():
            aggregated[key] = torch.zeros_like(global_state[key])
        else:
            aggregated[key] = None

    for idx, (client_model, client_size) in enumerate(zip(client_models, client_sizes)):
        client_state = client_model.state_dict()
        weight = client_size / total_size

        for key in global_state.keys():
            if global_state[key].is_floating_point():
                aggregated[key] += client_state[key] * weight
            elif idx == 0:
                aggregated[key] = client_state[key].clone()

    global_model.load_state_dict(aggregated)


def run_fedavg_standard(num_clients, num_rounds=10, local_epochs=5, lr=0.003):
    """
    Standard FedAvg Implementation:
    - Each round: clients receive global model, train locally, send back
    - Optimizer created fresh each round (standard FedAvg behavior)
    - Or: Use SGD without momentum for true FedAvg
    """

    print(f"\n{'='*60}")
    print(f"STANDARD FEDAVG: {num_clients} Clients")
    print('='*60)

    start_time = time.time()

    # Load config
    with open('phase0_corrected_config.json', 'r') as f:
        config = json.load(f)

    # Load dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Fixed seed for reproducibility
    torch.manual_seed(42)

    # Train/test split
    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test")

    # Distribute data
    samples_per_client = len(train_images) // num_clients
    client_data = []

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(train_images)
        client_imgs = train_images[start:end]
        client_lbls = train_labels[start:end]
        print(f"  Client {i+1}: {len(client_imgs)} samples")
        client_data.append((client_imgs, client_lbls, len(client_imgs)))

    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                            batch_size=64, shuffle=False)

    # Initialize global model
    factory = ImprovedModelFactory()
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = FederatedModel(global_backbone, 62).to(device)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: {num_rounds} rounds x {local_epochs} epochs")
    print(f"Using SGD (standard FedAvg)")

    # Federated training
    for fl_round in range(num_rounds):
        round_start = time.time()

        # Create client models for this round (fresh from global)
        client_models = []
        client_sizes = []

        for client_imgs, client_lbls, client_size in client_data:
            # Create new client model initialized from global
            client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
            client_model = FederatedModel(client_backbone, 62).to(device)
            client_model.load_state_dict(global_model.state_dict())

            # Create fresh optimizer for this client's local training
            # Using SGD as in standard FedAvg
            optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)

            # Create data loader
            loader = DataLoader(TensorDataset(client_imgs, client_lbls),
                              batch_size=64, shuffle=True, drop_last=True)

            # Local training
            client_model.train()
            for epoch in range(local_epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            client_models.append(client_model)
            client_sizes.append(client_size)

        # Aggregation
        fedavg_aggregate(global_model, client_models, client_sizes)

        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        round_time = time.time() - round_start

        print(f"  Round {fl_round+1:2d}: {acc:.3f} ({round_time:.1f}s)")

        # Clean up client models
        del client_models

    final_acc = evaluate(global_model, test_loader, device)
    total_time = time.time() - start_time

    print(f"\n  Final: {final_acc:.3f}")
    print(f"  Time:  {total_time:.1f}s")

    return {'final_accuracy': final_acc, 'total_time': total_time}


def run_fedavg_persistent_optimizer(num_clients, num_rounds=10, local_epochs=5, lr=0.003):
    """
    FedAvg with Persistent Optimizers (Alternative approach):
    - Clients maintain their optimizer state across rounds
    - Don't reset to global model each round
    - Only aggregate model weights, not optimizer state
    """

    print(f"\n{'='*60}")
    print(f"FEDAVG PERSISTENT: {num_clients} Clients")
    print('='*60)

    start_time = time.time()

    # Load config
    with open('phase0_corrected_config.json', 'r') as f:
        config = json.load(f)

    # Load dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    torch.manual_seed(42)

    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test")

    # Distribute data
    samples_per_client = len(train_images) // num_clients
    client_loaders = []
    client_sizes = []

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(train_images)
        client_imgs = train_images[start:end]
        client_lbls = train_labels[start:end]
        print(f"  Client {i+1}: {len(client_imgs)} samples")

        loader = DataLoader(TensorDataset(client_imgs, client_lbls),
                          batch_size=64, shuffle=True, drop_last=True)
        client_loaders.append(loader)
        client_sizes.append(len(client_imgs))

    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                            batch_size=64, shuffle=False)

    # Initialize models
    factory = ImprovedModelFactory()
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = FederatedModel(global_backbone, 62).to(device)

    # Create persistent client models and optimizers
    client_models = []
    client_optimizers = []

    for _ in range(num_clients):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = FederatedModel(client_backbone, 62).to(device)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

        # Adam with persistent state
        optimizer = optim.Adam(client_model.parameters(), lr=lr, weight_decay=1e-4)
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: {num_rounds} rounds x {local_epochs} epochs")
    print(f"Using Adam with PERSISTENT optimizer state")

    for fl_round in range(num_rounds):
        round_start = time.time()

        # Client training (NO model reset - continue from local state)
        for client_model, loader, optimizer in zip(client_models, client_loaders, client_optimizers):
            client_model.train()

            for epoch in range(local_epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # Aggregation (update global model only)
        fedavg_aggregate(global_model, client_models, client_sizes)

        # KEY DIFFERENCE: Clients don't reset to global model
        # They continue from their local state (FedProx-like behavior)

        # Evaluate global model
        acc = evaluate(global_model, test_loader, device)
        round_time = time.time() - round_start

        print(f"  Round {fl_round+1:2d}: {acc:.3f} ({round_time:.1f}s)")

    final_acc = evaluate(global_model, test_loader, device)
    total_time = time.time() - start_time

    print(f"\n  Final: {final_acc:.3f}")
    print(f"  Time:  {total_time:.1f}s")

    return {'final_accuracy': final_acc, 'total_time': total_time}


def main():
    """Phase 2: Compare FedAvg implementations"""

    print("PHASE 2 FIXED: MULTI-CLIENT FEDERATION")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    results = {}

    # Test both approaches for 4 clients
    print("\n--- STANDARD FEDAVG (SGD, fresh each round) ---")
    results['standard_4'] = run_fedavg_standard(num_clients=4)

    print("\n--- PERSISTENT OPTIMIZER (Adam, no reset) ---")
    results['persistent_4'] = run_fedavg_persistent_optimizer(num_clients=4)

    # Full test with best approach
    print("\n--- FULL MULTI-CLIENT TEST (Standard FedAvg) ---")
    for num_clients in [1, 2, 4]:
        key = f'standard_{num_clients}'
        results[key] = run_fedavg_standard(num_clients=num_clients)

    # Analysis
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS COMPARISON")
    print("=" * 60)

    print(f"\n{'Config':<25} {'Accuracy':<12} {'Time':<10}")
    print("-" * 50)

    for key, result in results.items():
        print(f"{key:<25} {result['final_accuracy']:.1%}        {result['total_time']:.0f}s")

    # Check 1-client baseline
    if 'standard_1' in results:
        baseline = results['standard_1']['final_accuracy']
        print(f"\n1-client baseline: {baseline:.1%}")

        if 'standard_4' in results:
            deg = baseline - results['standard_4']['final_accuracy']
            print(f"4-client degradation: {deg:+.1%}")

            if abs(deg) < 0.10:
                print("\nVERDICT: Federation logic WORKS")
                print("Ready for research comparison")
            else:
                print("\nVERDICT: Still debugging needed")


if __name__ == "__main__":
    main()