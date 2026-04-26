"""
STEP 2 PROPER: Multi-Client Validation with Validated Config
===========================================================

Uses the VALIDATED configuration that achieved 75.5%:
- 2000 samples (1600 train, 400 test)
- 10 rounds × 5 epochs = 50 total epochs
- Batch size: 64
- LR: 0.003
- Adam optimizer (persistent)

Tests: 1-client, 2-clients, 4-clients
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


class ProperModel(nn.Module):
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
    """Proper FedAvg aggregation with BatchNorm handling"""
    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    for key in global_state.keys():
        if global_state[key].is_floating_point():
            global_state[key] = torch.zeros_like(global_state[key])

    for client_model, client_size in zip(client_models, client_sizes):
        client_state = client_model.state_dict()
        weight = client_size / total_size

        for key in global_state.keys():
            if not global_state[key].is_floating_point():
                # Integer tensors: copy from first client
                if client_model == client_models[0]:
                    global_state[key] = client_state[key].clone()
            else:
                # Float tensors: weighted average
                global_state[key] += client_state[key] * weight

    global_model.load_state_dict(global_state)


def proper_multiclient_test(num_clients):
    """Proper multi-client test with validated configuration"""

    print(f"\n{num_clients}-CLIENT TEST (Validated Config)")
    print("=" * 50)

    # Load validated config
    with open('phase0_corrected_config.json', 'r') as f:
        config = json.load(f)

    # Load VALIDATED dataset size (2000 samples)
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Train/test split (80/20)
    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    print(f"Total: {len(train_images)} train, {len(test_images)} test")

    # Split across clients
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
    torch.manual_seed(42)
    factory = ImprovedModelFactory()

    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = ProperModel(global_backbone, 62)

    client_models = []
    client_optimizers = []

    for _ in range(num_clients):
        backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        model = ProperModel(backbone, 62)
        model.load_state_dict(global_model.state_dict())
        client_models.append(model)

        # CRITICAL: Persistent optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: 10 rounds × {config['local_epochs']} epochs = {10 * config['local_epochs']} total")

    # Federated training
    for fl_round in range(10):
        # Client training
        for client_model, client_loader, optimizer in zip(client_models, client_loaders, client_optimizers):
            client_model.train()

            for local_epoch in range(config['local_epochs']):
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
        print(f"  Round {fl_round+1:2d}: {acc:.3f}")

    return evaluate(global_model, test_loader)


def main():
    """Step 2: Proper multi-client validation"""

    print("STEP 2 PROPER: Multi-Client Validation")
    print("=" * 60)
    print("Using VALIDATED configuration (same as 75.5% success)")
    print()

    results = {}

    for num_clients in [1, 2, 4]:
        acc = proper_multiclient_test(num_clients)
        results[num_clients] = acc

    # Analysis
    print("\n" + "=" * 60)
    print("STEP 2 RESULTS: Multi-Client Aggregation")
    print("=" * 60)

    for num_clients, acc in results.items():
        print(f"{num_clients} client(s): {acc:.1%}")

    # Check degradation
    baseline = results[1]
    deg_2 = baseline - results[2]
    deg_4 = baseline - results[4]

    print(f"\nPerformance vs 1-client baseline:")
    print(f"  2 clients: {deg_2:+.1%}")
    print(f"  4 clients: {deg_4:+.1%}")

    max_deg = max(abs(deg_2), abs(deg_4))

    if max_deg < 0.05:
        print("\nVERDICT: Multi-client aggregation WORKS")
        print("READY FOR PHASE 2: Research validation")
        return True
    else:
        print("\nVERDICT: Multi-client aggregation has issues")
        print(f"Max degradation: {max_deg:.1%} (threshold: 5%)")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nSTEP 2 STATUS: {'VALIDATED' if success else 'NEEDS DEBUG'}")
