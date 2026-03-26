"""
STEP 2: MULTI-CLIENT FEDERATION DEBUGGING
=========================================

With 1-client baseline fixed (59.5%), test if multi-client federation
maintains performance or introduces additional issues.

Tests:
- 2-client FedAvg (simplest multi-client scenario)
- 4-client FedAvg (moderate federation)
- Aggregation correctness verification
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import json
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class FederatedModel(nn.Module):
    """Model for federated learning debugging"""

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


def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
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
    """Split data across multiple clients"""

    print(f"Creating {num_clients}-client federated setup")

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Train/test split
    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    # Split training data across clients (IID split)
    samples_per_client = len(train_images) // num_clients
    client_loaders = []

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(train_images)

        client_images = train_images[start_idx:end_idx]
        client_labels = train_labels[start_idx:end_idx]

        client_dataset = TensorDataset(client_images, client_labels)
        client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True, drop_last=True)
        client_loaders.append(client_loader)

        print(f"  Client {i+1}: {len(client_images)} samples, {len(client_loader)} batches")

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Test set: {len(test_images)} samples")

    return client_loaders, test_loader, config_data


def fedavg_aggregate(global_model, client_models, client_weights=None):
    """FedAvg aggregation: weighted average of client models"""

    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)

    global_state = global_model.state_dict()

    # Initialize with zeros
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    # Weighted sum of client parameters
    for client_model, weight in zip(client_models, client_weights):
        client_state = client_model.state_dict()
        for key in global_state:
            global_state[key] += weight * client_state[key].float()

    global_model.load_state_dict(global_state)


def test_multi_client_fedavg(num_clients, num_rounds=10, local_epochs=5):
    """Test FedAvg with multiple clients using FIXED optimizer handling"""

    print(f"\n{'='*60}")
    print(f"MULTI-CLIENT FEDAVG TEST: {num_clients} CLIENTS")
    print(f"{'='*60}")

    client_loaders, test_loader, config_data = create_federated_data(num_clients)

    torch.manual_seed(42)

    # Create global model
    factory = ImprovedModelFactory()
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = FederatedModel(global_backbone, 62)

    # Create client models and PERSISTENT optimizers (CRITICAL FIX)
    client_models = []
    client_optimizers = []

    for i in range(num_clients):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = FederatedModel(client_backbone, 62)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

        # CRITICAL: Create optimizer ONCE per client (persistent across rounds)
        optimizer = optim.Adam(
            client_model.parameters(),
            lr=config_data['learning_rate'],
            weight_decay=1e-4
        )
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: {num_rounds} rounds x {local_epochs} local epochs")
    print(f"FIXED: Persistent optimizers across rounds")

    # Federated training loop
    for fl_round in range(num_rounds):
        round_losses = []

        # CLIENT TRAINING
        for client_idx in range(num_clients):
            client_model = client_models[client_idx]
            optimizer = client_optimizers[client_idx]
            client_loader = client_loaders[client_idx]

            # Sync client with global model
            client_model.load_state_dict(global_model.state_dict())
            client_model.train()

            # Local training
            for local_epoch in range(local_epochs):
                epoch_loss = 0
                for batch_x, batch_y in client_loader:
                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                round_losses.append(epoch_loss / len(client_loader))

        # AGGREGATION
        fedavg_aggregate(global_model, client_models)

        # EVALUATION
        accuracy = evaluate_model(global_model, test_loader)
        avg_loss = sum(round_losses) / len(round_losses)

        print(f"  Round {fl_round+1:2d}: Acc={accuracy:.3f}, Loss={avg_loss:.3f}")

    final_accuracy = evaluate_model(global_model, test_loader)
    print(f"\nFINAL {num_clients}-CLIENT ACCURACY: {final_accuracy:.3f}")

    return final_accuracy


def analyze_multi_client_results(results):
    """Analyze multi-client federation results"""

    print("\n" + "="*60)
    print("STEP 2 ANALYSIS: MULTI-CLIENT FEDERATION")
    print("="*60)

    single_client_baseline = 0.595  # Our fixed 1-client result

    print(f"1-Client Baseline (Step 1): {single_client_baseline:.1%}")

    for num_clients, accuracy in results.items():
        gap = single_client_baseline - accuracy
        print(f"{num_clients}-Client FedAvg:          {accuracy:.1%} (gap: {gap:+.1%})")

    # Determine federation health
    two_client_acc = results.get(2, 0)
    four_client_acc = results.get(4, 0)

    if two_client_acc >= single_client_baseline * 0.95:  # Within 5%
        print("\nVERDICT: Multi-client federation WORKS")
        print("Next: Proceed to Phase 2 (FLEX vs Baselines)")
    elif two_client_acc >= single_client_baseline * 0.85:  # Within 15%
        print("\nVERDICT: Multi-client federation has MODERATE issues")
        print("Next: Optimize aggregation or client sync")
    else:
        print("\nVERDICT: Multi-client federation BROKEN")
        print("Next: Debug aggregation mechanism specifically")


def main():
    """Step 2: Multi-client federation debugging"""

    print("STEP 2: MULTI-CLIENT FEDERATION DEBUGGING")
    print("="*60)
    print("Purpose: Test if multi-client FedAvg works with fixed training")
    print("1-Client Baseline: 59.5% (after optimizer bug fix)")
    print()

    results = {}

    # Test 2-client FedAvg
    results[2] = test_multi_client_fedavg(num_clients=2, num_rounds=10, local_epochs=5)

    # Test 4-client FedAvg
    results[4] = test_multi_client_fedavg(num_clients=4, num_rounds=10, local_epochs=5)

    # Analyze results
    analyze_multi_client_results(results)

    print("\n" + "="*60)
    print("STEP 2 COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    results = main()