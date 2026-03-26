"""
STEP 2 DEBUG: BatchNorm vs GroupNorm in Federated Learning
=========================================================

BatchNorm running statistics don't aggregate properly in FedAvg.
Test hypothesis: Replace BatchNorm with GroupNorm to fix multi-client.
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


class FederatedModelNoBN(nn.Module):
    """Model WITHOUT BatchNorm for federated learning debugging"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        backbone_dim = backbone.output_dim

        # NO BatchNorm - just simple layers
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
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


class FederatedModelLayerNorm(nn.Module):
    """Model with LayerNorm instead of BatchNorm"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        backbone_dim = backbone.output_dim

        # LayerNorm instead of BatchNorm (no running statistics)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
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
    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    # Split training data across clients
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

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return client_loaders, test_loader, config_data


def fedavg_aggregate(global_model, client_models):
    """Simple FedAvg aggregation"""
    with torch.no_grad():
        global_state = global_model.state_dict()
        n_clients = len(client_models)

        for key in global_state:
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            for client_model in client_models:
                global_state[key] += client_model.state_dict()[key].float() / n_clients

        global_model.load_state_dict(global_state)


def test_fedavg_no_batchnorm(num_clients, model_class, model_name):
    """Test FedAvg without BatchNorm issues"""

    print(f"\n{'='*60}")
    print(f"{model_name}: {num_clients} CLIENTS")
    print(f"{'='*60}")

    client_loaders, test_loader, config_data = create_federated_data(num_clients)

    torch.manual_seed(42)

    # Create global model
    factory = ImprovedModelFactory()
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = model_class(global_backbone, 62)

    # Create client models
    client_models = []
    client_optimizers = []

    for i in range(num_clients):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = model_class(client_backbone, 62)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

        # Fresh optimizer per client (standard FedAvg approach)
        optimizer = optim.Adam(
            client_model.parameters(),
            lr=config_data['learning_rate'],
            weight_decay=1e-4
        )
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: 10 rounds x 5 local epochs")

    # Federated training loop
    for fl_round in range(10):
        # CLIENT TRAINING
        for client_idx in range(num_clients):
            client_model = client_models[client_idx]
            optimizer = client_optimizers[client_idx]
            client_loader = client_loaders[client_idx]

            # Sync client with global model
            client_model.load_state_dict(global_model.state_dict())

            # Reset optimizer for fresh start each round (standard FedAvg)
            optimizer = optim.Adam(
                client_model.parameters(),
                lr=config_data['learning_rate'],
                weight_decay=1e-4
            )
            client_optimizers[client_idx] = optimizer

            client_model.train()

            # Local training
            for local_epoch in range(5):
                for batch_x, batch_y in client_loader:
                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # AGGREGATION
        fedavg_aggregate(global_model, client_models)

        # EVALUATION
        accuracy = evaluate_model(global_model, test_loader)
        print(f"  Round {fl_round+1:2d}: {accuracy:.3f}")

    final_accuracy = evaluate_model(global_model, test_loader)
    print(f"\nFINAL {num_clients}-CLIENT ({model_name}): {final_accuracy:.3f}")

    return final_accuracy


def main():
    """Test BatchNorm hypothesis in federated learning"""

    print("STEP 2 DEBUG: BATCHNORM HYPOTHESIS")
    print("="*60)
    print("Hypothesis: BatchNorm running statistics break FedAvg")
    print("Test: Remove BatchNorm and compare performance")
    print()

    results = {}

    # Test without BatchNorm
    print("\n--- NO BatchNorm (Simple ReLU) ---")
    results['no_bn_2'] = test_fedavg_no_batchnorm(2, FederatedModelNoBN, "NO_BATCHNORM")
    results['no_bn_4'] = test_fedavg_no_batchnorm(4, FederatedModelNoBN, "NO_BATCHNORM")

    # Test with LayerNorm
    print("\n--- LayerNorm (No Running Stats) ---")
    results['ln_2'] = test_fedavg_no_batchnorm(2, FederatedModelLayerNorm, "LAYERNORM")
    results['ln_4'] = test_fedavg_no_batchnorm(4, FederatedModelLayerNorm, "LAYERNORM")

    # Analysis
    print("\n" + "="*60)
    print("BATCHNORM HYPOTHESIS ANALYSIS")
    print("="*60)
    print(f"1-Client Baseline (with BN): 59.5%")
    print(f"\nWithout BatchNorm:")
    print(f"  2-Client: {results['no_bn_2']:.1%}")
    print(f"  4-Client: {results['no_bn_4']:.1%}")
    print(f"\nWith LayerNorm:")
    print(f"  2-Client: {results['ln_2']:.1%}")
    print(f"  4-Client: {results['ln_4']:.1%}")

    # Check if removing BatchNorm helps
    if results['no_bn_2'] > 0.3 or results['ln_2'] > 0.3:
        print("\nCONCLUSION: BatchNorm IS causing federated issues")
        print("FIX: Use LayerNorm or GroupNorm for federated models")
    else:
        print("\nCONCLUSION: BatchNorm is NOT the primary issue")
        print("INVESTIGATE: Other federated training mechanics")

    return results


if __name__ == "__main__":
    results = main()