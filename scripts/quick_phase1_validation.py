"""
Quick Phase 1 Fix: BatchNorm-safe federated data splits
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class SimpleFederatedModel(nn.Module):
    """Simplified model without BatchNorm for federated testing"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        # Simplified architecture without BatchNorm (avoids batch size issues)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Xavier initialization
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_representation=False):
        features = self.backbone(x)

        # Forward through classifier
        h = features
        for i in range(4):  # Through second ReLU
            h = self.classifier[i](h)
        representation = h  # 128-dim

        # Final layers
        if self.training:
            h = self.classifier[4](h)  # Dropout
            logits = self.classifier[5](h)  # Final linear
        else:
            logits = self.classifier[5](h)  # Skip dropout

        if return_representation:
            return logits, representation
        return logits


def create_safe_federated_splits():
    """Create federated splits with guaranteed minimum batch sizes"""

    print("Creating BatchNorm-safe federated splits")

    # Load config
    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    # Load dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)  # Smaller for reliable splits

    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    print(f"Dataset: {len(images)} images")

    # Create simpler federated setup
    num_clients = 10  # Fewer clients for more samples per client
    samples_per_client = len(images) // num_clients
    min_samples = 50  # Ensure minimum samples per client

    client_loaders = []

    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = min(start_idx + samples_per_client, len(images))

        client_images = images[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]

        # Ensure minimum samples
        if len(client_images) < min_samples and client_id < num_clients - 1:
            needed = min_samples - len(client_images)
            extra_images = images[end_idx:end_idx + needed]
            extra_labels = labels[end_idx:end_idx + needed]

            client_images = torch.cat([client_images, extra_images])
            client_labels = torch.cat([client_labels, extra_labels])

        dataset = TensorDataset(client_images, client_labels)
        # Use smaller batch size but ensure >1 for BatchNorm
        batch_size = min(32, max(2, len(dataset) // 4))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        client_loaders.append(loader)

        print(f"  Client {client_id}: {len(dataset)} samples, batch_size={batch_size}")

    # Test set
    test_size = 400
    test_indices = torch.randperm(len(images))[:test_size]
    test_dataset = TensorDataset(images[test_indices], labels[test_indices])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Test set: {len(test_dataset)} samples")

    return client_loaders, test_loader, config_data


def quick_fedavg_test(client_loaders, test_loader, config_data):
    """Quick FedAvg test with simplified model"""

    print("\nQuick FedAvg Test (BatchNorm-safe)")

    # Create models
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = SimpleFederatedModel(backbone, 62)

    client_models = []
    for i in range(len(client_loaders)):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = SimpleFederatedModel(client_backbone, 62)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

    # Test federated training
    for fl_round in range(5):
        # Client training
        for client_model, client_loader in zip(client_models, client_loaders):
            if len(client_loader) == 0:
                continue

            client_model.train()
            optimizer = optim.Adam(client_model.parameters(), lr=0.003)
            criterion = nn.CrossEntropyLoss()

            for local_epoch in range(3):
                for batch_x, batch_y in client_loader:
                    if len(batch_x) < 2:  # Skip problematic batches
                        continue

                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # FedAvg aggregation
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            client_params = [client_models[i].state_dict()[key] for i in range(len(client_models))]
            if client_params[0].dtype in [torch.long, torch.int]:
                global_dict[key] = client_params[0]
            else:
                global_dict[key] = torch.stack(client_params).float().mean(dim=0)
        global_model.load_state_dict(global_dict)

        # Update clients
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

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
        print(f"  Round {fl_round+1}: {accuracy:.3f}")

    print(f"Final accuracy: {accuracy:.3f}")

    if accuracy > 0.1:
        print("SUCCESS: FedAvg converges in federated setting")
        return True
    else:
        print("FAILURE: FedAvg does not converge adequately")
        return False


def main():
    """Quick Phase 1 validation without BatchNorm issues"""

    print("QUICK PHASE 1 VALIDATION (BatchNorm-safe)")
    print("=" * 50)

    try:
        # Create safe federated splits
        client_loaders, test_loader, config_data = create_safe_federated_splits()

        # Test FedAvg
        success = quick_fedavg_test(client_loaders, test_loader, config_data)

        print("\nRESULT:")
        if success:
            print("+ FedAvg baseline works in federated setting")
            print("+ Phase 1 sanity check PASSED")
            print("+ Ready to proceed with proper Phase 1 implementation")
        else:
            print("- FedAvg baseline fails in federated setting")
            print("- Phase 1 sanity check FAILED")
            print("- Need to debug federated training further")

        return success

    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    main()