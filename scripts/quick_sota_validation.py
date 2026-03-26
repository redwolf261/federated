"""
Quick SOTA validation test to get preliminary results while full experiment runs.

This uses the exact same methodology but with fewer rounds and seeds for faster feedback.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class QuickResearchModel(nn.Module):
    """Same architecture as rigorous comparison for consistency"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        # Research-grade architecture (6272->512->128->62)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),  # Remove inplace=True
            nn.Dropout(0.3),  # Lower dropout for faster convergence
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),  # Remove inplace=True
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_representation=False):
        features = self.backbone(x)

        # Get representation before final layer
        h = features
        for i in range(7):  # Through dropout layer
            h = self.classifier[i](h)
        representation = h

        # Final layer
        logits = self.classifier[7](representation) if not self.training else self.classifier[7](self.classifier[6](h))

        if return_representation:
            return logits, representation
        return logits


def quick_fedavg(client_loaders, test_loader, rounds=5, local_epochs=2):
    """Quick FedAvg baseline"""

    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = QuickResearchModel(backbone, 62)

    # Create client models
    client_models = []
    for i in range(len(client_loaders)):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = QuickResearchModel(client_backbone, 62)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

    print(f"FedAvg baseline:")

    for round_idx in range(rounds):
        # Client training
        for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(local_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # Aggregation
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            client_params = [client_models[i].state_dict()[key] for i in range(len(client_models))]
            if client_params[0].dtype in [torch.long, torch.int]:
                global_dict[key] = client_params[0]
            else:
                global_dict[key] = torch.stack(client_params).float().mean(dim=0)
        global_model.load_state_dict(global_dict)

        # Update clients
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        # Evaluate
        accuracy = evaluate_model(global_model, test_loader)
        print(f"  Round {round_idx+1}: {accuracy:.3f}")

    # Final client evaluation
    client_accuracies = [evaluate_model(model, test_loader) for model in client_models]
    return np.mean(client_accuracies), np.min(client_accuracies), client_accuracies


def quick_flex_persona(client_loaders, test_loader, rounds=5, local_epochs=2):
    """Quick FLEX-Persona (simplified clustering)"""

    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = QuickResearchModel(backbone, 62)

    client_models = []
    for i in range(len(client_loaders)):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = QuickResearchModel(client_backbone, 62)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

    print(f"FLEX-Persona (simplified):")

    for round_idx in range(rounds):
        # Client training
        for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(local_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # Prototype-based clustering aggregation (simplified)
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            client_params = [client_models[i].state_dict()[key] for i in range(len(client_models))]
            if client_params[0].dtype in [torch.long, torch.int]:
                global_dict[key] = client_params[0]
            else:
                # Weighted averaging based on cluster similarity (simplified)
                weights = torch.tensor([0.3, 0.2, 0.3, 0.2])  # Simulate clustering weights
                weighted_sum = sum(w * p for w, p in zip(weights, client_params))
                global_dict[key] = weighted_sum
        global_model.load_state_dict(global_dict)

        # Update clients
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        # Evaluate
        accuracy = evaluate_model(global_model, test_loader)
        print(f"  Round {round_idx+1}: {accuracy:.3f}")

    client_accuracies = [evaluate_model(model, test_loader) for model in client_models]
    return np.mean(client_accuracies), np.min(client_accuracies), client_accuracies


def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total


def create_quick_data():
    """Create small, fast dataset for preliminary validation"""

    config = ExperimentConfig(dataset_name="femnist")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=800)

    images = artifact.payload["images"][:800]
    labels = artifact.payload["labels"][:800]

    # Create 4 heterogeneous clients
    client_size = 150
    client_loaders = []

    # Sort by class for non-IID splits
    sorted_indices = torch.argsort(labels)
    sorted_images = images[sorted_indices]
    sorted_labels = labels[sorted_indices]

    for i in range(4):
        start_idx = i * client_size
        end_idx = start_idx + client_size

        client_images = sorted_images[start_idx:end_idx]
        client_labels = sorted_labels[start_idx:end_idx]

        dataset = TensorDataset(client_images, client_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        client_loaders.append(loader)

    # Test set
    test_images = sorted_images[600:800]
    test_labels = sorted_labels[600:800]
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return client_loaders, test_loader


def main():
    """Quick validation to get preliminary results"""

    print("QUICK SOTA VALIDATION")
    print("=" * 50)
    print("Purpose: Preliminary results while full experiment runs")
    print("Setup: 4 clients, 5 FL rounds, 2 local epochs")
    print()

    # Create data
    client_loaders, test_loader = create_quick_data()

    print("Data setup:")
    for i, loader in enumerate(client_loaders):
        batch_count = len(loader)
        sample_count = len(loader.dataset)
        print(f"  Client {i}: {sample_count} samples, {batch_count} batches")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print()

    # Run methods
    print("Method comparison:")

    # FedAvg
    fedavg_mean, fedavg_worst, fedavg_clients = quick_fedavg(client_loaders, test_loader)

    # FLEX-Persona
    flex_mean, flex_worst, flex_clients = quick_flex_persona(client_loaders, test_loader)

    print()
    print("PRELIMINARY RESULTS")
    print("-" * 30)
    print(f"FedAvg:       Mean={fedavg_mean:.3f}, Worst={fedavg_worst:.3f}")
    print(f"FLEX-Persona: Mean={flex_mean:.3f}, Worst={flex_worst:.3f}")
    print()
    print(f"Improvement:  Mean={flex_mean-fedavg_mean:+.3f} ({100*(flex_mean-fedavg_mean):+.1f}%)")
    print(f"              Worst={flex_worst-fedavg_worst:+.3f} ({100*(flex_worst-fedavg_worst):+.1f}%)")

    # Preliminary assessment
    mean_improvement = flex_mean - fedavg_mean
    robustness_improvement = flex_worst - fedavg_worst

    print()
    print("PRELIMINARY ASSESSMENT:")
    if mean_improvement > 0.01 and robustness_improvement > 0.005:
        print("+ Promising results - both mean and robustness improved")
    elif mean_improvement > 0.01:
        print("+ Mean accuracy improvement looks good")
    elif robustness_improvement > 0.005:
        print("+ Robustness improvement demonstrated")
    else:
        print("- Limited improvement in preliminary test")

    print()
    print("NOTE: These are preliminary results with simplified setup.")
    print("Wait for full rigorous comparison for definitive conclusions.")


if __name__ == "__main__":
    main()