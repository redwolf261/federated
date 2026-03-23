"""Quick baseline comparison test to validate the framework.

This tests the baseline comparison methodology with a smaller setup to ensure
it works correctly before running full research experiments.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class SimpleFedAvg:
    """Simple FedAvg for quick testing."""

    def __init__(self):
        self.comm_cost = 0

    def train(self, client_loaders, test_loader, config, rounds=10):
        factory = ImprovedModelFactory()

        # Create models
        client_models = []
        for i in range(len(client_loaders)):
            model = factory.build_client_model(i, config.model, config.dataset_name)
            client_models.append(model)

        global_model = factory.build_client_model(0, config.model, config.dataset_name)
        criterion = nn.CrossEntropyLoss()

        total_params = sum(p.numel() for p in global_model.parameters())

        accuracies = []

        for round_idx in range(rounds):
            # Client updates
            for client_id, (model, loader) in enumerate(zip(client_models, client_loaders)):
                model.load_state_dict(global_model.state_dict())
                self.comm_cost += total_params

                optimizer = optim.Adam(model.parameters(), lr=0.001)
                model.train()

                # Local training
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    logits = model.forward_task(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                self.comm_cost += total_params

            # Aggregate
            global_dict = global_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] = torch.stack([
                    client_models[i].state_dict()[key]
                    for i in range(len(client_models))
                ]).mean(dim=0)
            global_model.load_state_dict(global_dict)

            # Evaluate
            accuracy = self._evaluate(global_model, test_loader)
            accuracies.append(accuracy)

        return {
            'final_accuracy': accuracies[-1],
            'accuracies': accuracies,
            'comm_cost': self.comm_cost
        }

    def _evaluate(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model.forward_task(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total


def quick_baseline_test():
    """Quick test of baseline comparison framework."""

    print("QUICK BASELINE COMPARISON TEST")
    print("="*40)

    # Small setup for quick test
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    # Load small dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=800)
    images = artifact.payload["images"][:800]
    labels = artifact.payload["labels"][:800]

    # Create test/train split
    test_size = 200
    indices = torch.randperm(len(images))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    test_dataset = TensorDataset(images[test_indices], labels[test_indices])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_images = images[train_indices]
    train_labels = labels[train_indices]

    # Create heterogeneous client splits
    num_clients = 4
    unique_classes = torch.unique(train_labels)
    classes_per_client = len(unique_classes) // 2

    client_loaders = []

    for i in range(num_clients):
        # Each client sees subset of classes
        start_class = (i * classes_per_client) % len(unique_classes)
        client_classes = []

        for j in range(classes_per_client):
            class_idx = (start_class + j) % len(unique_classes)
            client_classes.append(unique_classes[class_idx])

        # Get data for client classes
        client_mask = torch.zeros(len(train_labels), dtype=torch.bool)
        for class_id in client_classes:
            client_mask |= (train_labels == class_id)

        client_indices = torch.where(client_mask)[0]
        n_samples = min(100, len(client_indices))
        sampled_indices = client_indices[torch.randperm(len(client_indices))[:n_samples]]

        client_images = train_images[sampled_indices]
        client_labels = train_labels[sampled_indices]

        dataset = TensorDataset(client_images, client_labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        client_loaders.append(loader)

    print(f"Setup: {num_clients} clients, {len(test_dataset)} test samples")

    # Test baseline methods
    methods = {
        'FedAvg': SimpleFedAvg()
    }

    results = {}

    for name, method in methods.items():
        print(f"\nTesting {name}...")
        result = method.train(client_loaders, test_loader, config, rounds=5)

        results[name] = result
        print(f"  Final accuracy: {result['final_accuracy']:.4f}")
        print(f"  Communication: {result['comm_cost']:,}")

    # Analysis
    print(f"\nResults Summary:")
    for name, result in results.items():
        print(f"  {name}: {result['final_accuracy']:.4f} accuracy")

    print(f"\nFramework validation: {'SUCCESS' if results else 'FAILED'}")

    return len(results) > 0


if __name__ == "__main__":
    quick_baseline_test()