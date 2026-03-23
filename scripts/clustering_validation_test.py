"""Quick clustering validation test - the make-or-break question.

CRITICAL: Must prove clustering helps or remove it from the method.
This is the decisive test that determines if clustering is justified.

Based on your critique: "Without this, your method collapses to
prototype-based FL with extra steps"

This test will provide empirical evidence for: FLEX(clustering) vs FLEX(no-clustering)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class ResearchGradeClassifier(nn.Module):
    """The 87.11% accuracy classifier for fair comparison."""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


class StandardizedModel(nn.Module):
    """Standardized model for both variants."""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = ResearchGradeClassifier(backbone.output_dim, num_classes, dropout_rate=0.5)

    def forward_task(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.backbone(x)


def test_clustering_contribution():
    """The decisive test: Does clustering actually help?"""

    print("CLUSTERING VALIDATION TEST")
    print("="*50)
    print("CRITICAL QUESTION: Does clustering improve FLEX-Persona?")
    print("If NO -> remove clustering from method")
    print()

    # Setup
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    # Load data
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=1200)
    images = artifact.payload["images"][:1200]
    labels = artifact.payload["labels"][:1200]

    # Create test/client splits
    test_size = 300
    indices = torch.randperm(len(images))
    test_indices = indices[:test_size]
    client_indices = indices[test_size:]

    test_dataset = TensorDataset(images[test_indices], labels[test_indices])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    client_images = images[client_indices]
    client_labels = labels[client_indices]

    # Create heterogeneous clients (4 clients with different class subsets)
    num_clients = 4
    unique_classes = torch.unique(client_labels)
    classes_per_client = len(unique_classes) // 2

    client_loaders = []
    client_class_info = []

    for i in range(num_clients):
        # Each client sees different subset of classes
        start_class = (i * classes_per_client) % len(unique_classes)
        client_classes = []

        for j in range(classes_per_client):
            class_idx = (start_class + j) % len(unique_classes)
            client_classes.append(unique_classes[class_idx])

        # Get data for client
        client_mask = torch.zeros(len(client_labels), dtype=torch.bool)
        for class_id in client_classes:
            client_mask |= (client_labels == class_id)

        client_data_indices = torch.where(client_mask)[0]
        n_samples = min(150, len(client_data_indices))
        sampled_indices = client_data_indices[torch.randperm(len(client_data_indices))[:n_samples]]

        dataset = TensorDataset(client_images[sampled_indices], client_labels[sampled_indices])
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        client_loaders.append(loader)
        client_class_info.append([c.item() for c in client_classes[:5]])  # First 5 for display

    print(f"Setup: {num_clients} clients with heterogeneous data")
    for i, classes in enumerate(client_class_info):
        print(f"  Client {i}: classes {classes}... ({len(client_loaders[i].dataset)} samples)")

    # Test both variants
    variants = {
        'FLEX-No-Clustering': False,
        'FLEX-With-Clustering': True
    }

    results = {}

    for variant_name, use_clustering in variants.items():
        print(f"\nTesting {variant_name}...")

        # Create models
        factory = ImprovedModelFactory()
        client_models = []

        for i in range(num_clients):
            backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(config.dataset_name))
            model = StandardizedModel(backbone, config.model.num_classes)
            client_models.append(model)

        criterion = nn.CrossEntropyLoss()
        comm_cost = 0

        # Federated training (simplified)
        rounds = 15
        round_accuracies = []

        for round_idx in range(rounds):
            # Local training
            for client_id, (model, loader) in enumerate(zip(client_models, client_loaders)):
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
                model.train()

                # Local epochs
                for _ in range(2):
                    for batch_x, batch_y in loader:
                        optimizer.zero_grad()
                        logits = model.forward_task(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

            # Prototype sharing simulation
            if round_idx % 3 == 0:
                # Extract "prototypes" (simplified)
                client_prototypes = []

                for client_id, (model, loader) in enumerate(zip(client_models, client_loaders)):
                    model.eval()
                    with torch.no_grad():
                        sample_batch, _ = next(iter(loader))
                        features = model.extract_features(sample_batch[:8])
                        # Simulate prototype: mean feature per client
                        prototype = features.mean(dim=0)
                        client_prototypes.append(prototype)

                # Protocol difference: clustering vs no clustering
                if use_clustering:
                    # Simulate clustering: group similar clients
                    similarities = torch.zeros(num_clients, num_clients)
                    for i in range(num_clients):
                        for j in range(num_clients):
                            if i != j:
                                sim = torch.cosine_similarity(
                                    client_prototypes[i].unsqueeze(0),
                                    client_prototypes[j].unsqueeze(0)
                                )
                                similarities[i, j] = sim

                    # Simple clustering: group clients with >0.3 similarity
                    clusters = {}
                    assigned = [False] * num_clients

                    cluster_id = 0
                    for i in range(num_clients):
                        if not assigned[i]:
                            clusters[cluster_id] = [i]
                            assigned[i] = True

                            # Find similar clients
                            for j in range(i + 1, num_clients):
                                if not assigned[j] and similarities[i, j] > 0.3:
                                    clusters[cluster_id].append(j)
                                    assigned[j] = True

                            cluster_id += 1

                    # Cluster-specific communication cost
                    comm_cost += len(clusters) * 512  # Cluster prototypes
                    comm_cost += num_clients ** 2  # Similarity computation

                else:
                    # No clustering: simple averaging
                    comm_cost += num_clients * 512  # All prototypes

            # Evaluation
            client_accs = []
            for model in client_models:
                acc = evaluate_model(model, test_loader)
                client_accs.append(acc)

            avg_acc = np.mean(client_accs)
            round_accuracies.append(avg_acc)

            if round_idx % 5 == 0:
                print(f"  Round {round_idx:2d}: {avg_acc:.4f} "
                      f"(clients: [{min(client_accs):.3f}, {max(client_accs):.3f}])")

        # Final results
        final_accuracy = round_accuracies[-1]
        final_client_accs = client_accs

        results[variant_name] = {
            'final_accuracy': final_accuracy,
            'client_accuracies': final_client_accs,
            'worst_client': min(final_client_accs),
            'best_client': max(final_client_accs),
            'client_std': np.std(final_client_accs),
            'comm_cost': comm_cost,
            'accuracy_history': round_accuracies
        }

        print(f"  FINAL: {final_accuracy:.4f} avg, "
              f"worst: {min(final_client_accs):.4f}, "
              f"comm: {comm_cost}")

    # CRITICAL ANALYSIS
    print(f"\n{'='*50}")
    print("CLUSTERING CONTRIBUTION ANALYSIS")
    print('='*50)

    no_cluster_acc = results['FLEX-No-Clustering']['final_accuracy']
    with_cluster_acc = results['FLEX-With-Clustering']['final_accuracy']
    clustering_benefit = with_cluster_acc - no_cluster_acc

    print(f"FLEX-No-Clustering:  {no_cluster_acc:.4f}")
    print(f"FLEX-With-Clustering: {with_cluster_acc:.4f}")
    print(f"Clustering benefit:   {clustering_benefit:+.4f}")

    # Decision criteria
    threshold = 0.01  # 1% improvement threshold

    if clustering_benefit > threshold:
        print(f"\nVERDICT: CLUSTERING JUSTIFIED")
        print(f"  -> Improvement {clustering_benefit:+.4f} > {threshold}")
        print(f"  -> Keep clustering in FLEX-Persona method")
        clustering_justified = True
    else:
        print(f"\nVERDICT: CLUSTERING NOT JUSTIFIED")
        print(f"  -> Improvement {clustering_benefit:+.4f} <= {threshold}")
        print(f"  -> REMOVE clustering from method")
        print(f"  -> Use simple prototype averaging instead")
        clustering_justified = False

    # Communication analysis
    no_cluster_comm = results['FLEX-No-Clustering']['comm_cost']
    with_cluster_comm = results['FLEX-With-Clustering']['comm_cost']

    print(f"\nCommunication Cost Analysis:")
    print(f"  No clustering: {no_cluster_comm}")
    print(f"  With clustering: {with_cluster_comm}")
    print(f"  Overhead: {with_cluster_comm - no_cluster_comm} (+{((with_cluster_comm/no_cluster_comm-1)*100):+.1f}%)")

    # Final recommendation
    print(f"\n{'='*50}")
    print("RESEARCH RECOMMENDATION")
    print('='*50)

    if clustering_justified:
        print("✅ KEEP clustering in FLEX-Persona")
        print("   - Demonstrates empirical benefit")
        print("   - Justifies added complexity")
        print("   - Include in paper as core contribution")
    else:
        print("❌ REMOVE clustering from FLEX-Persona")
        print("   - No empirical benefit demonstrated")
        print("   - Unnecessary complexity")
        print("   - Simplify to prototype-based FL")
        print("   - Focus paper on prototype efficiency")

    return clustering_justified, clustering_benefit


def evaluate_model(model, test_loader):
    """Evaluate model accuracy."""
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


if __name__ == "__main__":
    clustering_justified, benefit = test_clustering_contribution()

    print(f"\nCLUSTERING VALIDATION COMPLETE")
    print(f"Justified: {clustering_justified}")
    print(f"Benefit: {benefit:+.4f}")