"""
FINAL DIAGNOSTIC: Federated Training Inefficiency Root Cause
=========================================================

With optimizer bug fixed, identify the specific cause of persistent
federated performance loss across all test configurations.

Systematic investigation of remaining factors:
1. Data loading order differences
2. Random seed state disruption
3. Gradient accumulation patterns
4. BatchNorm statistics differences
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class DiagnosticModel(nn.Module):
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


def setup_controlled_data():
    """Create exact same data setup for all tests"""

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Fixed deterministic split
    np.random.seed(42)
    indices = np.random.permutation(len(images))
    train_size = int(0.8 * len(images))

    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    # Use same DataLoader settings
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=64, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=64, shuffle=False
    )

    return train_loader, test_loader, config_data


def test_data_ordering_hypothesis():
    """Test if data ordering in federated rounds affects learning"""

    print("HYPOTHESIS 1: Data ordering in federated structure")

    train_loader, test_loader, config_data = setup_controlled_data()

    # Test 1: Centralized with standard data loading
    print("  Centralized (standard):")
    torch.manual_seed(42)
    factory = ImprovedModelFactory()
    backbone1 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model1 = DiagnosticModel(backbone1, 62)
    optimizer1 = optim.Adam(model1.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model1.train()
    for epoch in range(50):  # Extended training
        for batch_x, batch_y in train_loader:
            optimizer1.zero_grad()
            outputs = model1(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer1.step()

    acc1 = evaluate_model(model1, test_loader)
    print(f"    Final accuracy: {acc1:.3f}")

    # Test 2: Simulated federated (same data, federated structure)
    print("  Federated simulation:")
    torch.manual_seed(42)
    backbone2 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model2 = DiagnosticModel(backbone2, 62)
    optimizer2 = optim.Adam(model2.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)

    for fl_round in range(10):  # 10 rounds × 5 epochs = 50 total
        model2.train()
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer2.zero_grad()
                outputs = model2(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer2.step()

        # Round evaluation (key difference)
        round_acc = evaluate_model(model2, test_loader)
        print(f"    Round {fl_round+1:2d}: {round_acc:.3f}")

    acc2 = evaluate_model(model2, test_loader)
    print(f"    Final accuracy: {acc2:.3f}")
    print(f"    Gap: {acc1 - acc2:+.3f}")


def test_gradient_accumulation_hypothesis():
    """Test if gradient accumulation patterns differ"""

    print("\nHYPOTHESIS 2: Gradient accumulation consistency")

    train_loader, test_loader, config_data = setup_controlled_data()

    # Test identical training with different evaluation frequencies
    results = []
    eval_frequencies = [1, 5, 10, 50]  # Epochs between evaluations

    for eval_freq in eval_frequencies:
        torch.manual_seed(42)
        factory = ImprovedModelFactory()
        backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        model = DiagnosticModel(backbone, 62)
        optimizer = optim.Adam(model.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(50):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Periodic evaluation
            if (epoch + 1) % eval_freq == 0:
                _ = evaluate_model(model, test_loader)

        final_acc = evaluate_model(model, test_loader)
        results.append(final_acc)
        print(f"  Eval every {eval_freq:2d} epochs: {final_acc:.3f}")

    max_acc = max(results)
    min_acc = min(results)
    print(f"  Evaluation frequency impact: {max_acc - min_acc:.3f}")


def test_batchnorm_statistics_hypothesis():
    """Test BatchNorm behavior differences"""

    print("\nHYPOTHESIS 3: BatchNorm statistics disruption")

    train_loader, test_loader, config_data = setup_controlled_data()

    def get_bn_running_stats(model):
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                stats[name] = {
                    'mean': module.running_mean.clone(),
                    'var': module.running_var.clone()
                }
        return stats

    # Test 1: Normal training
    torch.manual_seed(42)
    factory = ImprovedModelFactory()
    backbone1 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model1 = DiagnosticModel(backbone1, 62)
    optimizer1 = optim.Adam(model1.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model1.train()
    for epoch in range(50):
        for batch_x, batch_y in train_loader:
            optimizer1.zero_grad()
            outputs = model1(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer1.step()

    final_stats1 = get_bn_running_stats(model1)
    acc1 = evaluate_model(model1, test_loader)
    print(f"  Continuous training:     {acc1:.3f}")

    # Test 2: Federated-style with eval mode interruptions
    torch.manual_seed(42)
    backbone2 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model2 = DiagnosticModel(backbone2, 62)
    optimizer2 = optim.Adam(model2.parameters(), lr=config_data['learning_rate'], weight_decay=1e-4)

    for fl_round in range(10):
        model2.train()
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer2.zero_grad()
                outputs = model2(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer2.step()

        # Evaluation interrupts training mode
        _ = evaluate_model(model2, test_loader)

    final_stats2 = get_bn_running_stats(model2)
    acc2 = evaluate_model(model2, test_loader)
    print(f"  Eval-interrupted training: {acc2:.3f}")
    print(f"  Performance gap: {acc1 - acc2:+.3f}")

    # Compare BatchNorm statistics
    print("  BatchNorm statistics comparison:")
    for layer_name in final_stats1:
        mean_diff = torch.norm(final_stats1[layer_name]['mean'] - final_stats2[layer_name]['mean'])
        var_diff = torch.norm(final_stats1[layer_name]['var'] - final_stats2[layer_name]['var'])
        print(f"    {layer_name}: mean_diff={mean_diff:.4f}, var_diff={var_diff:.4f}")


def main():
    """Comprehensive final diagnostic"""

    print("FINAL DIAGNOSTIC: FEDERATED TRAINING INEFFICIENCY")
    print("=" * 60)
    print("Goal: Identify root cause of persistent federated performance loss")
    print("After fixing optimizer recreation bug")
    print()

    test_data_ordering_hypothesis()
    test_gradient_accumulation_hypothesis()
    test_batchnorm_statistics_hypothesis()

    print("\n" + "=" * 60)
    print("FINAL DIAGNOSTIC COMPLETE")
    print("This should pinpoint the specific cause of federated inefficiency")


if __name__ == "__main__":
    main()