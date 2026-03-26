"""
TRAINING REGIME COMPARISON: Epochs vs Rounds
==========================================

Tests if breaking training into federated rounds affects convergence
compared to continuous training with same total epochs.
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


class DebugModel(nn.Module):
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
    """Quick evaluation"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def training_regime_test():
    """Compare continuous vs round-based training"""

    print("TRAINING REGIME TEST: Continuous vs Round-Based")
    print("="*50)

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    # Load larger dataset for more reliable results
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Create train/test split
    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=64, shuffle=False)

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test")
    print(f"Total planned epochs: 50 (10 rounds × 5 epochs)")

    criterion = nn.CrossEntropyLoss()

    # TEST 1: Continuous Training (50 epochs straight)
    print("\nTest 1: Continuous Training (50 epochs)")
    torch.manual_seed(42)

    factory = ImprovedModelFactory()
    backbone1 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model_continuous = DebugModel(backbone1, 62)
    optimizer_continuous = optim.Adam(model_continuous.parameters(),
                                    lr=config_data['learning_rate'], weight_decay=1e-4)

    model_continuous.train()
    for epoch in range(50):
        for batch_x, batch_y in train_loader:
            optimizer_continuous.zero_grad()
            outputs = model_continuous(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_continuous.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            acc = evaluate_model(model_continuous, test_loader)
            print(f"  Epoch {epoch+1:2d}: {acc:.3f}")

    final_continuous = evaluate_model(model_continuous, test_loader)

    # TEST 2: Round-based Training (10 rounds × 5 epochs)
    print("\nTest 2: Round-based Training (10 rounds × 5 epochs)")
    torch.manual_seed(42)

    backbone2 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model_rounds = DebugModel(backbone2, 62)
    optimizer_rounds = optim.Adam(model_rounds.parameters(),
                                lr=config_data['learning_rate'], weight_decay=1e-4)

    for fl_round in range(10):
        model_rounds.train()

        # 5 epochs per round
        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer_rounds.zero_grad()
                outputs = model_rounds(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer_rounds.step()

        # Evaluate after each round
        acc = evaluate_model(model_rounds, test_loader)
        print(f"  Round {fl_round+1:2d}: {acc:.3f}")

    final_rounds = evaluate_model(model_rounds, test_loader)

    # TEST 3: Round-based with BatchNorm reset check
    print("\nTest 3: Round-based with BatchNorm Analysis")
    torch.manual_seed(42)

    backbone3 = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    model_bn_check = DebugModel(backbone3, 62)
    optimizer_bn = optim.Adam(model_bn_check.parameters(),
                            lr=config_data['learning_rate'], weight_decay=1e-4)

    # Check BatchNorm statistics before/after rounds
    def get_bn_stats(model):
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'num_batches_tracked': module.num_batches_tracked.clone()
                }
        return stats

    initial_stats = get_bn_stats(model_bn_check)

    for fl_round in range(10):
        model_bn_check.train()

        round_start_stats = get_bn_stats(model_bn_check)

        for local_epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer_bn.zero_grad()
                outputs = model_bn_check(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer_bn.step()

        round_end_stats = get_bn_stats(model_bn_check)

        # Check if BatchNorm stats changed significantly
        bn_change = False
        for name in initial_stats:
            start_mean = round_start_stats[name]['running_mean']
            end_mean = round_end_stats[name]['running_mean']
            if torch.norm(end_mean - start_mean) > 0.01:
                bn_change = True
                break

        acc = evaluate_model(model_bn_check, test_loader)
        print(f"  Round {fl_round+1:2d}: {acc:.3f} (BN change: {bn_change})")

    final_bn_check = evaluate_model(model_bn_check, test_loader)

    # RESULTS COMPARISON
    print("\n" + "="*50)
    print("TRAINING REGIME COMPARISON RESULTS")
    print("="*50)
    print(f"Continuous training (50 epochs):     {final_continuous:.1%}")
    print(f"Round-based training (10×5 epochs):  {final_rounds:.1%}")
    print(f"Round-based with BN analysis:        {final_bn_check:.1%}")

    continuous_vs_rounds = final_continuous - final_rounds
    print(f"\nPerformance difference: {continuous_vs_rounds:+.1%}")

    if abs(continuous_vs_rounds) < 0.02:
        print("CONCLUSION: Training regime is NOT the issue")
        print("NEXT: Investigate other architectural or data handling differences")
    elif continuous_vs_rounds > 0.02:
        print("CONCLUSION: Round-based training HURTS performance")
        print("HYPOTHESIS: Breaking training disrupts convergence patterns")
    else:
        print("CONCLUSION: Round-based training HELPS performance")
        print("HYPOTHESIS: Round structure provides beneficial regularization")


if __name__ == "__main__":
    training_regime_test()