"""
STEP 1 FIXED: 1-CLIENT SANITY TEST WITH CORRECTED OPTIMIZER HANDLING
================================================================

Fixes the critical bug: optimizer recreation every federated round
Tests: Fixed 1-client FedAvg should now achieve ~76% (similar to centralized)
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


class DebugFederatedModel(nn.Module):
    """Model for debugging federation issues"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        # VALIDATED architecture (same as centralized success)
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

        # Xavier initialization (validated)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def create_single_client_data():
    """Create single client with substantial data"""

    print("Creating single-client federated setup")

    # Load validated config
    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

    # Load dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)

    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    print(f"Total dataset: {len(images)} samples")

    # Train/test split
    train_size = int(0.8 * len(images))
    test_size = len(images) - train_size

    indices = torch.randperm(len(images))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Single client gets ALL training data
    client_images = images[train_indices]
    client_labels = labels[train_indices]

    test_images = images[test_indices]
    test_labels = labels[test_indices]

    print(f"Single client: {len(client_images)} training samples")
    print(f"Test set: {len(test_images)} samples")

    # Create loaders
    client_dataset = TensorDataset(client_images, client_labels)
    client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return [client_loader], test_loader, config_data


def fixed_single_client_fedavg_test(client_loaders, test_loader, config_data):
    """FIXED: Single-client FedAvg with proper optimizer handling"""

    print("\nFIXED SINGLE-CLIENT FEDAVG TEST")
    print("Expected: ~76% (after fixing optimizer bug)")
    print("Fix: Persistent optimizer across federated rounds")

    torch.manual_seed(42)

    # Create model (same as validated centralized)
    factory = ImprovedModelFactory()
    backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = DebugFederatedModel(backbone, 62)

    # Single client model
    client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    client_model = DebugFederatedModel(client_backbone, 62)
    client_model.load_state_dict(global_model.state_dict())

    client_loader = client_loaders[0]

    # CRITICAL FIX: Create optimizer ONCE (not every round)
    optimizer = optim.Adam(
        client_model.parameters(),
        lr=config_data['learning_rate'],  # 0.003
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    print(f"Client batches: {len(client_loader)}")
    print("FIXED: Persistent optimizer maintains momentum across FL rounds")

    # Federated training loop
    for fl_round in range(10):
        # CLIENT TRAINING (with PERSISTENT optimizer)
        client_model.train()

        # NO OPTIMIZER RECREATION - Use the SAME optimizer instance
        # This preserves Adam's momentum and variance estimates

        # Local epochs (5 from config)
        for local_epoch in range(config_data['local_epochs']):
            epoch_loss = 0
            for batch_x, batch_y in client_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # AGGREGATION (trivial with 1 client)
        global_model.load_state_dict(client_model.state_dict())

        # UPDATE CLIENT (trivial with 1 client)
        # client_model already has the right weights

        # EVALUATION
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
        print(f"  Round {fl_round+1:2d}: {accuracy:.3f}")

    final_accuracy = accuracy
    print(f"\nFINAL 1-CLIENT ACCURACY (FIXED): {final_accuracy:.3f}")

    return final_accuracy


def analyze_fixed_result(accuracy):
    """Analyze fixed single-client test result"""

    print(f"\n" + "="*60)
    print("STEP 1 FIXED ANALYSIS: OPTIMIZER BUG FIX")
    print("="*60)

    centralized_target = 0.76  # Our validated centralized performance
    federated_threshold = 0.70  # Minimum acceptable

    print(f"Fixed 1-Client FedAvg: {accuracy:.1%}")
    print(f"Centralized target:    {centralized_target:.1%}")
    print(f"Acceptable threshold:  {federated_threshold:.1%}")

    gap = centralized_target - accuracy
    print(f"Performance gap:       {gap:+.1%}")

    if accuracy >= federated_threshold:
        if accuracy >= centralized_target * 0.95:  # Within 5% of centralized
            verdict = "OPTIMIZER BUG FIXED - TRAINING PIPELINE WORKS"
            next_step = "Proceed to Step 2: Multi-client federation debugging"
            bug_location = "Multi-client federation logic (original bug was optimizer recreation)"
        else:
            verdict = "PARTIAL FIX - TRAINING PIPELINE IMPROVED"
            next_step = "Investigate remaining training efficiency issues"
            bug_location = "Remaining training efficiency in federated context"
    else:
        verdict = "OPTIMIZER FIX INSUFFICIENT"
        next_step = "Investigate additional training loop issues"
        bug_location = "Additional federated training problems"

    print(f"\nVERDICT: {verdict}")
    print(f"Bug location: {bug_location}")
    print(f"Next step: {next_step}")

    return accuracy >= federated_threshold


def main():
    """Step 1 Fixed: Single-client test with corrected optimizer handling"""

    print("STEP 1 FIXED: 1-CLIENT TEST WITH OPTIMIZER BUG FIX")
    print("="*60)
    print("Purpose: Test if optimizer recreation was the critical bug")
    print("Critical fix: Persistent optimizer across federated rounds")
    print()

    try:
        # Create single-client setup
        client_loaders, test_loader, config_data = create_single_client_data()

        # Run FIXED single-client FedAvg
        accuracy = fixed_single_client_fedavg_test(client_loaders, test_loader, config_data)

        # Analyze result
        training_works = analyze_fixed_result(accuracy)

        print(f"\n" + "="*60)
        print("STEP 1 FIXED COMPLETE")
        print("="*60)

        if training_works:
            print("RESULT: Optimizer bug fixed - training pipeline validated")
            print("ISSUE: Original bug was optimizer recreation every FL round")
            print("NEXT: Proceed to Step 2 (multi-client debugging)")
            return accuracy, True
        else:
            print("RESULT: Optimizer fix insufficient - additional issues remain")
            print("ISSUE: More complex federated training problems")
            print("NEXT: Deeper investigation of federated training mechanics")
            return accuracy, False

    except Exception as e:
        print(f"CRITICAL ERROR in Step 1 Fixed: {e}")
        return 0.0, False


if __name__ == "__main__":
    accuracy, success = main()

    print(f"\nSTEP 1 FIXED RESULT: {accuracy:.3f}")
    if success:
        print("✓ Optimizer bug fixed - Ready for Step 2: Multi-client debugging")
    else:
        print("✗ Additional federated training issues remain")