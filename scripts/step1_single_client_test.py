"""
STEP 1: 1-CLIENT SANITY COLLAPSE TEST
===================================

Critical test: 1-client FedAvg should ~= centralized performance (76%)

If 1-client FedAvg < 70%:
  -> Training loop is BROKEN (not federation issue)
  -> STOP everything, fix training first

If 1-client FedAvg ~= 76%:
  -> Training works, bug is in multi-client federation
  -> Proceed to Step 2 (multi-client debugging)

This test definitively separates training bugs from federation bugs.
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


def single_client_fedavg_test(client_loaders, test_loader, config_data):
    """Test FedAvg with single client (should ~= centralized performance)"""

    print("\nSINGLE-CLIENT FEDAVG TEST")
    print("Expected: ~76% (similar to centralized)")
    print("If <70%: Training loop BROKEN")

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

    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    print(f"Client batches: {len(client_loader)}")

    # Federated training loop
    for fl_round in range(10):
        # CLIENT TRAINING (using validated parameters)
        client_model.train()

        optimizer = optim.Adam(
            client_model.parameters(),
            lr=config_data['learning_rate'],  # 0.003
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

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
    print(f"\nFINAL 1-CLIENT ACCURACY: {final_accuracy:.3f}")

    return final_accuracy


def analyze_single_client_result(accuracy):
    """Analyze single-client test result"""

    print(f"\n" + "="*60)
    print("STEP 1 ANALYSIS: 1-CLIENT SANITY TEST")
    print("="*60)

    centralized_target = 0.76  # Our validated centralized performance
    federated_threshold = 0.70  # Minimum acceptable

    print(f"1-Client FedAvg:     {accuracy:.1%}")
    print(f"Centralized target:  {centralized_target:.1%}")
    print(f"Acceptable threshold: {federated_threshold:.1%}")

    gap = centralized_target - accuracy
    print(f"Performance gap:     {gap:+.1%}")

    if accuracy >= federated_threshold:
        if accuracy >= centralized_target * 0.95:  # Within 5% of centralized
            verdict = "TRAINING PIPELINE WORKS"
            next_step = "Proceed to Step 2: Multi-client federation debugging"
            bug_location = "Multi-client federation logic"
        else:
            verdict = "TRAINING PIPELINE MARGINAL"
            next_step = "Proceed to Step 2, but investigate training efficiency"
            bug_location = "Training efficiency in federated context"
    else:
        verdict = "TRAINING PIPELINE BROKEN"
        next_step = "STOP: Fix single-client training before federation"
        bug_location = "Basic federated training loop"

    print(f"\nVERDICT: {verdict}")
    print(f"Bug location: {bug_location}")
    print(f"Next step: {next_step}")

    return accuracy >= federated_threshold


def main():
    """Step 1: Single-client sanity collapse test"""

    print("STEP 1: 1-CLIENT SANITY COLLAPSE TEST")
    print("="*50)
    print("Purpose: Separate training bugs from federation bugs")
    print("Critical test: 1-client FedAvg should ~= centralized (76%)")
    print()

    try:
        # Create single-client setup
        client_loaders, test_loader, config_data = create_single_client_data()

        # Run single-client FedAvg
        accuracy = single_client_fedavg_test(client_loaders, test_loader, config_data)

        # Analyze result
        training_works = analyze_single_client_result(accuracy)

        print(f"\n" + "="*50)
        print("STEP 1 COMPLETE")
        print("="*50)

        if training_works:
            print("RESULT: Training pipeline validated")
            print("ISSUE: Bug is in multi-client federation")
            print("NEXT: Proceed to Step 2 (multi-client debugging)")
            return accuracy, True
        else:
            print("RESULT: Training pipeline BROKEN")
            print("ISSUE: Basic federated training fails")
            print("NEXT: Fix single-client training before proceeding")
            return accuracy, False

    except Exception as e:
        print(f"CRITICAL ERROR in Step 1: {e}")
        return 0.0, False


if __name__ == "__main__":
    accuracy, success = main()

    print(f"\nSTEP 1 RESULT: {accuracy:.3f}")
    if success:
        print("✓ Ready for Step 2: Multi-client debugging")
    else:
        print("✗ Must fix training loop first")