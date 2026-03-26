"""
PHASE 2: Multi-Client Federation with GPU Support
================================================

Full multi-client federated validation:
- 2+ clients with proper aggregation
- GPU acceleration for speed
- Validated configuration (75.5% baseline)
- Weight synchronization at each round
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FederatedModel(nn.Module):
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


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def fedavg_aggregate(global_model, client_models, client_sizes):
    """FedAvg aggregation with proper handling of all tensor types"""
    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    # Initialize aggregation buffers
    aggregated = {}
    for key in global_state.keys():
        if global_state[key].is_floating_point():
            aggregated[key] = torch.zeros_like(global_state[key])
        else:
            # Non-float tensors: will copy from first client
            aggregated[key] = None

    # Weighted sum
    for idx, (client_model, client_size) in enumerate(zip(client_models, client_sizes)):
        client_state = client_model.state_dict()
        weight = client_size / total_size

        for key in global_state.keys():
            if global_state[key].is_floating_point():
                aggregated[key] += client_state[key] * weight
            elif idx == 0:
                # Integer tensors: copy from first client
                aggregated[key] = client_state[key].clone()

    global_model.load_state_dict(aggregated)


def run_multiclient_federated(num_clients, num_rounds=10, local_epochs=5):
    """Run multi-client federated training with validated configuration"""

    print(f"\n{'='*60}")
    print(f"MULTI-CLIENT FEDERATED: {num_clients} Clients")
    print('='*60)

    start_time = time.time()

    # Load config
    with open('phase0_corrected_config.json', 'r') as f:
        config = json.load(f)

    # Load dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=2000)
    images = artifact.payload["images"][:2000]
    labels = artifact.payload["labels"][:2000]

    # Train/test split
    train_size = int(0.8 * len(images))
    indices = torch.randperm(len(images))
    train_images = images[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    test_images = images[indices[train_size:]]
    test_labels = labels[indices[train_size:]]

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test")

    # Distribute data across clients
    samples_per_client = len(train_images) // num_clients
    client_loaders = []
    client_sizes = []

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(train_images)
        client_imgs = train_images[start:end]
        client_lbls = train_labels[start:end]

        print(f"  Client {i+1}: {len(client_imgs)} samples")

        loader = DataLoader(TensorDataset(client_imgs, client_lbls),
                          batch_size=64, shuffle=True, drop_last=True)
        client_loaders.append(loader)
        client_sizes.append(len(client_imgs))

    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                            batch_size=64, shuffle=False)

    # Initialize models
    torch.manual_seed(42)
    factory = ImprovedModelFactory()

    # Global model
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = FederatedModel(global_backbone, 62).to(device)

    # Client models with persistent optimizers
    client_models = []
    client_optimizers = []

    for _ in range(num_clients):
        backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        model = FederatedModel(backbone, 62).to(device)
        model.load_state_dict(global_model.state_dict())
        client_models.append(model)

        # CRITICAL: Persistent optimizer (NOT recreated each round)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        client_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()

    total_epochs = num_rounds * local_epochs
    print(f"Training: {num_rounds} rounds x {local_epochs} epochs = {total_epochs} total")
    print(f"Device: {device}")

    # Track per-client accuracy
    round_accuracies = []

    # Federated training loop
    for fl_round in range(num_rounds):
        round_start = time.time()

        # --- CLIENT TRAINING ---
        for client_id, (model, loader, optimizer) in enumerate(
            zip(client_models, client_loaders, client_optimizers)):

            model.train()

            for epoch in range(local_epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # --- WEIGHT SYNCHRONIZATION (FedAvg) ---
        fedavg_aggregate(global_model, client_models, client_sizes)

        # --- UPDATE CLIENTS with global model ---
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        # --- EVALUATION ---
        acc = evaluate(global_model, test_loader, device)
        round_time = time.time() - round_start
        round_accuracies.append(acc)

        print(f"  Round {fl_round+1:2d}: {acc:.3f} ({round_time:.1f}s)")

    # Final evaluation
    final_acc = evaluate(global_model, test_loader, device)
    total_time = time.time() - start_time

    # Per-client evaluation (for worst-client tracking)
    print(f"\n  Per-client final accuracy:")
    client_accs = []
    for i, model in enumerate(client_models):
        client_acc = evaluate(model, test_loader, device)
        client_accs.append(client_acc)
        print(f"    Client {i+1}: {client_acc:.3f}")

    worst_client = min(client_accs)
    best_client = max(client_accs)

    print(f"\n  Final Global: {final_acc:.3f}")
    print(f"  Worst Client: {worst_client:.3f}")
    print(f"  Best Client:  {best_client:.3f}")
    print(f"  Total time:   {total_time:.1f}s")

    return {
        'num_clients': num_clients,
        'final_accuracy': final_acc,
        'worst_client': worst_client,
        'best_client': best_client,
        'round_accuracies': round_accuracies,
        'total_time': total_time
    }


def main():
    """Phase 2: Multi-Client Federation Validation"""

    print("PHASE 2: MULTI-CLIENT FEDERATION VALIDATION")
    print("=" * 60)
    print(f"Device: {device}")
    print("Testing: 1-client, 2-clients, 4-clients, 8-clients")
    print()

    results = {}

    for num_clients in [1, 2, 4, 8]:
        result = run_multiclient_federated(num_clients, num_rounds=10, local_epochs=5)
        results[num_clients] = result

    # --- ANALYSIS ---
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS: MULTI-CLIENT FEDERATION")
    print("=" * 60)

    baseline = results[1]['final_accuracy']

    print(f"\n{'Clients':<10} {'Final Acc':<12} {'Worst Client':<14} {'vs Baseline':<12} {'Time':<10}")
    print("-" * 60)

    for num_clients, result in results.items():
        acc = result['final_accuracy']
        worst = result['worst_client']
        diff = acc - baseline
        time_taken = result['total_time']

        print(f"{num_clients:<10} {acc:.1%}        {worst:.1%}          {diff:+.1%}        {time_taken:.0f}s")

    # Check for significant degradation
    max_degradation = 0
    for num_clients, result in results.items():
        if num_clients > 1:
            degradation = baseline - result['final_accuracy']
            max_degradation = max(max_degradation, degradation)

    print(f"\nMax degradation from 1-client: {max_degradation:.1%}")

    if max_degradation < 0.05:
        print("\nVERDICT: Multi-client federation WORKS")
        print("Aggregation logic validated")
        print("READY FOR RESEARCH COMPARISON (FedAvg vs MOON vs FLEX)")
        success = True
    elif max_degradation < 0.10:
        print("\nVERDICT: Minor degradation detected")
        print("Federation works but with small efficiency loss")
        print("Proceed with caution to research comparison")
        success = True
    else:
        print("\nVERDICT: Significant degradation detected")
        print("Federation logic needs debugging")
        success = False

    # Save results
    import json
    with open('phase2_multiclient_results.json', 'w') as f:
        # Convert for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[str(k)] = {
                'final_accuracy': v['final_accuracy'],
                'worst_client': v['worst_client'],
                'best_client': v['best_client'],
                'total_time': v['total_time']
            }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: phase2_multiclient_results.json")

    return success


if __name__ == "__main__":
    success = main()
    print(f"\nPHASE 2 STATUS: {'VALIDATED' if success else 'NEEDS DEBUG'}")