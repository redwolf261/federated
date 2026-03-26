"""
PHASE 2 Enhanced: Multi-Client with Performance Tracking
======================================================

Tests multi-client federation with detailed per-client analytics:
- Per-client accuracy monitoring
- Worst-client performance tracking
- Aggregation efficiency analysis
- GPU acceleration enabled
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class TrackedFedModel(nn.Module):
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


def evaluate_detailed(model, test_loader, client_loaders=None):
    """Enhanced evaluation with per-client tracking"""
    model.eval()
    results = {}

    with torch.no_grad():
        # Global test accuracy
        correct = total = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        global_acc = correct / total
        results['global'] = global_acc

        # Per-client local accuracy (if client data provided)
        if client_loaders:
            client_accs = []
            for i, client_loader in enumerate(client_loaders):
                client_correct = client_total = 0
                for batch_x, batch_y in client_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, preds = torch.max(outputs, 1)
                    client_correct += (preds == batch_y).sum().item()
                    client_total += batch_y.size(0)

                client_acc = client_correct / client_total if client_total > 0 else 0
                client_accs.append(client_acc)
                results[f'client_{i+1}'] = client_acc

            results['worst_client'] = min(client_accs)
            results['best_client'] = max(client_accs)
            results['client_std'] = np.std(client_accs)

    return results


def fedavg_aggregate_gpu(global_model, client_models, client_sizes):
    """GPU-accelerated FedAvg with proper BatchNorm handling"""
    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    for key in global_state.keys():
        if global_state[key].is_floating_point():
            global_state[key] = torch.zeros_like(global_state[key])

    for client_model, client_size in zip(client_models, client_sizes):
        client_state = client_model.state_dict()
        weight = client_size / total_size

        for key in global_state.keys():
            if not global_state[key].is_floating_point():
                # Integer tensors (BatchNorm num_batches_tracked)
                if client_model == client_models[0]:
                    global_state[key] = client_state[key].clone()
            else:
                # Float tensors: weighted average
                global_state[key] += client_state[key] * weight

    global_model.load_state_dict(global_state)


def create_federated_data_gpu(num_clients):
    """Create federated data setup with GPU support"""

    with open('phase0_corrected_config.json', 'r') as f:
        config_data = json.load(f)

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

    # Split training data across clients
    samples_per_client = len(train_images) // num_clients
    client_loaders = []

    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else len(train_images)

        client_images = train_images[start_idx:end_idx]
        client_labels = train_labels[start_idx:end_idx]

        client_dataset = TensorDataset(client_images, client_labels)
        # Ensure minimum batch size for BatchNorm
        client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True, drop_last=True)
        client_loaders.append(client_loader)

        print(f"  Client {client_id+1}: {len(client_images)} samples")

    test_loader = DataLoader(TensorDataset(test_images, test_labels),
                            batch_size=64, shuffle=False)

    return client_loaders, test_loader, config_data


def test_enhanced_multiclient(num_clients):
    """Enhanced multi-client test with performance tracking"""

    print(f"\n{'='*70}")
    print(f"ENHANCED {num_clients}-CLIENT FEDAVG (GPU + Performance Tracking)")
    print('='*70)

    # Create federated data
    client_loaders, test_loader, config_data = create_federated_data_gpu(num_clients)

    # Initialize models on GPU
    torch.manual_seed(42)
    factory = ImprovedModelFactory()

    # Global model
    global_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
    global_model = TrackedFedModel(global_backbone, 62).to(device)

    # Client models
    client_models = []
    client_optimizers = []
    client_sizes = []

    for i in range(num_clients):
        client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec("femnist"))
        client_model = TrackedFedModel(client_backbone, 62).to(device)
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)

        # CRITICAL: Persistent optimizer per client
        optimizer = optim.Adam(client_model.parameters(),
                              lr=config_data['learning_rate'], weight_decay=1e-4)
        client_optimizers.append(optimizer)

        # Track client data sizes for weighted aggregation
        client_size = sum(len(batch[1]) for batch in client_loaders[i])
        client_sizes.append(client_size)

    criterion = nn.CrossEntropyLoss()

    print(f"Training: 15 rounds x {config_data['local_epochs']} local epochs")
    print("Enhanced tracking: Global + per-client + worst-client performance")
    print()

    # Track performance over rounds
    performance_history = {'rounds': [], 'global': [], 'worst_client': [], 'client_std': []}

    # Federated training with enhanced tracking
    for fl_round in range(15):  # 15 rounds for good convergence
        # CLIENT TRAINING (parallel simulation)
        for client_id, (client_model, client_loader, optimizer) in enumerate(
            zip(client_models, client_loaders, client_optimizers)):

            client_model.train()

            # Local training
            for local_epoch in range(config_data['local_epochs']):
                for batch_x, batch_y in client_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = client_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        # AGGREGATION
        fedavg_aggregate_gpu(global_model, client_models, client_sizes)

        # UPDATE CLIENTS with global model
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # ENHANCED EVALUATION
        results = evaluate_detailed(global_model, test_loader, client_loaders)

        performance_history['rounds'].append(fl_round + 1)
        performance_history['global'].append(results['global'])
        performance_history['worst_client'].append(results['worst_client'])
        performance_history['client_std'].append(results['client_std'])

        print(f"  Round {fl_round+1:2d}: Global={results['global']:.3f}, "
              f"Worst={results['worst_client']:.3f}, Std={results['client_std']:.3f}")

    # Final detailed evaluation
    final_results = evaluate_detailed(global_model, test_loader, client_loaders)

    print(f"\nFINAL RESULTS:")
    print(f"  Global accuracy:     {final_results['global']:.1%}")
    print(f"  Worst client:        {final_results['worst_client']:.1%}")
    print(f"  Best client:         {final_results['best_client']:.1%}")
    print(f"  Client std dev:      {final_results['client_std']:.3f}")
    print(f"  Performance gap:     {final_results['best_client'] - final_results['worst_client']:.1%}")

    return final_results, performance_history


def main():
    """Phase 2 Enhanced: Multi-client federation with performance tracking"""

    print("PHASE 2 ENHANCED: Multi-Client Federation Performance Analysis")
    print("=" * 80)
    print(f"Device: {device}")
    print("Enhanced monitoring: Per-client accuracy + worst-client tracking")
    print()

    results = {}
    histories = {}

    for num_clients in [1, 2, 4]:
        final_res, history = test_enhanced_multiclient(num_clients)
        results[num_clients] = final_res
        histories[num_clients] = history

    # PHASE 2 ANALYSIS
    print("\n" + "=" * 80)
    print("PHASE 2 ANALYSIS: Multi-Client Federation Results")
    print("=" * 80)

    baseline_global = results[1]['global']

    print("GLOBAL ACCURACY:")
    for num_clients in [1, 2, 4]:
        global_acc = results[num_clients]['global']
        degradation = baseline_global - global_acc
        print(f"  {num_clients} client(s): {global_acc:.1%} (vs baseline: {degradation:+.1%})")

    print("\nWORST-CLIENT PERFORMANCE:")
    for num_clients in [1, 2, 4]:
        worst_acc = results[num_clients]['worst_client']
        print(f"  {num_clients} client(s): {worst_acc:.1%}")

    print("\nCLIENT FAIRNESS (Best - Worst):")
    for num_clients in [1, 2, 4]:
        if num_clients > 1:  # Only meaningful for multi-client
            fairness_gap = results[num_clients]['best_client'] - results[num_clients]['worst_client']
            print(f"  {num_clients} client(s): {fairness_gap:.1%}")

    # Decision criteria
    max_global_degradation = max(abs(baseline_global - results[2]['global']),
                                abs(baseline_global - results[4]['global']))
    worst_client_min = min(results[2]['worst_client'], results[4]['worst_client'])

    print(f"\nAGGREGATION EFFICIENCY:")
    print(f"  Max global degradation: {max_global_degradation:.1%}")
    print(f"  Minimum worst-client:   {worst_client_min:.1%}")

    if max_global_degradation < 0.10 and worst_client_min > 0.50:  # 10% max degradation, 50% min worst-client
        print("\nVERDICT: ✅ Multi-client federation VALIDATED")
        print("READY FOR RESEARCH VALIDATION: SOTA baselines comparison")
        return True
    else:
        print("\nVERDICT: ❌ Multi-client federation needs optimization")
        print("Issues: Aggregation inefficiency or client fairness problems")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nPHASE 2 STATUS: {'VALIDATED - Ready for SOTA comparison' if success else 'NEEDS DEBUGGING'}")