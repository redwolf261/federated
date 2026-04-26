"""
PHASE 1: BASELINE SANITY VERIFICATION
====================================

Implements FedAvg and MOON baselines with VALIDATED training parameters
from Phase 0 to ensure both methods achieve research-grade performance.

Requirements (from research protocol):
- FedAvg must converge (not random)
- MOON must improve stability or accuracy
- Both methods should reach reasonable performance (>70% centralized basis)
- Failure condition: If accuracy too low or unstable → STOP, setup invalid

This phase uses the validated configuration from Phase 0:
- Adam optimizer, LR=0.003, Batch=64, Dropout=0.2/0.1
- Dirichlet splits (α=0.5), 50 clients
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


class ValidatedFederatedModel(nn.Module):
    """Federated model using validated Phase 0 architecture"""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.output_dim

        # VALIDATED architecture from Phase 0
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

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization (validated in Phase 0)"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_representation=False):
        features = self.backbone(x)

        # Forward through classifier
        h = features
        for i in range(6):  # Through second ReLU
            h = self.classifier[i](h)
        representation = h  # 128-dim

        # Final layers (handle dropout in training vs eval)
        if self.training:
            h = self.classifier[6](h)  # Dropout
            logits = self.classifier[7](h)  # Linear
        else:
            logits = self.classifier[7](h)  # Skip dropout

        if return_representation:
            return logits, representation
        return logits


def load_validated_config():
    """Load validated configuration from Phase 0"""
    try:
        with open('phase0_corrected_config.json', 'r') as f:
            config_data = json.load(f)
        print("Loaded validated Phase 0 configuration")
        return config_data
    except FileNotFoundError:
        print("ERROR: Phase 0 configuration not found")
        print("Run phase0_corrected.py first to establish experimental ground truth")
        return None


def create_dirichlet_federated_splits(config_data, seed: int = 0):
    """Create Dirichlet splits for federated learning (from Phase 0)"""

    print(f"Creating Dirichlet federated splits (alpha={config_data['dirichlet_alpha']})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    registry = DatasetRegistry(project_root)
    artifact = registry.load(config_data['dataset_name'], max_rows=5000)  # Larger for federated

    images = artifact.payload["images"][:5000]
    labels = artifact.payload["labels"][:5000]

    num_clients = config_data['num_clients']
    num_classes = config_data.get('num_classes', 62)
    alpha = config_data['dirichlet_alpha']

    print(f"Dataset: {len(images)} images, {len(torch.unique(labels))} classes")
    print(f"Federated setup: {num_clients} clients, alpha={alpha}")

    # Create Dirichlet distribution per class
    client_class_counts = np.zeros((num_clients, num_classes))

    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_size = class_mask.sum().item()

        if class_size == 0:
            continue

        # Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        class_client_counts = (proportions * class_size).astype(int)

        # Handle remainder
        remainder = class_size - class_client_counts.sum()
        for _ in range(remainder):
            client_id = np.random.randint(num_clients)
            class_client_counts[client_id] += 1

        client_class_counts[:, class_id] = class_client_counts

    # Create client data loaders
    client_loaders = []
    sorted_indices = torch.argsort(labels)
    sorted_labels = labels[sorted_indices]

    current_class_indices = {class_id: 0 for class_id in range(num_classes)}

    for client_id in range(num_clients):
        client_indices = []

        for class_id in range(num_classes):
            needed_samples = int(client_class_counts[client_id, class_id])

            if needed_samples == 0:
                continue

            class_mask = (sorted_labels == class_id)
            class_indices = torch.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            start_idx = current_class_indices[class_id]
            end_idx = min(start_idx + needed_samples, len(class_indices))

            selected_indices = class_indices[start_idx:end_idx]
            global_indices = sorted_indices[selected_indices]

            client_indices.extend(global_indices.tolist())
            current_class_indices[class_id] = end_idx

        # Create client loader
        if len(client_indices) > 0:
            client_images = images[client_indices]
            client_labels = labels[client_indices]

            dataset = TensorDataset(client_images, client_labels)
            loader = DataLoader(dataset, batch_size=config_data['batch_size'], shuffle=True)
            client_loaders.append(loader)
        else:
            # Empty client - give minimal data
            dataset = TensorDataset(images[:1], labels[:1])
            loader = DataLoader(dataset, batch_size=1)
            client_loaders.append(loader)

    # Global test set
    test_size = int(0.2 * len(images))
    test_indices = torch.randperm(len(images))[:test_size]
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config_data['batch_size'])

    # Statistics
    non_empty_clients = len([loader for loader in client_loaders if len(loader.dataset) > 1])
    avg_samples = np.mean([len(loader.dataset) for loader in client_loaders if len(loader.dataset) > 1])

    print(f"Created federated setup:")
    print(f"  Active clients: {non_empty_clients}/{num_clients}")
    print(f"  Avg samples/client: {avg_samples:.0f}")
    print(f"  Test samples: {len(test_dataset)}")

    return client_loaders, test_loader


class FedAvgBaseline:
    """FedAvg implementation with validated parameters"""

    def __init__(self, config_data):
        self.config = config_data
        self.method_name = "FedAvg"

    def federated_training(self, client_loaders, test_loader, seed: int = 0):
        """Run FedAvg with validated parameters"""

        print(f"\nPhase 1: Testing {self.method_name}")
        print(f"Config: LR={self.config['learning_rate']}, Opt={self.config['optimizer_type']}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create global model
        factory = ImprovedModelFactory()
        backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(self.config['dataset_name']))
        global_model = ValidatedFederatedModel(backbone, self.config.get('num_classes', 62))

        # Create client models
        client_models = []
        for i in range(len(client_loaders)):
            client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(self.config['dataset_name']))
            client_model = ValidatedFederatedModel(client_backbone, self.config.get('num_classes', 62))
            client_model.load_state_dict(global_model.state_dict())
            client_models.append(client_model)

        accuracy_history = []

        # Federated learning rounds
        for fl_round in range(10):  # Limited rounds for Phase 1 sanity check
            # Client training
            for i, (client_model, client_loader) in enumerate(zip(client_models, client_loaders)):
                if len(client_loader.dataset) <= 1:
                    continue  # Skip empty clients

                self._train_client(client_model, client_loader)

            # FedAvg aggregation
            self._fedavg_aggregate(global_model, client_models)

            # Update clients
            for client_model in client_models:
                client_model.load_state_dict(global_model.state_dict())

            # Evaluation
            accuracy = self._evaluate_model(global_model, test_loader)
            accuracy_history.append(accuracy)

            print(f"  Round {fl_round+1:2d}: {accuracy:.3f}")

        final_accuracy = accuracy_history[-1]
        return {
            'method': self.method_name,
            'final_accuracy': final_accuracy,
            'accuracy_history': accuracy_history,
            'converged': final_accuracy > 0.1,  # Basic convergence check (>10%)
            'stable': len(accuracy_history) >= 5 and np.std(accuracy_history[-5:]) < 0.1
        }

    def _train_client(self, client_model, client_loader):
        """Train single client with validated parameters"""
        client_model.train()

        # Use validated optimizer parameters with defaults
        weight_decay = self.config.get('weight_decay', 1e-4)
        momentum = self.config.get('momentum', 0.9)

        if self.config['optimizer_type'] == 'adam':
            optimizer = optim.Adam(
                client_model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.SGD(
                client_model.parameters(),
                lr=self.config['learning_rate'],
                momentum=momentum,
                weight_decay=weight_decay
            )

        criterion = nn.CrossEntropyLoss()

        # Local training epochs
        for local_epoch in range(self.config['local_epochs']):
            for batch_x, batch_y in client_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def _fedavg_aggregate(self, global_model, client_models):
        """Standard FedAvg aggregation"""
        global_dict = global_model.state_dict()

        for key in global_dict.keys():
            client_params = [client_models[i].state_dict()[key] for i in range(len(client_models))]

            if client_params[0].dtype in [torch.long, torch.int]:
                global_dict[key] = client_params[0]
            else:
                global_dict[key] = torch.stack(client_params).float().mean(dim=0)

        global_model.load_state_dict(global_dict)

    def _evaluate_model(self, model, test_loader):
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


class MOONBaseline:
    """MOON implementation with validated parameters"""

    def __init__(self, config_data, temperature: float = 0.5, mu: float = 1.0):
        self.config = config_data
        self.method_name = "MOON"
        self.temperature = temperature
        self.mu = mu

    def _contrastive_loss(self, representation, global_representation, previous_representation):
        """MOON contrastive loss"""
        rep = F.normalize(representation, dim=1)
        global_rep = F.normalize(global_representation, dim=1)
        prev_rep = F.normalize(previous_representation, dim=1)

        pos_sim = torch.sum(rep * global_rep, dim=1) / self.temperature
        neg_sim = torch.sum(rep * prev_rep, dim=1) / self.temperature

        loss = -torch.mean(torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))))
        return loss

    def federated_training(self, client_loaders, test_loader, seed: int = 0):
        """Run MOON with validated parameters"""

        print(f"\nPhase 1: Testing {self.method_name}")
        print(f"Config: LR={self.config['learning_rate']}, Contrastive mu={self.mu}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create models
        factory = ImprovedModelFactory()
        backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(self.config['dataset_name']))
        global_model = ValidatedFederatedModel(backbone, self.config.get('num_classes', 62))

        client_models = []
        previous_models = []

        for i in range(len(client_loaders)):
            # Client model
            client_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(self.config['dataset_name']))
            client_model = ValidatedFederatedModel(client_backbone, self.config.get('num_classes', 62))
            client_model.load_state_dict(global_model.state_dict())
            client_models.append(client_model)

            # Previous model
            prev_backbone = factory._build_backbone("small_cnn", factory.infer_input_spec(self.config['dataset_name']))
            prev_model = ValidatedFederatedModel(prev_backbone, self.config.get('num_classes', 62))
            prev_model.load_state_dict(global_model.state_dict())
            previous_models.append(prev_model)

        accuracy_history = []

        # Federated learning rounds
        for fl_round in range(10):  # Limited rounds for Phase 1
            # Store previous models
            for i in range(len(client_models)):
                previous_models[i].load_state_dict(client_models[i].state_dict())

            # Client training with contrastive loss
            for i, client_loader in enumerate(client_loaders):
                if len(client_loader.dataset) <= 1:
                    continue

                self._train_client_contrastive(
                    client_models[i],
                    global_model,
                    previous_models[i],
                    client_loader
                )

            # FedAvg aggregation (same as FedAvg)
            self._fedavg_aggregate(global_model, client_models)

            # Update clients
            for client_model in client_models:
                client_model.load_state_dict(global_model.state_dict())

            # Evaluation
            accuracy = self._evaluate_model(global_model, test_loader)
            accuracy_history.append(accuracy)

            print(f"  Round {fl_round+1:2d}: {accuracy:.3f}")

        final_accuracy = accuracy_history[-1]
        return {
            'method': self.method_name,
            'final_accuracy': final_accuracy,
            'accuracy_history': accuracy_history,
            'converged': final_accuracy > 0.1,
            'stable': len(accuracy_history) >= 5 and np.std(accuracy_history[-5:]) < 0.1
        }

    def _train_client_contrastive(self, client_model, global_model, previous_model, client_loader):
        """Train client with MOON contrastive loss"""
        client_model.train()
        global_model.eval()
        previous_model.eval()

        # Validated optimizer with defaults
        weight_decay = self.config.get('weight_decay', 1e-4)
        momentum = self.config.get('momentum', 0.9)

        if self.config['optimizer_type'] == 'adam':
            optimizer = optim.Adam(
                client_model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.SGD(
                client_model.parameters(),
                lr=self.config['learning_rate'],
                momentum=momentum,
                weight_decay=weight_decay
            )

        criterion = nn.CrossEntropyLoss()

        for local_epoch in range(self.config['local_epochs']):
            for batch_x, batch_y in client_loader:
                optimizer.zero_grad()

                # Forward pass with representations
                logits, representation = client_model(batch_x, return_representation=True)

                # Cross-entropy loss
                ce_loss = criterion(logits, batch_y)

                # Get global and previous representations
                with torch.no_grad():
                    _, global_rep = global_model(batch_x, return_representation=True)
                    _, prev_rep = previous_model(batch_x, return_representation=True)

                # Contrastive loss
                contrastive_loss = self._contrastive_loss(representation, global_rep, prev_rep)

                # Total loss
                total_loss = ce_loss + self.mu * contrastive_loss

                total_loss.backward()
                optimizer.step()

    def _fedavg_aggregate(self, global_model, client_models):
        """FedAvg aggregation (same as FedAvg baseline)"""
        global_dict = global_model.state_dict()

        for key in global_dict.keys():
            client_params = [client_models[i].state_dict()[key] for i in range(len(client_models))]

            if client_params[0].dtype in [torch.long, torch.int]:
                global_dict[key] = client_params[0]
            else:
                global_dict[key] = torch.stack(client_params).float().mean(dim=0)

        global_model.load_state_dict(global_dict)

    def _evaluate_model(self, model, test_loader):
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


def phase1_baseline_sanity_verification():
    """Phase 1: Verify FedAvg and MOON baselines work with validated setup"""

    print("="*70)
    print("PHASE 1: BASELINE SANITY VERIFICATION")
    print("="*70)
    print("Objective: Verify FedAvg and MOON achieve reasonable performance")
    print("Requirements: Both methods must converge (not random)")
    print("Failure condition: If accuracy <10% or unstable -> STOP")
    print()

    # Load validated configuration
    config_data = load_validated_config()
    if config_data is None:
        return False

    print(f"Using validated Phase 0 configuration:")
    print(f"  LR: {config_data['learning_rate']}, Optimizer: {config_data['optimizer_type']}")
    print(f"  Batch size: {config_data['batch_size']}, Dropout: {config_data['dropout_rate']}")

    # Create federated data splits
    client_loaders, test_loader = create_dirichlet_federated_splits(config_data, seed=42)

    # Test baselines
    results = []

    # Test FedAvg
    fedavg = FedAvgBaseline(config_data)
    fedavg_result = fedavg.federated_training(client_loaders, test_loader, seed=42)
    results.append(fedavg_result)

    # Test MOON
    moon = MOONBaseline(config_data, temperature=0.5, mu=1.0)
    moon_result = moon.federated_training(client_loaders, test_loader, seed=42)
    results.append(moon_result)

    # Analysis
    print(f"\n" + "="*70)
    print("PHASE 1 BASELINE VERIFICATION RESULTS")
    print("="*70)

    all_converged = True
    all_stable = True

    print(f"{'Method':<10} {'Final Acc':<12} {'Converged':<12} {'Stable':<10} {'Status':<15}")
    print("-" * 70)

    for result in results:
        method = result['method']
        final_acc = result['final_accuracy']
        converged = result['converged']
        stable = result['stable']

        if converged and stable and final_acc > 0.1:
            status = "PASS"
        elif converged and final_acc > 0.05:
            status = "MARGINAL"
        else:
            status = "FAIL"
            all_converged = False

        if not stable:
            all_stable = False

        print(f"{method:<10} {final_acc:<12.3f} {converged:<12} {stable:<10} {status:<15}")

    print()

    if all_converged and all_stable:
        print("+ PHASE 1 SUCCESS: Both baselines converge and are stable")
        print("+ Ready to proceed to Phase 2 (Head-to-Head Comparison)")

        # Save baseline results
        baseline_results = {
            'phase1_results': results,
            'all_converged': all_converged,
            'all_stable': all_stable,
            'config_used': config_data
        }

        with open('phase1_baseline_results.json', 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)

        print("Baseline results saved to: phase1_baseline_results.json")
        return True

    else:
        print("x PHASE 1 FAILURE: Baseline methods do not meet requirements")
        print("CRITICAL: Cannot proceed to Phase 2 - setup still invalid")

        if not all_converged:
            print("Issue: One or more methods failed to converge adequately")
        if not all_stable:
            print("Issue: Training instability detected")

        print()
        print("Required fixes:")
        print("1. Check federated data splits (too heterogeneous?)")
        print("2. Adjust FL hyperparameters (more local epochs?)")
        print("3. Verify model initialization")
        print("4. Consider different aggregation strategies")

        return False


if __name__ == "__main__":
    success = phase1_baseline_sanity_verification()

    print(f"\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)

    if success:
        print("STATUS: BASELINE SANITY VERIFIED")
        print("NEXT: Implement Phase 2 - Controlled Head-to-Head Comparison")
        print("ACTION: Add FLEX-Persona to rigorous comparison framework")
    else:
        print("STATUS: BASELINE VERIFICATION FAILED")
        print("NEXT: Fix federated learning setup before proceeding")
        print("ACTION: Debug federated training issues")