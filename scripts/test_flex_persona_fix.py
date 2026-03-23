#!/usr/bin/env python3
"""Apply the architectural fix to FLEX-Persona backbone and test full system."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.model_factory import ModelFactory
from flex_persona.data.client_data_manager import ClientDataManager
from flex_persona.data.dataset_registry import DatasetRegistry

def test_full_flex_persona_with_fix():
    """Test complete FLEX-Persona system with the architectural fix applied."""
    print("="*70)
    print("TESTING FULL FLEX-PERSONA WITH ARCHITECTURAL FIX")
    print("="*70)

    # First, let's create a patched version for testing
    # (In production, we'd update the actual backbones.py file)

    print("Creating temporary fixed backbone for testing...")

    # Monkey-patch the SmallCNNBackbone for this test
    from flex_persona.models import backbones

    original_backbone_class = backbones.SmallCNNBackbone

    class FixedSmallCNNBackbone(nn.Module):
        """Fixed SmallCNNBackbone that preserves spatial information."""
        def __init__(self, in_channels: int = 3) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # CRITICAL FIX: Use Flatten instead of AdaptiveAvgPool2d
                nn.Flatten(),
            )
            # Fixed output dimension
            self.output_dim = 128 * 7 * 7  # 6272

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.features(x)

    # Temporarily replace the backbone class
    backbones.SmallCNNBackbone = FixedSmallCNNBackbone

    try:
        # Test the full FLEX-Persona pipeline with the fix
        config = ExperimentConfig(dataset_name="femnist")
        config.model.num_classes = 62
        config.num_clients = 2  # Small test
        config.training.max_samples_per_client = 500  # Limited for speed
        config.training.learning_rate = 0.001
        config.training.batch_size = 32

        print(f"Configuration:")
        print(f"  - Dataset: {config.dataset_name}")
        print(f"  - Classes: {config.model.num_classes}")
        print(f"  - Clients: {config.num_clients}")
        print(f"  - Max samples per client: {config.training.max_samples_per_client}")
        print()

        # Test single client with fixed architecture
        print("Testing single client with fixed FLEX-Persona architecture...")

        # Create client data
        manager = ClientDataManager(project_root, config)
        bundles = manager.build_client_bundles()
        client_bundle = bundles[0]

        print(f"Client data: {client_bundle.num_samples} samples, {len(client_bundle.class_histogram)} classes")

        # Create model using ModelFactory (now with fixed backbone)
        model = ModelFactory.build_client_model(
            client_id=0,
            model_config=config.model,
            dataset_name=config.dataset_name,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Backbone output dim: {model.backbone.output_dim}")
        print(f"Adapter: {model.adapter.input_dim} -> {model.adapter.shared_dim}")
        print(f"Classifier: {model.classifier.in_features} -> {model.classifier.out_features}")

        # Test forward pass
        test_batch_x, test_batch_y = next(iter(client_bundle.train_loader))
        print(f"\nTesting forward pass:")
        print(f"Input: {test_batch_x.shape}")

        with torch.no_grad():
            # Test FLEX-Persona forward methods
            features = model.extract_features(test_batch_x)
            shared_repr = model.project_shared(features)
            task_output = model.forward_task(test_batch_x)

            print(f"Features: {features.shape}, range=[{features.min():.3f}, {features.max():.3f}], std={features.std():.3f}")
            print(f"Shared repr: {shared_repr.shape}, range=[{shared_repr.min():.3f}, {shared_repr.max():.3f}], std={shared_repr.std():.3f}")
            print(f"Task output: {task_output.shape}, range=[{task_output.min():.3f}, {task_output.max():.3f}], std={task_output.std():.3f}")

        # Quick training test
        print(f"\nTraining test (3 epochs):")
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(3):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in client_bundle.train_loader:
                optimizer.zero_grad()

                logits = model.forward_task(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            # Evaluate
            model.eval()
            eval_correct = 0
            eval_total = 0
            with torch.no_grad():
                for batch_x, batch_y in client_bundle.eval_loader:
                    logits = model.forward_task(batch_x)
                    preds = torch.argmax(logits, dim=1)
                    eval_correct += (preds == batch_y).sum().item()
                    eval_total += batch_y.size(0)
            model.train()

            train_acc = correct / total
            eval_acc = eval_correct / eval_total
            avg_loss = epoch_loss / len(client_bundle.train_loader)

            print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Eval={eval_acc:.4f}, Loss={avg_loss:.4f}")

        final_eval_acc = eval_acc
        print(f"\nFinal evaluation accuracy: {final_eval_acc:.4f} ({final_eval_acc:.1%})")

        # Results analysis
        print(f"\n{'='*70}")
        print("RESULTS ANALYSIS")
        print('='*70)

        if final_eval_acc > 0.5:
            print("[EXCELLENT] >50% accuracy - Fix completely resolves the issue!")
            print("[SUCCESS] FLEX-Persona now ready for proper federated experiments")
            print(f"Expected federated performance: 70%+ (vs previous 7%)")
        elif final_eval_acc > 0.3:
            print("[GOOD] 30-50% accuracy - Major improvement, may need hyperparameter tuning")
        elif final_eval_acc > 0.15:
            print("[IMPROVED] 15-30% accuracy - Significant improvement but still issues")
        else:
            print("[ISSUES REMAIN] <15% accuracy - Additional problems need fixing")

        print(f"\nNext steps:")
        if final_eval_acc > 0.3:
            print("1. ✅ Apply this fix to the actual SmallCNNBackbone class")
            print("2. ✅ Update adapter input dimension to handle new backbone size")
            print("3. ✅ Re-run comprehensive federated learning experiments")
            print("4. ✅ Should now achieve proper FEMNIST baseline performance")
        else:
            print("1. Investigate remaining architectural issues")
            print("2. Check for additional FLEX-Persona implementation problems")

    finally:
        # Restore original backbone class
        backbones.SmallCNNBackbone = original_backbone_class
        print(f"\n[NOTE] Backbone class restored to original (fix was temporary for testing)")

if __name__ == "__main__":
    test_full_flex_persona_with_fix()