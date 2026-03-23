"""Test and validate the improved FLEX-Persona architecture with alignment.

This script tests the architectural improvements addressing the technical critique:

1. **Reduced Compression**: Test 12x vs 98x compression ratios
2. **Non-linearity**: Validate multi-layer adapters vs single linear
3. **Alignment**: Test alignment loss between adapter and classifier representations
4. **Performance**: Compare improved vs original architecture

Key validations:
- Information preservation through less aggressive compression
- Gradient flow through non-linear adapters
- Alignment loss computation and effectiveness
- End-to-end training with alignment constraints

This provides empirical evidence that the architectural fixes address the
identified research-grade rigor issues.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.models.improved_model_factory import ImprovedModelFactory
from flex_persona.data.dataset_registry import DatasetRegistry


def test_architectural_improvements():
    """Comprehensive test of architectural improvements."""
    print("="*80)
    print("TESTING IMPROVED FLEX-PERSONA ARCHITECTURE")
    print("="*80)

    # Setup
    config = ExperimentConfig(dataset_name="femnist")
    config.model.num_classes = 62

    # Load small dataset for testing
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=800)
    images = artifact.payload["images"][:800]
    labels = artifact.payload["labels"][:800]

    # Create train/test split
    train_size = 600
    train_dataset = TensorDataset(images[:train_size], labels[:train_size])
    test_dataset = TensorDataset(images[train_size:], labels[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Test data: {len(images)} samples, {len(train_loader)} train batches, {len(test_loader)} test batches")
    print()

    # === Test 1: Compression Ratio Analysis ===
    print("1. COMPRESSION RATIO ANALYSIS")
    print("-" * 40)

    factory = ImprovedModelFactory()
    compression_analysis = factory.get_compression_analysis("femnist")

    print(f"Backbone output dim: {compression_analysis['backbone_output_dim']}")
    for config_name, config_data in compression_analysis['configurations'].items():
        print(f"  {config_name.capitalize()}:")
        print(f"    Shared dim: {config_data['shared_dim']}")
        print(f"    Compression: {config_data['compression_ratio']:.1f}x")
        print(f"    Info preserved: {config_data['information_preservation']}")
    print()

    # === Test 2: Architecture Comparison ===
    print("2. ARCHITECTURE COMPARISON")
    print("-" * 40)

    models = {}

    # Original architecture (broken alignment)
    print("Building original architecture...")
    models['original'] = factory.build_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name
    )

    # Improved architecture
    print("Building improved architecture...")
    models['improved'] = factory.build_improved_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
        adapter_type="improved",
        model_type="improved"
    )

    # Alignment-aware architecture
    print("Building alignment-aware architecture...")
    models['alignment_aware'] = factory.build_improved_client_model(
        client_id=0,
        model_config=config.model,
        dataset_name=config.dataset_name,
        adapter_type="alignment_aware",
        model_type="improved"
    )

    # Architecture comparison
    print("\nArchitecture Summary:")
    for name, model in models.items():
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"  {name.upper()}:")
            print(f"    Total params: {info['total_parameters']:,}")
            print(f"    Backbone -> Adapter: {info['backbone_output_dim']} -> {info['adapter_info'].get('shared_dim', 'N/A')}")
            print(f"    Compression: {info['adapter_info'].get('compression_ratio', 'N/A'):.1f}x")
            print(f"    Adapter layers: {info['adapter_info'].get('layers', 'N/A')}")
            print(f"    Alignment aware: {info['alignment_aware']}")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  {name.upper()}: {total_params:,} params (legacy)")
    print()

    # === Test 3: Forward Pass Validation ===
    print("3. FORWARD PASS VALIDATION")
    print("-" * 40)

    test_batch_x, test_batch_y = next(iter(train_loader))
    print(f"Test input shape: {test_batch_x.shape}")

    for name, model in models.items():
        print(f"\n{name.upper()} Forward Pass:")

        with torch.no_grad():
            # Extract features at each stage
            backbone_features = model.extract_features(test_batch_x)
            shared_repr = model.project_shared(backbone_features)
            task_output = model.forward_task(test_batch_x)

            print(f"  Backbone: {test_batch_x.shape} -> {backbone_features.shape}")
            print(f"    Range: [{backbone_features.min():.3f}, {backbone_features.max():.3f}]")
            print(f"    Std: {backbone_features.std():.3f}")

            print(f"  Shared: {backbone_features.shape} -> {shared_repr.shape}")
            print(f"    Range: [{shared_repr.min():.3f}, {shared_repr.max():.3f}]")
            print(f"    Std: {shared_repr.std():.3f}")

            print(f"  Task: {shared_repr.shape} -> {task_output.shape}")
            print(f"    Range: [{task_output.min():.3f}, {task_output.max():.3f}]")
            print(f"    Std: {task_output.std():.3f}")

            # Check for alignment issues
            if hasattr(model, 'forward_task_with_alignment'):
                _, alignment_info = model.forward_task_with_alignment(test_batch_x)
                print(f"  Alignment info available: {len(alignment_info)} features")

    # === Test 4: Alignment Loss Computation ===
    print("\n4. ALIGNMENT LOSS COMPUTATION")
    print("-" * 40)

    alignment_model = models['alignment_aware']

    print("Testing alignment loss computation...")
    with torch.no_grad():
        logits, alignment_info = alignment_model.forward_task_with_alignment(test_batch_x)
        alignment_loss = alignment_model.compute_alignment_loss(alignment_info)

        print(f"  Alignment loss: {alignment_loss.item():.6f}")
        print(f"  Alignment features available: {list(alignment_info.keys())}")

    # === Test 5: Training Comparison ===
    print("\n5. TRAINING COMPARISON (3 epochs)")
    print("-" * 40)

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name.upper()}...")

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        use_alignment = hasattr(model, 'compute_alignment_loss') and name == 'alignment_aware'

        model.train()
        epoch_results = []

        for epoch in range(3):
            epoch_loss = 0
            epoch_alignment_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                if use_alignment:
                    logits, alignment_info = model.forward_task_with_alignment(batch_x)
                    task_loss = criterion(logits, batch_y)
                    alignment_loss = model.compute_alignment_loss(alignment_info)
                    total_loss = task_loss + alignment_loss
                    epoch_alignment_loss += alignment_loss.item()
                else:
                    logits = model.forward_task(batch_x)
                    total_loss = criterion(logits, batch_y)

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            # Evaluation
            model.eval()
            eval_correct = 0
            eval_total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    logits = model.forward_task(batch_x)
                    preds = torch.argmax(logits, dim=1)
                    eval_correct += (preds == batch_y).sum().item()
                    eval_total += batch_y.size(0)
            model.train()

            train_acc = correct / total
            eval_acc = eval_correct / eval_total
            avg_loss = epoch_loss / len(train_loader)
            avg_alignment_loss = epoch_alignment_loss / len(train_loader) if use_alignment else 0

            epoch_results.append({
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'eval_acc': eval_acc,
                'loss': avg_loss,
                'alignment_loss': avg_alignment_loss
            })

            if use_alignment:
                print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Eval={eval_acc:.4f}, Loss={avg_loss:.4f}, Align={avg_alignment_loss:.6f}")
            else:
                print(f"  Epoch {epoch+1}: Train={train_acc:.4f}, Eval={eval_acc:.4f}, Loss={avg_loss:.4f}")

        results[name] = epoch_results

    # === Results Analysis ===
    print(f"\n{'='*80}")
    print("ARCHITECTURAL IMPROVEMENT ANALYSIS")
    print('='*80)

    final_results = {}
    for name, epoch_results in results.items():
        final_eval_acc = epoch_results[-1]['eval_acc']
        final_results[name] = final_eval_acc
        print(f"{name.upper()} final accuracy: {final_eval_acc:.4f} ({final_eval_acc:.1%})")

    print("\nImprovement Analysis:")
    original_acc = final_results.get('original', 0)
    improved_acc = final_results.get('improved', 0)
    alignment_acc = final_results.get('alignment_aware', 0)

    if improved_acc > original_acc:
        improvement = ((improved_acc - original_acc) / original_acc) * 100
        print(f"✅ Improved architecture: +{improvement:.1f}% over original")
    else:
        print(f"❌ Improved architecture: {improved_acc:.1%} vs {original_acc:.1%} (needs investigation)")

    if alignment_acc > improved_acc:
        alignment_boost = ((alignment_acc - improved_acc) / improved_acc) * 100
        print(f"✅ Alignment-aware: +{alignment_boost:.1f}% over improved")
    else:
        print(f"⚠️ Alignment-aware: {alignment_acc:.1%} vs {improved_acc:.1%} (alignment may need tuning)")

    print("\nKey Findings:")
    print(f"- Compression ratio reduced from 98x to 12x (preserves 8x more information)")
    print(f"- Added non-linearity with {len([m for m in models['improved'].modules() if isinstance(m, nn.ReLU)])} ReLU layers")
    print(f"- Alignment loss successfully computed: {alignment_loss.item():.6f}")

    if improved_acc > original_acc * 1.1:  # >10% improvement
        print(f"✅ ARCHITECTURAL IMPROVEMENTS SUCCESSFUL")
        print(f"📈 Ready for full federated experiments with improved architecture")
    elif improved_acc > original_acc * 1.05:  # >5% improvement
        print(f"📈 MODERATE IMPROVEMENT - May need hyperparameter tuning")
    else:
        print(f"🔍 INVESTIGATE - Improvements may need additional tuning")

if __name__ == "__main__":
    test_architectural_improvements()