#!/usr/bin/env python3
"""Test federated data splitting to identify potential issues."""

import sys
from pathlib import Path
from collections import Counter
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager

def test_federated_data_splitting():
    """Test how data is split among federated clients."""
    print("="*70)
    print("FEDERATED DATA SPLITTING ANALYSIS")
    print("="*70)

    # Test with different configurations
    configs = [
        ("Normal", {"clients": 10, "max_samples": None}),
        ("Limited samples", {"clients": 10, "max_samples": 200}),
        ("Many clients", {"clients": 20, "max_samples": None}),
    ]

    for config_name, config_params in configs:
        print(f"\n{config_name} Configuration:")
        print("-" * 50)

        # Create config
        config = ExperimentConfig(dataset_name="femnist")
        config.num_clients = config_params["clients"]
        config.training.max_samples_per_client = config_params["max_samples"]
        config.model.num_classes = 62

        print(f"  Settings: {config_params['clients']} clients, max_samples={config_params['max_samples']}")

        # Build client data
        try:
            manager = ClientDataManager(project_root, config)
            bundles = manager.build_client_bundles()

            print(f"  Created {len(bundles)} client bundles")

            # Analyze each client
            total_samples = 0
            total_classes = set()
            client_stats = []

            for i, bundle in enumerate(bundles):
                stats = {
                    'client_id': bundle.client_id,
                    'samples': bundle.num_samples,
                    'classes': len(bundle.class_histogram),
                    'class_list': sorted(bundle.class_histogram.keys()),
                    'train_batches': len(bundle.train_loader),
                    'eval_batches': len(bundle.eval_loader),
                }

                # Check class distribution
                hist = bundle.class_histogram
                if hist:
                    stats['most_common_class'] = max(hist.values())
                    stats['least_common_class'] = min(hist.values())
                    stats['class_imbalance'] = stats['most_common_class'] / stats['least_common_class']
                else:
                    stats['most_common_class'] = 0
                    stats['least_common_class'] = 0
                    stats['class_imbalance'] = float('inf')

                client_stats.append(stats)
                total_samples += bundle.num_samples
                total_classes.update(bundle.class_histogram.keys())

            # Summary statistics
            sample_counts = [s['samples'] for s in client_stats]
            class_counts = [s['classes'] for s in client_stats]

            print(f"  Total samples: {total_samples}")
            print(f"  Total unique classes across clients: {len(total_classes)}")
            print(f"  Samples per client: min={min(sample_counts)}, max={max(sample_counts)}, avg={np.mean(sample_counts):.1f}")
            print(f"  Classes per client: min={min(class_counts)}, max={max(class_counts)}, avg={np.mean(class_counts):.1f}")

            # Show individual client details
            print(f"\n  Individual client analysis:")
            for stats in client_stats[:5]:  # Show first 5 clients
                print(f"    Client {stats['client_id']}: {stats['samples']} samples, {stats['classes']} classes")
                print(f"      Classes: {stats['class_list'][:10]}{'...' if len(stats['class_list']) > 10 else ''}")

                if stats['class_imbalance'] != float('inf'):
                    print(f"      Class imbalance: {stats['class_imbalance']:.1f}x")
                else:
                    print(f"      No class data")

                # Test data loader
                try:
                    bundle = bundles[stats['client_id']]
                    batch_x, batch_y = next(iter(bundle.train_loader))
                    unique_labels_in_batch = len(torch.unique(batch_y))
                    print(f"      Sample batch: {batch_x.shape}, {unique_labels_in_batch} unique labels")
                except Exception as e:
                    print(f"      Data loader error: {e}")

            # Check for potential issues
            print(f"\n  Potential issues detected:")
            issues_found = False

            if min(sample_counts) < 50:
                print(f"    - Some clients have very few samples ({min(sample_counts)})")
                issues_found = True

            if min(class_counts) < 5:
                print(f"    - Some clients have very few classes ({min(class_counts)})")
                issues_found = True

            # Check class coverage
            expected_classes = set(range(62))  # FEMNIST has 0-61
            missing_classes = expected_classes - total_classes
            if missing_classes:
                print(f"    - Missing classes from dataset: {sorted(missing_classes)}")
                issues_found = True

            # Check for empty clients
            empty_clients = [s for s in client_stats if s['samples'] == 0]
            if empty_clients:
                print(f"    - Empty clients: {[s['client_id'] for s in empty_clients]}")
                issues_found = True

            # Check extreme class imbalance within clients
            high_imbalance = [s for s in client_stats if s['class_imbalance'] > 50]
            if high_imbalance:
                print(f"    - Clients with extreme class imbalance (>50x): {[s['client_id'] for s in high_imbalance]}")
                issues_found = True

            if not issues_found:
                print(f"    No major issues detected")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            print(traceback.format_exc())

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Key things to check for federated learning performance:")
    print("1. Each client should have reasonable number of samples (>50)")
    print("2. Each client should have multiple classes (>5)")
    print("3. No clients should be empty")
    print("4. Class imbalance within clients shouldn't be extreme (>100x)")
    print("5. All expected classes should appear across clients")

if __name__ == "__main__":
    test_federated_data_splitting()