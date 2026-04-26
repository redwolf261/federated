#!/usr/bin/env python3
"""Test the standardized artifact manager with a minimal experimental run."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.artifact_manager import ExperimentArtifactManager, SeedResult

def test_artifact_manager():
    """Test the artifact manager with mock data."""

    print("Testing Standardized Artifact Manager")
    print("=" * 60)

    workspace = Path(__file__).parent.parent

    # Create test experiment
    artifact_mgr = ExperimentArtifactManager(workspace, "test_experiment")

    # Initialize config
    config = artifact_mgr.initialize_config(
        experiment_name="test_experiment_demo",
        description="Testing the standardized artifact management system",
        method="prototype",
        regime="high_het",
        dataset_name="femnist",
        seed_list=[11, 22, 33],
        num_clients=10,
        rounds=20,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.01,
        max_samples_per_client=256
    )

    print(f"Created experiment: {config.experiment_id}")
    print(f"Git commit: {config.git_commit_hash}")

    # Add mock seed results
    mock_results = [
        SeedResult(
            seed=11,
            method="prototype",
            regime="high_het",
            mean_accuracy=0.8234,
            worst_accuracy=0.7123,
            p10_accuracy=0.7456,
            bottom3_accuracy=0.7234,
            collapsed=False,
            collapsed_sensitive=False,
            stability_variance=0.0123,
            rounds_data=[
                {"round": 0, "mean_client_accuracy": 0.2345},
                {"round": 1, "mean_client_accuracy": 0.5678},
                {"round": 2, "mean_client_accuracy": 0.8234}
            ],
            execution_time_seconds=45.2
        ),
        SeedResult(
            seed=22,
            method="prototype",
            regime="high_het",
            mean_accuracy=0.8456,
            worst_accuracy=0.7345,
            p10_accuracy=0.7678,
            bottom3_accuracy=0.7456,
            collapsed=False,
            collapsed_sensitive=False,
            stability_variance=0.0098,
            rounds_data=[
                {"round": 0, "mean_client_accuracy": 0.2456},
                {"round": 1, "mean_client_accuracy": 0.5789},
                {"round": 2, "mean_client_accuracy": 0.8456}
            ],
            execution_time_seconds=47.8
        ),
        SeedResult(
            seed=33,
            method="prototype",
            regime="high_het",
            mean_accuracy=0.8012,
            worst_accuracy=0.6890,
            p10_accuracy=0.7123,
            bottom3_accuracy=0.6999,
            collapsed=False,
            collapsed_sensitive=False,
            stability_variance=0.0156,
            rounds_data=[
                {"round": 0, "mean_client_accuracy": 0.2123},
                {"round": 1, "mean_client_accuracy": 0.5456},
                {"round": 2, "mean_client_accuracy": 0.8012}
            ],
            execution_time_seconds=43.1
        )
    ]

    # Add results one by one (simulating real experiment)
    for result in mock_results:
        print(f"Adding result for seed {result.seed}...")
        artifact_mgr.add_seed_result(result)

    # Finalize experiment
    print("\nFinalizing experiment...")
    aggregates = artifact_mgr.finalize_experiment()

    # Display results
    print(f"\n[SUCCESS] Test completed successfully!")
    print(f"   - Experiment ID: {artifact_mgr.experiment_id}")
    print(f"   - Seeds processed: {len(mock_results)}")
    print(f"   - Mean accuracy: {aggregates.mean_accuracy_avg:.4f} ± {aggregates.mean_accuracy_std:.4f}")
    print(f"   - Collapse rate: {aggregates.collapse_rate:.1%}")
    print(f"   - Artifacts directory: {artifact_mgr.experiment_dir}")

    print(f"\nArtifacts created:")
    for file_path in artifact_mgr.experiment_dir.glob("*"):
        if file_path.is_file():
            print(f"   - {file_path.name}")

    print(f"\nExperiment registry updated at:")
    print(f"   - {workspace / 'experiments' / 'experiment_registry.json'}")

    return True

if __name__ == "__main__":
    success = test_artifact_manager()
    if success:
        print("\n[SUCCESS] Artifact manager test PASSED")
    else:
        print("\n❌ Artifact manager test FAILED")
        sys.exit(1)