#!/usr/bin/env python3
"""Instrument SCAFFOLD to log K values"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.data.client_data_manager import ClientDataManager
import torch

# Config
alpha = 0.1
num_clients = 10
batch_size = 64
max_samples = 4000
local_epochs = 1

cfg = ExperimentConfig(
    experiment_name=f"instrument_scaffold",
    dataset_name="femnist",
    num_clients=num_clients,
    random_seed=42,
    partition_mode="dirichlet",
    dirichlet_alpha=alpha,
    output_dir=str(PROJECT_ROOT / "outputs"),
)
cfg.training.batch_size = batch_size
cfg.training.max_samples_per_client = max_samples // num_clients

dm = ClientDataManager(str(PROJECT_ROOT), cfg)
bundles = dm.build_client_bundles()

# Inspect batch sizes
batch_sizes = []
for bundle in bundles:
    num_batches = len(bundle.train_loader)
    batch_sizes.append(num_batches)
    K = local_epochs * num_batches
    print(f"Client {bundle.client_id}: train_loader={num_batches} batches, K={K}, 1/(K*0.003)={1/(K*0.003):.1f}x")

print()
avg_batches = sum(batch_sizes) / len(batch_sizes)
avg_K = local_epochs * avg_batches
avg_scale = 1 / (avg_K * 0.003)
print(f"Average: {avg_batches:.1f} batches/client, K={avg_K:.1f}, 1/(K*lr)={avg_scale:.1f}x")
print()
print(f"If control/grad ratio is 372x, then we'd need:")
print(f"  K = 1 / (372 * 0.003) = {1/(372*0.003):.1f}")
print(f"  Or multiply denominator by: 372 / {avg_scale:.1f} = {372/avg_scale:.1f}x")
