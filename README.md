# FLEX-Persona

A comprehensive **federated learning framework** designed to enable collaborative model training across heterogeneous clients with non-IID data, while preserving data privacy.

## 🔍 The Problem

Traditional federated learning methods rely on **shared model architectures and parameter aggregation**, which has significant limitations:

- **Architectural Inflexibility**: All clients must run identical models, ignoring differences in computational capacity
- **Limited Data Heterogeneity**: Parameter averaging (FedAvg) struggles with non-IID data distributions
- **Poor Personalization**: A single global model often underperforms for individual clients with unique data patterns
- **Communication Overhead**: Sharing full model parameters across network boundaries is expensive
- **Worst-Client Problem**: Global optimization can harm specialized clients with distinct objectives

## 💡 The Solution

FLEX-Persona introduces a **representation-based collaboration** approach:

Instead of sharing model parameters, clients exchange **low-dimensional representation summaries** through **adapter networks** that map local features into a shared latent space. The server leverages these representations to:

1. **Cluster similar clients** using Wasserstein distance on prototype distributions
2. **Generate cluster-level knowledge** by aggregating cluster prototypes
3. **Guide personalized updates** by providing cluster guidance back to each client

This enables seamless cross-architecture collaboration while improving personalization and worst-client performance with less communication overhead.

## 🎯 Overview

FLEX-Persona enables federated learning experiments where:
- **Heterogeneous Clients**: Different model architectures per client (CNN, ResNet, MLP) collaborate without shared parameters
- **Non-IID Data**: Support for naturally partitioned datasets (FEMNIST writer-based, CIFAR-100 by class)
- **Representation-Based Collaboration**: Clients exchange compact prototype distributions instead of model weights
- **Cluster-Aware Training**: Server-side clustering groups similar clients; cluster guidance regularizes local training
- **Improved Personalization**: Each client balances local loss with cluster alignment, preserving individuality
- **Communication Efficiency**: Compact representation summaries reduce bandwidth compared to parameter sharing
- **GPU Acceleration**: Full CUDA/cuDNN support for efficient training on NVIDIA GPUs
- **Transparent Simulation**: Explicit round-by-round orchestration with visibility into all federated phases

## ✨ Key Features

### Data Management
- **FEMNIST**: Character recognition with 3597 writers (naturally partitioned by writer-id)
- **CIFAR-100**: 100-class image classification with configurable client partitioning
- **Smart Splitting**: 80/20 train/eval splits per client with deterministic seeding
- **Efficient Loading**: Parquet-based (FEMNIST) and pickle-based (CIFAR-100) datasets

### Model Architectures
- **SmallCNN**: 2-layer lightweight CNN for image tasks (~100K params)
- **ResNet8**: Lightweight residual network (8 layers)
- **MLP**: Feedforward neural network for flexible architectures
- **Adapter Pattern**: Decoupled backbone + adapter + classifier for heterogeneity

### Federated Learning
- **Server-Side Clustering**: Spectral clustering on Wasserstein distance matrix
- **Client Guidance**: Cluster prototype-based regularization during local training
- **Multiple Aggregation Modes**: `prototype`, `fedavg`, and `fedprox`
- **Round-Based Simulation**: Explicit phases: local_train → upload → cluster → broadcast → guidance → eval
- **Convergence Tracking**: Logging of cluster assignments, distances, accuracy per round

### Technical Stack
- **Python 3.12.8** with virtual environment isolation
- **PyTorch 2.10.0+cu126** for CUDA 12.6 GPU acceleration
- **scikit-learn** for spectral clustering
- **POT** (Python Optimal Transport) for Wasserstein distance computation
- **pytest** for automated testing

## 📊 Project Structure

```
flex_persona/
├── config/                      # Experiment configuration
│   ├── experiment_config.py    # Main config with dataset/model/training/clustering settings
│   ├── defaults.py             # Constants and default values
│   └── types.py                # Type definitions
├── data/                        # Data loading and partitioning
│   ├── dataset_registry.py     # FEMNIST/CIFAR-100 dataset registry
│   ├── client_data_manager.py  # Per-client data bundling with train/eval split
│   └── partition_strategies.py # Partitioning algorithms (writer-based, class-based)
├── models/                      # Model architectures
│   ├── backbones.py            # SmallCNN, ResNet8, MLP implementations
│   ├── client_model.py         # Model wrapper with adapter logic
│   └── model_factory.py        # Model creation factory
├── training/                    # Training procedures
│   ├── local_trainer.py        # Standard local SGD training
│   ├── cluster_aware_trainer.py # Cluster-guided training with Wasserstein regularization
│   └── optimizer_factory.py    # Optimizer creation
├── similarity/                  # Prototype & similarity computation
│   ├── prototype_distribution.py # Client model to prototype extraction
│   ├── wasserstein_distance.py  # Wasserstein distance on distributions
│   └── similarity_calculator.py # Distance → similarity transformation
├── clustering/                  # Server-side clustering
│   ├── spectral_clusterer.py   # Spectral clustering on affinity matrix
│   └── cluster_analyzer.py     # Cluster metrics and diagnostics
├── federated/                   # Federated learning orchestration
│   ├── client.py               # Client-side logic (train, guidance, eval)
│   ├── server.py               # Server-side clustering and aggregation
│   ├── simulator.py            # Multi-round federated simulation
│   └── round_state.py          # Round state tracking
├── evaluation/                  # Metrics & logging
│   ├── evaluator.py            # Accuracy, fairness, utility metrics
│   ├── communication_tracker.py # Byte-level communication logging
│   ├── convergence_logger.py   # Round-by-round logging
│   └── report_builder.py       # Result aggregation
├── training/                    # Additional training utilities
│   └── loss_composer.py        # Loss function composition
└── prototypes/                  # Prototype utilities
    ├── prototype_factory.py    # Prototype creation
    └── prototype_cache.py      # Lightweight caching

tests/
├── test_model_interfaces.py         # ✅ Model backbone & adapter tests
├── test_prototype_pipeline.py       # ✅ Prototype extraction tests
├── test_similarity_and_clustering.py # ✅ Wasserstein & clustering tests
├── test_federated_round_smoke.py    # ✅ Full round simulation
└── test_data_pipeline.py            # 📋 Data loading/partitioning tests
```

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.12+** (tested with 3.12.8)
- **NVIDIA GPU** with CUDA 12.1+ (optional, falls back to CPU)
- **~3 GB** for Python dependencies + datasets

### Step 1: Clone Repository
```bash
git clone https://github.com/redwolf261/federated.git
cd federated
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt pytest
```

### Step 4: (Optional) Enable GPU Support
If you have an NVIDIA GPU, install CUDA-enabled PyTorch:
```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

Verify GPU is detected:
```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 5: Run Tests
```bash
# Light tests (< 20s each)
pytest tests/test_model_interfaces.py tests/test_prototype_pipeline.py tests/test_similarity_and_clustering.py -v

# Full round simulation (transparent output, ~30s)
pytest tests/test_federated_round_smoke.py -v

# All tests
pytest tests/ -v
```

## 💻 Quick Start

### Run a Federated Learning Round

```python
from pathlib import Path
from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator

# Configure experiment
workspace_root = Path(".")
config = ExperimentConfig(dataset_name="femnist")
config.num_clients = 10
config.training.rounds = 5
config.training.local_epochs = 2
config.training.cluster_aware_epochs = 1
config.clustering.num_clusters = 3
config.validate()

# Build simulator
simulator = FederatedSimulator(workspace_root=workspace_root, config=config)

# Run one round
round_idx = 1
print(f"Running round {round_idx}...")

# Phase 1: Local training
for client in simulator.clients:
    metrics = client.train_local(
        local_epochs=config.training.local_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    print(f"  {client.client_id}: loss={metrics['local_loss']:.4f}")

# Phase 2: Upload
upload_messages = [c.build_upload_message(round_idx) for c in simulator.clients]
simulator.server.receive_client_messages(upload_messages)

# Phase 3: Server clustering
distances = simulator.server.compute_wasserstein_matrix()
similarity, adjacency = simulator.server.build_similarity_and_adjacency(distances)
assignments = simulator.server.cluster_clients(similarity)
clusters = simulator.server.compute_cluster_distributions(assignments)

# Phase 4: Broadcast
broadcast_msgs = simulator.server.build_broadcast_messages(
    round_idx=round_idx,
    cluster_assignments=assignments,
    cluster_distributions=clusters,
    affinity_matrix=similarity,
)

# Phase 5: Cluster guidance
for client, msg in zip(simulator.clients, broadcast_msgs):
    metrics = client.apply_cluster_guidance(
        message=msg,
        cluster_aware_epochs=config.training.cluster_aware_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        lambda_cluster=config.training.lambda_cluster,
    )
    print(f"  {client.client_id}: objective={metrics['total_objective']:.4f}")

# Phase 6: Evaluation
accuracies = {c.client_id: c.evaluate_accuracy() for c in simulator.clients}
mean_acc = simulator.evaluator.mean_client_accuracy(accuracies)
print(f"Round {round_idx} complete. Mean accuracy: {mean_acc:.4f}")
```

## 🏗️ Architecture

### Federated Learning Flow

```
Round t:
  [Client Phase]
    1. Load local dataset (train split)
    2. train_local(): Standard SGD for local_epochs
    3. build_upload_message(): Extract prototype distribution
    ↓
  [Server Phase]
    4. compute_wasserstein_matrix(): Pairwise client similarity
    5. cluster_clients(): Spectral clustering on similarity
    6. compute_cluster_distributions(): Aggregate prototypes per cluster
    7. build_broadcast_messages(): Send cluster guidance to clients
    ↓
  [Client Phase - Cluster Aware]
    8. apply_cluster_guidance(): Train with cluster prototype regularization
       Loss = local_loss + λ_cluster × alignment_loss
    ↓
  [Evaluation Phase]
    9. evaluate_accuracy(): Test on eval split (separate from train)
```

### Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `ExperimentConfig` | Unified config schema | ✅ Working |
| `DatasetRegistry` | FEMNIST/CIFAR-100 loading | ✅ Working |
| `ClientDataManager` | Per-client 80/20 split | ✅ Fixed & Working |
| `BackboneFactory` | Heterogeneous architectures | ✅ Working |
| `PrototypeDistribution` | Extract client prototypes | ✅ Working |
| `WassersteinDistance` | Client similarity | ✅ Working |
| `SpectralClusterer` | Server-side clustering | ✅ Working |
| `ClusterAwareTrainer` | Guidance + regularization | ✅ Fixed & Working |
| `FederatedSimulator` | Round orchestration | ✅ Fixed & Working |
| `Evaluator` | Accuracy metrics | ✅ Working |
| `CommunicationTracker` | Byte-level logging | ✅ Fixed & Working |

## 🔧 Configuration

All experiment settings are controlled via `ExperimentConfig`:

```python
config = ExperimentConfig(dataset_name="femnist")

# Data
config.num_clients = 20
config.training.max_samples_per_client = 128

# Model
config.model.shared_dim = 64
config.model.client_backbones = [
  "small_cnn", "resnet8", "mlp", "small_cnn",
]

# Training
config.training.rounds = 10
config.training.local_epochs = 3
config.training.cluster_aware_epochs = 1
config.training.batch_size = 32
config.training.learning_rate = 0.01
config.training.weight_decay = 1e-5
config.training.lambda_cluster = 0.1  # Cluster guidance weight
config.training.aggregation_mode = "prototype"  # or "fedavg", "fedprox"
config.training.fedprox_mu = 0.01  # used when aggregation_mode="fedprox"
config.training.early_stopping_enabled = True
config.training.early_stopping_patience = 5
config.training.early_stopping_min_delta = 0.001

# Clustering
config.clustering.num_clusters = 5
config.clustering.num_neighbors = 10  # For similarity graph

# Reproducibility
config.random_seed = 42

config.validate()
```

## 📈 Testing & Validation

### Test Coverage

```
✅ test_model_interfaces.py (2 tests, ~1s)
   - SmallCNN, ResNet8, MLP forward passes
   - Adapter pattern with different input sizes

✅ test_prototype_pipeline.py (1 test, ~2s)
   - Prototype extraction from trained models
   - Distribution creation and serialization

✅ test_similarity_and_clustering.py (1 test, ~13s)
   - Wasserstein distance computation
   - Spectral clustering on similarity matrix

✅ test_federated_round_smoke.py (1 test, ~250-300s)
   - Full federated round with 3 clients
   - All 6 phases: train → upload → cluster → broadcast → guidance → eval
   - Communication tracking and metrics

📋 test_data_pipeline.py
   - FEMNIST/CIFAR-100 loading
   - Partition strategies validation
```

### Recent Fixes

| File | Issue | Fix | Status |
|------|-------|-----|--------|
| `cluster_aware_trainer.py` | Cluster loss zeroed on line 54 | Compute actual MSE loss between client & cluster prototypes | ✅ Fixed |
| `client_data_manager.py` | Train/eval used same data | Implement 80/20 split with seeded RNG per client | ✅ Fixed |
| `pyproject.toml` | Submodules not packaged | Switch to setuptools dynamic discovery | ✅ Fixed |
| `simulator.py:97` | Keyword arg mismatch | Change `c2s_round_bytes=` to `c2s_bytes=` | ✅ Fixed |
| `simulator.py:46` | Hardcoded CPU device | Add `device = "cuda" if torch.cuda.is_available() else "cpu"` | ✅ Fixed |

## 📊 Example Output

Running a 2-client, 1-round simulation:

```
[INIT] building simulator → 2.59s
  clients=2

[PHASE 1] local training → 8.03s
  client 1 done in 7.81s local_loss=1.4329
  client 2 done in 0.22s local_loss=1.5847

[PHASE 2] client upload → 0.03s
  upload complete: c2s_bytes=44,720

[PHASE 3] server clustering → 9.85s
  distance_matrix: (2, 2)
  cluster_assignments=[0, 1]

[PHASE 4] server broadcast → 0.01s
  s2c_bytes=367,251

[PHASE 5] cluster guidance → 0.69s
  client 1 guidance in 0.57s total_objective=1.3295
  client 2 guidance in 0.11s total_objective=1.4521

[PHASE 6] evaluation → 0.06s
  mean_client_accuracy=0.623456
  worst_client_accuracy=0.481923

[SUMMARY]
  cluster_assignments: [0, 1]
  distance_matrix_shape: (2, 2)
  similarity_matrix_shape: (2, 2)
  communication: c2s=44_720 bytes, s2c=367_251 bytes, total=411_971 bytes

Total wall time: 21.28s
GPU utilization: 33% (NVIDIA GeForce RTX 2050)
```

## 🎓 Concepts

### Cluster-Aware Training
Each client receives guidance from its assigned cluster (set of similar clients). During training, the client loss combines:
- **Local Loss**: Standard supervised learning loss
- **Alignment Loss**: MSE between client prototypes and cluster prototype (scaled by λ_cluster)

This encourages heterogeneous clients to align on shared features while preserving personalization.

### Prototype Distribution
A compact representation of a client's learned features extracted from intermediate layers. Used for efficient similarity computation without sending full model weights.

### Wasserstein Distance
Optimal transport distance between prototype distributions. More meaningful than L2 distance for comparing learned representations.

## 📋 Dependencies

See [requirements.txt](requirements.txt) for complete list. Key packages:
- `torch` 2.10.0 (GPU-enabled)
- `numpy` 2.4.3
- `pandas` 2.3.3
- `scikit-learn` 1.8.0
- `scipy` 1.17.1
- `POT` 0.9.6 (Optimal Transport)

## 🐛 Known Issues & Limitations

- CIFAR-100 test currently skipped (CPU-only testing mode)
- Spectral clustering with >10 clients may require KNN graph tuning
- Communication overhead is logged but not minimized (no compression)

## 🔮 Future Work

- [x] Model aggregation strategies (FedAvg, FedProx)
- [ ] Adaptive clustering (dynamic num_clusters)
- [ ] Stronger FL baselines (SCAFFOLD, MOON)
- [ ] Communication compression (quantization, sparsification)
- [ ] Multi-GPU support
- [ ] Production-grade async simulator
- [ ] Tensorboard integration

## 📖 References

- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- **Non-IID Data**: Yurochkin et al., "Federated Learning with Matched Averaging" (Clustered FL)
- **Optimal Transport**: Peyré & Cuturi, "Computational Optimal Transport"
- **FEMNIST Dataset**: Caldas et al., "LEAF: A Benchmark for Federated Settings"

## 📝 License

See [LICENSE](LICENSE) if present, or refer to original repository.

## 🤝 Contributing

Contributions welcome! Key areas:
- Additional datasets (Shakespeare, StackOverflow)
- Advanced aggregation methods
- Heterogeneous model training improvements
- Performance optimizations

## ✉️ Contact

Based on [FLEX-Persona](https://github.com/redwolf261/federated) by redwolf261
