"""Reproducibility helper functions."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Global seed value.
        deterministic: If True, enforce deterministic CUDA/cuDNN behavior where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # cuBLAS determinism requirements for CUDA matrix ops.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Raise on nondeterministic operations to prevent silent drift.
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
