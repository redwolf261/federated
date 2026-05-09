"""Project-wide constants."""

from __future__ import annotations

import torch

DEFAULT_RANDOM_SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_NUM_CLIENTS = 4
MIN_CLIENTS = 1
MAX_CLIENTS = 100
