"""Local CIFAR-10 file loader with torchvision download fallback."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from torchvision.datasets import CIFAR10

from .transforms import numpy_images_to_nchw_tensor


class Cifar10Loader:
    """Loads CIFAR-10 from the standard torchvision dataset layout."""

    def __init__(self, cifar_dir: Path) -> None:
        self.cifar_dir = cifar_dir

    def load(
        self,
        max_train_samples: int | None = None,
        max_test_samples: int | None = None,
    ) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean but got `align=0`.*",
                category=Warning,
            )
            train_ds = CIFAR10(root=str(self.cifar_dir), train=True, download=True)
            test_ds = CIFAR10(root=str(self.cifar_dir), train=False, download=True)

        train_images_np = train_ds.data
        train_labels_np = torch.as_tensor(train_ds.targets, dtype=torch.int64).numpy()
        test_images_np = test_ds.data
        test_labels_np = torch.as_tensor(test_ds.targets, dtype=torch.int64).numpy()

        if max_train_samples is not None:
            train_images_np = train_images_np[:max_train_samples]
            train_labels_np = train_labels_np[:max_train_samples]
        if max_test_samples is not None:
            test_images_np = test_images_np[:max_test_samples]
            test_labels_np = test_labels_np[:max_test_samples]

        return {
            "train_images": numpy_images_to_nchw_tensor(train_images_np),
            "train_labels": torch.from_numpy(train_labels_np).long(),
            "test_images": numpy_images_to_nchw_tensor(test_images_np),
            "test_labels": torch.from_numpy(test_labels_np).long(),
            "class_names": list(train_ds.classes),
        }