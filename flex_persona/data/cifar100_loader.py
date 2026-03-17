"""Local CIFAR-100 file loader (python format)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .transforms import numpy_images_to_nchw_tensor


class Cifar100Loader:
    """Loads CIFAR-100 train/test/meta from local python-format files."""

    def __init__(self, cifar_dir: Path) -> None:
        self.cifar_dir = cifar_dir

    def load(
        self,
        max_train_samples: int | None = None,
        max_test_samples: int | None = None,
    ) -> dict[str, Any]:
        train_path = self.cifar_dir / "train"
        test_path = self.cifar_dir / "test"
        meta_path = self.cifar_dir / "meta"

        for path in (train_path, test_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing CIFAR-100 file: {path}")

        train_images_np, train_labels_np = self._decode_split(train_path)
        test_images_np, test_labels_np = self._decode_split(test_path)

        if max_train_samples is not None:
            train_images_np = train_images_np[:max_train_samples]
            train_labels_np = train_labels_np[:max_train_samples]
        if max_test_samples is not None:
            test_images_np = test_images_np[:max_test_samples]
            test_labels_np = test_labels_np[:max_test_samples]

        meta = self._load_pickle(meta_path)
        label_key = b"fine_label_names" if b"fine_label_names" in meta else "fine_label_names"
        class_names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in meta[label_key]
        ]

        return {
            "train_images": numpy_images_to_nchw_tensor(train_images_np),
            "train_labels": torch.from_numpy(train_labels_np).long(),
            "test_images": numpy_images_to_nchw_tensor(test_images_np),
            "test_labels": torch.from_numpy(test_labels_np).long(),
            "class_names": class_names,
        }

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any] | dict[bytes, Any]:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="bytes")

    def _decode_split(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        split_dict = self._load_pickle(path)
        data_key = b"data" if b"data" in split_dict else "data"

        if b"fine_labels" in split_dict:
            label_key: str | bytes = b"fine_labels"
        elif "fine_labels" in split_dict:
            label_key = "fine_labels"
        elif b"labels" in split_dict:
            label_key = b"labels"
        else:
            label_key = "labels"

        flat_images = np.array(split_dict[data_key], dtype=np.uint8)
        labels = np.array(split_dict[label_key], dtype=np.int64)
        images = flat_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, labels
