import io
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetInfo:
    name: str
    num_samples: int
    num_classes: int
    image_tensor_shape: Tuple[int, ...]
    label_tensor_shape: Tuple[int, ...]


def preprocess_images(images: np.ndarray) -> torch.Tensor:
    """
    Convert image arrays to float32 torch tensors and normalize to [0, 1].
    """
    tensor = torch.from_numpy(images).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _load_pickle(file_path: str) -> Dict[Any, Any]:
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _decode_cifar_split(split_path: str) -> Tuple[np.ndarray, np.ndarray]:
    split_dict = _load_pickle(split_path)

    data_key = b"data" if b"data" in split_dict else "data"
    label_key = (
        b"fine_labels"
        if b"fine_labels" in split_dict
        else "fine_labels"
        if "fine_labels" in split_dict
        else b"labels"
        if b"labels" in split_dict
        else "labels"
    )

    flat_images = np.array(split_dict[data_key], dtype=np.uint8)
    labels = np.array(split_dict[label_key], dtype=np.int64)

    # CIFAR stores images as (N, 3072) in channel-first blocks: R(1024), G(1024), B(1024)
    images = flat_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_cifar_dataset(
    cifar_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor], List[str]]:
    """
    Load CIFAR-100 from local files (`train`, `test`, `meta`), normalize,
    convert to tensors, and create train/test DataLoaders.
    """
    train_path = os.path.join(cifar_dir, "train")
    test_path = os.path.join(cifar_dir, "test")
    meta_path = os.path.join(cifar_dir, "meta")

    for required_path in (train_path, test_path, meta_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing required CIFAR-100 file: {required_path}")

    train_images_np, train_labels_np = _decode_cifar_split(train_path)
    test_images_np, test_labels_np = _decode_cifar_split(test_path)

    meta = _load_pickle(meta_path)
    label_key = b"fine_label_names" if b"fine_label_names" in meta else "fine_label_names"
    fine_label_names_raw = meta[label_key]
    fine_label_names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in fine_label_names_raw
    ]

    train_images = preprocess_images(train_images_np).permute(0, 3, 1, 2)  # NCHW
    test_images = preprocess_images(test_images_np).permute(0, 3, 1, 2)
    train_labels = torch.from_numpy(train_labels_np).long()
    test_labels = torch.from_numpy(test_labels_np).long()

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    tensors = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

    return train_loader, test_loader, tensors, fine_label_names


def _find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _encode_label_series(series: pd.Series) -> np.ndarray:
    """
    Encode FEMNIST labels as int64.
    If labels are already numeric, keep values. Otherwise, map categories to codes.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.int64)

    categorical = pd.Categorical(series.astype(str))
    return categorical.codes.astype(np.int64)


def _encode_writer_series(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preserve raw writer IDs and provide deterministic numeric codes for tensor use.
    """
    raw_ids = series.astype(str).to_numpy()
    writer_codes = pd.Categorical(raw_ids).codes.astype(np.int64)
    return raw_ids, writer_codes


def _extract_image_array(value: Any) -> np.ndarray:
    """
    Extract an image from common FEMNIST parquet cell formats and return 28x28 uint8 array.
    """
    if isinstance(value, dict):
        if "array" in value:
            arr = np.array(value["array"], dtype=np.uint8)
        elif "bytes" in value and value["bytes"] is not None:
            pil_img = Image.open(io.BytesIO(value["bytes"]))
            arr = np.array(pil_img, dtype=np.uint8)
        elif "path" in value and value["path"]:
            pil_img = Image.open(value["path"])
            arr = np.array(pil_img, dtype=np.uint8)
        else:
            raise ValueError("Unsupported dict image format in FEMNIST parquet cell.")
    elif isinstance(value, bytes):
        try:
            pil_img = Image.open(io.BytesIO(value))
            arr = np.array(pil_img, dtype=np.uint8)
        except Exception:
            arr = np.frombuffer(value, dtype=np.uint8)
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=np.uint8)
    elif isinstance(value, str):
        if os.path.exists(value):
            pil_img = Image.open(value)
            arr = np.array(pil_img, dtype=np.uint8)
        else:
            stripped = value.replace(",", " ").strip()
            arr = np.fromstring(stripped, sep=" ", dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image value type: {type(value)}")

    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 28 * 28:
        arr = arr.reshape(28, 28)
    elif arr.ndim == 2 and arr.shape == (28, 28):
        pass
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    else:
        raise ValueError(f"Unexpected FEMNIST image shape: {arr.shape}")

    return arr.astype(np.uint8)


def load_femnist_dataset(
    parquet_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, Dict[str, Any], pd.DataFrame]:
    """
    Load FEMNIST parquet data, extract image/label/writer IDs,
    normalize images, convert to tensors, and create a DataLoader.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"FEMNIST parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    image_col = _find_first_existing_column(df, ["image", "pixels", "img", "x"])
    label_col = _find_first_existing_column(df, ["label", "labels", "y", "class", "character", "char"])
    writer_col = _find_first_existing_column(
        df,
        ["writer_id", "client_id", "writer", "user_id", "user", "client"],
    )

    if image_col is None:
        raise ValueError(f"Could not find image column. Available columns: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find label column. Available columns: {list(df.columns)}")
    if writer_col is None:
        raise ValueError(
            "Could not find writer/client ID column. "
            f"Available columns: {list(df.columns)}"
        )

    image_arrays = np.stack([_extract_image_array(v) for v in df[image_col].tolist()], axis=0)
    labels_np = _encode_label_series(df[label_col])
    writer_ids_raw, writer_ids_codes = _encode_writer_series(df[writer_col])

    image_tensors = preprocess_images(image_arrays)  # N, 28, 28
    image_tensors = image_tensors.unsqueeze(1)  # N, 1, 28, 28
    label_tensors = torch.from_numpy(labels_np).long()

    dataset = TensorDataset(image_tensors, label_tensors)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    tensors = {
        "images": image_tensors,
        "labels": label_tensors,
        "writer_ids": torch.from_numpy(writer_ids_codes).long(),
    }

    info_df = pd.DataFrame(
        {
            "label": labels_np,
            "writer_id": writer_ids_raw,
            "writer_id_code": writer_ids_codes,
        }
    )

    return loader, tensors, info_df


def print_dataset_info(info: DatasetInfo) -> None:
    print(f"\n{info.name} dataset info")
    print(f"- number of samples: {info.num_samples}")
    print(f"- number of classes: {info.num_classes}")
    print(f"- image tensor shape: {info.image_tensor_shape}")
    print(f"- label tensor shape: {info.label_tensor_shape}")


def show_sample_images(
    cifar_images: torch.Tensor,
    cifar_labels: torch.Tensor,
    femnist_images: torch.Tensor,
    femnist_labels: torch.Tensor,
    num_samples: int = 6,
) -> None:
    num_samples = max(1, num_samples)
    fig, axes = plt.subplots(2, num_samples, figsize=(2.5 * num_samples, 6))

    for i in range(num_samples):
        cifar_idx = i % len(cifar_images)
        femnist_idx = i % len(femnist_images)

        cifar_img = cifar_images[cifar_idx].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(cifar_img)
        axes[0, i].set_title(f"CIFAR y={int(cifar_labels[cifar_idx])}")
        axes[0, i].axis("off")

        femnist_img = femnist_images[femnist_idx].squeeze(0).cpu().numpy()
        axes[1, i].imshow(femnist_img, cmap="gray")
        axes[1, i].set_title(f"FEMNIST y={int(femnist_labels[femnist_idx])}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    base_dataset_dir = os.path.join("dataset")
    cifar_dir = os.path.join(base_dataset_dir, "cifar-100-python")
    femnist_parquet = os.path.join(base_dataset_dir, "femnist", "train-00000-of-00001.parquet")

    cifar_train_loader, cifar_test_loader, cifar_tensors, cifar_class_names = load_cifar_dataset(
        cifar_dir=cifar_dir,
        batch_size=64,
        num_workers=0,
    )
    femnist_loader, femnist_tensors, femnist_info_df = load_femnist_dataset(
        parquet_path=femnist_parquet,
        batch_size=64,
        num_workers=0,
    )

    _ = cifar_train_loader, cifar_test_loader, femnist_loader  # keeps explicit references used in pipelines

    cifar_info = DatasetInfo(
        name="CIFAR-100 (train)",
        num_samples=cifar_tensors["train_images"].shape[0],
        num_classes=len(cifar_class_names),
        image_tensor_shape=tuple(cifar_tensors["train_images"].shape),
        label_tensor_shape=tuple(cifar_tensors["train_labels"].shape),
    )
    femnist_info = DatasetInfo(
        name="FEMNIST",
        num_samples=femnist_tensors["images"].shape[0],
        num_classes=int(femnist_tensors["labels"].unique().numel()),
        image_tensor_shape=tuple(femnist_tensors["images"].shape),
        label_tensor_shape=tuple(femnist_tensors["labels"].shape),
    )

    print_dataset_info(cifar_info)
    print_dataset_info(femnist_info)

    print("\nFEMNIST writer/client ID examples:")
    print(femnist_info_df.head())

    show_sample_images(
        cifar_images=cifar_tensors["train_images"],
        cifar_labels=cifar_tensors["train_labels"],
        femnist_images=femnist_tensors["images"],
        femnist_labels=femnist_tensors["labels"],
        num_samples=6,
    )


if __name__ == "__main__":
    main()
