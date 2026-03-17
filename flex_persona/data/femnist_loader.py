"""FEMNIST parquet loader with robust image extraction."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .transforms import numpy_grayscale_to_nchw_tensor


class FemnistLoader:
    """Loads FEMNIST parquet and extracts images, labels, and writer ids."""

    IMAGE_CANDIDATES = ["image", "pixels", "img", "x"]
    LABEL_CANDIDATES = ["label", "labels", "y", "class", "character", "char"]
    WRITER_CANDIDATES = ["writer_id", "client_id", "writer", "user_id", "user", "client"]

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

    def load(self, max_rows: int | None = None) -> dict[str, Any]:
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"FEMNIST parquet not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)
        if max_rows is not None:
            df = df.head(max_rows).copy()

        image_col = self._find_column(df, self.IMAGE_CANDIDATES)
        label_col = self._find_column(df, self.LABEL_CANDIDATES)
        writer_col = self._find_column(df, self.WRITER_CANDIDATES)

        if image_col is None:
            raise ValueError(f"No image column found in FEMNIST columns: {list(df.columns)}")
        if label_col is None:
            raise ValueError(f"No label column found in FEMNIST columns: {list(df.columns)}")
        if writer_col is None:
            raise ValueError(f"No writer column found in FEMNIST columns: {list(df.columns)}")

        image_arrays = np.stack([self._extract_image_array(v) for v in df[image_col].tolist()], axis=0)
        labels_np = self._encode_labels(df[label_col])
        writer_ids = df[writer_col].astype(str).to_numpy()

        return {
            "images": numpy_grayscale_to_nchw_tensor(image_arrays),
            "labels": torch.from_numpy(labels_np).long(),
            "writer_ids": writer_ids,
            "frame": pd.DataFrame({"label": labels_np, "writer_id": writer_ids}),
        }

    @staticmethod
    def _encode_labels(series: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            return numeric.to_numpy(dtype=np.int64)

        # For categorical/string labels, build deterministic integer class ids.
        categorical = pd.Categorical(series.astype(str))
        return categorical.codes.astype(np.int64)

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _extract_image_array(self, value: Any) -> np.ndarray:
        if isinstance(value, dict):
            if "array" in value:
                arr = np.array(value["array"], dtype=np.uint8)
            elif "bytes" in value and value["bytes"] is not None:
                arr = np.array(Image.open(io.BytesIO(value["bytes"])), dtype=np.uint8)
            elif "path" in value and value["path"]:
                arr = np.array(Image.open(value["path"]), dtype=np.uint8)
            else:
                raise ValueError("Unsupported FEMNIST dict image format")
        elif isinstance(value, bytes):
            try:
                arr = np.array(Image.open(io.BytesIO(value)), dtype=np.uint8)
            except Exception:
                arr = np.frombuffer(value, dtype=np.uint8)
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=np.uint8)
        elif isinstance(value, str):
            possible_path = Path(value)
            if possible_path.exists():
                arr = np.array(Image.open(possible_path), dtype=np.uint8)
            else:
                arr = np.fromstring(value.replace(",", " ").strip(), sep=" ", dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported FEMNIST image type: {type(value)}")

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
