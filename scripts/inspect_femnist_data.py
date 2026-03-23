#!/usr/bin/env python3
"""Quick data inspection to check for preprocessing issues."""

import sys
from pathlib import Path
from collections import Counter
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flex_persona.data.dataset_registry import DatasetRegistry

def inspect_femnist_data():
    """Inspect FEMNIST data for potential preprocessing issues."""
    print("="*60)
    print("FEMNIST DATA INSPECTION")
    print("="*60)

    # Load FEMNIST data
    print("Loading FEMNIST data...")
    registry = DatasetRegistry(project_root)
    artifact = registry.load("femnist", max_rows=10000)  # Sample for inspection

    images = artifact.payload["images"]
    labels = artifact.payload["labels"]
    writer_ids = artifact.payload["writer_ids"]

    print(f"Data shapes:")
    print(f"  Images: {images.shape} ({images.dtype})")
    print(f"  Labels: {labels.shape} ({labels.dtype})")
    print(f"  Writer IDs: {writer_ids.shape}")
    print()

    print(f"Data ranges:")
    print(f"  Images: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels: [{labels.min()}, {labels.max()}]")
    print()

    # Check label distribution
    label_counts = Counter(labels.numpy())
    print(f"Label distribution:")
    print(f"  Unique labels: {len(label_counts)}")
    print(f"  Label range: {min(label_counts.keys())} to {max(label_counts.keys())}")

    # Check for class imbalance
    counts = list(label_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    print(f"  Samples per class: min={min_count}, max={max_count}, avg={np.mean(counts):.1f}")

    if max_count / min_count > 100:
        print("  [WARNING] Severe class imbalance detected!")
    elif max_count / min_count > 10:
        print("  [NOTE] Moderate class imbalance")

    print()

    # Show first few labels and their counts
    most_common = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most common classes:")
    for label, count in most_common[:10]:
        print(f"  Label {label}: {count} samples")

    least_common = sorted(label_counts.items(), key=lambda x: x[1])
    print("Least common classes:")
    for label, count in least_common[:10]:
        print(f"  Label {label}: {count} samples")

    print()

    # Check if labels are continuous
    sorted_labels = sorted(label_counts.keys())
    expected_labels = list(range(len(sorted_labels)))
    if sorted_labels != expected_labels:
        print(f"[WARNING] Labels not continuous!")
        print(f"  Expected: 0-{len(sorted_labels)-1}")
        print(f"  Actual: {sorted_labels}")

        # Check for label gaps
        gaps = []
        for i in range(max(sorted_labels)):
            if i not in sorted_labels:
                gaps.append(i)
        if gaps:
            print(f"  Missing labels: {gaps}")
    else:
        print(f"[SUCCESS] Labels are continuous: 0-{max(sorted_labels)}")

    print()

    # Sample a few images to check visually
    print("Sampling images for visual inspection:")

    for i in range(min(5, len(images))):
        img = images[i].squeeze()  # Remove channel dimension for grayscale
        label = labels[i].item()

        print(f"  Sample {i}: Label={label}, Shape={img.shape}, Range=[{img.min():.3f}, {img.max():.3f}]")

        # Check if image looks reasonable
        if img.max() <= 0.1:
            print(f"    [WARNING] Very dark image - may be improperly normalized")
        elif img.min() >= 0.9:
            print(f"    [WARNING] Very bright image - may be improperly normalized")
        elif (img == 0).sum() > img.numel() * 0.8:
            print(f"    [WARNING] >80% of pixels are zero - may be corrupted")

    # Check for any NaN or infinite values
    if torch.isnan(images).any():
        print(f"[ERROR] Found NaN values in images!")
    if torch.isinf(images).any():
        print(f"[ERROR] Found infinite values in images!")
    if torch.isnan(labels.float()).any():
        print(f"[ERROR] Found NaN values in labels!")

    print()

    # Check writer distribution (federated aspect)
    writer_counts = Counter(writer_ids)
    print(f"Writer (client) distribution:")
    print(f"  Unique writers: {len(writer_counts)}")
    writer_sample_counts = list(writer_counts.values())
    print(f"  Samples per writer: min={min(writer_sample_counts)}, max={max(writer_sample_counts)}, avg={np.mean(writer_sample_counts):.1f}")

    print()
    print("Potential issues to investigate:")

    if min_count < 10:
        print(f"  - Very few samples for some classes ({min_count} minimum)")
    if len(writer_counts) < 50:
        print(f"  - Very few writers/clients ({len(writer_counts)})")
    if images.std() < 0.1:
        print(f"  - Low image variance ({images.std():.3f}) - may indicate normalization issues")
    if images.std() > 0.5:
        print(f"  - High image variance ({images.std():.3f}) - may indicate normalization issues")

if __name__ == "__main__":
    inspect_femnist_data()