#!/usr/bin/env python
"""
Script to visualize predictions on the test/val/train set.

This script reads one or more CSV files containing model predictions and
displays each image alongside its predicted label(s). It assumes that the
predictions CSV and the dataset’s pickle file are located in the same
directory.

Usage:
    python visualize_predictions.py --directory path/to/folder \
                                    --csv-filenames resnet_preds.csv \
                                    --dataset-split test \
                                    --num-images 25

Dataset split mapping (explicit):
    train -> train_data.pkl
    val   -> val_data.pkl
    test  -> test_data.pkl
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# List of class names corresponding to prediction indices.
CLASSES = ["trafficlight", "stop", "speedlimit", "crosswalk"]

# Explicit dataset filename mapping.
DATASET_FILES = {
    "train": "train_data.pkl",
    "val": "val_data.pkl",
    "test": "test_data.pkl",
}


def visualize_predictions(
    directory: str,
    csv_filenames: str | None = None,
    dataset_split: str = "test",
    num_images: int = 25,
) -> None:
    """Display images from a dataset split with predictions from one or more CSV files."""
    # Resolve csv_filenames into a list of file names. If None, discover automatically.
    if csv_filenames is None:
        candidates = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
        selected: list[str] = []
        for fname in candidates:
            try:
                df = pd.read_csv(os.path.join(directory, fname), nrows=1)
            except Exception:
                continue
            if "preds" in df.columns:
                selected.append(fname)
        if not selected:
            raise ValueError(f"No CSV files with a 'preds' column were found in {directory}.")
        csv_list = selected
    else:
        csv_list = [c.strip() for c in csv_filenames.split(",") if c.strip()]
        if not csv_list:
            raise ValueError("csv_filenames string provided but no valid filenames parsed.")

    # Ensure each specified CSV exists and contains a preds column
    preds_map: dict[str, list[int]] = {}
    for fname in csv_list:
        csv_path = os.path.join(directory, fname)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Prediction file '{csv_path}' does not exist.")
        df = pd.read_csv(csv_path)
        if "preds" not in df.columns:
            raise ValueError(f"CSV file '{fname}' must contain a 'preds' column.")
        preds_map[fname] = df["preds"].tolist()

    # Load the dataset from the explicit split->filename mapping
    if dataset_split not in DATASET_FILES:
        raise ValueError(
            f"Invalid dataset_split '{dataset_split}'. "
            f"Choose from {list(DATASET_FILES.keys())}"
        )

    pkl_path = os.path.join(directory, DATASET_FILES[dataset_split])
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Dataset file '{pkl_path}' not found. "
            f"(Expected for split '{dataset_split}': {DATASET_FILES[dataset_split]})"
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Extract only images; ignore labels
    dataset = [img for img, _ in data]
    total_images = len(dataset)
    num_images = min(num_images, total_images)

    # Convert a tensor to a PIL Image (no torchvision dependency).
    def tensor_to_pil(img_tensor):
        arr = img_tensor.detach().cpu().numpy()  # (C,H,W)
        arr = np.transpose(arr, (1, 2, 0))       # (H,W,C)
        min_val, max_val = arr.min(), arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)

    # Determine grid dimensions (square or near square)
    cols = int(num_images ** 0.5)
    cols = max(cols, 1)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx in range(num_images):
        img_tensor = dataset[idx]
        img = tensor_to_pil(img_tensor)

        # Build title with predictions from each file
        title_parts = []
        for fname, preds in preds_map.items():
            if idx >= len(preds):
                label_str = f"{fname}: [no pred]"
            else:
                pred_idx = int(preds[idx])
                if 0 <= pred_idx < len(CLASSES):
                    label_str = f"{fname}: {CLASSES[pred_idx]}"
                else:
                    label_str = f"{fname}: Invalid({pred_idx})"
            title_parts.append(label_str)

        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(" | ".join(title_parts), fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize traffic sign predictions from one or more CSV files. "
            "By default, the script searches the given directory for CSV files "
            "containing a 'preds' column and uses them all."
        )
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help=(
            "Path to the folder containing the dataset pickle and prediction CSV files "
            "(default: current directory)"
        ),
    )
    parser.add_argument(
        "--csv-filenames",
        type=str,
        default=None,
        help=(
            "Comma-separated list of CSV files to use (e.g., 'resnet.csv,cnn.csv'). "
            "If omitted, all CSVs with a 'preds' column in the directory are used."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=list(DATASET_FILES.keys()),
        help=(
            "Dataset split to visualize (default: 'test'). "
            "Maps to train_data.pkl / val_data.pkl / test_data.pkl."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=25,
        help="Number of images to display (default: 25)",
    )
    args = parser.parse_args()

    visualize_predictions(
        directory=args.directory,
        csv_filenames=args.csv_filenames,
        dataset_split=args.dataset_split,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()
