
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"

SEED = 26504


def load_dataset_and_split():
    """Load raw flattened faces and the saved, fixed train/validation/test split."""
    faces = fetch_olivetti_faces()
    X = faces.images.reshape(len(faces.images), -1)
    y = faces.target

    split_path = DATA_DIR / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Saved split not found: {split_path}. "
            "Run src/dataset_setup.py first."
        )

    with np.load(split_path) as split:
        train_idx = split["train_indices"].copy()
        val_idx = split["validation_indices"].copy()
        test_idx = split["test_indices"].copy()

    # Check that the saved split is valid and has no overlap.
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Train, validation, and test indices overlap.")

    if len(all_idx) != len(y) or not np.array_equal(
        np.sort(all_idx), np.arange(len(y))
    ):
        raise ValueError("Saved split does not cover the dataset exactly.")

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
        train_idx, val_idx, test_idx,
    )


def make_masked_features(X):
    """Zero rows and columns 24–39 in each flattened 64x64 image."""
    images = X.reshape(-1, 64, 64).copy()
    images[:, 24:40, :] = 0.0
    images[:, :, 24:40] = 0.0
    return images.reshape(len(images), -1)
