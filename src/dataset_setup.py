
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Roll number: m26ecm504
SEED = 26504

# Load the dataset
faces = fetch_olivetti_faces()

images = faces.images
labels = faces.target

# Flatten raw pixels: 64 x 64 -> 4096 features
X = images.reshape(len(images), -1)
y = labels

print("Dataset loaded successfully")
print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Number of identities:", len(np.unique(y)))
print("Random seed:", SEED)

# Create a reproducible split within each identity
rng = np.random.default_rng(SEED)

train_indices = []
validation_indices = []
test_indices = []

for identity in np.unique(y):
    identity_indices = np.where(y == identity)[0]
    shuffled_indices = rng.permutation(identity_indices)

    train_indices.extend(shuffled_indices[:6])
    validation_indices.extend(shuffled_indices[6:8])
    test_indices.extend(shuffled_indices[8:10])

train_indices = np.array(train_indices, dtype=np.int64)
validation_indices = np.array(validation_indices, dtype=np.int64)
test_indices = np.array(test_indices, dtype=np.int64)

# Save the split so all experiments can reuse it
np.savez(
    DATA_DIR / "split_indices.npz",
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
)

print("Split saved to:", DATA_DIR / "split_indices.npz")
print("Training images:", len(train_indices))
print("Validation images:", len(validation_indices))
print("Test images:", len(test_indices))
