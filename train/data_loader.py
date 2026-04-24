import os
import numpy as np
from PIL import Image


CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IMG_SIZE = 64
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3


def load_dataset(data_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Load EuroSAT_RGB, split into train/val/test, return normalized float32 arrays."""
    images, labels = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith('.jpg'):
                continue
            img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32).flatten())
            labels.append(CLASS_TO_IDX[cls])

    X = np.stack(images)   # (N, 12288)
    y = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    X_test,  y_test  = X[:n_test],        y[:n_test]
    X_val,   y_val   = X[n_test:n_test+n_val], y[n_test:n_test+n_val]
    X_train, y_train = X[n_test+n_val:],  y[n_test+n_val:]

    # normalize with train statistics
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    stats = {'mean': mean, 'std': std}
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), stats


def batch_iter(X, y, batch_size, shuffle=True, seed=None):
    """Yield (X_batch, y_batch) mini-batches."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]
