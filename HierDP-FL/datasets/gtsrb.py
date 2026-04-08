"""
datasets/gtsrb.py
────────────────────────────────────────────────────────────────────────────
GTSRB (German Traffic Sign Recognition Benchmark) dataset loader.

Download options (in order of preference):
  1. torchvision.datasets.GTSRB  – if torchvision is installed, downloads
     automatically to `dataset_root/gtsrb/`.
  2. tensorflow_datasets.load('german_traffic_sign')  – if tensorflow_datasets
     is installed.
  3. Manual CSV+image folder layout (fallback).

All options produce the same output: (x_tr, y_tr), (x_te, y_te) as uint8
numpy arrays of shape (N, 32, 32, 3) and int32 labels.

Federated partitioning mirrors cifar10.py (IID / Dirichlet / Shard).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD  = (0.2724, 0.2608, 0.2669)
NUM_CLASSES = 43
IMG_SIZE = 32


# ── Loading raw data ──────────────────────────────────────────────────────

def _load_via_torchvision(root: str) -> Tuple:
    """Load GTSRB via torchvision (PyTorch), convert to numpy."""
    try:
        import torchvision
        import torchvision.transforms as T
        from torchvision.datasets import GTSRB
        from torch.utils.data import DataLoader
        import torch

        transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
        train_ds = GTSRB(root=root, split="train", download=True, transform=transform)
        test_ds  = GTSRB(root=root, split="test",  download=True, transform=transform)

        def ds_to_numpy(ds):
            loader = DataLoader(ds, batch_size=512, num_workers=0)
            xs, ys = [], []
            for xb, yb in loader:
                # ToTensor gives (C,H,W) in [0,1]; convert to uint8 (H,W,C)
                xb_np = (xb.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                xs.append(xb_np)
                ys.extend(yb.numpy().tolist())
            return np.concatenate(xs), np.array(ys, dtype=np.int32)

        logger.info("Loading GTSRB via torchvision…")
        x_tr, y_tr = ds_to_numpy(train_ds)
        x_te, y_te = ds_to_numpy(test_ds)
        return (x_tr, y_tr), (x_te, y_te)

    except ImportError:
        return None


def _load_via_tfds(root: str) -> Optional[Tuple]:
    """Load GTSRB via tensorflow_datasets."""
    try:
        import tensorflow_datasets as tfds
        logger.info("Loading GTSRB via tensorflow_datasets…")
        ds_tr, ds_te = tfds.load(
            "german_traffic_sign",
            split=["train", "test"],
            data_dir=root,
            as_supervised=False,
        )

        def process(ds, split_name):
            xs, ys = [], []
            for ex in ds:
                img = ex["image"].numpy()
                label = int(ex["label"].numpy())
                img_r = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy().astype(np.uint8)
                xs.append(img_r)
                ys.append(label)
            return np.stack(xs), np.array(ys, dtype=np.int32)

        x_tr, y_tr = process(ds_tr, "train")
        x_te, y_te = process(ds_te, "test")
        return (x_tr, y_tr), (x_te, y_te)
    except Exception as e:
        logger.warning(f"tensorflow_datasets GTSRB load failed: {e}")
        return None


def load_gtsrb_raw(dataset_root: str = "data") -> Tuple:
    """
    Attempt to load GTSRB, trying torchvision then tensorflow_datasets.

    Returns
    -------
    (x_tr, y_tr), (x_te, y_te)  – numpy uint8 images, int32 labels
    """
    result = _load_via_torchvision(dataset_root)
    if result is not None:
        return result
    result = _load_via_tfds(dataset_root)
    if result is not None:
        return result
    raise RuntimeError(
        "Could not load GTSRB automatically. "
        "Install either 'torchvision' (pip install torchvision) or "
        "'tensorflow_datasets' (pip install tensorflow-datasets) and re-run."
    )


# ── tf.data pipeline ──────────────────────────────────────────────────────

def _normalise_gtsrb(x, y):
    mean_t = tf.constant(GTSRB_MEAN, dtype=tf.float32)
    std_t  = tf.constant(GTSRB_STD,  dtype=tf.float32)
    return (tf.cast(x, tf.float32) / 255.0 - mean_t) / std_t, y


def _augment_gtsrb(img, label):
    """Light augmentation for GTSRB (traffic sign recognition is rotation-sensitive)."""
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_flip_left_right(img)
    return img, label


def _build_ds(x, y, indices, batch_size, augment=True, shuffle=True, seed=42):
    x_sub = x[indices]
    y_sub = y[indices]
    ds = tf.data.Dataset.from_tensor_slices((x_sub, y_sub))
    if shuffle:
        ds = ds.shuffle(len(x_sub), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_normalise_gtsrb, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment_gtsrb, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


# ── Partition helpers (reuse from cifar10) ───────────────────────────────

def _iid_partition(y, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [idx[i::num_clients] for i in range(num_clients)]


def _dirichlet_partition(y, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c_idx in class_indices:
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions /= proportions.sum()
        splits = (proportions * len(c_idx)).astype(int)
        splits[-1] = len(c_idx) - splits[:-1].sum()
        pos = 0
        for i, n in enumerate(splits):
            client_indices[i].extend(c_idx[pos:pos + n].tolist())
            pos += n
    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def _shard_partition(y, num_clients, shards_per_client=2, seed=42):
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(y, kind="stable")
    total_shards = num_clients * shards_per_client
    shard_size = len(y) // total_shards
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size]
              for i in range(total_shards)]
    shard_ids = rng.permutation(total_shards)
    return [
        np.concatenate([shards[s] for s in shard_ids[i * shards_per_client:(i + 1) * shards_per_client]])
        for i in range(num_clients)
    ]


# ── Public API ────────────────────────────────────────────────────────────

def get_gtsrb_dataloaders(args):
    """
    Build federated GTSRB data loaders.

    Returns same tuple as get_cifar_dataloaders.
    """
    (x_tr, y_tr), (x_te, y_te) = load_gtsrb_raw(
        dataset_root=getattr(args, "dataset_root", "data")
    )

    iid = getattr(args, "iid", 0)
    alpha = getattr(args, "dirichlet_alpha", 0.5)
    spc = getattr(args, "classes_per_client", 5)

    if iid == 1:
        partitions = _iid_partition(y_tr, args.num_clients, seed=args.seed)
    elif iid == -1:
        partitions = _dirichlet_partition(y_tr, args.num_clients, alpha=alpha, seed=args.seed)
    else:
        partitions = _shard_partition(y_tr, args.num_clients, shards_per_client=spc, seed=args.seed)

    train_datasets = [
        _build_ds(x_tr, y_tr, partitions[i], args.batch_size,
                  augment=True, shuffle=True, seed=args.seed + i)
        for i in range(args.num_clients)
    ]
    test_datasets = [
        _build_ds(x_te, y_te, np.arange(len(y_te)),
                  args.batch_size * 4, augment=False, shuffle=False)
        for _ in range(args.num_clients)
    ]
    v_train_ds = _build_ds(x_tr, y_tr, np.arange(len(y_tr)),
                            args.batch_size * 4, augment=False, shuffle=False)
    v_test_ds  = _build_ds(x_te, y_te, np.arange(len(y_te)),
                            args.batch_size * 4, augment=False, shuffle=False)

    num_train_samples = [len(p) for p in partitions]
    logger.info(
        f"Dataset: GTSRB | Clients: {args.num_clients} | "
        f"Avg samples/client: {int(np.mean(num_train_samples))}"
    )
    return train_datasets, test_datasets, v_train_ds, v_test_ds, num_train_samples
