"""
datasets/cifar10.py
────────────────────────────────────────────────────────────────────────────
CIFAR-10 / CIFAR-100 data loaders with federated partitioning.

Supports:
  • IID partition   – each client receives a random uniform slice
  • non-IID (Dirichlet) – Dirichlet(α) label allocation (realistic FL setting)
  • non-IID (shard)    – each client gets k classes only (HierFL-compatible)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# ── Normalisation constants ──────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def _normalise(mean, std):
    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t  = tf.constant(std,  dtype=tf.float32)
    return lambda x, y: ((tf.cast(x, tf.float32) / 255.0 - mean_t) / std_t, y)


# ── Raw data loading ─────────────────────────────────────────────────────

def _load_raw(dataset: str = "cifar10"):
    if dataset == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == "cifar100":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar100.load_data()
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y_tr = y_tr.flatten().astype(np.int32)
    y_te = y_te.flatten().astype(np.int32)
    return (x_tr, y_tr), (x_te, y_te), mean, std


# ── Partition helpers ─────────────────────────────────────────────────────

def _iid_partition(y: np.ndarray, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    """Return list of index arrays, one per client, uniformly distributed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [idx[i::num_clients] for i in range(num_clients)]


def _dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Dirichlet(α) non-IID partition.
    Small α → highly non-IID; α → ∞ → IID.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c_idx in class_indices:
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([
            p * (len(ci) < len(y) // num_clients + 1)
            for p, ci in zip(proportions, client_indices)
        ])
        proportions /= proportions.sum()
        splits = (proportions * len(c_idx)).astype(int)
        splits[-1] = len(c_idx) - splits[:-1].sum()
        pos = 0
        for i, n in enumerate(splits):
            client_indices[i].extend(c_idx[pos:pos + n].tolist())
            pos += n

    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def _shard_partition(
    y: np.ndarray,
    num_clients: int,
    shards_per_client: int = 2,
    seed: int = 42,
) -> List[np.ndarray]:
    """HierFL-style shard partition: each client gets `shards_per_client` classes."""
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(y, kind="stable")
    total_shards = num_clients * shards_per_client
    shard_size = len(y) // total_shards
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size]
              for i in range(total_shards)]
    shard_ids = rng.permutation(total_shards)
    client_indices = []
    for i in range(num_clients):
        assigned = shard_ids[i * shards_per_client:(i + 1) * shards_per_client]
        client_indices.append(np.concatenate([shards[s] for s in assigned]))
    return client_indices


# ── tf.data pipeline ──────────────────────────────────────────────────────

def _to_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    mean, std,
    augment: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    x_sub = x[indices]
    y_sub = y[indices]

    ds = tf.data.Dataset.from_tensor_slices((x_sub, y_sub))
    if shuffle:
        ds = ds.shuffle(len(x_sub), seed=seed, reshuffle_each_iteration=True)

    norm = _normalise(mean, std)
    ds = ds.map(norm, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def aug(img, label):
            img = tf.image.pad_to_bounding_box(
                tf.expand_dims(img, 0), 4, 4, 40, 40)[0]
            img = tf.image.random_crop(img, (32, 32, 3))
            img = tf.image.random_flip_left_right(img)
            return img, label
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _to_tf_dataset_test(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    mean, std,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_normalise(mean, std), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Public API ────────────────────────────────────────────────────────────

def get_cifar_dataloaders(args) -> Tuple[
    List[tf.data.Dataset],  # per-client train datasets
    List[tf.data.Dataset],  # per-client test datasets
    tf.data.Dataset,        # global validation train dataset
    tf.data.Dataset,        # global validation test dataset
    List[int],              # num train samples per client
]:
    """
    Build federated data loaders for CIFAR-10 or CIFAR-100.

    args attributes used:
        dataset         : 'cifar10' | 'cifar100'
        num_clients     : int
        batch_size      : int
        iid             : int   (1 = IID, 0 = shard, -1 = Dirichlet)
        dirichlet_alpha : float (used when iid == -1)
        classes_per_client : int (used when iid == 0)
        seed            : int
    """
    (x_tr, y_tr), (x_te, y_te), mean, std = _load_raw(args.dataset)

    iid = getattr(args, "iid", 0)
    alpha = getattr(args, "dirichlet_alpha", 0.5)
    spc = getattr(args, "classes_per_client", 2)

    if iid == 1:
        partitions = _iid_partition(y_tr, args.num_clients, seed=args.seed)
    elif iid == -1:
        partitions = _dirichlet_partition(y_tr, args.num_clients,
                                          alpha=alpha, seed=args.seed)
    else:  # shard (iid == 0 or -2)
        partitions = _shard_partition(y_tr, args.num_clients,
                                      shards_per_client=spc, seed=args.seed)

    # Log distribution summary
    for i, p in enumerate(partitions):
        classes, counts = np.unique(y_tr[p], return_counts=True)
        logger.debug(f"Client {i}: {len(p)} samples, classes {dict(zip(classes.tolist(), counts.tolist()))}")

    train_datasets = [
        _to_tf_dataset(x_tr, y_tr, partitions[i], args.batch_size,
                       mean, std, augment=True, shuffle=True, seed=args.seed + i)
        for i in range(args.num_clients)
    ]

    # Each client's test set = global test set (for evaluation consistency with HierFL)
    test_datasets = [
        _to_tf_dataset_test(x_te, y_te, args.batch_size * 4, mean, std)
        for _ in range(args.num_clients)
    ]

    v_train_ds = _to_tf_dataset(
        x_tr, y_tr, np.arange(len(y_tr)),
        args.batch_size * 4, mean, std, augment=False, shuffle=False
    )
    v_test_ds = _to_tf_dataset_test(x_te, y_te, args.batch_size * 4, mean, std)

    num_train_samples = [len(p) for p in partitions]
    logger.info(
        f"Dataset: {args.dataset} | "
        f"Clients: {args.num_clients} | "
        f"Partition: {'IID' if iid==1 else 'Dirichlet' if iid==-1 else 'Shard'} | "
        f"Avg samples/client: {int(np.mean(num_train_samples))}"
    )
    return train_datasets, test_datasets, v_train_ds, v_test_ds, num_train_samples
