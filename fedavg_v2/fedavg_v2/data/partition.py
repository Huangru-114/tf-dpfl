import numpy as np
import tensorflow as tf


def make_client_dataset(images_np, labels_np, indices, config):
    """
    给定索引列表，构建单个客户端的 tf.data.Dataset。

    Args:
        images_np: 全量图片 numpy 数组 (N, H, W, C)
        labels_np: 全量标签 numpy 数组 (N,)
        indices:   当前客户端拥有的样本索引
        config:    读取 batch_size

    Returns:
        tf.data.Dataset，可直接用于训练循环
    """
    client_images = images_np[indices]
    client_labels = labels_np[indices]

    ds = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
    ds = (
        ds
        .shuffle(buffer_size=len(indices))
        .batch(config["data"]["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


def iid_partition(images_np, labels_np, n_clients, config):
    """
    IID 分区：随机均匀分配，每个客户端数据量相同，分布接近全局分布。

    Returns:
        List[tf.data.Dataset]，长度为 n_clients
    """
    n_total = len(labels_np)
    indices = np.random.permutation(n_total)
    splits  = np.array_split(indices, n_clients)

    client_datasets = []
    for idx_split in splits:
        ds = make_client_dataset(images_np, labels_np, idx_split, config)
        client_datasets.append(ds)

    print(f"[Partition] IID | {n_clients} clients | "
          f"~{n_total // n_clients} samples each")
    return client_datasets


def noniid_partition(images_np, labels_np, n_clients, config, alpha=0.5):
    """
    Non-IID 分区：用 Dirichlet(α) 分布控制异质程度。
    α 越小，各客户端数据分布差异越大。

    Returns:
        List[tf.data.Dataset]，长度为 n_clients
    """
    num_classes    = len(np.unique(labels_np))
    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels_np == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(alpha=np.ones(n_clients) * alpha)
        cut_points  = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits      = np.split(class_indices, cut_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    client_datasets = []
    for indices in client_indices:
        ds = make_client_dataset(images_np, labels_np, np.array(indices), config)
        client_datasets.append(ds)

    _print_distribution(client_indices, labels_np, n_clients, num_classes, alpha)
    return client_datasets


def _print_distribution(client_indices, labels_np, n_clients, num_classes, alpha):
    """打印每个客户端的样本数和类别分布，用于验证 Non-IID 效果。"""
    print(f"\n[Partition] Non-IID | α={alpha} | "
          f"{n_clients} clients | {num_classes} classes")
    print(f"{'Client':<12} {'Samples':<10} Top-3 classes")
    print("-" * 52)
    for i, indices in enumerate(client_indices):
        indices      = np.array(indices)
        label_counts = np.bincount(labels_np[indices], minlength=num_classes)
        top3         = np.argsort(label_counts)[::-1][:3]
        top3_str     = ", ".join([f"cls{c}:{label_counts[c]}" for c in top3])
        print(f"client_{i:<5} {len(indices):<10} {top3_str}")
    print()
