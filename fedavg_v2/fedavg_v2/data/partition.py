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


def make_per_client_test_datasets(
    x_test_np: np.ndarray,
    y_test_np: np.ndarray,
    client_train_indices: list,   # List[np.ndarray]，每个客户端的训练样本索引
    y_train_np: np.ndarray,       # 训练集标签，用于推断各客户端持有的类别
    config: dict,
) -> list:
    """
    为每个客户端生成同类型（same-distribution）测试集。

    逻辑：
        1. 从 y_train_np[client_train_indices[i]] 推断客户端 i 持有哪些类别。
        2. 在测试集里过滤出这些类别对应的样本。
        3. 构建 tf.data.Dataset 并返回。

    适用于 pathological / noniid / iid 三种分区方式：
        - pathological：每个客户端只有少数类别，测试集也只包含这些类别。
        - noniid (Dirichlet)：每个客户端理论上持有所有类别，
          但某些类别样本极少（可能为 0）；过滤后只保留训练集中实际出现的类别。
        - iid：每个客户端持有几乎所有类别，测试集接近全局测试集。

    Args:
        x_test_np            : 测试集图片 (M, H, W, C)
        y_test_np            : 测试集标签 (M,)
        client_train_indices : 每个客户端训练样本的索引列表
        y_train_np           : 训练集标签（用于查各客户端的类别集合）
        config               : 读取 batch_size

    Returns:
        List[tf.data.Dataset]，长度 == len(client_train_indices)。
        若某客户端在测试集中一个匹配样本都找不到（极端情况），
        对应位置返回 None，调用方需做判空。
    """
    client_test_datasets = []

    for i, train_idx in enumerate(client_train_indices):

        # ── 步骤1：推断该客户端持有的类别集合 ────────────────────────────
        # 取训练索引对应的标签，去重得到类别集合
        client_classes = set(np.unique(y_train_np[train_idx]).tolist())

        # ── 步骤2：在测试集中过滤出这些类别的样本 ────────────────────────
        mask = np.isin(y_test_np, list(client_classes))
        test_idx = np.where(mask)[0]

        if len(test_idx) == 0:
            # 极端情况：测试集里完全没有该客户端的类别（理论上不应发生）
            print(
                f"  [Partition] WARNING: Client {i} has no matching test samples "
                f"(classes={sorted(client_classes)}). Returning None."
            )
            client_test_datasets.append(None)
            continue

        # ── 步骤3：构建 Dataset ───────────────────────────────────────────
        # 测试集不 shuffle、不 drop_last，与训练集 Dataset 构建方式区分
        test_images = x_test_np[test_idx]
        test_labels = y_test_np[test_idx]

        ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        ds = (
            ds
            .batch(config["data"]["batch_size"])
            .prefetch(tf.data.AUTOTUNE)
        )
        client_test_datasets.append(ds)

    # ── 打印分配摘要 ──────────────────────────────────────────────────────
    _print_test_distribution(
        client_train_indices, y_train_np, y_test_np, client_test_datasets
    )

    return client_test_datasets


def _print_test_distribution(
    client_train_indices, y_train_np, y_test_np, client_test_datasets
):
    """打印每个客户端测试集的样本数和类别，供人工验证。"""
    print(f"\n[Partition] Per-client test datasets")
    print(f"{'Client':<12} {'Train cls':<24} {'Test samples':<14} Test classes")
    print("-" * 72)
    for i, (train_idx, ds) in enumerate(
        zip(client_train_indices, client_test_datasets)
    ):
        train_cls = sorted(set(np.unique(y_train_np[train_idx]).tolist()))
        if ds is None:
            print(f"client_{i:<5} {str(train_cls):<24} {'N/A':<14} N/A")
            continue

        # 遍历 dataset 统计测试集实际类别分布
        all_labels = np.concatenate([y.numpy() for _, y in ds])
        test_cls   = sorted(set(all_labels.tolist()))
        n_test     = len(all_labels)
        print(f"client_{i:<5} {str(train_cls):<24} {n_test:<14} {test_cls}")
    print()


def iid_partition(images_np, labels_np, n_clients, config):
    """
    IID 分区：随机均匀分配，每个客户端数据量相同，分布接近全局分布。

    Returns:
        client_datasets : List[tf.data.Dataset]，长度为 n_clients
        client_indices  : List[np.ndarray]，每个客户端的样本索引
    """
    n_total = len(labels_np)
    indices = np.random.permutation(n_total)
    splits  = np.array_split(indices, n_clients)

    client_datasets = []
    client_indices  = []
    for idx_split in splits:
        ds = make_client_dataset(images_np, labels_np, idx_split, config)
        client_datasets.append(ds)
        client_indices.append(idx_split)

    print(f"[Partition] IID | {n_clients} clients | "
          f"~{n_total // n_clients} samples each")
    return client_datasets, client_indices


def noniid_partition(images_np, labels_np, n_clients, config, alpha=0.5):
    """
    Non-IID 分区：用 Dirichlet(α) 分布控制异质程度。
    α 越小，各客户端数据分布差异越大。

    Returns:
        client_datasets : List[tf.data.Dataset]，长度为 n_clients
        client_indices  : List[np.ndarray]，每个客户端的样本索引
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

    client_datasets   = []
    client_indices_np = []
    for indices in client_indices:
        idx_arr = np.array(indices)
        ds = make_client_dataset(images_np, labels_np, idx_arr, config)
        client_datasets.append(ds)
        client_indices_np.append(idx_arr)

    _print_distribution(client_indices, labels_np, n_clients, num_classes, alpha)
    return client_datasets, client_indices_np


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


def pathological_noniid_partition(images_np, labels_np, n_clients, config,
                                   classes_per_client=2):
    """
    Pathological non-IID 分区（论文 Hier-pFedMe Section IV-A 所用方法）。

    原理：将每个类别的样本按 shard 切分，再将 shard 分配给客户端，
          保证每个客户端恰好拥有 classes_per_client 个类别的数据。
          这是最严苛的 non-IID 设置，类别分布差异极大。

    对应论文数据集配置：
        CIFAR-10:       classes_per_client=2   (10 classes)
        CIFAR-100:      classes_per_client=10  (100 classes)
        Tiny-ImageNet:  classes_per_client=20  (200 classes)

    Args:
        images_np          : 全量图片 numpy 数组 (N, H, W, C)
        labels_np          : 全量标签 numpy 数组 (N,)
        n_clients          : 客户端总数
        config             : 读取 batch_size
        classes_per_client : 每个客户端拥有的类别数

    Returns:
        client_datasets : List[tf.data.Dataset]，长度为 n_clients
        client_indices  : List[np.ndarray]，每个客户端的样本索引
    """
    num_classes = len(np.unique(labels_np))

    # 总 shard 数 = n_clients * classes_per_client
    # 每个类别切出 shards_per_class 个 shard
    total_shards     = n_clients * classes_per_client
    shards_per_class = max(1, total_shards // num_classes)

    # 将每个类别的样本切成 shards_per_class 份
    all_shards = []
    for c in range(num_classes):
        class_idx = np.where(labels_np == c)[0]
        np.random.shuffle(class_idx)
        all_shards.extend(np.array_split(class_idx, shards_per_class))

    # 打乱 shard 顺序后轮流分配给各客户端
    shard_order    = np.random.permutation(len(all_shards))
    client_indices = [[] for _ in range(n_clients)]
    for rank, shard_id in enumerate(shard_order):
        client_indices[rank % n_clients].extend(all_shards[shard_id].tolist())

    _print_pathological_distribution(
        client_indices, labels_np, n_clients, num_classes, classes_per_client
    )

    client_datasets   = []
    client_indices_np = []
    for indices in client_indices:
        idx_arr = np.array(indices)
        client_datasets.append(make_client_dataset(images_np, labels_np, idx_arr, config))
        client_indices_np.append(idx_arr)

    return client_datasets, client_indices_np


def _print_pathological_distribution(client_indices, labels_np, n_clients,
                                      num_classes, classes_per_client):
    """打印每个客户端实际持有的类别及各类别数量，验证 pathological 分区效果。"""
    print(f"\n[Pathological non-IID] {n_clients} clients, "
          f"{classes_per_client} classes/client, {num_classes} total classes")
    print(f"{'Client':<12} {'Samples':<10} Classes held (cls:count)")
    print("-" * 72)
    for i, indices in enumerate(client_indices):
        idx_arr      = np.array(indices)
        label_counts = np.bincount(labels_np[idx_arr], minlength=num_classes)
        cls_str      = ", ".join([
            f"cls{c}:{label_counts[c]}"
            for c in range(num_classes)
            if label_counts[c] > 0
        ])
        print(f"client_{i:<5} {len(idx_arr):<10} [{cls_str}]")
    print()