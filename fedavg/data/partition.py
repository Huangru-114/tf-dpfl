import numpy as np
import tensorflow as tf


def extract_numpy(dataset: tf.data.Dataset):
    """
    把 unbatched 的 tf.data.Dataset 提取成 numpy 数组。
    调用前确保 dataset 已经 .unbatch()。
    这一步只需要在实验开始时做一次。
    """
    images, labels = [], []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)


def _augment(image, label):
    """Standard CIFAR augmentation: random horizontal flip + random crop."""
    shape = tf.shape(image)         # capture once, before any modification
    image = tf.image.random_flip_left_right(image)
    image = tf.image.pad_to_bounding_box(image, 4, 4, shape[0] + 8, shape[1] + 8)
    # tf.stack required: passing a Python list of symbolic tensors fails in graph mode
    image = tf.image.random_crop(image, tf.stack([shape[0], shape[1], shape[2]]))
    return image, label


def make_client_dataset(images_np, labels_np, indices, config, shuffle=True):
    """
    给定索引列表，构建单个客户端的 tf.data.Dataset。
    shuffle=True (训练集)：打乱 + 数据增强。
    shuffle=False (测试集)：确定性顺序，无增强。
    """
    client_images = images_np[indices]
    client_labels = labels_np[indices]

    ds = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices))
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config["data"]["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds
 
 
def split_client_train_test(images_np, labels_np, client_indices, config,
                            test_ratio=0.2):
    """
    PFLlib 式划分：从每个 client 自身的训练样本中按 test_ratio 随机抽取测试集。
    剩余 (1-test_ratio) 部分作为训练集。

    Returns:
        train_datasets : List[tf.data.Dataset]，(1-test_ratio) 训练集
        train_indices  : List[np.ndarray]，训练样本索引
        test_datasets  : List[tf.data.Dataset]，test_ratio 测试集（无增强）
    """
    train_datasets, train_indices_list, test_datasets = [], [], []
    for indices in client_indices:
        n     = len(indices)
        n_test = max(1, int(n * test_ratio))
        perm  = np.random.permutation(n)
        test_idx  = indices[perm[:n_test]]
        train_idx = indices[perm[n_test:]]
        train_datasets.append(
            make_client_dataset(images_np, labels_np, train_idx, config, shuffle=True)
        )
        train_indices_list.append(train_idx)
        test_datasets.append(
            make_client_dataset(images_np, labels_np, test_idx, config, shuffle=False)
        )

    avg_train = np.mean([len(idx) for idx in train_indices_list])
    avg_test  = avg_train * test_ratio / (1 - test_ratio)
    print(f"[Train/Test split] {len(client_indices)} clients | "
          f"avg train={avg_train:.0f}, avg test={avg_test:.0f} samples (ratio={test_ratio})")
    return train_datasets, train_indices_list, test_datasets


def make_per_client_test_datasets(test_images_np, test_labels_np,
                                   client_train_indices_np, train_labels_np,
                                   config):
    client_test_datasets = []
    print(f"\n[Test split] Verifying per-client train/test class alignment:")
    print(f"{'Client':<12} {'Train classes':<30} {'Test classes':<30} {'Test samples'}")
    print("-" * 90)
    for i, train_indices in enumerate(client_train_indices_np):
        held_classes = np.unique(train_labels_np[train_indices])
        mask         = np.isin(test_labels_np, held_classes)
        test_indices = np.where(mask)[0]
        test_classes = np.unique(test_labels_np[test_indices])

        # 验证 train/test 类别是否完全一致
        match = set(held_classes.tolist()) == set(test_classes.tolist())
        flag  = "" if match else "  ← MISMATCH"

        print(f"client_{i:<5} {str(sorted(held_classes.tolist())):<30} "
              f"{str(sorted(test_classes.tolist())):<30} "
              f"{len(test_indices)}{flag}")

        if len(test_indices) == 0:
            test_indices = np.arange(len(test_labels_np))

        ds = make_client_dataset(
            test_images_np, test_labels_np, test_indices, config, shuffle=False
        )
        client_test_datasets.append(ds)

    print()
    return client_test_datasets

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

    print(f"[IID] {n_clients} clients, ~{n_total // n_clients} samples each")
    return client_datasets, splits


def _print_distribution(client_indices, labels_np, n_clients, num_classes, alpha):
    """打印每个客户端的样本数和类别分布，用于验证 Non-IID 效果。"""
    print(f"\n[Non-IID] α={alpha}, {n_clients} clients, {num_classes} classes")
    print(f"{'Client':<12} {'Samples':<10} All classes (cls:count)")
    print("-" * 72)
    for i, indices in enumerate(client_indices):
        indices      = np.array(indices)
        label_counts = np.bincount(labels_np[indices], minlength=num_classes)
        all_cls_str  = ", ".join([
            f"cls{c}:{label_counts[c]}"
            for c in range(num_classes)
            if label_counts[c] > 0
        ])
        print(f"client_{i:<5} {len(indices):<10} {all_cls_str}")
    print()


def noniid_partition(images_np, labels_np, n_clients, config, alpha=0.5):
    """
    Non-IID 分区：用 Dirichlet(α) 分布控制异质程度。
    α 越小，各客户端数据分布差异越大。

    原理：对每个类别从 Dirichlet(α) 采样分配比例，
          决定该类的样本如何分配给各客户端。

    Returns:
        List[tf.data.Dataset]，长度为 n_clients
    """
    num_classes    = len(np.unique(labels_np))
    client_indices = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels_np == c)[0]
        np.random.shuffle(class_indices)

        proportions  = np.random.dirichlet(alpha=np.ones(n_clients) * alpha)
        cut_points   = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits       = np.split(class_indices, cut_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    _print_distribution(client_indices, labels_np, n_clients, num_classes, alpha)

    client_datasets = []
    client_indices_np = []

    for indices in client_indices:
        ds = make_client_dataset(images_np, labels_np, np.array(indices), config)
        client_datasets.append(ds)
        client_indices_np.append(np.array(indices))


    return client_datasets, client_indices_np


def pathological_noniid_partition(images_np, labels_np, n_clients, config,
                                   classes_per_client=10):
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


def superclass_edge_partition(images_np, labels_np, n_clients, config,
                               edge_fine_classes):
    """
    超类感知分区：每个 edge 只分配数据属于该 edge 语义超类的客户端。

    与标准 pathological_noniid_partition 的区别：
    - 标准版：所有客户端从全部 100 个类中各取 classes_per_client 个类
    - 本函数：每个 edge 的客户端只从该 edge 的 50 个类中取 classes_per_client 个类

    效果：将每个 edge model 的实际分类任务从 100 分类降低到 50 分类（n_edges=2 时），
    与论文中每个 edge 只负责其语义超类范围内的分类假设一致。

    Args:
        images_np        : 全量图片 numpy 数组 (N, H, W, C)
        labels_np        : 全量标签 numpy 数组 (N,)，保持原始标签 ID（不重映射）
        n_clients        : 总客户端数，均匀分配给各 edge
        config           : 读取 batch_size 和 federation.classes_per_client
        edge_fine_classes: List[set]，每个 edge 拥有的细粒度类 ID 集合
                           可由 clustering._superclass_groups_to_fineclasses() 生成

    Returns:
        client_datasets  : List[tf.data.Dataset]，长度为 n_clients
        client_indices   : List[np.ndarray]，每个客户端的样本全局索引
        assignments      : List[int]，assignments[i] = 第 i 个客户端所属 edge ID
                           客户端按 edge 顺序排列（edge 0 的所有客户端在前）
    """
    n_edges            = len(edge_fine_classes)
    classes_per_client = int(config["federation"].get("classes_per_client", 10))

    # 均匀分配每个 edge 的客户端数（divmod 保证总数精确）
    base, remainder    = divmod(n_clients, n_edges)
    clients_per_edge   = [base + (1 if e < remainder else 0) for e in range(n_edges)]

    all_datasets  = []
    all_indices   = []
    all_assignments = []

    for edge_id, fine_classes in enumerate(edge_fine_classes):
        n_ec             = clients_per_edge[edge_id]
        fine_cls_sorted  = sorted(fine_classes)
        n_edge_classes   = len(fine_cls_sorted)

        # 筛选出属于该 edge 的所有样本（使用全局索引）
        mask                = np.isin(labels_np, fine_cls_sorted)
        edge_global_indices = np.where(mask)[0]
        edge_labels         = labels_np[edge_global_indices]

        # Pathological non-IID：在该 edge 的类别范围内切 shard
        total_shards     = n_ec * classes_per_client
        shards_per_class = max(1, total_shards // n_edge_classes)

        all_shards = []
        for c in fine_cls_sorted:
            class_global_idx = edge_global_indices[edge_labels == c]
            np.random.shuffle(class_global_idx)
            all_shards.extend(np.array_split(class_global_idx, shards_per_class))

        # 打乱后轮流分配给该 edge 内的客户端
        shard_order      = np.random.permutation(len(all_shards))
        edge_client_idx  = [[] for _ in range(n_ec)]
        for rank, shard_id in enumerate(shard_order):
            edge_client_idx[rank % n_ec].extend(all_shards[shard_id].tolist())

        for indices in edge_client_idx:
            idx_arr = np.array(indices, dtype=np.int64)
            all_datasets.append(
                make_client_dataset(images_np, labels_np, idx_arr, config)
            )
            all_indices.append(idx_arr)
            all_assignments.append(edge_id)

    _print_superclass_edge_distribution(
        all_indices, all_assignments, labels_np,
        n_edges, edge_fine_classes, classes_per_client
    )
    return all_datasets, all_indices, all_assignments


def _print_superclass_edge_distribution(client_indices, assignments, labels_np,
                                         n_edges, edge_fine_classes, classes_per_client):
    print(f"\n[Superclass-aware partition] {n_edges} edges, "
          f"{classes_per_client} classes/client")
    for e in range(n_edges):
        edge_pool    = edge_fine_classes[e]
        edge_clients = [i for i, a in enumerate(assignments) if a == e]
        print(f"\n  Edge {e} (class pool: {sorted(edge_pool)}):")
        print(f"  {'Client':<12} {'Samples':<10} {'Classes held':<35} {'Out-of-pool?'}")
        print("  " + "-" * 75)
        for i in edge_clients:
            idx_arr     = client_indices[i]
            held        = set(np.unique(labels_np[idx_arr]).tolist())
            out_of_pool = held - edge_pool
            flag        = f"LEAK: {out_of_pool}" if out_of_pool else "OK"
            print(f"  client_{i:<5} {len(idx_arr):<10} "
                  f"{str(sorted(held)):<35} {flag}")
    print()


def make_per_edge_test_datasets(test_images_np, test_labels_np,
                                 edge_fine_classes, config):
    """
    为每个 edge 生成仅包含该 edge 语义超类细粒度类别的测试集。
    配合 superclass_edge_partition 使用，保证 EM 评估与训练分布一致。

    Args:
        test_images_np   : 全量测试图片 (N_test, H, W, C)
        test_labels_np   : 全量测试标签 (N_test,)
        edge_fine_classes: List[set]，每个 edge 的细粒度类 ID 集合
        config           : 读取 batch_size

    Returns:
        List[tf.data.Dataset]，长度为 n_edges
    """
    edge_test_datasets = []
    for e, fine_classes in enumerate(edge_fine_classes):
        mask         = np.isin(test_labels_np, sorted(fine_classes))
        test_indices = np.where(mask)[0]
        if len(test_indices) == 0:
            test_indices = np.arange(len(test_labels_np))
        ds = make_client_dataset(
            test_images_np, test_labels_np, test_indices, config, shuffle=False
        )
        edge_test_datasets.append(ds)
        print(f"  [Test split] Edge {e}: {len(test_indices)} test samples "
              f"({len(fine_classes)} classes)")
    return edge_test_datasets


def merge_test_datasets(clients, batch_size):
    """合并一个服务器下所有客户端的测试数据集"""
    datasets = []
    for client in clients:
        ds = client.get_test_dataset()  # 返回 tf.data.Dataset，可能已经 batching
        # 为了合并，我们去掉 batch 维度，变成每个样本
        ds = ds.unbatch()
        datasets.append(ds)
    # 使用 concatenate 依次合并
    merged = datasets[0]
    for ds in datasets[1:]:
        merged = merged.concatenate(ds)
    # 重新进行批处理
    merged = merged.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return merged

if __name__ == "__main__":
    import yaml
    from data.dataset import load_gtsrb

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    train_ds, _ = load_gtsrb(config)

    print("Extracting numpy arrays...")
    images_np, labels_np = extract_numpy(train_ds.unbatch())
    print(f"Extracted: images={images_np.shape}, labels={labels_np.shape}")

    noniid_clients = noniid_partition(
        images_np, labels_np, n_clients=10, config=config, alpha=0.5
    )
    for images, labels in noniid_clients[0].take(1):
        print("client_0 batch shape:", images.shape)
