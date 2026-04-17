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


def make_client_dataset(images_np, labels_np, indices, config, shuffle=True):
    """
    给定索引列表，构建单个客户端的 tf.data.Dataset。
    shuffle=False 用于测试集，保持确定性顺序。
    """
    client_images = images_np[indices]
    client_labels = labels_np[indices]
 
    ds = tf.data.Dataset.from_tensor_slices((client_images, client_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices))
    ds = ds.batch(config["data"]["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds
 
 
def make_per_client_test_datasets(test_images_np, test_labels_np,
                                   client_train_indices_np, train_labels_np,
                                   config):
    """
    为每个 client 生成与其训练集同分布的测试子集（论文评估方式）。
 
    原理：从每个 client 的训练索引中找出其持有的类别，
          然后从全体测试集中筛选出这些类别的测试样本。
          这样评估时 client 只在自己见过的类上打分，
          与论文 pathological non-IID 设置的评估方式一致。
 
    Args:
        test_images_np         : 全量测试图片 (N_test, H, W, C)
        test_labels_np         : 全量测试标签 (N_test,)
        client_train_indices_np: List[np.ndarray]，训练集 client 划分索引
        train_labels_np        : 全量训练标签，用于确定每个 client 的持有类别
        config                 : 读取 batch_size
 
    Returns:
        List[tf.data.Dataset]，长度为 n_clients，每个只含该 client 的类别
    """
    client_test_datasets = []
    for train_indices in client_train_indices_np:
        held_classes = np.unique(train_labels_np[train_indices])
        mask         = np.isin(test_labels_np, held_classes)
        test_indices = np.where(mask)[0]
 
        if len(test_indices) == 0:
            # 极端情况：退化为全体测试集
            test_indices = np.arange(len(test_labels_np))
 
        ds = make_client_dataset(
            test_images_np, test_labels_np, test_indices, config, shuffle=False
        )
        client_test_datasets.append(ds)
 
    print(f"[Test split] {len(client_train_indices_np)} per-client test sets, "
          f"avg {np.mean([np.sum(np.isin(test_labels_np, np.unique(train_labels_np[idx]))) for idx in client_train_indices_np]):.0f} samples each")
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
