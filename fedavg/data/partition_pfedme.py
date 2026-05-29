import numpy as np
import tensorflow as tf

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
