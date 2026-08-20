"""
data/hierarchical_partition.py  –  层级（两段式）Non-IID 数据划分

核心思想（任务核心贡献点）：
  Step 1（inter-edge）：先把全量数据按 `inter_cfg` 分给 n_edges 个边缘节点
                        （把每个边缘节点视为一个"大客户端"）。
  Step 2（intra-edge）：再在每个边缘节点内部按 `intra_cfg` 把该 edge 的样本
                        进一步划分给该 edge 下的各客户端。

inter / intra 都支持 type ∈ {"iid", "dirichlet", "pathological"}，
通过 `HIERARCHICAL_DISTRIBUTION_CONFIGS` 组合出 H1–H4 等配置。

与 data/partition.py::superclass_edge_partition 保持**相同返回签名**：
  (client_datasets, client_indices, assignments)
其中 assignments[i] = 第 i 个客户端所属 edge id（客户端按 edge 顺序排列），
因此可直接走 main.py 的 `precomputed_assignments`（baked）通道，跳过再分配。

实现要点：
  - 全程基于"样本全局索引"操作，最后才用 make_client_dataset 构建 tf.data.Dataset。
  - 保证每个客户端样本数量大致均衡，设置最小样本数阈值 min_samples，
    不足的客户端从同 edge 内最富余的客户端借样本，避免空 / 极小客户端。
  - 使用 np.random，可由 main.py 的 set_seed 统一控制复现。
"""

import numpy as np

# 注：make_client_dataset 依赖 TensorFlow，改为函数内惰性导入，
# 使索引级划分逻辑（及离线冒烟检查）可在无 TF 环境下独立运行。


# ════════════════════════════════════════════════════════════════════════════
# 索引级划分原语：在给定的样本索引池 pool_idx 上，把样本分成 n_parts 份
# ════════════════════════════════════════════════════════════════════════════

def _split_iid(pool_idx, labels_np, n_parts, rng):
    """IID：随机均匀切分。"""
    perm = rng.permutation(len(pool_idx))
    return [pool_idx[chunk] for chunk in np.array_split(perm, n_parts)]


def _split_dirichlet(pool_idx, labels_np, n_parts, alpha, rng):
    """Dirichlet(α)：对池内每个类别按 Dirichlet 比例分给 n_parts 份。"""
    parts = [[] for _ in range(n_parts)]
    pool_labels = labels_np[pool_idx]
    for c in np.unique(pool_labels):
        cls_idx = pool_idx[pool_labels == c]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha=np.ones(n_parts) * alpha)
        cut = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for p, split in enumerate(np.split(cls_idx, cut)):
            parts[p].extend(split.tolist())
    return [np.array(p, dtype=np.int64) for p in parts]


def _split_pathological(pool_idx, labels_np, n_parts, classes_per_part, rng):
    """
    Pathological：把池内每个类切成若干 shard，轮流分配给 n_parts 份，
    使每份大致只含 classes_per_part 个类别。
    """
    pool_labels = labels_np[pool_idx]
    present = np.unique(pool_labels)
    n_present = len(present)
    classes_per_part = max(1, min(classes_per_part, n_present))

    total_shards = n_parts * classes_per_part
    shards_per_class = max(1, total_shards // n_present)

    all_shards = []
    for c in present:
        cls_idx = pool_idx[pool_labels == c]
        rng.shuffle(cls_idx)
        all_shards.extend(np.array_split(cls_idx, shards_per_class))

    order = rng.permutation(len(all_shards))
    parts = [[] for _ in range(n_parts)]
    for rank, sid in enumerate(order):
        parts[rank % n_parts].extend(all_shards[sid].tolist())
    return [np.array(p, dtype=np.int64) for p in parts]


def _partition_indices(pool_idx, labels_np, n_parts, cfg, rng):
    """按 cfg["type"] 把 pool_idx 分成 n_parts 份索引数组。"""
    ptype = cfg.get("type", "iid")
    if ptype == "iid":
        return _split_iid(pool_idx, labels_np, n_parts, rng)
    if ptype == "dirichlet":
        return _split_dirichlet(pool_idx, labels_np, n_parts,
                                float(cfg.get("alpha", 0.5)), rng)
    if ptype == "pathological":
        # inter 层用 classes_per_edge，intra 层用 classes_per_client；二者择一
        cpp = int(cfg.get("classes_per_edge",
                          cfg.get("classes_per_client", 2)))
        return _split_pathological(pool_idx, labels_np, n_parts, cpp, rng)
    raise ValueError(f"Unknown partition type in hierarchical cfg: {ptype!r}")


# ════════════════════════════════════════════════════════════════════════════
# 最小样本阈值：在同一 edge 内做均衡，避免空 / 极小客户端
# ════════════════════════════════════════════════════════════════════════════

def _enforce_min_samples(client_idx_lists, min_samples, rng):
    """
    对同一 edge 内的客户端索引列表做就地均衡：
    对样本数 < min_samples 的客户端，从当前最富余的客户端借样本补足。
    若总样本不足以让所有客户端达到阈值，则尽量均摊（不抛错）。
    """
    lists = [np.array(x, dtype=np.int64) for x in client_idx_lists]
    n = len(lists)
    if n == 0:
        return lists

    total = sum(len(x) for x in lists)
    feasible_min = min(min_samples, total // n)  # 总量不足时降级阈值

    for _ in range(n * 4):  # 有限次再平衡，防御性上限
        sizes = [len(x) for x in lists]
        i_min = int(np.argmin(sizes))
        if sizes[i_min] >= feasible_min:
            break
        i_max = int(np.argmax(sizes))
        if i_max == i_min or sizes[i_max] <= feasible_min:
            break
        need = feasible_min - sizes[i_min]
        movable = sizes[i_max] - feasible_min
        k = max(1, min(need, movable))
        donor = lists[i_max]
        take = rng.choice(len(donor), k, replace=False)
        mask = np.ones(len(donor), dtype=bool)
        mask[take] = False
        lists[i_min] = np.concatenate([lists[i_min], donor[take]])
        lists[i_max] = donor[mask]
    return lists


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def hierarchical_partition(images_np, labels_np, n_clients, n_edges,
                           inter_cfg, intra_cfg, config,
                           min_samples=10, seed=None, build_datasets=True):
    """
    层级两段式划分。

    Args:
        images_np, labels_np : 已标准化的全量图像 / 标签 numpy
        n_clients            : 客户端总数（均匀分配到各 edge）
        n_edges              : 边缘节点数
        inter_cfg            : 跨边缘节点分布，如 {"type":"dirichlet","alpha":0.5}
        intra_cfg            : 边缘节点内部分布，如 {"type":"pathological","classes_per_client":2}
        config               : 读取 batch_size
        min_samples          : 每客户端最小样本阈值
        seed                 : 可选，独立 RNG 种子（None 则用全局 np.random 状态）
        build_datasets       : True 构建 tf.data.Dataset（需 TF）；
                               False 仅返回索引/分配（离线冒烟检查用，client_datasets=None）

    Returns:
        client_datasets : List[tf.data.Dataset] 或 None（build_datasets=False）
        client_indices  : List[np.ndarray]
        assignments     : List[int]，assignments[i] = 客户端 i 的 edge id
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    make_client_dataset = None
    if build_datasets:
        from data.partition import make_client_dataset  # 惰性导入（依赖 TF）

    all_idx = np.arange(len(labels_np), dtype=np.int64)

    # ── Step 1：inter-edge —— 把全量数据分给 n_edges 个 edge ────────────────
    edge_pools = _partition_indices(all_idx, labels_np, n_edges, inter_cfg, rng)

    # 均匀分配每个 edge 的客户端数
    base, rem = divmod(n_clients, n_edges)
    clients_per_edge = [base + (1 if e < rem else 0) for e in range(n_edges)]

    all_datasets, all_indices, all_assignments = [], [], []

    for e in range(n_edges):
        pool = edge_pools[e]
        n_ec = clients_per_edge[e]

        # ── Step 2：intra-edge —— 在该 edge 内部分给 n_ec 个客户端 ──────────
        client_idx_lists = _partition_indices(pool, labels_np, n_ec, intra_cfg, rng)
        client_idx_lists = _enforce_min_samples(client_idx_lists, min_samples, rng)

        for idx_arr in client_idx_lists:
            if build_datasets:
                all_datasets.append(
                    make_client_dataset(images_np, labels_np, idx_arr, config))
            all_indices.append(idx_arr)
            all_assignments.append(e)

    _print_hierarchical_distribution(
        all_indices, all_assignments, labels_np, n_edges, inter_cfg, intra_cfg
    )
    return (all_datasets if build_datasets else None), all_indices, all_assignments


def _print_hierarchical_distribution(client_indices, assignments, labels_np,
                                     n_edges, inter_cfg, intra_cfg):
    num_classes = len(np.unique(labels_np))
    print(f"\n[Hierarchical partition] {n_edges} edges | "
          f"inter={inter_cfg} | intra={intra_cfg}")
    for e in range(n_edges):
        members = [i for i, a in enumerate(assignments) if a == e]
        edge_idx = np.concatenate([client_indices[i] for i in members]) \
            if members else np.array([], dtype=np.int64)
        edge_classes = sorted(np.unique(labels_np[edge_idx]).tolist()) \
            if len(edge_idx) else []
        print(f"\n  Edge {e}: {len(members)} clients | {len(edge_idx)} samples | "
              f"classes={edge_classes}")
        print(f"  {'Client':<12} {'Samples':<10} Classes held (cls:count)")
        print("  " + "-" * 70)
        for i in members:
            idx_arr = client_indices[i]
            counts = np.bincount(labels_np[idx_arr], minlength=num_classes)
            cls_str = ", ".join(f"cls{c}:{counts[c]}"
                                for c in range(num_classes) if counts[c] > 0)
            print(f"  client_{i:<5} {len(idx_arr):<10} [{cls_str}]")
    print()


# ════════════════════════════════════════════════════════════════════════════
# 离线冒烟检查（不依赖 GPU / wandb）：python -m data.hierarchical_partition
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 用合成标签做纯划分逻辑校验，无需真实数据 / TF 计算图。
    n_total, num_classes = 6000, 10
    labels = np.repeat(np.arange(num_classes), n_total // num_classes)
    np.random.shuffle(labels)
    images = np.zeros((n_total, 4, 4, 3), dtype=np.float32)  # 占位
    cfg = {"data": {"batch_size": 16}}

    test_configs = {
        "H1_patho_in_dir_out": (
            {"type": "dirichlet", "alpha": 0.5},                  # inter
            {"type": "pathological", "classes_per_client": 2},    # intra
        ),
        "H2_dir_in_patho_out": (
            {"type": "pathological", "classes_per_edge": 5},      # inter
            {"type": "dirichlet", "alpha": 0.5},                  # intra
        ),
        "H3_high_both": (
            {"type": "dirichlet", "alpha": 0.5},
            {"type": "dirichlet", "alpha": 0.1},
        ),
        "H4_iid_in_dir_out": (
            {"type": "dirichlet", "alpha": 0.1},
            {"type": "iid"},
        ),
    }

    for name, (inter, intra) in test_configs.items():
        print("\n" + "=" * 78)
        print(f"  Smoke test: {name}")
        print("=" * 78)
        _, indices, assignments = hierarchical_partition(
            images, labels, n_clients=20, n_edges=2,
            inter_cfg=inter, intra_cfg=intra, config=cfg,
            min_samples=10, seed=42, build_datasets=False,
        )
        sizes = [len(x) for x in indices]
        assert min(sizes) >= 1, f"{name}: empty client detected!"
        assert sum(sizes) == n_total, \
            f"{name}: sample leak (got {sum(sizes)} vs {n_total})"
        print(f"  [OK] {name}: clients={len(indices)} | "
              f"min={min(sizes)} max={max(sizes)} | total={sum(sizes)}")
    print("\n[Smoke test] all hierarchical configs passed.\n")
