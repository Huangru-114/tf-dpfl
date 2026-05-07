import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def collect_gradient_updates(clients, global_weights):
    """
    让所有客户端各做一次前向+反向，收集梯度更新方向。
    这是数据分布的隐私友好代理，不暴露原始数据。

    注意：这里收集的是模型更新量（本地权重 - 全局权重），
    不是单步梯度，所以需要先让客户端训练几步。
    """
    updates = []
    for client in clients:
        local_w = client.model.get_weights()
        delta   = [lw - gw for lw, gw in zip(local_w, global_weights)]
        flat    = np.concatenate([d.flatten() for d in delta])
        updates.append(flat)
    return updates


def cosine_similarity_matrix(updates):
    """计算所有客户端梯度更新的两两余弦相似度矩阵。"""
    n          = len(updates)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ni = np.linalg.norm(updates[i])
            nj = np.linalg.norm(updates[j])
            if ni > 0 and nj > 0:
                sim_matrix[i][j] = (
                    np.dot(updates[i], updates[j]) / (ni * nj)
                )
    return sim_matrix


def cluster_by_similarity(sim_matrix, n_clusters):
    """
    用层级聚类把相似度矩阵转换为 n_clusters 个簇的标签。
    相似度高 = 距离小 → 分到同一个 Edge Server。
    """
    dist_matrix = np.clip(1.0 - sim_matrix, 0, 2)
    clustering  = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    return clustering.fit_predict(dist_matrix)


# ══════════════════════════════════════════════════════════════
# 语义分组表
# ══════════════════════════════════════════════════════════════

# ── CIFAR-10 ──────────────────────────────────────────────────
# airplane=0, automobile=1, bird=2, cat=3, deer=4,
# dog=5, frog=6, horse=7, ship=8, truck=9
_CIFAR10_SEMANTIC_GROUPS = {
    2: [
        {2, 3, 4, 5, 6, 7},   # edge 0：动物
        {0, 1, 8, 9},          # edge 1：交通工具
    ],
    4: [
        {2, 6},                # edge 0：飞行/水生动物（bird, frog）
        {3, 4, 5, 7},          # edge 1：陆地动物（cat, deer, dog, horse）
        {0, 8},                # edge 2：空中/水上交通（airplane, ship）
        {1, 9},                # edge 3：陆地交通（automobile, truck）
    ],
}

# ── CIFAR-100 ─────────────────────────────────────────────────
# 20 个超类到细粒度类 ID 的映射（CIFAR-100 官方定义）
# 超类 ID → fine-class ID 列表
_CIFAR100_SC_TO_FINE = {
    0:  [4, 30, 55, 72, 95],   # aquatic mammals
    1:  [1, 32, 67, 73, 91],   # fish
    2:  [54, 62, 70, 82, 92],  # flowers
    3:  [9, 10, 16, 28, 61],   # food containers
    4:  [0, 51, 53, 57, 83],   # fruit & vegetables
    5:  [22, 39, 40, 86, 87],  # household electrical devices
    6:  [5, 20, 25, 84, 94],   # household furniture
    7:  [6, 7, 14, 18, 24],    # insects
    8:  [3, 42, 43, 88, 97],   # large carnivores
    9:  [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores & herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],   # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],   # vehicles 1
    19: [41, 69, 81, 85, 89],  # vehicles 2
}

# 逆向映射：fine-class ID → 超类 ID（供 _fine_to_superclass 快速查询）
_CIFAR100_FINE_TO_SC = {
    fine: sc
    for sc, fines in _CIFAR100_SC_TO_FINE.items()
    for fine in fines
}

# CIFAR-100 超类的语义分组：n_edges → [超类 ID 集合, ...]
# 分组原则：同一 edge 内的超类在视觉语义上尽量相近
#
# n_edges=2 ：生物界 vs 人造物
# n_edges=4 ：水生/小型动物 | 大型陆地动物 | 植物/自然/人 | 物体/交通/建筑
# n_edges=5 ：水生 | 陆地动物 | 植物自然 | 人与室内 | 交通与建筑
# n_edges=10：每 2 个语义相近的超类一组
# n_edges=20：每个超类单独一个 edge（最极端设置）
_CIFAR100_SEMANTIC_GROUPS = {
    # n_edges=2：生命体 vs 无生命体/场景
    # 每组恰好 10 个超类 = 50 个细粒度类，保证 edge 间客户端数量平衡
    # edge 0：所有动物 + 人（生命体）
    # edge 1：植物/食物/自然场景 + 人造物（无生命体或场景）
    2: [
        {0, 1, 7, 8, 11, 12, 13, 14, 15, 16},  # 生命体：水生动物、鱼、昆虫、大型食肉、草食、中型哺乳、无脊椎、人、爬行、小型哺乳
        {2, 3, 4, 5, 6,  9, 10, 17, 18, 19},   # 非生命/场景：花、食品容器、果蔬、家电、家具、户外建筑、自然场景、树木、交通1、交通2
    ],
    4: [
        {0, 1, 7, 13},        # 水生动物 + 昆虫/无脊椎动物
        {8, 11, 12, 15, 16},  # 大型/中型/小型陆地动物
        {2, 4, 10, 14, 17},   # 植物、自然场景、人
        {3, 5, 6, 9, 18, 19}, # 食品容器、家电、家具、建筑、交通工具
    ],
    5: [
        {0, 1, 13},            # 水生生物（aquatic mammals, fish, invertebrates）
        {7, 8, 11, 12, 15, 16},# 陆地动物（insects, carnivores, herbivores, mammals, reptiles）
        {2, 4, 10, 17},        # 植物与自然（flowers, fruit/veg, natural outdoor, trees）
        {3, 5, 6, 14},         # 室内场景（food containers, electrical, furniture, people）
        {9, 18, 19},           # 交通与建筑（man-made outdoor, vehicles 1&2）
    ],
    10: [
        {0, 1},                # 水生动物（aquatic mammals + fish）
        {7, 13},               # 小型无脊椎动物（insects + non-insect invertebrates）
        {8, 15},               # 爬行动物与猛兽（large carnivores + reptiles）
        {11, 12, 16},          # 哺乳动物（large herbivores + medium + small mammals）
        {2, 17},               # 植物（flowers + trees）
        {4, 10},               # 食物与自然（fruit/veg + natural outdoor scenes）
        {14},                  # 人
        {3, 6},                # 室内物品（food containers + household furniture）
        {5},                   # 家用电器
        {9, 18, 19},           # 交通与建筑（man-made outdoor + vehicles 1&2）
    ],
    20: [{sc} for sc in range(20)],  # 每个超类独占一个 edge
}


def _build_cifar100_edge_groups(n_edges):
    """
    为未在 _CIFAR100_SEMANTIC_GROUPS 中预定义的 n_edges 构建 CIFAR-100 超类分组。

    策略：将 20 个超类按 round-robin 均匀分配，保证：
    1. 每个 edge 的超类数量相同（20/n_edges，需 n_edges 能整除 20）
    2. 同一超类的 5 个细粒度类始终在同一 edge（保留超类语义完整性）
    3. 每个 edge 持有相同数量的细粒度类（100/n_edges）

    注意：此函数只是几何均匀分配，无语义优化。
    对于常用 n_edges（2,4,5,10,20），应直接使用预定义的语义分组表。
    """
    superclass_ids = list(range(20))
    groups = [set() for _ in range(n_edges)]
    for i, sc in enumerate(superclass_ids):
        groups[i % n_edges].add(sc)
    return groups


def _superclass_groups_to_fineclasses(superclass_groups):
    """
    将超类 ID 集合列表转换为细粒度类 ID 集合列表。
    供 semantic_assignment 使用。
    """
    return [
        {fine for sc in sc_set for fine in _CIFAR100_SC_TO_FINE[sc]}
        for sc_set in superclass_groups
    ]


# ══════════════════════════════════════════════════════════════
# 核心：客户端 → Edge Server 的分配方案
# ══════════════════════════════════════════════════════════════

def random_assignment(clients, n_edges):
    """
    基础版：随机均匀分配，作为 baseline。
    返回 assignments[i] = 第 i 个客户端归属的 Edge 编号。
    """
    n          = len(clients)
    indices    = np.random.permutation(n)
    splits     = np.array_split(indices, n_edges)
    assignments = np.zeros(n, dtype=int)
    for edge_id, split in enumerate(splits):
        for idx in split:
            assignments[idx] = edge_id

    print(f"[Assignment] Random | {n_edges} edges")
    _print_assignment(assignments, n_edges)
    return assignments.tolist()


def semantic_assignment(clients, n_edges, config):
    """
    基于语义相似性的 edge 分组（论文 Hier-pFedMe 的分组方式）。

    原理：对每个 client 统计其持有的细粒度类别，然后将其分配到
    与这些类别语义最近的 edge 组（以细粒度类别集合交集大小衡量）。

    支持数据集：
        cifar10   → 硬编码语义分组（N=2：动物 vs 交通工具；N=4：进一步细分）
        cifar100  → 基于官方 20 个超类的语义分组
                    预定义 N∈{2,4,5,10,20}；其他 N 自动按超类轮流分配
        其他      → 按类别 ID 区间均匀划分（Tiny-ImageNet 等）

    Args:
        clients  : List[FLClient]
        n_edges  : edge 数量
        config   : 读取 dataset 名称和 num_classes

    Returns:
        List[int]，长度为 n_clients，每个值为 [0, n_edges) 内的 edge id
    """
    dataset     = config["data"]["dataset"].lower()
    num_classes = config["data"]["num_classes"]

    # ── 构建每个 edge 组的细粒度类别集合 ──────────────────────────────
    if dataset == "cifar10" and n_edges in _CIFAR10_SEMANTIC_GROUPS:
        # CIFAR-10：直接使用细粒度类 ID 分组
        edge_class_groups = _CIFAR10_SEMANTIC_GROUPS[n_edges]

    elif dataset == "cifar100":
        # CIFAR-100：先确定超类分组，再展开为细粒度类 ID 集合
        if n_edges in _CIFAR100_SEMANTIC_GROUPS:
            sc_groups = _CIFAR100_SEMANTIC_GROUPS[n_edges]
        else:
            # 不在预定义表中：按超类轮流分配，保持超类语义完整性
            print(f"  [Semantic] n_edges={n_edges} not in predefined CIFAR-100 groups, "
                  f"falling back to round-robin superclass assignment")
            sc_groups = _build_cifar100_edge_groups(n_edges)

        edge_class_groups = _superclass_groups_to_fineclasses(sc_groups)

        # 打印分组信息，便于核查
        print(f"\n  [Semantic] CIFAR-100 edge groups (superclass level):")
        sc_names = {
            0:"aquatic mammals", 1:"fish", 2:"flowers", 3:"food containers",
            4:"fruit/veg", 5:"electrical", 6:"furniture", 7:"insects",
            8:"large carnivores", 9:"man-made outdoor", 10:"natural outdoor",
            11:"herbivores", 12:"medium mammals", 13:"invertebrates",
            14:"people", 15:"reptiles", 16:"small mammals", 17:"trees",
            18:"vehicles1", 19:"vehicles2",
        }
        for e, sc_set in enumerate(sc_groups):
            names = [sc_names[sc] for sc in sorted(sc_set)]
            print(f"    Edge {e}: {names}  "
                  f"(fine classes: {len(edge_class_groups[e])})")

    else:
        # Tiny-ImageNet 等：按类别 ID 区间均匀划分
        classes_per_edge  = num_classes // n_edges
        edge_class_groups = [
            set(range(
                e * classes_per_edge,
                (e + 1) * classes_per_edge if e < n_edges - 1 else num_classes
            ))
            for e in range(n_edges)
        ]

    # ── 统计每个 client 持有的细粒度类别 ──────────────────────────────
    assignments = []
    for client in clients:
        label_counts = np.zeros(num_classes, dtype=int)
        for _, labels in client.dataset.unbatch():
            label_counts[labels.numpy()] += 1
        client_classes = set(np.where(label_counts > 0)[0].tolist())

        # 计算与每个 edge 组的交集大小，取最大值对应的 edge
        overlaps  = [len(client_classes & grp) for grp in edge_class_groups]
        best_edge = int(np.argmax(overlaps)) if max(overlaps) > 0 \
                    else np.random.randint(n_edges)
        assignments.append(best_edge)

    print(f"\n[Assignment] Semantic | {n_edges} edges | dataset={dataset}")
    _print_assignment(assignments, n_edges)
    return assignments


def warmup_gradient_assignment(clients, global_model,
                                n_edges, config):
    """
    进阶版：基于 warm-up 的梯度相似度聚类分配。

    流程：
        1. 用全局模型广播给所有客户端
        2. 所有客户端做 warmup_rounds 轮本地训练
        3. 收集每个客户端的梯度更新
        4. 计算两两余弦相似度
        5. 层级聚类 → 得到每个客户端的 Edge 归属

    调用时机：build_edge_servers() 里，在正式训练开始前。
    """
    warmup_rounds  = config["federation"].get("warmup_rounds", 5)
    global_weights = global_model.get_weights()

    print(f"\n[Assignment] Gradient clustering | "
          f"warm-up {warmup_rounds} rounds...")

    # Step 1：广播全局权重，所有客户端从同一起点开始
    for client in clients:
        client.set_weights(global_weights)

    # Step 2：warm-up 训练，积累梯度信息
    # 每轮结束后不聚合，让各客户端沿自己的数据分布方向发散
    # 这样发散方向的差异就反映了数据分布的差异
    for r in range(1, warmup_rounds + 1):
        for client in clients:
            client.local_train(round_idx=r)
        print(f"  Warm-up round {r}/{warmup_rounds} done")

    # Step 3：收集所有客户端训练后的梯度更新方向
    updates    = collect_gradient_updates(clients, global_weights)
    sim_matrix = cosine_similarity_matrix(updates)

    # Step 4：聚类
    labels = cluster_by_similarity(sim_matrix, n_edges)

    # Step 5：恢复全局权重（warm-up 不应该影响正式训练的起点）
    for client in clients:
        client.set_weights(global_weights)

    print(f"[Assignment] Clustering done | {n_edges} edges")
    _print_assignment(labels, n_edges, sim_matrix)

    return labels.tolist()


def _print_assignment(assignments, n_edges, sim_matrix=None):
    """打印分配结果，如果有相似度矩阵则显示簇内相似度。"""
    for e in range(n_edges):
        members = [i for i, a in enumerate(assignments) if a == e]
        if sim_matrix is not None and len(members) > 1:
            idxs     = members
            intra    = np.mean([
                sim_matrix[i][j]
                for i in idxs for j in idxs if i != j
            ])
            sim_str  = f" | intra-sim={intra:.3f}"
        else:
            sim_str  = ""
        print(f"  Edge {e}: clients {members}{sim_str}")


def histogram_assignment(clients, n_edges, config):
    """
    直接用标签分布直方图聚类，完全不需要训练。
    从几分钟压缩到几秒。

    可选加 DP 噪声保护隐私（见 config clustering_epsilon）。
    """
    num_classes = config["data"]["num_classes"]
    epsilon     = config["federation"].get("clustering_epsilon", None)

    print(f"\n[Assignment] Histogram clustering | "
          f"{len(clients)} clients → {n_edges} edges"
          + (f" | ε={epsilon}" if epsilon else " | no DP"))

    histograms = []
    for client in clients:
        # 直接从 dataset 统计标签分布
        label_counts = np.zeros(num_classes)
        for _, labels in client.dataset.unbatch():
            label_counts[labels.numpy()] += 1

        hist = label_counts / (label_counts.sum() + 1e-8)

        # 可选：加 Laplace 噪声保护
        if epsilon is not None:
            sensitivity = 1.0 / max(client.n_samples, 1)
            noise       = np.random.laplace(
                0, sensitivity / epsilon, num_classes
            )
            hist = np.clip(hist + noise, 0, 1)

        histograms.append(hist)

    histograms = np.array(histograms)

    # 用 KL 散度作为距离
    from sklearn.cluster import AgglomerativeClustering
    dist_matrix = _kl_distance_matrix(histograms)
    clustering  = AgglomerativeClustering(
        n_clusters=n_edges,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist_matrix)

    _print_assignment(labels, n_edges)
    return labels.tolist()


def _kl_distance_matrix(histograms):
    """
    计算标签分布之间的对称 KL 散度矩阵。
    比余弦距离更适合比较概率分布。
    """
    n    = len(histograms)
    dist = np.zeros((n, n))
    eps  = 1e-8
    for i in range(n):
        for j in range(n):
            p = histograms[i] + eps
            q = histograms[j] + eps
            # 对称 KL = (KL(p||q) + KL(q||p)) / 2
            kl_pq = np.sum(p * np.log(p / q))
            kl_qp = np.sum(q * np.log(q / p))
            dist[i][j] = (kl_pq + kl_qp) / 2.0
    return dist