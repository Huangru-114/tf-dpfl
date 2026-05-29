import numpy as np
import tensorflow as tf


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
# 分组原则：同一 edge 内的超类在视觉语义上尽量相近，且每组超类数量相等
_CIFAR100_SEMANTIC_GROUPS = {
    # N=2：生命体 vs 无生命体/场景
    # 每组 10 超类 = 50 细粒度类（完全平衡）
    # edge 0：所有动物 + 人（有生命体）
    # edge 1：植物/食物/自然场景 + 人造物
    2: [
        {0, 1, 7, 8, 11, 12, 13, 14, 15, 16},  # aquatic mammals, fish, insects, large carnivores,
                                                  # herbivores, medium mammals, invertebrates,
                                                  # people, reptiles, small mammals
        {2, 3, 4, 5, 6, 9, 10, 17, 18, 19},    # flowers, food containers, fruit/veg,
                                                  # electrical, furniture, man-made outdoor,
                                                  # natural outdoor, trees, vehicles1, vehicles2
    ],
    # N=4：5+5+5+5 超类 = 25+25+25+25 细粒度类（完全平衡）
    # edge 0：水生及小型无脊椎动物
    # edge 1：大型陆地动物 + 人
    # edge 2：植物、自然食物与户外场景
    # edge 3：人造物品与建成环境
    4: [
        {0, 1, 7, 13, 15},   # aquatic mammals, fish, insects, non-insect invertebrates, reptiles
        {8, 11, 12, 14, 16}, # large carnivores, herbivores, medium mammals, people, small mammals
        {2, 3, 4, 10, 17},   # flowers, food containers, fruit/veg, natural outdoor, trees
        {5, 6, 9, 18, 19},   # electrical devices, furniture, man-made outdoor, vehicles1, vehicles2
    ],
}


def _build_cifar100_edge_groups(n_edges):
    """
    为未在 _CIFAR100_SEMANTIC_GROUPS 中预定义的 n_edges 构建 CIFAR-100 超类分组。
    按 round-robin 均匀分配，无语义优化。建议只使用预定义的 N=2/4 配置。
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