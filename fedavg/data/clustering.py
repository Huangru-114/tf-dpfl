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