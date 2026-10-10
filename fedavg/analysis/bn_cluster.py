"""
analysis/bn_cluster.py  —  CCS 的聚类部分（离线检测）：用客户端上传的 BN moving 统计量做 HDBSCAN、剔除离群端

阶段三 P0 的附加读数（PLAN §4 P0「CCS-clu 离线检测」；只报告，不进判定）：快照里各端的私有 BN 统计量 → 每个 edge 聚一次 →
剔除集合对恶意端的 TPR / FPR。零训练成本，先看 CCS 的聚类在本设定下分不分得开（F-085：统计量通道单独 AUROC 约 0.90）。

照 CCS 官方 `src/defence/AT/hdbscan_detect.py`（FINDINGS F-090）：
  · 每端一条向量 = 全部 BN 层的 running_mean 与 running_var 拼起来；
  · **余弦距离**；`min_cluster_size = n // 2 + 2`、`min_samples = 1`、`allow_single_cluster = True`；
  · 保留**最大**的簇，其余（含噪声）剔除；全部是噪声 → 不剔除。
与官方的差别（记为偏差）：官方每轮只对**本轮被选中**的端聚类；这里一个 edge 的全部有私有统计量的端一起聚
（离线快照里没有「本轮被选中」的概念）。本设定每个 edge 轮每个 edge 只选 2–3 端，`n // 2 + 2` 会超过 n →
官方那种逐轮聚类在本设定里根本成不了簇（在 `min_cluster_size > n` 时这里报告 `degenerate`、不剔除）。

纯 numpy + scikit-learn（≥ 1.3 的 `sklearn.cluster.HDBSCAN`），不 import TF。
"""

from __future__ import annotations

import numpy as np


def stat_vector(stats: list) -> np.ndarray:
    """一端的 BN 统计量（get_weights 里统计量索引处的数组，按索引升序）→ 一条向量。"""
    return np.concatenate([np.asarray(s, np.float64).reshape(-1) for s in stats])


def cosine_distance_matrix(V) -> np.ndarray:
    V = np.asarray(V, np.float64)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = V / n
    D = 1.0 - U @ U.T
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 2.0)


def detect(vectors: dict) -> dict:
    """
    vectors：{client_id: 向量}。返回 {admitted, rejected, labels, min_cluster_size, status}。
    status：ok / all_noise（不剔除）/ degenerate（端太少，不剔除）/ empty。
    """
    ids = sorted(int(k) for k in vectors)
    n = len(ids)
    mcs = n // 2 + 2
    if n == 0:
        return {"admitted": [], "rejected": [], "labels": {}, "min_cluster_size": mcs, "status": "empty"}
    if mcs > n:
        return {"admitted": ids, "rejected": [], "labels": {}, "min_cluster_size": mcs,
                "status": "degenerate"}
    from sklearn.cluster import HDBSCAN
    D = cosine_distance_matrix(np.stack([vectors[i] if i in vectors else vectors[str(i)] for i in ids]))
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="precomputed",
                  allow_single_cluster=True, copy=True).fit(D).labels_
    labels = {i: int(l) for i, l in zip(ids, lab)}
    clusters = [l for l in set(lab.tolist()) if l >= 0]
    if not clusters:
        return {"admitted": ids, "rejected": [], "labels": labels, "min_cluster_size": mcs,
                "status": "all_noise"}
    sizes = {l: int(np.sum(lab == l)) for l in clusters}
    keep = max(sorted(clusters), key=lambda l: sizes[l])          # 最大簇；并列取编号小的（确定性）
    admitted = [i for i in ids if labels[i] == keep]
    rejected = [i for i in ids if labels[i] != keep]
    return {"admitted": admitted, "rejected": rejected, "labels": labels,
            "min_cluster_size": mcs, "status": "ok"}


def rates(rejected, malicious, population) -> dict:
    """剔除集合对恶意端的 TPR、对良性端的 FPR（population = 参与聚类的端）。分母为 0 → None。"""
    pop = set(int(i) for i in population)
    mal = pop & set(int(i) for i in malicious)
    ben = pop - mal
    rej = set(int(i) for i in rejected) & pop
    return {"n": len(pop), "n_malicious": len(mal),
            "tpr": (len(rej & mal) / len(mal)) if mal else None,
            "fpr": (len(rej & ben) / len(ben)) if ben else None}
