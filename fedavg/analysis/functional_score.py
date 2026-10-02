"""
analysis/functional_score.py  —  「功能分数」c_k 的纯 numpy 部分（S6a；D-085 方案 §二）

问题：一个 body（edge 模型的特征提取部分）对「把输入推向类 k」有多容易？
  c_k = 在固定 ε、少步数的**定向 PGD**（推向类 k）下，失败的比例（失败 = 仍没被判成 k）。
  c_k 小 = 这个 body 里类 k 的特征方向本来就近 / 容易到达 → 可能已经「预接线」到目标类。
  3-D 想问的是：edge 的 body 和受害 edge 的 body 在 c_{y_t} 上是否分得开。

判成哪一类用 **NCM head**（nearest class mean）：每个 body 配一个在本 edge 良性端留出分片上取类均值原型的
最近均值分类器（不需要私有 head）。本模块只含不依赖 TF 的部分：
  · class_prototypes / ncm_logits：原型与最近均值打分；
  · ck_from_reached：由「PGD 之后是否被判成 k」的布尔矩阵得 c_k（只在真实类 ≠ k 的样本上算）；
  · auroc：秩 AUROC（平局取一半），G1P 离线读几何分数时也用它；
  · edge_contrast：某个 edge 与其余 edge 在 c_k 上的差。
空组 → None（陷阱 #13：无定义不是 0）。TF 部分在 analysis/ck_snapshot.py。
"""

from __future__ import annotations

import numpy as np


def class_prototypes(features, labels, n_classes: int):
    """类均值原型。返回 (protos [K, d] float64，没有样本的类整行 NaN；counts [K])。"""
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    d = f.shape[1] if f.ndim == 2 else 0
    protos = np.full((n_classes, d), np.nan)
    counts = np.zeros(n_classes, dtype=np.int64)
    for k in range(n_classes):
        sel = y == k
        counts[k] = int(sel.sum())
        if counts[k]:
            protos[k] = f[sel].mean(axis=0)
    return protos, counts


def ncm_logits(features, protos):
    """每个样本对每个类的打分 = −‖f − p_k‖²；没有原型的类（NaN）= −inf（永远不会被选中）。"""
    f = np.asarray(features, dtype=np.float64)
    p = np.asarray(protos, dtype=np.float64)
    d2 = ((f[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    d2 = np.where(np.isnan(d2), np.inf, d2)
    return -d2


def ncm_predict(features, protos):
    return np.argmax(ncm_logits(features, protos), axis=1)


def ck_from_reached(y, reached):
    """
    y: [n] 真实类；reached: [n, K] 布尔，reached[i, k] = 对样本 i 做推向 k 的 PGD 之后被判成 k。
    返回 (c [K], n_eligible [K])：c_k = 1 − 命中率，只统计 y ≠ k 的样本（真实类就是 k 的不算「推向」）；
    没有可统计的样本 → None。
    """
    y = np.asarray(y).reshape(-1).astype(np.int64)
    r = np.asarray(reached, dtype=bool)
    K = r.shape[1]
    c, n = [], []
    for k in range(K):
        elig = y != k
        n.append(int(elig.sum()))
        c.append(None if not elig.any() else float(1.0 - r[elig, k].mean()))
    return c, n


def auroc(pos, neg):
    """P(pos > neg) + 0.5·P(平局)，秩和公式（Mann–Whitney U）。任一组为空 → None。"""
    a = np.asarray(pos, dtype=np.float64).reshape(-1)
    b = np.asarray(neg, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return None
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < sorted_v.size:                         # 平局取平均秩
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[:a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def edge_contrast(c_by_edge: dict, ref_edge: int, k: int):
    """
    c_by_edge: {edge_id: [c_0 … c_{K−1}]}。返回 {ref, others_mean, diff}：
    ref = 参照 edge（攻击者所在 edge）的 c_k，others_mean = 其余 edge 的 c_k 均值（None 不参与），
    diff = others_mean − ref（> 0 ⇒ 参照 edge 的 body 更容易被推向 k）。缺任何一边 → 对应项 None。
    """
    ref = (c_by_edge.get(ref_edge) or [None] * (k + 1))[k]
    others = [v[k] for e, v in c_by_edge.items() if e != ref_edge and v and v[k] is not None]
    om = float(np.mean(others)) if others else None
    return {"ref": ref, "others_mean": om,
            "diff": None if (ref is None or om is None) else float(om - ref)}
