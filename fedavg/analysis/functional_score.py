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
  · tpr_at_fpr：FPR ≤ 5% 时的 TPR（G1 判定的次要读数）；
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


def tpr_at_fpr(pos, neg, fpr: float = 0.05):
    """
    FPR ≤ fpr 时的 TPR（N-007 的次要读数）。阈值 t = 负类从大到小第 ⌊fpr·n_neg⌋ + 1 个值；
    **严格大于 t 才判阳**，所以实际 FPR = #{neg > t} / n_neg ≤ ⌊fpr·n_neg⌋ / n_neg ≤ fpr，
    与 t 平局的一律判阴（保守，平局多时 TPR 偏低）。返回 #{pos > t} / n_pos；任一组为空 → None。
    """
    a = np.asarray(pos, dtype=np.float64).reshape(-1)
    b = np.asarray(neg, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return None
    k = int(np.floor(float(fpr) * b.size + 1e-12))
    if k >= b.size:                              # 允许全部负类判阳 → 阈值在负类之下
        return 1.0
    t = np.sort(b)[::-1][k]
    return float((a > t).mean())


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


def update_score(c_before, c_i, eps: float = 1e-6):
    """
    一个上传更新的功能分数（原文 §7）：Δ_k = c_k(θ_before) − c_k(θ_i)（越大 = 越被推向类 k），
        s_i = (max_k Δ_k − median_k Δ_k) / (MAD_k Δ_k + eps)，
    k* = argmax_k Δ_k。无定义的类（None）不参与；有定义的类 < 3 个 → (None, None)（MAD 没有意义）。
    返回 (s_i, k*)。
    """
    d = [(k, a - b) for k, (a, b) in enumerate(zip(c_before, c_i)) if a is not None and b is not None]
    if len(d) < 3:
        return None, None
    ks = [k for k, _ in d]
    v = np.array([x for _, x in d], dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    j = int(np.argmax(v))
    return float((v[j] - med) / (mad + eps)), int(ks[j])


def update_argmax_ties(c_before, c_i):
    """update_score 的 k* 的**全部**并列者：Δ_k = c_k(before) − c_k(i) 取到最大值的所有类（c_k 是量化的比例，
    并列常见；np.argmax 只取第一个，会偏向小编号的类 —— 目标类恰好是 0，G1 审查发现）。有定义的类 < 3 个 → []。"""
    d = [(k, a - b) for k, (a, b) in enumerate(zip(c_before, c_i)) if a is not None and b is not None]
    if len(d) < 3:
        return []
    mx = max(x for _, x in d)
    return [k for k, x in d if x == mx]
