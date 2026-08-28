"""
probe/occupation.py  —  Exp 0.3 Metric E/F：benign occupation 与它对 BD 位移的关系
（**纯 numpy，不 import tf**；Spearman 手写，不依赖 scipy —— 容器里有没有 scipy 未核实）。

Metric E：benign occupation
  O^abs_j  = |Δθ^clean_j|                       绝对占据（随收敛自然下降）
  O^rank_j = |Δθ^clean_j| 的全参数百分位 ∈ [0,1]  相对占据（**主用**）
  O^temp_j = |Δθ^clean_j| 超过该轮分位阈值的轮次占比

  用 rank 而不是 abs 的理由（用户 §11 已指出）：绝对量随训练收敛整体下移，
  于是后期**所有**参数都"看起来 dormant"，会造出伪 dormant space。

Metric F：occupation vs BD displacement

  ⚠️ **方法论限制，必须随结论一起写出去**：resnet10 有 ~5M 参数。在几百万个坐标上
  算相关系数，p 值毫无意义（任何微小效应都"显著"），而且 r 由 bulk 主导。更要命的是
  **层身份同时驱动两侧**——不同层的更新尺度差一两个数量级，于是全局 r 极容易只是
  一个 layer-identity 的伪相关。
  所以本模块的**主读数是分箱曲线**（occupation 十分位 → mean |Δθ_BD|），并且提供
  `per_group=` 逐层各算一份。r/ρ 只作附注。
"""
import numpy as np

_EPS = 1e-12


# ── Metric E ────────────────────────────────────────────────────────────────
def occupation_abs(delta_clean) -> np.ndarray:
    """O^abs = |Δθ^clean|。"""
    return np.abs(np.asarray(delta_clean, dtype=np.float64).ravel())


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    平均秩（并列取平均），秩从 1 开始。纯 numpy、**全向量化**，等价 scipy.stats.rankdata。

    并列必须取平均而不是随意打破：掩码投影、ReLU 死单元会造出成片的**精确 0**，
    数量常达总坐标的百分之几十。任意打破并列会让这些坐标在 occupation rank 上
    被人为摊开成一整段，凭空造出"占据度梯度"。
    向量化是必需的：resnet10 ~5M 参数，逐坐标 Python 循环要几秒且每个探测点都要跑。
    """
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    _, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))       # 每个并列组的 0-based 起点
    avg = (2.0 * starts + counts + 1.0) / 2.0                    # 组内 1-based 秩的均值
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[inv.ravel()]
    return ranks


def occupation_rank(delta_clean) -> np.ndarray:
    """
    O^rank ∈ [0,1]：|Δθ^clean| 在全部坐标上的百分位（0 = 最不被占据，1 = 占据最多）。

    并列取平均秩，再归一到 [0,1]。n=1 时返回 [0.5]（无从区分高低）。
    """
    a = occupation_abs(delta_clean)
    n = a.size
    if n == 0:
        return a
    if n == 1:
        return np.array([0.5])
    return (_rankdata_average(a) - 1.0) / (n - 1.0)


def occupation_temporal(deltas_clean, q: float = 0.5) -> np.ndarray:
    """
    O^temp_j = |Δθ^clean_j| 超过**该轮自身**分位阈值 τ_t 的轮次占比 ∈ [0,1]。

    阈值逐轮各取自己的分位数（而不是一个跨轮固定常数）—— 否则整体幅度随收敛下降
    会让后期所有轮都判为"未占据"，O^temp 退化成"训练早期占比"。

    Args:
        deltas_clean: [Δθ^clean(round_1), ...]，每个是同长度的展平向量
        q:            分位阈值，0.5 = 该轮中位数
    """
    arrs = [occupation_abs(d) for d in deltas_clean]
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError("各轮 Δθ^clean 长度必须一致")
    hits = np.zeros(n, dtype=np.float64)
    for a in arrs:
        hits += (a > np.quantile(a, q)).astype(np.float64)
    return hits / len(arrs)


# ── Metric F ────────────────────────────────────────────────────────────────
def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape 不一致: {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    if den < _EPS:
        return float("nan")          # 一侧是常数 → 相关系数无定义（不是 0）
    return float(np.dot(xc, yc) / den)


def spearman(x, y) -> float:
    """秩相关 = 对两侧的平均秩做 Pearson。手写，不依赖 scipy。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape 不一致: {x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    return pearson(_rankdata_average(x), _rankdata_average(y))


def binned_curve(occ_rank, bd_abs, n_bins: int = 10) -> dict:
    """
    **Metric F 的主读数**：把坐标按 occupation rank 等宽分箱，报每箱的 mean/median |Δθ_BD|。

    Figure 5 画这条曲线，不是散点 + 回归线 —— 几百万个点的散点图不可读，
    而单一 r 会把"低占据箱明显更高"这种非单调结构抹平。

    Returns:
        {"bin_edges": [...], "bin_center": [...], "count": [...],
         "bd_mean": [...], "bd_median": [...]}
        空箱的统计量为 nan（不是 0）。
    """
    o = np.asarray(occ_rank, dtype=np.float64).ravel()
    b = np.abs(np.asarray(bd_abs, dtype=np.float64).ravel())
    if o.shape != b.shape:
        raise ValueError(f"shape 不一致: {o.shape} vs {b.shape}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # 右闭到 1.0：rank 恰为 1.0 的坐标应落进最后一箱而不是溢出
    which = np.clip(np.digitize(o, edges[1:-1], right=False), 0, n_bins - 1)
    mean, med, cnt = [], [], []
    for k in range(n_bins):
        sel = b[which == k]
        cnt.append(int(sel.size))
        mean.append(float(sel.mean()) if sel.size else float("nan"))
        med.append(float(np.median(sel)) if sel.size else float("nan"))
    return {
        "bin_edges": edges.tolist(),
        "bin_center": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "count": cnt, "bd_mean": mean, "bd_median": med,
    }


def occupation_vs_bd(occ_rank, bd_abs, n_bins: int = 10, groups: dict = None) -> dict:
    """
    Metric F 汇总：分箱曲线（主）+ Pearson/Spearman（附注）。

    groups 非空时**逐组各算一份**（放在 "per_group" 下）——全局那份混了不同尺度的层，
    单看它容易把 layer-identity 读成 occupation 效应。
    """
    o = np.asarray(occ_rank, dtype=np.float64).ravel()
    b = np.abs(np.asarray(bd_abs, dtype=np.float64).ravel())
    out = {
        "n": int(o.size),
        "pearson": pearson(o, b),
        "spearman": spearman(o, b),
        "curve": binned_curve(o, b, n_bins),
    }
    if groups:
        out["per_group"] = {
            name: {
                "n": int(idx.size),
                "pearson": pearson(o[idx], b[idx]),
                "spearman": spearman(o[idx], b[idx]),
                "curve": binned_curve(o[idx], b[idx], n_bins),
            }
            for name, idx in groups.items()
        }
    return out
