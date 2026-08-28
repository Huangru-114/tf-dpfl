"""
probe/param_metrics.py  —  Exp 0.3 参数空间 Metric A–D（**纯 numpy，不 import tf**）。

全部函数只接受「展平向量」+「分组下标」，于是 global 与 layer-wise 共用一条码路
（分组下标由 probe/flatten.build_groups 产出）。

  Metric A  cos(Δθ_poison, Δθ_clean)   —— **辅助指标**。它主要反映「恶意端的本地更新
                                          是否仍沿正常任务方向」，**不是**后门相似度。
  Metric B  C_BD = cos(Δθ_BD, Δθ_clean) —— 本实验主指标。
              ≈ 1 → poison 诱导的位移与主任务同向
              ≈ 0 → 近似正交
              < 0 → 与主任务存在优化冲突
  Metric C  ConflictRatio = 符号不一致坐标占比
  Metric D  逐层 BD energy 与归一化占比 + top-k% 坐标的能量贡献
"""
import numpy as np

_EPS = 1e-12


# ── Metric A / B ────────────────────────────────────────────────────────────
def cosine(a, b) -> float:
    """两个展平向量的余弦。任一为零向量时返回 nan（而不是 0 —— 0 会被读成"正交"）。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def norm(a) -> float:
    """‖a‖₂。"""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


# ── Metric C ────────────────────────────────────────────────────────────────
def conflict_ratio(a, b, *, drop_zeros: bool = True) -> tuple:
    """
    符号冲突坐标占比：conflict_j = 1 if sign(a_j) != sign(b_j)。

    **为什么默认丢掉零坐标**：np.sign(0)=0，于是「一侧恰为 0」会被计成冲突。
    Neurotoxin 式掩码投影、ReLU 死单元、未被选中的私有 head 都会造出成片的精确 0，
    把它们算进去，ConflictRatio 会被这些与优化冲突无关的坐标顶起来。
    drop_zeros=True 时只在**两侧都非零**的坐标上统计。

    Returns:
        (ratio, n_used) —— n_used=0 时 ratio 为 nan（没有可比坐标，不是"零冲突"）。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")
    if drop_zeros:
        m = (a != 0.0) & (b != 0.0)
        a, b = a[m], b[m]
    n = a.size
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.sign(a) != np.sign(b))), int(n)


# ── Metric D ────────────────────────────────────────────────────────────────
def group_norms(delta, groups: dict) -> dict:
    """{group_name: ‖Δθ_group‖₂}。"""
    d = np.asarray(delta, dtype=np.float64).ravel()
    return {name: float(np.linalg.norm(d[idx])) for name, idx in groups.items()}


def norm_fractions(norms: dict) -> dict:
    """
    R_BD,l = ‖Δθ_BD,l‖₂ / Σ_k ‖Δθ_BD,k‖₂ —— **按用户 §十的字面定义**（范数占比）。

    注意这**不是**能量分解：Σ_l ‖·‖_l ≠ ‖·‖。真正可加的分解见 energy_fractions。
    两个都报，因为「集中在少数层」这个判断在两种口径下可能不同。
    """
    tot = sum(norms.values())
    if tot < _EPS:
        return {k: float("nan") for k in norms}
    return {k: float(v / tot) for k, v in norms.items()}


def energy_fractions(norms: dict) -> dict:
    """
    ‖Δθ_BD,l‖² / Σ_k ‖Δθ_BD,k‖² —— 真正可加的能量分解（Σ_l = 1 精确成立，
    因为分组互不重叠且并集是全体坐标时 Σ_l ‖·‖_l² = ‖·‖²）。
    """
    sq = {k: v * v for k, v in norms.items()}
    tot = sum(sq.values())
    if tot < _EPS:
        return {k: float("nan") for k in sq}
    return {k: float(v / tot) for k, v in sq.items()}


def topk_energy_fraction(delta, ks=(0.01, 0.05, 0.10)) -> dict:
    """
    按 |Δθ| 排序，最大的 k 比例坐标贡献了总能量（Σ Δθ²）的多少。

    用于回答「BD 位移是否集中在少数 **parameter**」（Metric D 的后半段）。
    k 向上取整到至少 1 个坐标。零向量返回 nan。

    Returns: {"top_1%": frac, "top_5%": frac, ...}
    """
    d = np.asarray(delta, dtype=np.float64).ravel()
    sq = d * d
    tot = float(sq.sum())
    n = d.size
    out = {}
    for k in ks:
        label = f"top_{k * 100:g}%"
        if n == 0 or tot < _EPS:
            out[label] = float("nan")
            continue
        m = max(1, int(np.ceil(k * n)))
        # 只需要最大的 m 个，不必全排序
        part = np.partition(sq, n - m)[n - m:]
        out[label] = float(part.sum() / tot)
    return out
