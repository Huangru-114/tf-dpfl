"""
tests/test_flame.py  —  FLAME (Nguyen et al., USENIX Security'22) 算法不变量

文献定义的三步（对更新 e_i = W_i − G_{t−1} 运算）：
  1. **聚类过滤**：对两两**余弦距离**跑 HDBSCAN
     （min_cluster_size = m//2 + 1, min_samples = 1, allow_single_cluster=True），
     取最大簇为 admitted，其余判为 outlier 丢弃。
  2. **自适应裁剪**：S_t = median_i(‖e_i‖₂)，每个 admitted 更新按 γ_i = min(1, S_t/‖e_i‖) 缩放。
  3. **加噪**：对聚合结果加 N(0, (λ·S_t)²)。
  聚合是 admitted 上的**无权平均**（1/L·Σ），**不按样本数加权**。

下列断言直接对着文献写。当前实现用 "majority-cosine 近似" 代替 HDBSCAN、
且按 n_samples 加权，因此会失败——这就是 CLAUDE.md 陷阱 #3 的数值证据。

纯 numpy，本地秒级可跑：  pytest tests/test_flame.py -v
"""

import numpy as np
import pytest

from defense.base_defense import flatten_weights
from defense.flame import FlameDefense


# ── 已知 bug（CLAUDE.md 陷阱 #3）：现实现是 majority-cosine 近似 + 样本加权，
#    非 USENIX'22 方案。标 xfail(strict=True) 的用意：L1 整体保持绿色，
#    这样它能当「有没有引入回归」的闸门；而一旦 FLAME 被改对，这些会 XPASS
#    → 套件变红 → 强制回来摘掉标记。规格本身始终是可执行的。
KNOWN_BUG = pytest.mark.xfail(strict=True, reason="陷阱 #3：FLAME 未按文献实现")


# ── 构造工具：把一维更新向量包成仓库约定的 client_updates 格式 ──────────────
REF_SHAPES = [(8, 8), (4,)]
DIM = 8 * 8 + 4


def _ref_weights():
    return [np.zeros(s, dtype=np.float32) for s in REF_SHAPES]


def _pack(updates, n_samples):
    """updates: (m, DIM) 的更新增量；返回 [(weights, n, loss, t), ...]（ref=0 故 W_i = e_i）。"""
    ref = _ref_weights()
    out = []
    for upd, n in zip(updates, n_samples):
        layers, off = [], 0
        for w in ref:
            layers.append(upd[off:off + w.size].reshape(w.shape).astype(np.float32))
            off += w.size
        out.append((layers, int(n), 0.0, 0.0))
    return out


def _unit(v):
    return v / np.linalg.norm(v)


def _benign_cluster(rng, n, direction, spread=0.05, scales=None):
    """围绕 direction 的紧簇：方向一致、范数可不同。"""
    scales = np.ones(n) if scales is None else np.asarray(scales, float)
    out = []
    for i in range(n):
        v = _unit(direction + spread * rng.normal(size=DIM) / np.sqrt(DIM))
        out.append(v * scales[i])
    return np.stack(out)


def _run(updates, n_samples, noise_scale=0.0):
    """跑一次 FLAME 聚合，返回 (聚合后的更新增量, defense 实例)。"""
    d = FlameDefense({"noise_scale": noise_scale})
    ref = _ref_weights()
    agg = d.aggregate(_pack(updates, n_samples), ref)
    return flatten_weights(agg) - flatten_weights(ref), d


def _admitted(defense):
    """取 admitted 索引集合。实现必须暴露 last_admitted（L2 的 admitted_count 也要用）。"""
    idx = getattr(defense, "last_admitted", None)
    if idx is None:
        pytest.fail("FlameDefense 未暴露 last_admitted —— 无法验证聚类过滤，也无法给 L2 "
                    "metrics.json 提供 admitted_count。")
    return set(int(i) for i in idx)


# ══════════════════════════════════════════════════════════════════════════
# 不变量 1：全良性输入 → 必须接纳全部
# ══════════════════════════════════════════════════════════════════════════
@KNOWN_BUG
def test_all_benign_admits_everyone(rng):
    """
    没有攻击者时，防御不该造成任何损失。这是最基本的「不误伤」要求：
    更新彼此接近时，接纳与否绝不能由数值噪声决定。
    """
    m = 20
    u = _unit(rng.normal(size=DIM))
    updates = _benign_cluster(rng, m, u)

    _, d = _run(updates, n_samples=[100] * m)

    assert _admitted(d) == set(range(m)), (
        f"全良性 {m} 个更新，只接纳了 {len(_admitted(d))} 个 —— 防御在无攻击时误伤良性客户端。")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 2：方向异常的更新必须被**精确**剔除
# ══════════════════════════════════════════════════════════════════════════
@KNOWN_BUG
def test_rejects_opposite_direction_attackers(rng):
    """16 个良性（方向 +u） + 4 个攻击者（方向 −u）→ 必须恰好接纳那 16 个。"""
    n_benign, n_attack = 16, 4
    u = _unit(rng.normal(size=DIM))
    benign = _benign_cluster(rng, n_benign, u)
    attack = _benign_cluster(rng, n_attack, -u)
    updates = np.concatenate([benign, attack])
    n_samples = [100] * (n_benign + n_attack)

    _, d = _run(updates, n_samples)

    assert _admitted(d) == set(range(n_benign)), (
        f"期望接纳良性 0..{n_benign - 1}，实际接纳 {sorted(_admitted(d))}")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 3：裁剪 —— 聚合结果范数不得超过 median 范数 S_t
# ══════════════════════════════════════════════════════════════════════════
def test_clipping_bounds_aggregate_norm(rng):
    """
    每个 admitted 更新被裁到 ‖·‖ ≤ S_t，凸组合仍 ≤ S_t。
    构造一个范数极大的「放大攻击」更新，方向与良性一致（骗过聚类），
    靠裁剪把它压住。
    """
    m = 20
    u = _unit(rng.normal(size=DIM))
    scales = np.ones(m)
    scales[-1] = 50.0                       # 放大攻击：方向对，范数 50 倍
    updates = _benign_cluster(rng, m, u, scales=scales)
    norms = np.linalg.norm(updates, axis=1)
    s_t = float(np.median(norms))

    agg, _ = _run(updates, n_samples=[100] * m, noise_scale=0.0)

    assert np.linalg.norm(agg) <= s_t * (1 + 1e-6), (
        f"聚合范数 {np.linalg.norm(agg):.4f} 超过 median 范数 S_t={s_t:.4f} —— 裁剪失效，"
        f"放大攻击可穿透。")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 4：文献是**无权平均**，不是样本数加权
# ══════════════════════════════════════════════════════════════════════════
@KNOWN_BUG
def test_aggregation_is_unweighted_over_admitted(rng):
    """
    FLAME 的动机是限制单个客户端的影响力；按 n_samples 加权会让「谎报样本数」
    成为免费的放大攻击通道，与防御目的直接冲突。
    这里所有更新范数相同 → 裁剪是恒等，聚合应恰好等于**算术平均**。
    """
    m = 9
    u = _unit(rng.normal(size=DIM))
    updates = _benign_cluster(rng, m, u, spread=0.30)   # 同簇但方向有可分辨的差异
    updates = updates / np.linalg.norm(updates, axis=1, keepdims=True)  # 范数全相同 → 不触发裁剪

    n_samples = [1] * (m - 1) + [1000]                  # 最后一个谎报 1000 倍样本数
    agg, d = _run(updates, n_samples, noise_scale=0.0)

    assert _admitted(d) == set(range(m)), "构造为单一良性簇，应全部接纳"

    unweighted = updates.mean(axis=0)
    weighted = np.average(updates, axis=0, weights=np.asarray(n_samples, float))

    err_unw = np.linalg.norm(agg - unweighted)
    err_wgt = np.linalg.norm(agg - weighted)
    assert err_unw < 1e-8, (
        f"聚合结果偏离无权平均 {err_unw:.3e}（偏离样本加权平均 {err_wgt:.3e}）—— "
        f"实现按 n_samples 加权，与文献不符。")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 5：noise_scale=0 时聚合必须完全确定
# ══════════════════════════════════════════════════════════════════════════
def test_zero_noise_is_deterministic(rng):
    """λ=0 时两次调用必须逐位相同——保证 smoke run 可复现。"""
    m = 12
    u = _unit(rng.normal(size=DIM))
    updates = _benign_cluster(rng, m, u)
    a1, _ = _run(updates, [100] * m, noise_scale=0.0)
    a2, _ = _run(updates, [100] * m, noise_scale=0.0)
    assert np.array_equal(a1, a2), "noise_scale=0 时聚合仍不确定"
