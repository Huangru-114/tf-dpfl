"""
tests/test_probe_occupation.py  —  Exp 0.3 Metric E/F 的不变量（纯 numpy，本地秒级）。

重点守两件事：
  1. 并列（成片的精确 0）必须取平均秩 —— 否则会凭空造出"占据度梯度"。
  2. Metric F 的主读数是**分箱曲线**，它必须能表达 r/ρ 表达不了的非单调结构。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from probe.occupation import (_rankdata_average, occupation_abs,          # noqa: E402
                              occupation_rank, occupation_temporal,
                              pearson, spearman, binned_curve,
                              occupation_vs_bd)


# ── 秩 ──────────────────────────────────────────────────────────────────────
def test_rankdata_average_breaks_ties_by_averaging():
    assert _rankdata_average(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_rankdata_average_handles_a_block_of_zeros():
    """掩码投影 / 死单元会造出成片精确 0，它们必须拿到同一个秩。"""
    r = _rankdata_average(np.array([0.0, 0.0, 0.0, 5.0]))
    assert r.tolist() == [2.0, 2.0, 2.0, 4.0]


def test_rankdata_average_matches_scipy_reference_values():
    """与 scipy.stats.rankdata 的已知输出对拍（硬编码，不依赖 scipy 存在）。"""
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    assert _rankdata_average(x).tolist() == [4.0, 1.5, 5.0, 1.5, 6.0, 8.0, 3.0, 7.0]


def test_rankdata_average_empty():
    assert _rankdata_average(np.zeros(0)).size == 0


# ── Metric E ────────────────────────────────────────────────────────────────
def test_occupation_abs_takes_magnitude():
    assert occupation_abs([1.0, -2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]


def test_occupation_rank_is_zero_to_one_percentile():
    assert occupation_rank([1.0, -2.0, 3.0]).tolist() == [0.0, 0.5, 1.0]


def test_occupation_rank_ignores_sign():
    assert occupation_rank([-3.0, 1.0]).tolist() == occupation_rank([3.0, 1.0]).tolist()


def test_occupation_rank_all_tied_is_flat_at_half():
    """全同占据 → 秩必须全为 0.5，不能被并列打破成 0..1 的假梯度。"""
    assert occupation_rank([2.0, 2.0, 2.0]).tolist() == [0.5, 0.5, 0.5]


def test_occupation_rank_single_coordinate_is_half():
    assert occupation_rank([7.0]).tolist() == [0.5]


def test_occupation_rank_is_invariant_to_global_rescaling():
    """相对占据必须对整体幅度收缩免疫 —— 这正是不用 O^abs 的理由。"""
    a = np.array([1.0, 5.0, 2.0, 9.0])
    assert occupation_rank(a).tolist() == occupation_rank(a * 1e-4).tolist()


def test_occupation_temporal_counts_rounds_above_that_rounds_own_threshold():
    """两轮里排名互换的坐标 → 各自恰好一半轮次在阈值之上。"""
    r1 = np.array([1.0, 2.0, 3.0, 4.0])
    r2 = np.array([4.0, 3.0, 2.0, 1.0])
    assert occupation_temporal([r1, r2], q=0.5).tolist() == [0.5, 0.5, 0.5, 0.5]


def test_occupation_temporal_persistently_occupied_coordinate_is_one():
    r1 = np.array([0.0, 0.0, 9.0])
    r2 = np.array([0.0, 0.0, 9.0])
    assert occupation_temporal([r1, r2], q=0.5).tolist() == [0.0, 0.0, 1.0]


def test_occupation_temporal_threshold_is_per_round_not_global():
    """
    第二轮整体缩小 1000 倍（模拟收敛）。若阈值是跨轮固定的，第二轮会全判"未占据"，
    该坐标的 O^temp 会掉到 0.5；per-round 阈值下它必须仍是 1.0。
    """
    r1 = np.array([0.0, 0.0, 9.0])
    r2 = r1 * 1e-3
    assert occupation_temporal([r1, r2], q=0.5).tolist() == [0.0, 0.0, 1.0]


def test_occupation_temporal_rejects_ragged_rounds():
    with pytest.raises(ValueError):
        occupation_temporal([np.zeros(3), np.zeros(4)])


# ── 相关系数 ────────────────────────────────────────────────────────────────
def test_pearson_perfect_linear():
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert pearson([1.0, 2.0, 3.0], [-2.0, -4.0, -6.0]) == pytest.approx(-1.0)


def test_pearson_constant_side_is_nan_not_zero():
    """一侧是常数时相关系数无定义。返回 0 会被读成"确认无相关"这个实质结论。"""
    assert np.isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_spearman_is_one_for_any_monotone_map():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)


def test_spearman_minus_one_for_reversed_order():
    assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_differs_from_pearson_on_nonlinear_monotone():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x ** 4
    assert spearman(x, y) == pytest.approx(1.0)
    assert pearson(x, y) < 0.95


# ── Metric F 主读数：分箱曲线 ───────────────────────────────────────────────
def test_binned_curve_assigns_rank_one_to_the_last_bin():
    """rank 恰为 1.0 的坐标必须落进最后一箱，不能溢出成一个空箱之外的类别。"""
    c = binned_curve([0.0, 1.0], [5.0, 7.0], n_bins=2)
    assert c["count"] == [1, 1]
    assert c["bd_mean"] == [5.0, 7.0]


def test_binned_curve_empty_bin_is_nan_not_zero():
    """空箱报 0 会在图上画成"该占据区间 BD 位移为零"，那是个假结论。"""
    c = binned_curve([0.0, 0.05], [1.0, 1.0], n_bins=4)
    assert c["count"] == [2, 0, 0, 0]
    assert c["bd_mean"][0] == pytest.approx(1.0)
    assert all(np.isnan(v) for v in c["bd_mean"][1:])


def test_binned_curve_uses_magnitude_of_bd():
    c = binned_curve([0.0, 1.0], [-5.0, 7.0], n_bins=2)
    assert c["bd_mean"] == [5.0, 7.0]


def test_binned_curve_captures_nonmonotone_structure_that_correlation_misses():
    """
    U 形：最低占据箱与最高占据箱的 BD 位移都高、中间低。
    Spearman ≈ 0（这正是"只报 r 会漏掉结构"的实例），但曲线必须显示出两端高。
    """
    occ = np.linspace(0.0, 1.0, 1001)
    bd = (occ - 0.5) ** 2
    assert abs(spearman(occ, bd)) < 0.05
    c = binned_curve(occ, bd, n_bins=10)
    assert c["bd_mean"][0] > c["bd_mean"][4]
    assert c["bd_mean"][-1] > c["bd_mean"][5]


def test_binned_curve_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        binned_curve([0.0, 1.0], [1.0])


# ── Metric F 汇总 ───────────────────────────────────────────────────────────
def test_occupation_vs_bd_detects_dormant_capacity_signature():
    """低占据 ↔ 高 BD 位移（Case A 的签名）→ ρ 必须显著为负。"""
    occ = np.linspace(0.0, 1.0, 500)
    bd = 1.0 - occ
    out = occupation_vs_bd(occ, bd)
    assert out["spearman"] == pytest.approx(-1.0)
    assert out["curve"]["bd_mean"][0] > out["curve"]["bd_mean"][-1]
    assert out["n"] == 500


def test_occupation_vs_bd_per_group_can_contradict_the_global_number():
    """
    方法论守卫：两层，组内 occupation 与 BD 位移**均为正相关**，但层间尺度差异
    让全局相关变负。若只报全局 r，就会把 layer-identity 读成 dormant-capacity 证据。
    """
    occ = np.concatenate([np.linspace(0.0, 0.4, 100),      # 低占据层
                          np.linspace(0.6, 1.0, 100)])     # 高占据层
    bd = np.concatenate([10.0 + np.linspace(0.0, 1.0, 100),
                         0.0 + np.linspace(0.0, 1.0, 100)])
    groups = {"low_layer": np.arange(0, 100), "high_layer": np.arange(100, 200)}
    out = occupation_vs_bd(occ, bd, groups=groups)
    assert out["spearman"] < -0.5                                  # 全局：负
    assert out["per_group"]["low_layer"]["spearman"] == pytest.approx(1.0)
    assert out["per_group"]["high_layer"]["spearman"] == pytest.approx(1.0)


def test_occupation_vs_bd_reports_both_correlations():
    out = occupation_vs_bd(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    assert set(out) >= {"n", "pearson", "spearman", "curve"}
