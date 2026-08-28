"""
tests/test_probe_param_metrics.py  —  Exp 0.3 参数空间 Metric A–D 的不变量
（纯 numpy，本地秒级；probe/param_metrics.py 与 probe/flatten.py 刻意不 import tf）。

断言全部**精确可手算**，不是"看起来差不多"。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from probe.flatten import flatten, flat_offsets, build_groups, delta      # noqa: E402
from probe.param_metrics import (cosine, norm, conflict_ratio, group_norms,  # noqa: E402
                                 norm_fractions, energy_fractions,
                                 topk_energy_fraction)


# ── flatten / 分组 ──────────────────────────────────────────────────────────
def test_flatten_preserves_order():
    w = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([5.0])]
    assert flatten(w).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_flatten_subset_only_takes_given_indices():
    w = [np.array([1.0, 2.0]), np.array([99.0]), np.array([3.0])]
    assert flatten(w, [0, 2]).tolist() == [1.0, 2.0, 3.0]


def test_build_groups_indices_point_into_the_subset_vector():
    """子集展平时，分组下标必须是**子集向量里**的位置，不是全量向量里的位置。"""
    w = [np.array([1.0, 2.0]), np.array([99.0]), np.array([3.0])]
    g = build_groups(w, {"a": [0], "b": [2]}, indices=[0, 2])
    assert g["a"].tolist() == [0, 1]
    assert g["b"].tolist() == [2]           # 不是 [3] —— 张量 1 被排除后 b 前移


def test_build_groups_drops_groups_outside_subset():
    """陷阱 #9 的实现方式：只在 backbone 索引上展平时，head 组直接不出现。"""
    w = [np.array([1.0]), np.array([2.0])]
    g = build_groups(w, {"base": [0], "head": [1]}, indices=[0])
    assert set(g) == {"base"}


def test_delta_matches_manual_subtraction():
    a = [np.array([5.0, 7.0])]
    b = [np.array([1.0, 2.0])]
    assert delta(a, b).tolist() == [4.0, 5.0]


# ── Metric A / B：cosine ────────────────────────────────────────────────────
def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    a, b = [1.0, 2.0, 3.0], [2.0, 1.0, 0.0]
    assert cosine(a, b) == pytest.approx(cosine(a, list(np.array(b) * 1e6)))


def test_cosine_zero_vector_is_nan_not_zero():
    """零向量返回 0 会被读成"正交"——那是个实质结论，不能由退化输入伪造出来。"""
    assert np.isnan(cosine([0.0, 0.0], [1.0, 1.0]))


def test_cosine_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0, 2.0], [1.0])


def test_norm_is_l2():
    assert norm([3.0, 4.0]) == pytest.approx(5.0)


# ── Metric C：conflict ratio ────────────────────────────────────────────────
def test_conflict_ratio_exact_on_known_sign_pattern():
    a = [1.0, -1.0, 1.0, -1.0]
    b = [1.0, 1.0, -1.0, -1.0]          # 位置 1、2 冲突
    r, n = conflict_ratio(a, b)
    assert r == pytest.approx(0.5)
    assert n == 4


def test_conflict_ratio_all_aligned_is_zero():
    r, _ = conflict_ratio([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert r == pytest.approx(0.0)


def test_conflict_ratio_all_opposed_is_one():
    r, _ = conflict_ratio([1.0, 2.0], [-1.0, -2.0])
    assert r == pytest.approx(1.0)


def test_conflict_ratio_drops_zero_coordinates():
    """精确 0（掩码投影 / 死单元）不该被 sign(0)=0 计成冲突。"""
    a = [0.0, 0.0, 1.0, -1.0]
    b = [5.0, -5.0, 1.0, 1.0]
    r, n = conflict_ratio(a, b, drop_zeros=True)
    assert n == 2                        # 只剩后两个坐标
    assert r == pytest.approx(0.5)       # 其中一个冲突


def test_conflict_ratio_keeps_zeros_when_asked():
    a = [0.0, 1.0]
    b = [5.0, 1.0]
    r, n = conflict_ratio(a, b, drop_zeros=False)
    assert n == 2
    assert r == pytest.approx(0.5)       # sign(0) != sign(5) 被计入


def test_conflict_ratio_no_usable_coordinate_is_nan():
    r, n = conflict_ratio([0.0, 0.0], [0.0, 0.0])
    assert n == 0 and np.isnan(r)


# ── Metric D：逐层能量 ──────────────────────────────────────────────────────
def test_group_norms_exact():
    d = np.array([3.0, 4.0, 5.0])
    g = {"a": np.array([0, 1]), "b": np.array([2])}
    n = group_norms(d, g)
    assert n["a"] == pytest.approx(5.0)
    assert n["b"] == pytest.approx(5.0)


def test_norm_fractions_sum_to_one():
    f = norm_fractions({"a": 3.0, "b": 1.0})
    assert f["a"] == pytest.approx(0.75)
    assert sum(f.values()) == pytest.approx(1.0)


def test_energy_fractions_are_the_additive_decomposition():
    """‖·‖² 可加：分组不重叠且并集为全体时，能量占比之和恰为 1。"""
    d = np.array([3.0, 4.0])
    g = {"a": np.array([0]), "b": np.array([1])}
    e = energy_fractions(group_norms(d, g))
    assert e["a"] == pytest.approx(9.0 / 25.0)
    assert e["b"] == pytest.approx(16.0 / 25.0)
    assert sum(e.values()) == pytest.approx(1.0)


def test_norm_and_energy_fractions_disagree_on_concentration():
    """
    两种口径给出不同的"集中度"，所以两个都要报 —— 只报一个会让结论依赖口径选择。
    """
    d = np.array([3.0, 4.0])
    g = {"a": np.array([0]), "b": np.array([1])}
    nf = norm_fractions(group_norms(d, g))
    ef = energy_fractions(group_norms(d, g))
    assert nf["b"] == pytest.approx(4.0 / 7.0)
    assert ef["b"] == pytest.approx(16.0 / 25.0)
    assert nf["b"] != pytest.approx(ef["b"])


def test_all_zero_group_norms_give_nan_fractions():
    f = norm_fractions({"a": 0.0, "b": 0.0})
    assert all(np.isnan(v) for v in f.values())


# ── Metric D：top-k% ────────────────────────────────────────────────────────
def test_topk_energy_fraction_on_exactly_sparse_vector():
    """100 个坐标，能量全在 1 个上 → top1% 必须精确等于 1.0。"""
    d = np.zeros(100)
    d[7] = 10.0
    f = topk_energy_fraction(d, ks=(0.01, 0.05))
    assert f["top_1%"] == pytest.approx(1.0)
    assert f["top_5%"] == pytest.approx(1.0)


def test_topk_energy_fraction_uniform_vector_is_proportional():
    """全同幅度 → top-k% 恰好占 k 的能量。"""
    d = np.ones(100)
    f = topk_energy_fraction(d, ks=(0.01, 0.10))
    assert f["top_1%"] == pytest.approx(0.01)
    assert f["top_10%"] == pytest.approx(0.10)


def test_topk_energy_fraction_uses_magnitude_not_sign():
    d = np.zeros(100)
    d[3] = -10.0
    assert topk_energy_fraction(d, ks=(0.01,))["top_1%"] == pytest.approx(1.0)


def test_topk_energy_fraction_zero_vector_is_nan():
    f = topk_energy_fraction(np.zeros(100), ks=(0.05,))
    assert np.isnan(f["top_5%"])


def test_topk_takes_at_least_one_coordinate():
    """k·n < 1 时向上取整到 1，不能取 0 个（会得到 0/tot=0，读成"完全不集中"）。"""
    d = np.zeros(10)
    d[0] = 1.0
    assert topk_energy_fraction(d, ks=(0.01,))["top_1%"] == pytest.approx(1.0)
