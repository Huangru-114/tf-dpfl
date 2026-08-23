"""
tests/test_drift_metrics.py  —  参数漂移 / 表示漂移数学核的不变量（纯 numpy，本地秒级）。

数学核刻意与 TF 解耦（attack/drift_metrics.py 不 import tf），故这条 L1 本地直接跑，
不必进容器。特征抽取（evaluate_drift）走 TF，另在集群 smoke 里验。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from attack.drift_metrics import param_drift, repr_shift   # noqa: E402


# ── 参数漂移 ────────────────────────────────────────────────────────────────
def test_param_drift_abs_and_rel():
    anchor = [np.array([3.0, 4.0]), np.array([10.0])]   # idx0=backbone, idx1=head
    edge   = [np.array([0.0, 0.0]), np.array([99.0])]
    absd, rel = param_drift(edge, anchor, base_idx=[0])
    assert absd == pytest.approx(5.0)        # ‖[-3,-4]‖ = 5
    assert rel == pytest.approx(1.0)         # 5 / ‖[3,4]‖ = 5/5


def test_param_drift_ignores_head_indices():
    """只在 backbone 索引上算 —— head 发散再大也不该进漂移（陷阱 #9）。"""
    anchor = [np.array([1.0, 0.0]), np.array([0.0])]
    edge   = [np.array([1.0, 0.0]), np.array([1000.0])]  # backbone 不变、head 巨变
    absd, rel = param_drift(edge, anchor, base_idx=[0])
    assert absd == pytest.approx(0.0)
    assert rel == pytest.approx(0.0)


def test_param_drift_zero_when_identical():
    w = [np.array([1.0, 2.0, 3.0]), np.array([5.0])]
    absd, rel = param_drift([a.copy() for a in w], w, base_idx=[0])
    assert absd == pytest.approx(0.0)
    assert rel == pytest.approx(0.0)


# ── 表示漂移 ────────────────────────────────────────────────────────────────
def test_repr_shift_mean_and_median():
    E_anchor = np.array([[1.0, 0.0], [0.0, 1.0]])
    E_edge   = np.array([[1.0, 0.0], [1.0, 0.0]])   # 样本0 同向(d=0)，样本1 正交(d=1)
    mean, med = repr_shift(E_edge, E_anchor)
    assert mean == pytest.approx(0.5)
    assert med == pytest.approx(0.5)


def test_repr_shift_median_robust_to_outlier():
    """median 版对少数极端样本更稳健 —— 3 个 0 + 1 个 1：mean=0.25，median=0。"""
    E_anchor = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    E_edge   = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.]])
    mean, med = repr_shift(E_edge, E_anchor)
    assert mean == pytest.approx(0.25, abs=1e-6)
    assert med == pytest.approx(0.0, abs=1e-6)


def test_repr_shift_zero_when_identical():
    E = np.random.default_rng(0).normal(size=(5, 8))
    mean, med = repr_shift(E, E.copy())
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert med == pytest.approx(0.0, abs=1e-9)
