"""
`fedavg/analysis/bn_cluster.py`：CCS 聚类部分的离线检测（P0 附加读数）。合成的 BN 统计量向量上的解析行为。
需要 scikit-learn ≥ 1.3（集群容器有；本地没有时 skip）。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from analysis import bn_cluster as BC                        # noqa: E402


def test_cosine_distance_matrix_hand_values():
    D = BC.cosine_distance_matrix([[1, 0], [0, 1], [2, 0]])
    np.testing.assert_allclose(D, [[0, 1, 0], [1, 0, 1], [0, 1, 0]], atol=1e-12)


def test_rates_and_degenerate_cases():
    r = BC.rates([3, 4], malicious={3, 9}, population=[1, 2, 3, 4])
    assert r == {"n": 4, "n_malicious": 1, "tpr": 1.0, "fpr": pytest.approx(1 / 3)}     # 良性 = {1, 2, 4}
    assert BC.rates([], malicious=set(), population=[1])["tpr"] is None
    d = BC.detect({1: np.ones(3), 2: np.ones(3)})               # n=2：min_cluster_size 3 > n
    assert d["status"] == "degenerate" and d["rejected"] == []
    assert BC.detect({})["status"] == "empty"


def test_hdbscan_rejects_the_odd_ones_out():
    pytest.importorskip("sklearn")
    import sklearn
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 3):
        pytest.skip("sklearn.cluster.HDBSCAN 需要 ≥ 1.3")
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(size=64)) + 1.0
    vec = {i: base + 0.01 * rng.normal(size=64) for i in range(20)}
    for i in (20, 21, 22):                                      # 方向明显不同的三端
        v = base.copy()
        v[:32] *= 4.0
        vec[i] = v + 0.01 * rng.normal(size=64)
    d = BC.detect(vec)
    assert d["status"] == "ok" and d["min_cluster_size"] == 23 // 2 + 2
    assert {20, 21, 22} <= set(d["rejected"])                  # 离群的三端一定被剔除
    r = BC.rates(d["rejected"], malicious={20, 21, 22}, population=list(vec))
    assert r["tpr"] == 1.0
    # 官方参数（allow_single_cluster + min_samples=1）下，HDBSCAN 也会把紧凑良性群里的一部分标成噪声 →
    # 良性端被误剔（这组合成数据上 sklearn 1.9 是 7 / 20；精确值随 sklearn 版本可能变，只钉范围）。
    # 这正是 P0 要报 FPR 的原因，不是实现错误。
    assert 0.0 < r["fpr"] < 0.5


def test_stat_vector_concatenates_in_order():
    v = BC.stat_vector([np.array([1.0, 2.0]), np.array([[3.0]])])
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


class _FakeHDBSCAN:
    """替身：直接返回给定的标签（测「保留最大簇 / 全噪声」的规则本身，与 sklearn 版本无关）。"""
    labels = None

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, D):
        self.labels_ = np.asarray(type(self).labels)
        return self


def _with_labels(monkeypatch, labels):
    sk = pytest.importorskip("sklearn.cluster")
    _FakeHDBSCAN.labels = labels
    monkeypatch.setattr(sk, "HDBSCAN", _FakeHDBSCAN)
    return {i: np.ones(3) + i for i in range(len(labels))}


def test_keeps_the_largest_cluster_ties_to_the_smaller_label(monkeypatch):
    seen = []
    orig_init = _FakeHDBSCAN.__init__
    monkeypatch.setattr(_FakeHDBSCAN, "__init__", lambda self, **kw: (seen.append(kw), orig_init(self, **kw))[1])
    d = BC.detect(_with_labels(monkeypatch, [1, 1, 1, 0, 0, -1]))
    assert d["status"] == "ok" and d["admitted"] == [0, 1, 2] and d["rejected"] == [3, 4, 5]
    assert seen == [{"min_cluster_size": 6 // 2 + 2, "min_samples": 1, "metric": "precomputed",
                     "allow_single_cluster": True, "copy": True}]          # CCS 官方参数
    d = BC.detect(_with_labels(monkeypatch, [1, 1, 0, 0, -1]))      # 并列 → 编号小的簇（0）
    assert d["admitted"] == [2, 3] and d["rejected"] == [0, 1, 4]


def test_all_noise_rejects_nobody(monkeypatch):
    d = BC.detect(_with_labels(monkeypatch, [-1, -1, -1, -1]))
    assert d["status"] == "all_noise" and d["rejected"] == [] and d["admitted"] == [0, 1, 2, 3]
