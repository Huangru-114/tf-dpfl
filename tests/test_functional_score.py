"""
tests/test_functional_score.py  —  S6a：c_k 的纯 numpy 部分（D-085；手算值，本地秒级）
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from analysis import functional_score as FS


def test_prototypes_are_class_means_and_empty_classes_are_nan():
    f = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    p, n = FS.class_prototypes(f, [0, 0, 2], 3)
    np.testing.assert_array_equal(p[0], [1.0, 0.0])
    np.testing.assert_array_equal(p[2], [10.0, 10.0])
    assert np.all(np.isnan(p[1])) and n.tolist() == [2, 0, 1]


def test_ncm_logits_are_negative_squared_distance_and_skip_missing_prototypes():
    p = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    z = FS.ncm_logits(np.array([[0.0, 0.0], [3.0, 4.0]]), p)
    np.testing.assert_array_equal(z[:, 0], [0.0, -25.0])
    np.testing.assert_array_equal(z[:, 2], [-25.0, 0.0])
    assert np.all(np.isneginf(z[:, 1]))
    assert FS.ncm_predict(np.array([[0.1, 0.0], [2.9, 4.0]]), p).tolist() == [0, 2]


def test_ck_counts_only_samples_whose_true_class_is_not_k():
    y = [0, 0, 1, 2]
    reached = np.array([[True, True, False],     # y=0：推向 0 不算；推向 1 命中；推向 2 没命中
                        [True, False, False],
                        [False, True, True],     # y=1
                        [True, True, False]])    # y=2
    c, n = FS.ck_from_reached(y, reached)
    assert n == [2, 3, 3]
    assert c[0] == pytest.approx(1 - 1 / 2)      # y≠0 的样本 [2,3]：命中 [F, T] → 1/2
    assert c[1] == pytest.approx(1 - 2 / 3)      # y≠1 的样本 [0,1,3]：命中 [T, F, T]
    assert c[2] == pytest.approx(1 - 1 / 3)      # y≠2 的样本 [0,1,2]：命中 [F, F, T]


def test_ck_is_none_when_nothing_to_push():
    c, n = FS.ck_from_reached([1, 1], np.array([[True, True], [False, True]]))
    assert c[1] is None and n[1] == 0 and c[0] == pytest.approx(0.5)


def test_auroc_hand_values_and_ties():
    assert FS.auroc([3, 4], [1, 2]) == 1.0
    assert FS.auroc([1, 2], [3, 4]) == 0.0
    assert FS.auroc([1, 2], [1, 2]) == 0.5                      # 全平局
    assert FS.auroc([2, 3], [1, 3]) == pytest.approx(0.625)     # 4 对：2>1 ✓、2<3 ✗、3>1 ✓、3=3 ½ → 2.5/4
    assert FS.auroc([], [1]) is None and FS.auroc([1], []) is None


def test_auroc_matches_the_brute_force_definition():
    r = np.random.default_rng(0)
    a, b = r.integers(0, 6, 40), r.integers(0, 6, 50)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert FS.auroc(a, b) == pytest.approx(brute)


def test_edge_contrast():
    c = {0: [0.2, 0.5], 1: [0.6, 0.9], 2: [0.8, None], 3: [0.7, 0.7]}
    r = FS.edge_contrast(c, ref_edge=0, k=0)
    assert r["ref"] == 0.2 and r["others_mean"] == pytest.approx(0.7) and r["diff"] == pytest.approx(0.5)
    r = FS.edge_contrast(c, ref_edge=0, k=1)
    assert r["others_mean"] == pytest.approx(0.8)               # None 不参与
    assert FS.edge_contrast({0: [None]}, 0, 0)["diff"] is None


def test_module_does_not_import_tf():
    tree = ast.parse(Path(FS.__file__).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            assert not any(m.split(".")[0] in ("tensorflow", "keras") for m in mods)


def test_update_score_hand_values():
    # Δ = c_before − c_i = [0, 0, 0.4, 0.1, −0.1]：median 0，MAD = median(|Δ|) = 0.1，max = 0.4 → s = 4
    s_, k = FS.update_score([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.1, 0.4, 0.6])
    assert k == 2 and s_ == pytest.approx(0.4 / (0.1 + 1e-6))
    # 无定义的类（None）不参与：只剩 3 个有定义的类仍可算
    s2, k2 = FS.update_score([0.5, None, 0.5, 0.5], [0.5, 0.0, 0.1, 0.5])
    assert k2 == 2 and s2 is not None


def test_update_score_needs_three_defined_classes():
    assert FS.update_score([0.5, None, 0.5], [0.1, 0.1, None]) == (None, None)
    assert FS.update_score([], []) == (None, None)
