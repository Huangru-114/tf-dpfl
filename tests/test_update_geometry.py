"""
tests/test_update_geometry.py  —  S6a ①：逐更新几何的纯算术（D-085；不 import TF，本地秒级）

断言都是手算值：
  · Δ = w[base] − edge_w[base]：只取 base 索引、不改入参；
  · 范数、留一余弦（对本 edge / 对全体 edge）的解析值；只有一个更新 → None（不是 0）；
  · CountSketch：对输入线性（精确）、1-稀疏向量范数保持（精确）、内积无偏（固定种子的统计断言）；
  · 哈希 / 符号只来自常量种子的独立 Generator —— 不碰全局 np.random / random（运行时 + AST）。
"""

import ast
import random
from pathlib import Path

import numpy as np
import pytest

from aggregation.client_update import ClientUpdate
from server import update_geometry as UG
from utils.kvline import collect_kv, parse_list

SRC = Path(UG.__file__)


def test_body_delta_takes_only_base_indices_and_does_not_mutate():
    w = [np.array([1.0, 2.0]), np.array([[5.0]]), np.array([9.0, 9.0])]
    edge = [np.array([0.5, 0.5]), np.array([[1.0]]), np.array([0.0, 0.0])]
    w0 = [a.copy() for a in w]
    e0 = [a.copy() for a in edge]
    d = UG.body_delta(w, edge, [0, 1])               # 索引 2 是 head，不进
    np.testing.assert_array_equal(d, np.array([0.5, 1.5, 4.0], np.float32))
    assert d.dtype == np.float32
    for a, b in zip(w + edge, w0 + e0):
        np.testing.assert_array_equal(a, b)


def test_leave_one_out_cosines_match_hand_values():
    g = UG.geometry({0: [np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)],
                     1: [np.array([1.0, 1.0], np.float32)]})
    r2 = np.sqrt(2.0)
    assert g[0]["norm"] == pytest.approx([1.0, 1.0])
    assert g[1]["norm"] == pytest.approx([r2])
    # edge 0：Δ0 与 Δ1 正交
    assert g[0]["cos_edge"] == pytest.approx([0.0, 0.0], abs=1e-12)
    # edge 1 只有一个更新：无「其他」→ None（陷阱 #13）
    assert g[1]["cos_edge"] == [None]
    # 全局：Δ0 vs (Δ1+Δ2)=[1,2]；Δ1 vs [2,1]；Δ2 vs [1,1]
    assert g[0]["cos_global"] == pytest.approx([1 / np.sqrt(5), 1 / np.sqrt(5)])
    assert g[1]["cos_global"] == pytest.approx([1.0])


def test_single_update_overall_has_no_cosine():
    g = UG.geometry({0: [np.array([3.0, 4.0], np.float32)]})
    assert g[0]["norm"] == pytest.approx([5.0])
    assert g[0]["cos_edge"] == [None] and g[0]["cos_global"] == [None]


def test_zero_update_or_zero_rest_gives_none_not_zero():
    z = np.zeros(3, np.float32)
    a = np.array([1.0, 0.0, 0.0], np.float32)
    g = UG.geometry({0: [z, a]})
    assert g[0]["cos_edge"][0] is None                 # Δ 为零向量
    assert g[0]["cos_edge"][1] is None                 # 其余之和是零向量
    # 一对相反的更新：其余之和 = −Δ → 余弦 −1
    g = UG.geometry({0: [a, -a, np.array([0, 1.0, 0], np.float32)]})
    assert g[0]["cos_edge"][0] == pytest.approx(-1 / np.sqrt(2))      # a vs (−a + e2)
    g2 = UG.geometry({0: [a, -a]})
    assert g2[0]["cos_edge"] == pytest.approx([-1.0, -1.0])


def test_countsketch_is_linear_and_preserves_one_sparse_norm():
    cs = UG.CountSketch(40, 16)
    r = np.random.default_rng(0)
    a, b = r.normal(size=40), r.normal(size=40)
    np.testing.assert_allclose(cs.apply(a + 2 * b), cs.apply(a) + 2 * cs.apply(b), atol=1e-5)
    e = np.zeros(40)
    e[7] = 3.0
    assert float(np.linalg.norm(cs.apply(e))) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        cs.apply(np.zeros(41))


def test_countsketch_inner_product_is_unbiased():
    r = np.random.default_rng(1)
    a, b = r.normal(size=60), r.normal(size=60)
    est = np.array([float(np.dot(UG.CountSketch(60, 16, seed=s).apply(a),
                                 UG.CountSketch(60, 16, seed=s).apply(b)))
                    for s in range(2000)])
    se = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - float(np.dot(a, b))) < 4 * se


def test_sketch_is_deterministic_and_does_not_touch_global_rngs():
    np.random.seed(123)
    random.seed(456)
    st_np, st_py = np.random.get_state()[1].copy(), random.getstate()
    x = np.arange(30, dtype=np.float32)
    s1 = UG.CountSketch(30, 8).apply(x)
    s2 = UG.CountSketch(30, 8).apply(x)
    np.testing.assert_array_equal(s1, s2)
    UG.geometry({0: [x, x[::-1].copy()]})
    np.testing.assert_array_equal(np.random.get_state()[1], st_np)
    assert random.getstate() == st_py


def test_module_has_no_tf_and_no_global_rng_calls():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            assert not any(m.split(".")[0] in ("tensorflow", "keras") for m in mods)
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
                and n.value.attr == "random" and getattr(n.value.value, "id", "") == "np"):
            assert n.attr == "default_rng", f"全局 np.random.{n.attr}"
        if isinstance(n, ast.Name) and n.id == "random":
            raise AssertionError("不应使用 Python random")


def _upd(cid, w):
    return ClientUpdate([np.asarray(w, np.float32), np.array([0.0], np.float32)], 10, 0.1, 0.0,
                        client_id=cid)


def test_observer_flush_prints_lines_and_clears_the_buffer():
    obs = UG.UpdateGeometry(malicious_ids={1}, sketch_dim=8)
    edge_w = [np.zeros(4, np.float32), np.zeros(1, np.float32)]
    obs.observe(0, 2, [_upd(0, [1, 0, 0, 0]), _upd(1, [0, 1, 0, 0])], edge_w, [0])
    obs.observe(1, 2, [_upd(5, [1, 1, 0, 0])], edge_w, [0])
    lines = obs.flush(3, 2)
    rows = collect_kv(lines, "[UpdateGeo]")
    assert [(d["round"], d["edge_id"], d["edge_round"]) for d in rows] == [(3, 0, 2), (3, 1, 2)]
    assert parse_list(rows[0]["cid"]) == [0, 1] and parse_list(rows[0]["mal"]) == [0, 1]
    assert parse_list(rows[0]["norm"]) == [1.0, 1.0]
    assert parse_list(rows[1]["cos_edge"]) == [None]
    assert obs.flush(3, 3) == []                       # 缓冲已清
    arr = obs.take_round_arrays()
    assert arr["sketch"].shape == (3, 8) and arr["sketch"].dtype == np.float16
    assert arr["client_id"].tolist() == [0, 1, 5] and arr["malicious"].tolist() == [False, True, False]
    assert arr["edge_round"].tolist() == [2, 2, 2]
    assert obs.take_round_arrays() is None


# ══════════════════════════════════════════════════════════════════════════
# S6b（D-087）：把 body Δ 拆成「可训练权重」与「BN 统计量」两部分
# ══════════════════════════════════════════════════════════════════════════
def test_stat_mask_marks_exactly_the_stat_tensors_in_base_order():
    ew = [np.zeros((2, 2)), np.zeros(3), np.zeros(1), np.zeros(2)]
    m = UG.stat_mask(ew, [0, 1, 3], [1])                 # 索引 2 不在 base；索引 1 是统计量
    assert m.tolist() == [False] * 4 + [True] * 3 + [False] * 2
    assert not UG.stat_mask(ew, [0, 1, 3], []).any()


def test_split_columns_are_hand_values_and_old_columns_are_unchanged():
    # base = {0: 权重 2 个坐标, 1: BN 统计量 2 个坐标}；Δ = [权重 | 统计量]
    edge_w = [np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(1, np.float32)]

    def upd(cid, w, st):
        return ClientUpdate([np.asarray(w, np.float32), np.asarray(st, np.float32),
                             np.zeros(1, np.float32)], 5, 0.1, 0.0, client_id=cid)
    obs = UG.UpdateGeometry(malicious_ids={1}, sketch_dim=8, stat_idx=[1])
    obs.observe(0, 1, [upd(0, [3, 0], [0, 4]), upd(1, [0, 1], [0, 0])], edge_w, [0, 1])
    obs.observe(1, 1, [upd(2, [1, 0], [0, 0])], edge_w, [0, 1])
    rows = collect_kv(obs.flush(1, 1), "[UpdateGeo]")
    r0 = rows[0]
    assert parse_list(r0["norm"]) == [5.0, 1.0]                      # 旧列：整个 body（3-4-5）
    assert parse_list(r0["norm_w"]) == [3.0, 1.0]                    # 权重部分
    assert parse_list(r0["norm_s"]) == [4.0, 0.0]                    # 统计量部分
    # edge 0 的权重部分：[3,0] 与 [0,1] 正交
    assert parse_list(r0["cos_edge_w"]) == pytest.approx([0.0, 0.0], abs=1e-4)
    # 全局（权重部分）：Δ0=[3,0] vs [0,1]+[1,0]=[1,1] → 1/√2
    assert parse_list(r0["cos_global_w"])[0] == pytest.approx(1 / np.sqrt(2), abs=1e-4)
    arr = obs.take_round_arrays()
    assert arr["sketch_w"].shape == arr["sketch"].shape == (3, 8)


def test_no_stat_idx_means_no_split_columns_and_no_sketch_w():
    obs = UG.UpdateGeometry(malicious_ids=(), sketch_dim=8)          # S6a 的用法：不拆
    ew = [np.zeros(4, np.float32), np.zeros(1, np.float32)]
    obs.observe(0, 1, [_upd(0, [1, 0, 0, 0]), _upd(1, [0, 1, 0, 0])], ew, [0])
    row = collect_kv(obs.flush(1, 1), "[UpdateGeo]")[0]
    assert "norm_w" not in row and "norm_s" not in row
    assert "sketch_w" not in obs.take_round_arrays()
