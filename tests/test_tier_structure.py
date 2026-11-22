"""
tests/test_tier_structure.py  —  fedavg/analysis/tier_structure.py 的纯 numpy 部分（补充实验 D0b，N-015）

合成「快照」：4 台边缘、3 个客户端，get_weights 索引 = [0 上云段 conv, 1 上云段统计量, 2 边缘段 conv, 3 边缘段统计量, 4 head]。
  · 结构正确（全局上云段 = 边缘加权平均、全局边缘段 = 初值不变、边缘段各不相同、私有部分各不相同）→ 全部 pass；
  · 每条各造一个反例：全局上云段不是平均 / 全局边缘段跨轮变了（被聚合了）/ 两台边缘的边缘段相同（被共享了）/
    两个客户端分类头相同 / 私有索引里混进边缘段的可训练权重 → 对应的那一条 FAIL、总判 FAIL；
  · 快照读写往返（格式 = BackdoorCloudServer._snapshot 的键名）。
TF 那一半（真实索引、真跑出来的快照形状）在 tests/test_three_tier_personalization.py。
"""

import copy
import json

import numpy as np
import pytest

from analysis import tier_structure as T

SETS = {"edge": [2, 3], "cloud": [0, 1], "head": [4], "stat": [1, 3]}
INIT_EDGE = [np.full((3, 3), 0.5, np.float32), np.full(3, 1.0, np.float32)]


def _snap(rnd, weights=(0.3, 0.2, 0.3, 0.2)):
    rng = np.random.default_rng(rnd)
    edges = {}
    for e in range(4):
        edges[e] = [rng.normal(size=(4, 4)).astype(np.float32), rng.random(4).astype(np.float32),
                    rng.normal(size=(3, 3)).astype(np.float32), rng.random(3).astype(np.float32),
                    np.zeros(2, np.float32)]
    g = [sum(w * edges[e][i].astype(np.float64) for e, w in enumerate(weights)).astype(np.float32) for i in (0, 1)]
    glob = g + [INIT_EDGE[0].copy(), INIT_EDGE[1].copy(), np.zeros(2, np.float32)]
    private = {cid: ([1, 3, 4], [rng.random(4).astype(np.float32), rng.random(3).astype(np.float32),
                                 rng.normal(size=2).astype(np.float32)]) for cid in range(3)}
    private[3] = ([], [])                                           # 从未被选中的客户端：私有为空，不参与比较
    return {"meta": {"round": rnd}, "global": glob, "edges": edges, "private": private}


def _snaps():
    return {6: _snap(6), 15: _snap(15), 60: _snap(60)}


def test_correct_structure_passes_every_check():
    res = T.check_run(_snaps(), SETS)
    assert res["pass"], json.dumps(res, default=str)[:400]
    r6 = res["rounds"]["6"]
    assert [round(w, 4) for w in r6["cloud_is_mixture"]["weights"]] == [0.3, 0.2, 0.3, 0.2]
    assert r6["cloud_is_mixture"]["rel_residual"] < 1e-6
    assert r6["private_distinct"]["n_clients_with_private"] == 3


def test_global_cloud_segment_that_is_not_an_average_fails():
    s = _snaps()
    s[15]["global"][0] = s[15]["global"][0] + 0.1
    res = T.check_run(s, SETS)
    assert not res["pass"] and not res["rounds"]["15"]["cloud_is_mixture"]["pass"]
    assert res["rounds"]["6"]["cloud_is_mixture"]["pass"]


def test_negative_weights_fail_even_with_zero_residual():
    s = _snaps()
    s[6] = _snap(6, weights=(1.2, -0.2, 0.0, 0.0))
    assert not T.check_run(s, SETS)["rounds"]["6"]["cloud_is_mixture"]["pass"]


def test_global_edge_segment_that_moves_between_rounds_fails():
    """边缘段被云聚合了（全局那一段不再是初值）。"""
    s = _snaps()
    s[60]["global"][2] = s[60]["edges"][0][2].copy() * 0.5 + s[60]["edges"][1][2] * 0.5
    res = T.check_run(s, SETS)
    assert not res["pass"] and not res["global_edge_seg_frozen"]["pass"]


def test_two_edges_with_the_same_edge_segment_fail():
    """边缘段被共享了（两台边缘一样）。"""
    s = _snaps()
    s[15]["edges"][2][2] = s[15]["edges"][1][2].copy()
    s[15]["edges"][2][3] = s[15]["edges"][1][3].copy()
    chk = T.check_run(s, SETS)["rounds"]["15"]["edge_segs_pairwise_distinct"]
    assert not chk["pass"] and chk["pairs"]["1-2"] is False and chk["pairs"]["0-1"] is True


def test_global_edge_segment_equal_to_an_edge_fails():
    s = _snaps()
    s[6]["edges"][3][2] = INIT_EDGE[0].copy()
    s[6]["edges"][3][3] = INIT_EDGE[1].copy()
    chk = T.check_run(s, SETS)["rounds"]["6"]["global_edge_seg_differs_from_every_edge"]
    assert not chk["pass"] and chk["per_edge"][3] is False


def test_duplicate_heads_fail():
    s = _snaps()
    idx, vals = s[6]["private"][1]
    s[6]["private"][1] = (idx, [vals[0], vals[1], s[6]["private"][0][1][2].copy()])
    chk = T.check_run(s, SETS)["rounds"]["6"]["private_distinct"]
    assert not chk["pass"] and chk["duplicate_head_pairs"] == 1 and chk["duplicate_stat_pairs"] == 0


def test_edge_trainables_in_private_state_fail():
    s = _snaps()
    s[6]["private"][2] = ([1, 2, 3, 4], [np.zeros(4), np.zeros((3, 3)), np.zeros(3), np.ones(2)])
    chk = T.check_run(s, SETS)["rounds"]["6"]["private_distinct"]
    assert not chk["pass"] and chk["clients_with_edge_trainables_in_private"] == [2]


def test_npz_roundtrip_uses_the_snapshot_writer_key_names(tmp_path):
    s = _snap(6)
    arrays = {f"global_{i:03d}": w for i, w in enumerate(s["global"])}
    for e, ws in s["edges"].items():
        arrays.update({f"edge{e}_{i:03d}": w for i, w in enumerate(ws)})
    for cid, (idx, vals) in s["private"].items():
        arrays[f"client{cid}_idx"] = np.asarray(idx, np.int32)
        arrays.update({f"client{cid}_{i:03d}": v for i, v in enumerate(vals)})
    arrays["meta_json"] = np.array(json.dumps({"round": 6, "edge_ids": [0, 1, 2, 3], "client_ids": [0, 1, 2, 3]}))
    p = tmp_path / "snapshot_r006.npz"
    np.savez(p, **arrays)
    back = T.load_npz(p)
    assert len(back["global"]) == 5 and sorted(back["edges"]) == [0, 1, 2, 3]
    assert back["private"][3] == ([], [])
    assert all(np.array_equal(a, b) for a, b in zip(back["global"], s["global"]))
    assert T.check_round(back, SETS)["pass"]


def test_module_does_not_import_tensorflow_at_import_time():
    src = (T.__file__ and open(T.__file__, encoding="utf-8").read())
    head = src.split("def index_sets")[0]
    assert "import tensorflow" not in head and "from models" not in head
