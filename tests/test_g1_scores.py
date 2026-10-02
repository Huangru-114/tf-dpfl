"""
tests/test_g1_scores.py  —  harness/g1_scores.py（S6b / D-087）。手算 / 构造值，纯 numpy。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g1_scores as GS                                           # noqa: E402

GEO = ["round", "edge_id", "edge_round", "cid", "mal", "norm", "cos_edge", "cos_global",
       "norm_w", "norm_s", "cos_edge_w", "cos_global_w"]


def _metrics(rows_spec, R=10):
    """rows_spec: [(round, edge, er, [(cid, mal, norm_w)...])]；其余几何列填占位。"""
    rows = []
    for rnd, e, er, ups in rows_spec:
        cids = [u[0] for u in ups]
        mal = [int(u[1]) for u in ups]
        nw = [u[2] for u in ups]
        rows.append([rnd, e, er, cids, mal, nw, [None] * len(ups), [None] * len(ups),
                     nw, [0.0] * len(ups), [None] * len(ups), [None] * len(ups)])
    return {"run": {"edge_rounds": R, "seed": 42, "provenance": {"run_id": "T"}},
            "update_geometry": {"columns": GEO, "rows": rows}}


def _heterogeneous(offsets, malicious_edges, n_per=8, rounds=3, bump=3.0, seed=0):
    """每个 edge 有自己的基线偏移；malicious_edges 里的 edge 一半更新是恶意（+bump）。"""
    r = np.random.default_rng(seed)
    spec, cid = [], 0
    for g in range(1, rounds + 1):
        for e, off in enumerate(offsets):
            ups = []
            for j in range(n_per):
                mal = e in malicious_edges and j % 2 == 0
                ups.append((cid, mal, off + (bump if mal else 0.0) + r.normal(0, 0.5)))
                cid += 1
            spec.append((g, e, 1, ups))
    return _metrics(spec)


def test_collect_joins_ck_by_round_edge_edgeround_and_computes_s():
    m = _metrics([(1, 0, 5, [(7, 1, 2.0), (8, 0, 1.0)])])
    m["ck_before"] = {"columns": ["round", "edge_id", "edge_round", "effective_round", "n_proto",
                                  "n_attack", "ncm_acc", "c"],
                      "rows": [[1, 0, 5, 5, 500, 64, 0.8, [0.5] * 5]]}
    m["ck_scores"] = {"columns": ["round", "edge_id", "edge_round", "effective_round", "cid", "mal",
                                  "ncm_acc", "c"],
                      "rows": [[1, 0, 5, 5, 7, 1, 0.8, [0.5, 0.5, 0.1, 0.4, 0.6]],
                               [1, 0, 5, 5, 8, 0, 0.8, [0.5, 0.5, 0.5, 0.5, 0.5]]]}
    ups = {u["cid"]: u for u in GS.collect_updates(m)}
    assert ups[7]["s_ck"] == pytest.approx(0.4 / (0.1 + 1e-6)) and ups[8]["s_ck"] is not None
    assert ups[7]["mal"] is True and ups[7]["norm_w"] == 2.0 and ups[7]["eff"] == 5


def test_edge_normalisation_beats_global_when_edges_have_different_baselines():
    m = _heterogeneous(offsets=[0, 10, 20, 30], malicious_edges={0, 1, 2, 3})
    r = GS.view_aurocs(GS.collect_updates(m), "norm_w", window=10, resamples=30)
    assert r["raw"] < 0.7                                    # 基线差把恶意 / 良性的分布混在一起
    assert r["edge"] > 0.95                                  # 在各自 edge 里归一化后分得很开
    assert r["delta"] > 0.2 and r["global_matched"] < r["edge"]


def test_homogeneous_edges_give_no_edge_advantage():
    m = _heterogeneous(offsets=[0, 0, 0, 0], malicious_edges={0, 1, 2, 3}, rounds=6)
    r = GS.view_aurocs(GS.collect_updates(m), "norm_w", window=10, resamples=40)
    assert abs(r["delta"]) < 0.05 and r["edge"] > 0.9       # 都分得开，只是视角没差别


def test_collocated_edge_view_loses_the_attacker_edge_offset():
    """集中布点：E0 全是恶意端。edge 视角把 E0 的整体偏高归一化掉；全局视角看得见。"""
    spec, cid = [], 0
    r = np.random.default_rng(1)
    for g in range(1, 5):
        for e in range(4):
            ups = []
            for j in range(8):
                ups.append((cid, e == 0, (4.0 if e == 0 else 0.0) + r.normal(0, 0.5)))
                cid += 1
            spec.append((g, e, 1, ups))
    res = GS.view_aurocs(GS.collect_updates(_metrics(spec)), "norm_w", window=10, resamples=40)
    assert res["global_full"] > 0.95 and res["edge"] < 0.7 and res["delta"] < -0.2


def test_undefined_cases_are_none_not_zero():
    res = GS.view_aurocs(GS.collect_updates(_metrics([(1, 0, 1, [(0, 0, 1.0), (1, 0, 2.0)])])),
                         "norm_w")
    assert res["raw"] is None and res["edge"] is None and res["delta"] is None   # 没有恶意端
    res = GS.view_aurocs([], "s_ck")
    assert res["n"] == 0 and res["raw"] is None
    # 老日志（没有 norm_w）→ 该分数整列 None
    m = _metrics([(1, 0, 1, [(0, 1, 1.0), (1, 0, 2.0)])])
    for row in m["update_geometry"]["rows"]:
        row[8] = None
    ups = GS.collect_updates(m)
    assert GS.view_aurocs(ups, "norm_w")["n"] == 0


def test_deterministic_given_the_seed():
    m = _heterogeneous([0, 5, 10, 15], {0, 2})
    a = GS.view_aurocs(GS.collect_updates(m), "norm_w", 10, 20, seed=3)
    b = GS.view_aurocs(GS.collect_updates(m), "norm_w", 10, 20, seed=3)
    assert a == b


def test_readout_and_main_run_on_a_synthetic_file(tmp_path, capsys):
    import json
    m = _heterogeneous([0, 10, 20, 30], {0, 1, 2, 3})
    p = tmp_path / "x.metrics.json"
    p.write_text(json.dumps(m))
    assert GS.main([str(p), "--window", "10", "--resamples", "10", "--json", str(tmp_path / "o.json")]) == 0
    out = capsys.readouterr().out
    assert "norm_w" in out and "Δ(edge" in out
    d = json.loads((tmp_path / "o.json").read_text())
    assert d["T"]["scores"]["norm_w"]["delta"] > 0.2
