"""
tests/test_g1_explore.py  —  harness/g1_explore.py（G1 的探索性读数，不进判定）。手算值，纯 numpy。
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g1_explore as E                                           # noqa: E402

LIGHT = list(E.EDGE_COLS)
DETAIL = ["edge_id", "margin_p50", "benign_asr_p90", "yt_clean_benign"]


def _run(R=10, n=4, post=0.9, full=0.6, light=0.75, e0=(0.2, 1.0), margin=(2.0, -1.0, 0.5)):
    """受害 edge：云聚合后 post、周期中点 light、云轮末 full；E0：云聚合后 e0[0]、云轮末 e0[1]。
    margin = (云聚合后, 云轮末, 轻评估点)，只给受害 edge；E0 的 margin 固定 9。"""
    per_edge, post_agg, light_rows, detail = {}, {}, {}, {}
    for g in range(1, n + 1):
        per_edge[str(g)] = ([{"edge_id": 0, "client_benign": e0[1]}]
                            + [{"edge_id": e, "client_benign": full} for e in (1, 2, 3)])
        detail[str(g)] = [[0, 9.0, 0, 0]] + [[e, margin[1], 0, 0] for e in (1, 2, 3)]
        if g >= 2:
            post_agg[str(g)] = [[0, 0, e0[0], None, 0, 0, 9.0, 0]] + [[e, 0, post, None, 0, 0, margin[0], 0] for e in (1, 2, 3)]
        light_rows[str((g - 1) * R + R // 2)] = ([[0, 0, 0.5, None, 0, 0, 9.0, 0]]
                                                 + [[e, 0, light, None, 0, 0, margin[2], 0] for e in (1, 2, 3)])
    return {"run": {"edge_rounds": R, "n_rounds": n}, "per_edge_rounds": per_edge,
            "per_edge_post_agg_rounds": post_agg, "per_edge_light_rounds": light_rows,
            "per_edge_light_columns": LIGHT, "per_edge_detail_rounds": detail, "per_edge_detail_columns": DETAIL}


def test_point_tables_place_light_points_inside_the_right_cycle():
    pts = E.point_tables(_run(R=10, n=3), "client_benign")
    assert pts[(2, 0)][1] == 0.9 and pts[(2, 5)][1] == 0.75 and pts[(2, 10)][1] == 0.6
    assert pts[(1, 5)][0] == 0.5                                   # 有效轮 5 → 第 1 云轮的第 5 个 edge 轮
    assert (1, 0) not in pts                                       # 第 1 云轮没有云聚合后点
    m = E.point_tables(_run(R=10, n=3), "margin_p50")
    assert m[(2, 0)][1] == 2.0 and m[(2, 10)][1] == -1.0 and m[(2, 5)][1] == 0.5


def test_cycle_profile_splits_halves_and_separates_e0():
    prof = E.cycle_profile(_run(R=10, n=4), "client_benign")
    assert set(prof["first"]) == {0, 5, 10} and set(prof["second"]) == {0, 5, 10}
    assert prof["first"][0] == {"victims": 0.9, "e0": 0.2}
    assert prof["first"][10] == {"victims": 0.6, "e0": 1.0}
    assert prof["first"][5]["victims"] == 0.75


def test_rates_by_hand():
    r = E.rates(_run(R=10, n=4))
    a = r["client_benign:first"]
    assert a["victim_r_down"] == pytest.approx(0.03)               # (0.9 − 0.6) / 10
    assert a["victim_d_jump"] == pytest.approx(0.3)                # 0.9 − 0.6
    assert a["victim_net_per_cycle"] == pytest.approx(0.0)         # = 云轮末的逐周期变化
    assert a["e0_drop_at_agg"] == pytest.approx(-0.8)              # 0.2 − 1.0
    assert a["e0_r_up"] == pytest.approx(0.08)                     # (1.0 − 0.2) / 10
    assert a["n_cycles"] == 1                                      # g = 2 … ⌊4/2⌋
    assert r["client_benign:second"]["n_cycles"] == 2              # g = 3, 4
    assert r["margin_p50:first"]["victim_r_down"] == pytest.approx(0.3)   # (2 − (−1)) / 10


def test_window_aurocs_on_a_tiny_geometry_table():
    m = _run()
    cols = ["round", "edge_id", "edge_round", "cid", "mal", "norm", "cos_edge", "cos_global",
            "norm_w", "norm_s", "cos_edge_w", "cos_global_w"]
    rows = []
    for g in (1, 3):                                               # 有效轮 1 与 21 → 两个 20 轮窗口
        rows.append([g, 0, 1, [1, 2, 3, 4], [1, 1, 0, 0], [0] * 4, [0] * 4, [0] * 4,
                     [5.0, 4.0, 1.0, 2.0], [0] * 4, [0] * 4, [0] * 4])
    m["update_geometry"] = {"columns": cols, "rows": rows}
    m["run"]["edge_rounds"] = 10
    w = E.window_aurocs(m, keys=("norm_w",))["norm_w"]
    assert [r["eff_from"] for r in w] == [1, 21]
    assert all(r["raw"] == 1.0 and r["n"] == 4 and r["n_mal"] == 2 for r in w)
