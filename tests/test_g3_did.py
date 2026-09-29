"""
tests/test_g3_did.py  —  3-B 差中差（harness/g3_did.py）：恒等式、分支、真实数据复现 F-066 / F-073

纯标准库，本地秒级。合成 metrics 只含脚本用到的字段。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g3_did as D                                               # noqa: E402


def _m(edge_vals, n=40, rho=0.2, stop=None, yt=(0.1, 0.1, 0.1, 0.1)):
    """edge_vals：{edge: 常数 或 callable(round)}。"""
    per = {}
    for r in range(1, n + 1):
        per[str(r)] = [{"edge_id": e, "client_benign": (v(r) if callable(v) else v)} for e, v in edge_vals.items()]
    return {"exit_code": 0, "client_failures": [], "per_edge_rounds": per,
            "run": {"poison_ratio": rho, "stopped_at_round": stop or n,
                    "data": {"per_edge": [{"edge_id": e, "yt_share": y} for e, y in enumerate(yt)]}}}


def _g0(vals, n=60):
    return _m({e: v for e, v in enumerate(vals)}, n=n, rho=0.0)


def _world(raw_c3, raw_c1, floor_c3, floor_c1):
    g3, g0 = {}, {}
    for s in D.SEEDS:
        g3[("C3", s)] = _m(dict(enumerate(raw_c3)))
        g3[("C1", s)] = _m(dict(enumerate(raw_c1)))
        g0[("C3", s)] = _g0(floor_c3)
        g0[("C1", s)] = _g0(floor_c1)
    return g3, g0


def test_contrast_and_identity():
    assert D.contrast({1: 0.8, 2: 0.6, 3: 0.9}) == pytest.approx(0.2)
    assert D.contrast({1: 0.8, 3: 0.9}) is None
    q = D.cell_run(_m({0: 1.0, 1: 0.95, 2: 0.95, 3: 0.95}), _g0([0.1, 0.10, 0.10, 0.005]))
    assert q["c_raw"] == pytest.approx(0.0)
    assert q["c_excess"] == pytest.approx(q["c_raw"] - q["c_floor"])
    assert q["c_excess"] == pytest.approx(0.095)                  # 天花板下 excess 的对比 = −floor 的对比


def test_window_is_the_last_ten_rounds_with_all_victims_and_floor_uses_the_same_rounds():
    g3 = _m({0: 1.0, 1: 0.9, 2: 0.9, 3: lambda r: None if r > 35 else 0.9}, n=40)
    g0 = _g0([0.1, 0.1, 0.1, 0.1])
    g0["per_edge_rounds"]["30"][3]["client_benign"] = 1.1          # 窗口内的一个 floor 点
    q = D.cell_run(g3, g0)
    assert q["window"] == [26, 35] and q["n_window"] == 10
    assert q["floor"][3] == pytest.approx(0.1 + 1.0 / 10)


def test_first_crossing_interpolates_in_cloud_rounds():
    assert D.first_crossing({1: 0.1, 2: 0.3, 3: 0.7}) == pytest.approx(2.5)
    assert D.first_crossing({1: 0.6, 2: 0.9}) == 1.0              # 首点已越过 → 记首点
    assert D.first_crossing({1: 0.1, 2: 0.2}) is None


def test_ceiling_with_floor_difference_gives_opposite_even_though_raw_is_flat():
    """F-073 的机制：原始对比为 0，但 C3 的 E3 floor 低 → excess DiD > 0 → opposite（预注册没有的一支）。"""
    g3, g0 = _world([1, .95, .95, .95], [1, .95, .95, .95], [.1, .1, .1, .005], [.1, .1, .1, .1])
    res = D.judge(g3, g0)
    assert res["overall"] == "opposite"
    assert all(d["c_raw"] == 0 for d in res["did_per_seed"].values())
    assert res["did_excess_mean"] == pytest.approx(0.095)
    assert len(res["ceiling_runs"]) == 6


def test_lower_e3_in_c3_is_natural_features():
    g3, g0 = _world([1, .8, .8, .5], [1, .8, .8, .8], [.1] * 4, [.1] * 4)
    res = D.judge(g3, g0)
    assert res["overall"] == "natural_features" and res["did_excess_ci"][1] < 0
    assert res["ceiling_runs"] == []


def test_mixed_signs_is_no_difference():
    g3, g0 = _world([1, .8, .8, .8], [1, .8, .8, .8], [.1] * 4, [.1] * 4)
    g3[("C3", 42)] = _m(dict(enumerate([1, .8, .8, .7])))
    g3[("C3", 43)] = _m(dict(enumerate([1, .8, .8, .9])))
    assert D.judge(g3, g0)["overall"] == "no_difference"


def test_validity_gates():
    g3, g0 = _world([1, .8, .8, .5], [1, .8, .8, .8], [.1] * 4, [.1] * 4)
    bad = dict(g0)
    bad[("C1", 43)] = _m(dict(enumerate([.1] * 4)), n=60, rho=0.2)          # G0 不是 ρ=0
    assert D.judge(g3, bad)["overall"] == "invalid"
    short = dict(g0)
    short[("C3", 44)] = _g0([.1] * 4, n=30)                                # floor 没覆盖窗口（31–40）
    assert D.judge(g3, short)["overall"] == "invalid"
    failed = dict(g3)
    failed[("C1", 42)] = dict(g3[("C1", 42)], client_failures=[{"client_id": 1}])
    assert D.judge(failed, g0)["overall"] == "invalid"
    missing = {k: v for k, v in g3.items() if k != ("C3", 44)}
    assert D.judge(missing, g0)["overall"] == "insufficient"
    no_floor = {k: v for k, v in g0.items() if k != ("C1", 42)}
    assert D.judge(g3, no_floor)["overall"] == "insufficient"
    assert D.judge({}, {})["overall"] == "missing"


def test_real_g3_g0_reproduce_findings_066_and_073():
    g3, g0, hd = D.load()
    if len(g3) < 12 or len(g0) < 12:
        pytest.skip("G3 / G0 的结果不在盘上")
    res = D.judge(g3, g0)
    raw = [res["did_per_seed"][str(s)]["c_raw"] for s in D.SEEDS]
    exc = [res["did_per_seed"][str(s)]["c_excess"] for s in D.SEEDS]
    assert raw == pytest.approx([0.0034, 0.0203, 0.0081], abs=2e-4)        # F-066
    assert exc == pytest.approx([0.0765, 0.0665, 0.0676], abs=2e-4)        # F-073
    assert res["overall"] == "opposite" and not res["invalid"]
    assert set(res["ceiling_runs"]) >= {"C3/s42", "C3/s43", "C3/s44"}
    for r in res["per_run"]:                                               # 恒等式在真实数据上逐 run 成立
        assert r["c_excess"] == pytest.approx(r["c_raw"] - r["c_floor"], abs=2e-4)
    assert len(D.explore_hdir(hd)["runs"]) == 12
