"""
tests/test_g6_verdict.py  —  3-E（G6）的判定（S7；规则 D-059 / D-071 预注册，脚本写于数据之后）

合成数据覆盖三种判定分支 + 两种 MTA 读法分歧的情形；真实数据把回传结果钉住。纯标准库。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g6_verdict as V                  # noqa: E402


def _run(arm, victim_asr, pm_acc, e0=1.0, n=12):
    """最小可判的 metrics：每轮 4 个 edge 的 client_benign 与 pm_acc 都是常数。"""
    edges = [{"edge_id": 0, "client_benign": e0}] + \
            [{"edge_id": e, "client_benign": victim_asr} for e in (1, 2, 3)]
    return {"run": {"edge_shared_blocks": V.ARMS[arm], "malicious_per_edge": [10, 0, 0, 0]},
            "exit_code": 0, "client_failures": [], "errors": [],
            "rounds": [{"round": r} for r in range(1, n + 1)],
            "per_edge_rounds": {str(r): edges for r in range(1, n + 1)},
            "acc_rounds": [{"round": r, "pm_acc": pm_acc, "pm_acc_stale": pm_acc + 0.03 if r % 2 else None}
                           for r in range(1, n + 1)]}


def _runs(asr, acc):
    """asr / acc：{arm: [s42, s43, s44]}。"""
    return {(a, s): _run(a, asr[a][i], acc[a][i]) for a in V.ARMS for i, s in enumerate(V.SEEDS)}


def test_blocks_when_asr_drops_in_every_seed_and_mta_cost_is_small():
    res = V.judge(_runs({"a": [0.8, 0.7, 0.9], "b": [0.3, 0.2, 0.4], "c": [0.1, 0.1, 0.1]},
                        {"a": [0.87] * 3, "b": [0.865] * 3, "c": [0.855] * 3}))
    assert res["overall"] == {"b": "blocks", "c": "blocks"}
    assert res["arms"]["b"]["d_asr"] == {"s42": 0.5, "s43": 0.5, "s44": 0.5}
    assert res["arms"]["c"]["d_mta_fresh_mean"] == pytest.approx(0.015)


def test_one_seed_without_a_drop_is_no_block():
    res = V.judge(_runs({"a": [0.8, 0.7, 0.9], "b": [0.3, 0.75, 0.4], "c": [0.1, 0.1, 0.1]},
                        {a: [0.87] * 3 for a in "abc"}))
    assert res["arms"]["b"]["verdict"] == "no_block" and res["arms"]["c"]["verdict"] == "blocks"


def test_mta_cost_above_the_line_is_costly():
    res = V.judge(_runs({"a": [0.8] * 3, "b": [0.3] * 3, "c": [0.1] * 3},
                        {"a": [0.87] * 3, "b": [0.865] * 3, "c": [0.84] * 3}))
    assert res["arms"]["c"]["verdict"] == "costly"


def test_the_two_mta_readings_are_reported_separately():
    """0.02 按均值成立、按逐 seed 不成立 → 不假装是 blocks。"""
    res = V.judge(_runs({"a": [0.8] * 3, "b": [0.3] * 3, "c": [0.1] * 3},
                        {"a": [0.87] * 3, "b": [0.865] * 3, "c": [0.845, 0.86, 0.86]}))
    c = res["arms"]["c"]
    assert c["mta_ok_mean"] is True and c["mta_ok_every_seed"] is False
    assert c["verdict"] == "mta_reading_matters"


def test_stale_column_is_read_on_its_own_evaluation_points():
    q = V.run_quantities(_run("a", 0.5, 0.87))
    assert q["pm_acc_stale"] == pytest.approx(0.90)       # 隔点列：只取有值的点，不当成 None 丢掉整条
    assert q["victim_asr"] == pytest.approx(0.5) and q["e0_benign_asr"] == 1.0


def test_switch_that_did_not_take_effect_is_invalid():
    runs = _runs({a: [0.5] * 3 for a in "abc"}, {a: [0.87] * 3 for a in "abc"})
    runs[("c", 43)]["run"]["edge_shared_blocks"] = 0
    res = V.judge(runs)
    assert res["overall"] == "invalid" and res["invalid"][0]["run"] == "c/s43"


def test_real_g6_verdict_is_what_the_report_says():
    """G6 回传（F-061 / F-065③）→ 两臂都 blocks（F-077）；数与 REPORT §5.4 的表一致。"""
    runs = V.load()
    if any(v is None for v in runs.values()):
        pytest.skip("G6 不全")
    res = V.judge(runs)
    assert res["overall"] == {"b": "blocks", "c": "blocks"} and not res["invalid"]
    assert res["arms"]["b"]["d_asr"] == {"s42": 0.5267, "s43": 0.4882, "s44": 0.5573}
    assert res["arms"]["c"]["d_asr"] == {"s42": 0.733, "s43": 0.5396, "s44": 0.9175}
    assert res["arms"]["c"]["d_mta_fresh"] == {"s42": 0.01942, "s43": 0.01971, "s44": 0.01226}
    got = {(p["arm"], p["seed"]): round(p["victim_asr"], 3) for p in res["per_run"]}
    assert got == {("a", 42): 0.794, ("a", 43): 0.657, ("a", 44): 0.995,
                   ("b", 42): 0.267, ("b", 43): 0.169, ("b", 44): 0.438,
                   ("c", 42): 0.061, ("c", 43): 0.117, ("c", 44): 0.078}
