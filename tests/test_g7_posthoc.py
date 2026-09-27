"""
tests/test_g7_posthoc.py  —  G7 的事后判定（D-049，**非预注册**）

两侧 + 左删失（第一个评估点就 ≥ 0.5 算「更早」）+ 真实 G7 数据。纯 python，不 import TF。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import g7_posthoc as G      # noqa: E402


def _m(t50, final=0.95, pm=0.88, first=0.1, n=20, grid=5):
    """良性 ASR 轨迹：从 first 起步，在有效轮 t50 处线性越过 0.5，末值 final。"""
    rounds, acc = [], []
    for i in range(1, n + 1):
        eff = i * grid
        v = first if t50 is None else min(final, max(first, 0.5 + (eff - t50) * 0.02))
        if i > n - 10:
            v = final
        rounds.append({"round": eff, "local_benign_asr": v})
        acc.append({"round": eff, "pm_acc": pm, "pm_acc_stale": pm + 0.01})
    return {"exit_code": 0, "run": {"edge_rounds": 1}, "rounds": rounds, "acc_rounds": acc}


def _pairs(std, off):
    return {s: (std, off) for s in G.SEEDS}


def test_yes_when_official_is_earlier_and_not_lower_in_every_seed():
    res = G.judge(_pairs(_m(40), _m(20, final=0.99, pm=0.78)))
    assert res["overall"] == "yes" and res["posthoc"] is True
    assert res["per_seed"][0]["d_pm_acc"] == -0.1


def test_no_when_asr_is_lower_even_if_faster():
    assert G.judge(_pairs(_m(40, final=0.95), _m(20, final=0.90)))["overall"] == "no"


def test_no_when_slower():
    assert G.judge(_pairs(_m(20), _m(40)))["overall"] == "no"


def test_left_censored_official_counts_as_earlier():
    off = _m(None, final=0.99, first=0.6)          # 第一个点就 ≥ 0.5
    assert G.crossing(off).left_censored
    assert G.judge(_pairs(_m(40), off))["overall"] == "yes"


def test_both_left_censored_is_not_earlier():
    a = _m(None, first=0.6, final=0.99)
    assert G.judge(_pairs(a, a))["overall"] == "no"


def test_missing_and_invalid():
    assert G.judge({42: (_m(40), None)})["overall"] == "missing"
    bad = _m(20, final=0.99)
    bad["client_failures"] = [{"client_id": 1, "error": "x"}]
    assert G.judge(_pairs(_m(40), bad))["overall"] == "invalid"


def test_real_g7_data_gives_yes_posthoc():
    """真实数据（`5edd4df`）：3 个 seed 都「更快且不低」；干净精度两臂差 −0.08 … −0.11（混杂）。"""
    res = G.judge(G.load())
    assert res["overall"] == "yes"
    assert [p["t50_official"] for p in res["per_seed"]][2] == "<首点"
    assert all(-0.12 < p["d_pm_acc"] < -0.08 for p in res["per_seed"])
