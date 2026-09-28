"""
tests/test_decay_verdict.py  —  G8 的预注册判定（D-075；纯标准库）

合成轨迹（受害 edge 在第 28–30 / 51–60 / 61–70 轮的水平可以直接写出来）：
  · 三种判定各自的边界（阈值沿用 FLR 的 0.05 / 0.10，D-068）；
  · 主窗口 = 第 51–60 轮（与 FLR 的 floor 窗口同一批有效轮），不是 61–70；
  · 停止轮没生效（run.attack_stop_round ≠ 31）/ 客户端失败 / 没跑到第 60 轮 → invalid；
  · 另报量：retention、E0、margin 斜率。
另测配置层面的配对前提：G8 与 G6(a) 同 seed 只差停止轮 / n_rounds / 两个存盘开关。
"""

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import decay_verdict as V                                       # noqa: E402

CONFIGS = ROOT / "experiments/attack/hfl-mechanism/configs"


def _m(level, n=70, stop=31, e0=1.0, pre=0.8, end=None, margin=None, failures=None):
    """受害 edge：第 1–30 轮 = pre，31–60 轮 = level，61–70 轮 = end（缺省同 level）。"""
    per, rounds = {}, []
    for r in range(1, n + 1):
        v = pre if r <= 30 else (level if r <= 60 else (level if end is None else end))
        per[str(r)] = [{"edge_id": 0, "client_benign": e0}] + \
                      [{"edge_id": e, "client_benign": v} for e in (1, 2, 3)]
        rounds.append({"round": r, "margin_p50": None if margin is None else margin(r)})
    return {"exit_code": 0, "client_failures": failures or [],
            "run": {"attack_stop_round": stop, "n_edges": 4, "poison_ratio": 0.2,
                    "malicious_ids": [1, 2]},
            "per_edge_rounds": per, "rounds": rounds}


def _flr(level=0.02, n=60):
    m = _m(level, n=n, stop=None, e0=level, pre=level)
    m["run"]["poison_ratio"] = 0.0
    return m


def _triples(levels, **kw):
    return {s: (_m(lv, **kw), _flr(), None) for s, lv in zip(V.SEEDS, levels)}


@pytest.mark.parametrize("levels,want", [
    ((0.60, 0.45, 0.30), "persists"),              # excess 0.58 / 0.43 / 0.28 ≥ 0.10
    ((0.125, 0.40, 0.50), "persists"),             # 最小的 excess 0.105 ≥ 0.10
    ((0.06, 0.065, 0.03), "decays_to_floor"),      # excess 0.04 / 0.045 / 0.01 ≤ 0.05
    ((0.115, 0.40, 0.50), "user_decides"),         # 0.095 差一点够不上 persists
    ((0.06, 0.30, 0.03), "user_decides"),          # seed 间不一致
    ((0.10, 0.10, 0.10), "user_decides"),          # 0.08：介于两者之间
])
def test_verdict_boundaries(levels, want):
    assert V.judge(_triples(levels))["overall"] == want


def test_main_window_is_51_to_60_not_the_end():
    """第 61–70 轮掉回 floor，但主窗口 51–60 仍高 → persists（61–70 只报告）。"""
    res = V.judge(_triples((0.5, 0.5, 0.5), end=0.02))
    assert res["overall"] == "persists"
    p = res["per_seed"][0]["victims"]
    assert p["excess"] == pytest.approx(0.48) and p["excess_end"] == pytest.approx(0.0)


def test_reported_quantities():
    res = V.judge(_triples((0.41, 0.41, 0.41), pre=0.82, margin=lambda r: 2.0 - 0.01 * r))
    p = res["per_seed"][0]
    assert p["victims"]["A_T"] == pytest.approx(0.82)
    assert p["victims"]["retention"] == pytest.approx((0.41 - 0.02) / (0.82 - 0.02))
    assert p["e0"]["A_main"] == pytest.approx(1.0) and p["e0"]["floor"] == pytest.approx(0.02)
    assert p["margin_p50_slope_31_70"] == pytest.approx(-0.01)
    assert p["same_malicious_ids"] is True


@pytest.mark.parametrize("kw,needle", [
    ({"stop": None}, "attack_stop_round"),         # 停止轮静默没生效
    ({"stop": 21}, "attack_stop_round"),
    ({"n": 55}, "主判定窗口"),
    ({"failures": [{"client_id": 3, "error": "x"}]}, "client_failures"),
])
def test_invalid_runs(kw, needle):
    t = _triples((0.5, 0.5, 0.5))
    t[42] = (_m(0.5, **kw), _flr(), None)
    res = V.judge(t)
    assert res["overall"] == "invalid"
    assert any(needle in r for r in res["per_seed"][0]["reasons"])


def test_flr_that_is_not_rho0_is_invalid():
    t = _triples((0.5, 0.5, 0.5))
    bad = _flr()
    bad["run"]["poison_ratio"] = 0.2
    t[43] = (_m(0.5), bad, None)
    assert V.judge(t)["overall"] == "invalid"


def test_missing_and_insufficient():
    assert V.judge({})["overall"] == "missing"
    t = _triples((0.5, 0.5, 0.5))
    t[44] = (None, _flr(), None)
    assert V.judge(t)["overall"] == "insufficient"


def test_no_results_yet_on_disk():
    """数据回来之前写定：现在盘上没有 G8 / FLR 的结果。"""
    assert V.judge(V.load())["overall"] == "missing"


# ── 配置层面的配对前提 ──────────────────────────────────────────────────────
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


@pytest.mark.parametrize("seed", V.SEEDS)
def test_g8_config_differs_from_g6a_only_in_stop_length_and_switches(seed):
    g8 = _flat(yaml.safe_load((CONFIGS / f"G8__a__s{seed}.yaml").read_text(encoding="utf-8")))
    g6 = _flat(yaml.safe_load((CONFIGS / f"G6__a__s{seed}.yaml").read_text(encoding="utf-8")))
    diff = {k for k in g8.keys() | g6.keys() if g8.get(k) != g6.get(k)}
    assert diff == {"backdoor.attack_stop_round", "federation.n_rounds",
                    "evaluation.dump_logits_every", "evaluation.snapshot_rounds",
                    "meta.group", "meta.run_id"}
    assert g8["backdoor.attack_stop_round"] == V.STOP_ROUND and g8["federation.n_rounds"] == 70
    assert g8["evaluation.snapshot_rounds"] == "30/70"


@pytest.mark.parametrize("arm", ["a", "b", "c"])
def test_g6d_config_differs_from_g6_only_in_placement(arm):
    d = _flat(yaml.safe_load((CONFIGS / f"G6D__{arm}__s42.yaml").read_text(encoding="utf-8")))
    g6 = _flat(yaml.safe_load((CONFIGS / f"G6__{arm}__s42.yaml").read_text(encoding="utf-8")))
    diff = {k for k in d.keys() | g6.keys() if d.get(k) != g6.get(k)}
    assert diff == {"backdoor.malicious_per_edge", "meta.group", "meta.run_id"}
    assert d["backdoor.malicious_per_edge"] == [3, 3, 2, 2]
