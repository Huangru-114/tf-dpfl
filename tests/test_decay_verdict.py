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


def test_on_disk_results_give_a_defined_verdict():
    """随盘上的数据状态断言（FLR 已于 `df4e98e` 回传，F-065）：
    G8 一个都没回 → missing；回来后 → 必须是预注册判定之一，不能报错。"""
    g8 = [V.RESULTS / "G8" / f"G8__a__s{s}.metrics.json" for s in (42, 43, 44)]
    overall = V.judge(V.load())["overall"]
    if not any(p.exists() for p in g8):
        assert overall == "missing"
    else:
        assert overall in {"persists", "decays_to_floor", "user_decides", "insufficient", "invalid", "missing"}


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


# ══════════════════════════════════════════════════════════════════════════
# flat 对照（G8F，D-077）：良性端池化列、窗口按有效轮对齐
# ══════════════════════════════════════════════════════════════════════════
def _flat_m(level, pre=0.8, end=None, n=350, stop=151, er=1, n_edges=1, every=5, failures=None):
    """flat：有效轮 = 轮号；每 5 轮一个评估点。≤150 = pre，151–300 = level，>300 = end。"""
    rounds = []
    for r in range(every, n + 1, every):
        eff = r * er
        v = pre if eff <= 150 else (level if eff <= 300 else (level if end is None else end))
        rounds.append({"round": r, "local_benign_asr": v})
    return {"exit_code": 0, "client_failures": failures or [],
            "run": {"attack_stop_round": stop, "n_edges": n_edges, "edge_rounds": er,
                    "malicious_ids": [1, 2]},
            "rounds": rounds}


def _flat_pairs(levels, **kw):
    return {s: (_flat_m(lv, **kw), None) for s, lv in zip(V.SEEDS, levels)}


@pytest.mark.parametrize("levels,want", [
    ((0.45, 0.40, 0.35), "flat_plateau"),          # 最小 0.35 ≥ HIGH（边界含）
    ((0.45, 0.40, 0.34), "user_decides"),          # 0.34 差一点
    ((0.18, 0.14, 0.25), "flat_decays"),           # 最大 0.25 ≤ LOW（边界含）
    ((0.18, 0.14, 0.2001), "flat_decays"),         # G8 自己的池化值（s44 = 0.2001）必须判得出「像 HFL」
    ((0.18, 0.14, 0.26), "user_decides"),
    ((0.10, 0.45, 0.10), "user_decides"),          # seed 间不一致
])
def test_flat_verdict_boundaries(levels, want):
    assert V.judge_flat(_flat_pairs(levels))["overall"] == want


def test_flat_windows_are_in_effective_rounds():
    """同一个量：flat（每轮 1 有效轮、每 5 轮一点）与 G8（每轮 5 有效轮、每轮一点）取到同一批有效轮。"""
    flat = _flat_m(0.3, pre=0.9)
    hfl = _flat_m(0.3, pre=0.9, n=70, er=5, n_edges=4, every=1, stop=31)
    for win in (V.FLAT_WIN_T, V.FLAT_WIN_MAIN, V.FLAT_WIN_END):
        assert V.pooled_window_mean(flat, win) == pytest.approx(V.pooled_window_mean(hfl, win))
    assert V.pooled_window_mean(flat, V.FLAT_WIN_T) == pytest.approx(0.9)
    assert V.pooled_window_mean(flat, V.FLAT_WIN_MAIN) == pytest.approx(0.3)
    # 主窗口正好 10 个评估点（有效轮 255, 260, …, 300）
    assert sum(1 for r in flat["rounds"] if 255 <= r["round"] <= 300) == 10


def test_flat_reports_pairing_with_g8():
    g8 = _m(0.1)
    g8["run"]["edge_rounds"] = 5
    g8["rounds"] = [{"round": r, "local_benign_asr": 0.9 if r <= 30 else 0.18} for r in range(1, 71)]
    res = V.judge_flat({s: (_flat_m(0.40), g8) for s in V.SEEDS})
    p = res["per_seed"][0]
    assert p["hfl"]["A_main"] == pytest.approx(0.18) and p["flat_minus_hfl"] == pytest.approx(0.22)
    assert res["overall"] == "flat_plateau"


@pytest.mark.parametrize("bad", [
    {"stop": 31},                                  # 停止轮没生效
    {"n_edges": 4},                                # 不是 flat
    {"n": 250},                                    # 没跑到主窗口末
    {"failures": [{"client_id": 3}]},              # 客户端异常被吞
])
def test_flat_invalid_runs_are_flagged(bad):
    lv = (0.4, 0.4, 0.4)
    pairs = _flat_pairs(lv)
    pairs[42] = (_flat_m(0.4, **bad), None)
    assert V.judge_flat(pairs)["overall"] == "invalid"


def test_flat_missing_and_insufficient():
    assert V.judge_flat({})["overall"] == "missing"
    pairs = _flat_pairs((0.4, 0.4, 0.4))
    pairs[44] = (None, None)
    assert V.judge_flat(pairs)["overall"] == "insufficient"


def test_flat_on_disk_results_give_a_defined_verdict():
    """随盘上的数据状态断言：G8F 没回 → missing；回来后必须是预注册判定之一。"""
    files = [V.RESULTS / "G8F" / f"G8F__std__s{s}.metrics.json" for s in V.SEEDS]
    overall = V.judge_flat(V.load_flat())["overall"]
    if not any(p.exists() for p in files):
        assert overall == "missing"
    else:
        assert overall in {"flat_plateau", "flat_decays", "user_decides", "insufficient",
                           "invalid", "missing"}


def test_real_g8_pooled_values_are_what_the_thresholds_were_set_against():
    """反向锚点（真实数据）：G8 池化列 0.1807 / 0.1424 / 0.2001 —— 阈值 0.25 / 0.35 就是照它定的。"""
    got = [V._pooled(g)["A_main"] for _, g in (V.load_flat()[s] for s in V.SEEDS)]
    assert got == [0.1807, 0.1424, 0.2001]
    assert max(got) <= V.FLAT_LOW


@pytest.mark.parametrize("seed", V.SEEDS)
def test_g8f_config_is_g8_made_flat(seed):
    """G8F 与 G8 只差「flat 化」与存盘开关；停止轮换算成同一批有效轮（D-077）。"""
    f = _flat(yaml.safe_load((CONFIGS / f"G8F__std__s{seed}.yaml").read_text(encoding="utf-8")))
    g8 = _flat(yaml.safe_load((CONFIGS / f"G8__a__s{seed}.yaml").read_text(encoding="utf-8")))
    diff = {k for k in f.keys() | g8.keys() if f.get(k) != g8.get(k)}
    assert diff == {"federation.n_edges", "federation.edge_rounds", "federation.n_rounds",
                    "backdoor.malicious_per_edge", "backdoor.attack_stop_round",
                    "backdoor.eval_interval", "evaluation.eval_interval",
                    "evaluation.dump_logits_every", "evaluation.snapshot_rounds",
                    "meta.group", "meta.run_id"}
    # 有效轮对齐：flat 的轮数 / 停止轮 = G8 的 × edge_rounds（停止轮从 1 起、左闭右开）
    assert f["federation.n_rounds"] == g8["federation.n_rounds"] * g8["federation.edge_rounds"]
    assert f["backdoor.attack_stop_round"] - 1 == (g8["backdoor.attack_stop_round"] - 1) * 5
    assert f["backdoor.attack_stop_round"] == V.FLAT_STOP_ROUND
    # 评估点对齐：flat 每 5 轮一点 = G8 每云轮一点
    assert f["backdoor.eval_interval"] * f["federation.edge_rounds"] == \
        g8["backdoor.eval_interval"] * g8["federation.edge_rounds"]
    assert "evaluation.dump_logits_every" not in f and "evaluation.snapshot_rounds" not in f
