"""
tests/test_g5_verdict.py  —  G5（3.3 收敛门控）的预注册判定（D-081）

纯标准库（+ yaml 读登记表），本地秒级。合成 metrics 只含判定用到的字段。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g5_verdict as V                                           # noqa: E402

REGISTRY = ROOT / "experiments/attack/hfl-mechanism/registry.yaml"


def _g5(t0, *, peak, floor_level=0.05, base=0.05, dil=0.08, failures=None, **run_over):
    c = V.CELLS[t0]
    rounds = []
    for r in range(1, c["n_rounds"] + 1):
        if r in c["window"]:
            v = peak if r == c["window"][-1] else peak - 0.1        # 最大值在窗口末
        elif r in c["dilution"]:
            v = dil
        else:
            v = base
        rounds.append({"round": r, "local_benign_asr": v})
    run = {"attack_start_round": c["start"], "attack_stop_round": c["stop"],
           "generator_schedule": "window", "poison_ratio": 0.2}
    run.update(run_over)
    return {"exit_code": 0, "client_failures": failures or [], "run": run, "rounds": rounds,
            "checksums": [{"round": r, "global": f"{t0}-{r}"} for r in range(1, c["n_rounds"] + 1)]}


def _g0(level=0.05, n=60, rho=0.0):
    return {"exit_code": 0, "client_failures": [], "run": {"poison_ratio": rho},
            "rounds": [{"round": r, "local_benign_asr": level} for r in range(1, n + 1)]}


def _world(peaks_by_seed, **kw):
    """peaks_by_seed：{seed: [5 个 t0 的峰值]}。"""
    g5 = {(t0, s): _g5(t0, peak=p, **kw) for s, ps in peaks_by_seed.items() for t0, p in zip(V.T0S, ps)}
    g0 = {s: _g0() for s in peaks_by_seed}
    return g5, g0


# ── 格子常数 ──────────────────────────────────────────────────────────────────
def test_cells_match_the_registry():
    yaml = pytest.importorskip("yaml")
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    g5 = reg["groups"]["G5"]
    assert g5["set"]["backdoor.generator_schedule"] == V.SCHEDULE
    assert sorted(g5["seeds"]) == list(V.SEEDS)
    cells = {c["cell"]: c["set"] for c in g5["cells"]}
    assert sorted(cells, key=lambda c: int(c[1:])) == [f"t{t}" for t in V.T0S]
    for t0 in V.T0S:
        s, c = cells[f"t{t0}"], V.CELLS[t0]
        assert (s["backdoor.attack_start_round"], s["backdoor.attack_stop_round"],
                s["federation.n_rounds"]) == (c["start"], c["stop"], c["n_rounds"])
        assert len(c["window"]) == 4 and c["dilution"][-1] == c["n_rounds"]


# ── 统计原语 ──────────────────────────────────────────────────────────────────
def test_ranks_average_ties():
    assert V.ranks([10, 20, 20, 5]) == [2.0, 3.5, 3.5, 1.0]


@pytest.mark.parametrize("y,want", [
    ([1, 2, 3, 4, 5], 1.0),
    ([5, 4, 3, 2, 1], -1.0),
    ([1, 3, 2, 5, 4], 0.8),                 # 1 − 6·Σd²/(n(n²−1)) = 1 − 6·4/120
    ([1, 1, 2, 2, 3], 0.9487),              # 并列：秩 [1.5,1.5,3.5,3.5,5] 的 Pearson = 9/√90
])
def test_spearman_values(y, want):
    assert V.spearman([20, 60, 100, 140, 180], y) == pytest.approx(want, abs=1e-4)


def test_spearman_constant_series_is_none():
    assert V.spearman([1, 2, 3, 4, 5], [0.3] * 5) is None


# ── 量的定义 ──────────────────────────────────────────────────────────────────
def test_peak_is_window_max_minus_same_round_floor():
    m = _g5(100, peak=0.6, dil=0.12)
    g0 = _g0()
    for r in V.CELLS[100]["window"]:                 # 窗口内 floor = 0.05 / 0.07 / 0.09 / 0.11 → 均值 0.08
        g0["rounds"][r - 1]["local_benign_asr"] = 0.05 + 0.02 * V.CELLS[100]["window"].index(r)
    q = V.quantities(m, g0, 100)
    assert q["peak"] == 0.6 and q["floor_window"] == pytest.approx(0.08)
    assert q["peak_excess"] == pytest.approx(0.52) and q["n_window_points"] == 4
    assert q["dilution_excess"] == pytest.approx(0.07)


# ── 判定 ──────────────────────────────────────────────────────────────────────
def test_all_seeds_rising_is_gated():
    g5, g0 = _world({42: [0.2, 0.3, 0.4, 0.5, 0.6], 43: [0.1, 0.3, 0.2, 0.5, 0.6],
                     44: [0.3, 0.35, 0.5, 0.45, 0.7]})
    res = V.judge(g5, g0)
    assert res["overall"] == "gated"
    assert res["rho_peak_ci"][0] == min(v["rho_peak"] for v in res["per_seed"].values())


def test_one_flat_or_falling_seed_is_not_gated():
    g5, g0 = _world({42: [0.2, 0.3, 0.4, 0.5, 0.6], 43: [0.1, 0.3, 0.2, 0.5, 0.6],
                     44: [0.6, 0.5, 0.4, 0.3, 0.2]})
    assert V.judge(g5, g0)["overall"] == "not_gated"


def test_all_seeds_falling_is_anti_gated():
    g5, g0 = _world({s: [0.6, 0.5, 0.45, 0.3, 0.2] for s in V.SEEDS})
    assert V.judge(g5, g0)["overall"] == "anti_gated"


def test_floor_that_rises_with_t0_can_cancel_a_raw_rise():
    """峰值原始值随 t0 上升，但同轮 floor 上升得更快 → excess 下降 → 不是门控。"""
    g5, g0 = _world({s: [0.30, 0.35, 0.40, 0.45, 0.50] for s in V.SEEDS})
    for s in V.SEEDS:
        for i, t0 in enumerate(V.T0S):
            for r in V.CELLS[t0]["window"]:
                g0[s]["rounds"][r - 1]["local_benign_asr"] = 0.05 + 0.1 * i
    assert V.judge(g5, g0)["overall"] == "anti_gated"


@pytest.mark.parametrize("over", [
    {"generator_schedule": "always"},             # 跑成了 B 臂
    {"attack_start_round": None},                 # 起点没生效
    {"attack_stop_round": 31},
])
def test_switches_that_did_not_take_effect_make_it_invalid(over):
    g5, g0 = _world({s: [0.2, 0.3, 0.4, 0.5, 0.6] for s in V.SEEDS})
    g5[(60, 43)] = _g5(60, peak=0.3, **over)
    assert V.judge(g5, g0)["overall"] == "invalid"


def test_invalid_floor_makes_it_invalid():
    g5, g0 = _world({s: [0.2, 0.3, 0.4, 0.5, 0.6] for s in V.SEEDS})
    g0[44] = _g0(rho=0.2)                          # 不是 ρ=0
    assert V.judge(g5, g0)["overall"] == "invalid"
    g0[44] = _g0(n=40)                             # 没跑到 t180 的稀释点
    assert V.judge(g5, g0)["overall"] == "invalid"


def test_swallowed_client_failure_is_invalid():
    g5, g0 = _world({s: [0.2, 0.3, 0.4, 0.5, 0.6] for s in V.SEEDS})
    g5[(180, 42)] = _g5(180, peak=0.6, failures=[{"client_id": 3}])
    assert V.judge(g5, g0)["overall"] == "invalid"


def test_missing_and_insufficient():
    assert V.judge({}, {})["overall"] == "missing"
    g5, g0 = _world({s: [0.2, 0.3, 0.4, 0.5, 0.6] for s in V.SEEDS})
    del g5[(140, 44)]
    assert V.judge(g5, g0)["overall"] == "insufficient"


def test_reproducibility_report_against_g5ab_arm_a():
    g5, g0 = _world({s: [0.2, 0.3, 0.4, 0.5, 0.6] for s in V.SEEDS})
    ref_same = _g5(20, peak=0.2)
    ref_diff = _g5(140, peak=0.5)
    ref_diff["checksums"][6]["global"] = "changed"
    rep = V.judge(g5, g0, {(20, 42): ref_same, (140, 43): ref_diff})["reproducibility_vs_G5AB_A"]
    by = {(r["cell"], r["seed"]): r for r in rep}
    assert by[("t20", 42)]["identical"] is True
    assert by[("t140", 43)]["identical"] is False and by[("t140", 43)]["first_diff_round"] == 7


def test_on_disk_results_give_a_defined_verdict():
    g5, g0, g5ab = V.load()
    assert V.judge(g5, g0, g5ab)["overall"] in {"gated", "anti_gated", "not_gated", "insufficient",
                                               "missing", "invalid"}


def test_real_g5ab_arm_a_and_g0_random_run_through_the_quantities():
    """真实数据的接线检查：把 G5AB-A（与 G5 的 t20 / t140 配置只差 meta）当作两格喂进去 → 两格有量、整体 insufficient。"""
    _, g0, g5ab = V.load()
    if not g5ab or any(g0.get(s) is None for s in (42, 43)):
        pytest.skip("G5AB / G0 的结果不在盘上")
    res = V.judge(g5ab, g0)
    assert res["overall"] in {"insufficient", "missing"} and not res["invalid"]
    got = {(p["t0"], p["seed"]): p for p in res["per_run"]}
    assert set(got) == set(g5ab)
    assert all(p["n_window_points"] == 4 and p["peak_excess"] is not None for p in got.values())
