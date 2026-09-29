"""
tests/test_g5ab_verdict.py  —  G5AB（生成器语义 A / B 对比，D-079）的预注册判定

纯标准库，本地秒级。合成的 metrics 只含判定用到的字段。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g5ab_verdict as V                                        # noqa: E402


def _m(cell, arm, peak=0.6, dil=0.3, base=0.1, failures=None, **run_over):
    c = V.CELLS[cell]
    rounds = []
    for r in range(1, c["n_rounds"] + 1):
        if c["start"] <= r < c["stop"]:
            v = peak if r == c["stop"] - 1 else peak - 0.05     # 最大值在窗口末
        elif r in c["dilution"]:
            v = dil
        else:
            v = base
        rounds.append({"round": r, "local_benign_asr": v})
    run = {"attack_start_round": c["start"], "attack_stop_round": c["stop"],
           "generator_schedule": V.ARMS[arm]}
    run.update(run_over)
    return {"exit_code": 0, "client_failures": failures or [], "run": run, "rounds": rounds}


def _all(delta_peak=0.0, delta_dil=0.0, overrides=None):
    runs = {}
    for cell in V.CELLS:
        for seed in V.SEEDS:
            runs[(cell, "A", seed)] = _m(cell, "A")
            dp = delta_peak(cell, seed) if callable(delta_peak) else delta_peak
            dd = delta_dil(cell, seed) if callable(delta_dil) else delta_dil
            runs[(cell, "B", seed)] = _m(cell, "B", peak=0.6 + dp, dil=0.3 + dd)
    runs.update(overrides or {})
    return runs


def test_quantities_read_the_window_max_and_the_three_dilution_points():
    q = V.quantities(_m("t140", "A", peak=0.7, dil=0.25), "t140")
    assert q["peak"] == 0.7 and q["n_peak_points"] == 4
    assert q["dilution"] == 0.25 and q["n_dilution_points"] == 3


def test_cells_match_the_g5_windows():
    """t20：投毒有效轮 21–40 = cloud 5–8；t140：141–160 = cloud 29–32；长度 t0+75；稀释 t0+65/70/75。"""
    for t0, cell in ((20, "t20"), (140, "t140")):
        c = V.CELLS[cell]
        assert c["start"] == t0 // 5 + 1 and c["stop"] == c["start"] + 4
        assert c["n_rounds"] == (t0 + 75) // 5
        assert c["dilution"] == tuple((t0 + d) // 5 for d in (65, 70, 75))


@pytest.mark.parametrize("dp,dd,want", [
    (0.0, 0.0, "insensitive"),
    (0.09, -0.09, "insensitive"),                        # 都 < 0.10
    (0.0, 0.10, "sensitive"),                            # 稀释在两个 seed 上都 +0.10（边界含）
    (lambda c, s: 0.20 if c == "t140" else 0.0, 0.0, "sensitive"),   # 只有 t140 的峰值差 → 成熟度混杂
    (lambda c, s: 0.15 if s == 42 else 0.0, 0.0, "user_decides"),    # 只有一个 seed
    (lambda c, s: 0.15 if s == 42 else -0.15, 0.0, "user_decides"),  # 两个 seed 反号
])
def test_verdict_boundaries(dp, dd, want):
    assert V.judge(_all(dp, dd))["overall"] == want


def test_mean_peak_delta_is_reported_per_cell():
    res = V.judge(_all(lambda c, s: 0.20 if c == "t140" else 0.02))
    assert res["mean_peak_delta"] == {"t20": 0.02, "t140": 0.2}


@pytest.mark.parametrize("over", [
    {"generator_schedule": "window"},                    # B 臂实际跑成了 A
    {"attack_start_round": None},                        # 起点没生效
    {"attack_stop_round": 31},
])
def test_switches_that_did_not_take_effect_make_it_invalid(over):
    runs = _all(overrides={("t20", "B", 42): _m("t20", "B", **over)})
    assert V.judge(runs)["overall"] == "invalid"


def test_swallowed_client_failures_make_it_invalid():
    runs = _all(overrides={("t140", "A", 43): _m("t140", "A", failures=[{"client_id": 7}])})
    assert V.judge(runs)["overall"] == "invalid"


def test_short_run_is_invalid():
    m = _m("t140", "A")
    m["rounds"] = m["rounds"][:40]
    assert V.judge(_all(overrides={("t140", "A", 42): m}))["overall"] == "invalid"


def test_missing_and_insufficient():
    assert V.judge({})["overall"] == "missing"
    runs = _all()
    runs[("t20", "B", 43)] = None
    assert V.judge(runs)["overall"] == "insufficient"


def test_on_disk_results_give_a_defined_verdict():
    overall = V.judge(V.load())["overall"]
    assert overall in {"insensitive", "sensitive", "user_decides", "insufficient",
                       "invalid", "missing"}


# ── 配置层面的前提 ─────────────────────────────────────────────────────────────
yaml = pytest.importorskip("yaml")
CONFIGS = ROOT / "experiments/attack/hfl-mechanism/configs"


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _cfg(name):
    return _flat(yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8")))


@pytest.mark.parametrize("cell", list(V.CELLS))
@pytest.mark.parametrize("seed", V.SEEDS)
def test_a_and_b_differ_only_in_the_generator_schedule(cell, seed):
    a, b = _cfg(f"G5AB__{cell}-A__s{seed}"), _cfg(f"G5AB__{cell}-B__s{seed}")
    diff = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert diff == {"backdoor.generator_schedule", "meta.run_id"}
    c = V.CELLS[cell]
    assert (a["backdoor.attack_start_round"], a["backdoor.attack_stop_round"],
            a["federation.n_rounds"]) == (c["start"], c["stop"], c["n_rounds"])


@pytest.mark.parametrize("seed", V.SEEDS)
def test_g5ab_is_g0_random_plus_the_window(seed):
    """B 臂在窗口之前 = ρ=0 影子攻击者 → 与 G0-random 同 seed 只差投毒率、窗口与长度（checksum 前提）。"""
    b, g0 = _cfg(f"G5AB__t140-B__s{seed}"), _cfg(f"G0__random__s{seed}")
    diff = {k for k in b.keys() | g0.keys() if b.get(k) != g0.get(k)}
    assert diff == {"backdoor.poison_ratio", "backdoor.attack_start_round",
                    "backdoor.attack_stop_round", "backdoor.generator_schedule",
                    "federation.n_rounds", "meta.group", "meta.run_id"}
    assert g0["backdoor.poison_ratio"] == 0.0 and b["backdoor.poison_ratio"] == 0.2
