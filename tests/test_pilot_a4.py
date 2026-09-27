"""
tests/test_pilot_a4.py  —  A4 的可行性 pilot：登记表、提交脚本、预注册判定

  · pilot 的每个格子 = P1 同格配置 + 对齐模板（+ 该组唯一的 set），别无其他差异；
  · submit_pilot.sh 只读 pilot 自己的 INDEX（上一级 submit.sh 的 D-006 门槛不被绕过）；
  · harness/pilot_a4.py 的判据与 D-029 / D-031 / D-028 逐字一致，通过 / 不通过两侧都测。

纯 python，不 import TF。
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import registry as R          # noqa: E402
import pilot_a4 as P          # noqa: E402
import alignment as AL        # noqa: E402

PILOT = ROOT / "experiments/attack/hfl-mechanism/pilot/registry.yaml"
P1 = ROOT / "experiments/attack/hfl-propagation"


def _declared(group, cell):
    reg = R.Registry(PILOT)
    run = next(r for r in reg.runs() if r["group"] == group and r["cell"] == cell)
    cfg = reg.declared_config(run)
    cfg.pop("meta")
    return cfg


def _p1_plus_template(name):
    cfg = yaml.safe_load((P1 / name).read_text(encoding="utf-8"))
    cfg["backdoor"]["malicious_strategy"] = "badpfl"      # P1 由 run_exp3.sh 的 CLI 传
    R.deep_merge(cfg, AL.template_nested())
    cfg["seed"] = 42
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# 登记表
# ══════════════════════════════════════════════════════════════════════════
def test_pilot_has_the_planned_runs():
    runs = R.Registry(PILOT).runs()
    assert sorted((r["group"], r["cell"]) for r in runs) == sorted(
        [("D029", "flat"), ("D029", "2edge_distributed"), ("A26", "flat"),
         ("A26", "2edge_distributed"), ("DET", "rep1"), ("DET", "rep2"),
         ("G2P", "4edge_distributed"), ("G2P", "10edge_distributed")])     # G2P：D-047
    reg = R.Registry(PILOT)
    assert reg.protocol == "P1"
    assert all(not (g.get("requires")) for g in reg.groups.values())


@pytest.mark.parametrize("cell,p1", [("flat", "flat_baseline.yaml"),
                                     ("2edge_distributed", "2edge_distributed.yaml")])
def test_d029_cells_are_p1_cells_plus_template(cell, p1):
    """D-029 与 P1 比 pm_acc：除了对齐模板，配置必须与 P1 同格一字不差。"""
    assert _declared("D029", cell) == _p1_plus_template(p1)


@pytest.mark.parametrize("cell", ["4edge_distributed", "10edge_distributed"])
def test_g2p_cells_are_p1_cells_plus_template(cell):
    """G2P 与旧 P1 比 T50：除了对齐模板，配置必须与 P1 同名格一字不差（与 D029 同一基配置）。"""
    assert _declared("G2P", cell) == _p1_plus_template(f"{cell}.yaml")


@pytest.mark.parametrize("cell", ["flat", "2edge_distributed"])
def test_a26_differs_from_d029_only_by_order(cell):
    a, d = _declared("A26", cell), _declared("D029", cell)
    assert a["training"].pop("fedrep_order") == "body_first"
    assert "fedrep_order" not in d["training"] and a == d


def test_det_pair_is_the_same_config():
    a, b = _declared("DET", "rep1"), _declared("DET", "rep2")
    assert a == b and a["federation"]["n_rounds"] == 5 and a["stopping"] == {}


def test_pilot_configs_are_full_p2_code_path():
    """用户拍板：pilot 跑完整 P2 代码路径（模板全开），口径版本仍记 P1。"""
    for g, c in (("D029", "flat"), ("A26", "2edge_distributed"), ("DET", "rep1")):
        assert AL.template_state(_declared(g, c)) == "p2"


def test_materialized_index_matches_the_registry():
    reg = R.Registry(PILOT)
    idx = (reg.configs_dir / "INDEX.tsv").read_text(encoding="utf-8").splitlines()
    got = {ln.split("\t")[0]: ln.split("\t")[4] for ln in idx[1:]}
    from utils.provenance import config_sha
    want = {r["run_id"]: config_sha(reg.declared_config(r)) for r in reg.runs()}
    assert got == want, "pilot/configs 过期了：改了登记表 / 模板 / P1 基配置之后要重新 materialize"


def test_submit_pilot_only_reads_the_pilot_index():
    src = (PILOT.parent / "submit_pilot.sh").read_text(encoding="utf-8")
    assert 'INDEX="$HERE/configs/INDEX.tsv"' in src
    assert "../configs" not in src and "AUDIT" not in src.split("set -euo pipefail")[1]
    assert 'JOB="$HERE/../cell.sbatch"' in src


# ══════════════════════════════════════════════════════════════════════════
# 预注册判定
# ══════════════════════════════════════════════════════════════════════════
def _m(pm_stale=None, pm=None, asr=None, stop="converged", n=12, checks=None):
    acc = [{"round": i, "pm_acc": pm, "pm_acc_stale": pm_stale} for i in range(1, n + 1)]
    rounds = [{"round": i, "local_benign_asr": asr} for i in range(1, n + 1)]
    return {"exit_code": 0, "run": {"stop_reason": stop, "stopped_at_effective": 200},
            "acc_rounds": acc, "rounds": rounds,
            "checksums": [{"round": r, "global": h} for r, h in (checks or [])]}


def test_d029_thresholds_are_the_decision_values():
    assert P.D029_PM_FLOOR == {"flat": 0.7367, "2edge_distributed": 0.7297}
    assert (P.D031_PM_TOL, P.D031_ASR_TOL) == (0.006, 0.07)


def test_d029_pass_and_fail():
    assert P.judge_d029(_m(pm_stale=0.7400), "flat")["verdict"] == "pass"
    assert P.judge_d029(_m(pm_stale=0.7360), "flat")["verdict"] == "fail"
    assert P.judge_d029(_m(pm_stale=0.7300), "2edge_distributed")["verdict"] == "pass"
    assert P.judge_d029(_m(pm_stale=0.80, stop="cap_reached"), "flat")["verdict"] == "fail"
    r = P.judge_d029(_m(pm=0.80), "flat")                 # 只有 fresh 列、没有陈旧列
    assert r["verdict"] == "fail" and "pm_acc_stale" in r["reasons"][0]
    assert P.judge_d029(None, "flat")["verdict"] == "missing"


def test_d029_uses_the_stale_column_not_the_fresh_one():
    r = P.judge_d029(_m(pm_stale=0.70, pm=0.90), "flat")
    assert r["verdict"] == "fail" and r["pm_acc_stale_last10"] == pytest.approx(0.70)


def test_d031_same_and_different():
    h = _m(pm=0.740, asr=0.70)
    assert P.judge_d031(h, _m(pm=0.745, asr=0.76), "flat")["verdict"] == "same"
    assert P.judge_d031(h, _m(pm=0.747, asr=0.70), "flat")["verdict"] == "different"
    assert P.judge_d031(h, _m(pm=0.740, asr=0.78), "flat")["verdict"] == "different"
    assert P.judge_d031(h, None, "flat")["verdict"] == "missing"


def test_d031_uses_last_ten_points():
    h = _m(pm=0.74, asr=0.70)
    b = _m(pm=0.74, asr=0.70)
    b["rounds"][0]["local_benign_asr"] = 0.0          # 早期的点不进末 10 点
    assert P.judge_d031(h, b, "flat")["verdict"] == "same"


def test_det_pass_fail_missing():
    ok = [(r, f"{r:012x}") for r in range(1, 6)]
    assert P.judge_det(_m(checks=ok), _m(checks=ok))["verdict"] == "pass"
    bad = ok[:2] + [(3, "ffffffffffff")] + ok[3:]
    r = P.judge_det(_m(checks=ok), _m(checks=bad))
    assert r["verdict"] == "fail" and r["first_diverging_round"] == 3
    assert P.judge_det(_m(checks=ok[:3]), _m(checks=ok))["verdict"] == "missing"


def test_overall_needs_both_cells():
    good = _m(pm_stale=0.75, pm=0.74, asr=0.7,
              checks=[(r, "a" * 12) for r in range(1, 6)])
    ms = {("D029", "flat"): good, ("D029", "2edge_distributed"): good,
          ("A26", "flat"): good, ("A26", "2edge_distributed"): good,
          ("DET", "rep1"): good, ("DET", "rep2"): good}
    res = P.judge_all(ms)
    assert res["D-029"]["overall"] == "pass" and res["D-031"]["overall"] == "same"
    assert res["A15-determinism"]["verdict"] == "pass"
    ms2 = dict(ms)
    ms2[("D029", "2edge_distributed")] = _m(pm_stale=0.70, pm=0.74, asr=0.7)
    assert P.judge_all(ms2)["D-029"]["overall"] == "fail"
    ms3 = dict(ms)
    ms3.pop(("A26", "flat"))
    assert P.judge_all(ms3)["D-031"]["overall"] == "missing"


# ══════════════════════════════════════════════════════════════════════════
# 有效性闸（D-044）：run 能不能算数，在判定之前、不改阈值
# ══════════════════════════════════════════════════════════════════════════
# 第一轮 pilot（e2ee9ca）的真实形状：恶意端的异常被吞、被踢出聚合
_SWALLOWED = [{"client_id": 49, "error": "UnimplementedError ..."},
              {"edge_id": 0, "dropped": [49]}, {"edge_id": 1, "dropped": [87]}]


def _broken(m, *, exit_code=0, failures=None):
    m = copy.deepcopy(m)
    m["exit_code"] = exit_code
    m["client_failures"] = failures or []
    return m


def test_clean_run_has_no_invalid_reasons():
    assert P.invalid_reasons(_m(pm_stale=0.75)) == []
    assert P.invalid_reasons(_broken(_m(pm_stale=0.75))) == []        # 空列表 = 有效


def test_swallowed_client_failure_makes_d029_invalid_not_pass():
    """反向锚点：同样的数字，client_failures 为空时 pass —— 闸是唯一的差别。"""
    good = _m(pm_stale=0.80)
    assert P.judge_d029(good, "flat")["verdict"] == "pass"
    r = P.judge_d029(_broken(good, failures=_SWALLOWED), "flat")
    assert r["verdict"] == "invalid" and "[49, 87]" in r["reasons"][0]


def test_crash_is_invalid_not_a_feasibility_fail():
    r = P.judge_d029(_broken(_m(pm_stale=0.27), exit_code=1), "flat")
    assert r["verdict"] == "invalid" and "exit_code=1" in r["reasons"][0]


def test_d031_is_invalid_if_either_arm_is():
    h = _m(pm=0.740, asr=0.0)
    b = _m(pm=0.741, asr=0.0)                      # 没有攻击者：两臂 ASR 都≈0 → 会判 same
    assert P.judge_d031(h, b, "flat")["verdict"] == "same"
    r = P.judge_d031(h, _broken(b, failures=_SWALLOWED), "flat")
    assert r["verdict"] == "invalid" and r["reasons"][0].startswith("body_first: ")
    assert P.judge_d031(_broken(h, exit_code=1), b, "flat")["verdict"] == "invalid"


def test_det_is_invalid_if_either_run_is():
    ok = [(r, "a" * 12) for r in range(1, 6)]
    a = _m(checks=ok)
    assert P.judge_det(a, _broken(a, failures=_SWALLOWED))["verdict"] == "invalid"


def test_overall_reports_invalid():
    good = _m(pm_stale=0.75, pm=0.74, asr=0.7,
              checks=[(r, "a" * 12) for r in range(1, 6)])
    ms = {("D029", "flat"): good, ("D029", "2edge_distributed"): good,
          ("A26", "flat"): good, ("A26", "2edge_distributed"): good,
          ("DET", "rep1"): good, ("DET", "rep2"): good}
    ms[("D029", "flat")] = _broken(good, failures=_SWALLOWED)
    res = P.judge_all(ms)
    assert res["D-029"]["overall"] == "invalid" and res["D-031"]["overall"] == "invalid"
    assert res["A15-determinism"]["verdict"] == "pass"


# ══════════════════════════════════════════════════════════════════════════
# G2P：G2 一致性复测（D-047）
# ══════════════════════════════════════════════════════════════════════════
def _t(t50, edge_rounds=1, grid=5, n=20):
    """一条 local_benign_asr 轨迹：在有效轮 t50 处线性越过 0.5（插值后恰好是 t50）。"""
    rounds = []
    for i in range(1, n + 1):
        eff = i * grid
        v = min(0.99, max(0.0, 0.5 + (eff - t50) * 0.01))
        rounds.append({"round": eff // edge_rounds, "local_benign_asr": v, "global_asr": v})
    return {"exit_code": 0, "run": {"edge_rounds": edge_rounds}, "rounds": rounds}


def _g2p(flat, e2, e4, e10, er=5):
    return {"flat": _t(flat), "2edge_distributed": _t(e2, er),
            "4edge_distributed": _t(e4, er), "10edge_distributed": _t(e10, er)}


P1_LIKE = _g2p(57.4, 46.8, 22.8, 85.5)          # 0.815 / 0.397 / 1.49


def test_g2p_rule_is_the_decision_values():
    assert P.G2P_BAND == (0.9, 1.1) and P.G2P_THETA == 0.5


def test_g2p_t50_is_interpolated_on_effective_rounds():
    assert P.t50(_t(33.4, edge_rounds=5)) == pytest.approx(33.4)


def test_g2p_consistent_when_every_ratio_is_on_the_same_side():
    r = P.judge_g2p(_g2p(45, 35, 20, 70), P1_LIKE)
    assert r["verdict"] == "consistent"
    assert r["cells"]["10edge_distributed"]["side_new"] == ">1"


def test_g2p_inconsistent_when_a_ratio_flips():
    r = P.judge_g2p(_g2p(45, 35, 20, 30), P1_LIKE)      # 10edge：P1 >1，新 <1
    assert r["verdict"] == "inconsistent"
    assert r["cells"]["10edge_distributed"]["agree"] is False


def test_g2p_band_near_one_is_not_a_disagreement():
    r = P.judge_g2p(_g2p(45, 35, 20, 47), P1_LIKE)      # 10edge：47/45 = 1.04 ≈ 1
    assert r["cells"]["10edge_distributed"]["side_new"] == "≈1"
    assert r["verdict"] == "consistent"


def test_g2p_censored_t50_is_undetermined_not_a_verdict():
    new = _g2p(45, 35, 20, 70)
    for row in new["10edge_distributed"]["rounds"]:
        row["local_benign_asr"] = 0.1                      # 从未越过 0.5
    assert P.judge_g2p(new, P1_LIKE)["verdict"] == "undetermined"


def test_g2p_missing_and_invalid():
    new = _g2p(45, 35, 20, 70)
    new["4edge_distributed"] = None
    assert P.judge_g2p(new, P1_LIKE)["verdict"] == "missing"
    new = _g2p(45, 35, 20, 70)
    new["4edge_distributed"]["client_failures"] = [{"client_id": 3, "error": "x"}]
    r = P.judge_g2p(new, P1_LIKE)
    assert r["verdict"] == "invalid" and r["reasons"][0].startswith("4edge_distributed: ")


def test_g2p_real_p1_reference_is_computed_not_copied():
    """反向锚点：旧 P1 的比值从 hfl-propagation/results 现算 —— 0.815 / 0.398 / 1.489。"""
    p1 = P.load_p1_reference()
    t = {c: P.t50(m) for c, m in p1.items()}
    ratios = [round(t[c] / t["flat"], 3) for c in P.G2P_HFL]
    assert ratios == [0.815, 0.398, 1.489]


def test_g2p_reuses_d029_for_flat_and_2edge():
    ms = {("D029", "flat"): _t(45), ("D029", "2edge_distributed"): _t(35, 5),
          ("G2P", "4edge_distributed"): _t(20, 5), ("G2P", "10edge_distributed"): _t(70, 5)}
    assert P.judge_all(ms, P1_LIKE)["G2P"]["verdict"] == "consistent"
