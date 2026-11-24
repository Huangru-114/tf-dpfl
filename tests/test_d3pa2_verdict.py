"""
P-a 追加探针（N-016 / D-140）：`harness/d3pa2_verdict.py` 的选择与判定。纯标准库 + d3_spec，本地秒级。

写于标定与正式数据之前：合成的 d3_pa2 JSON 覆盖选择规则与判定的每个分支、每道闸；底线读真实的 G0 metrics。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))
import d3pa2_verdict as V                                    # noqa: E402
from analysis import d3_spec as S                            # noqa: E402

EN = ROOT / "experiments/defense/edge-native"
CFG = S.load_pa2_config(EN / "d3pa2_configs.yaml")
FLOOR = {42: 0.0974, 43: 0.0633, 44: 0.0731}
GRID = [b["id"] for b in S.pa2_grid(CFG)]


def _asr(f, seed, base=1.0):
    """让 f 恰好等于给定值的臂 ASR（底线取真实 G0 值的 4 位小数，f 的容差足够）。"""
    return base - f * (base - V.floor_e0(V.G0_RESULTS, seed))


def _summ(asr, acc):
    return {"asr": asr, "margin_p50": 1.0, "acc": acc, "n": 15}


def _arm(asr, acc, diverged=False):
    if diverged:
        return {"diverged": True, "harden": {"steps": 3, "diverged_at": 3}}
    return {"diverged": False, "main": _summ(asr, acc), "fixed": _summ(asr, acc),
            "harden": {"steps": 75, "lr": 0.03, "dw_body_absmax": 0.1}}


def _round(budgets: dict, rep=True):
    """budgets = {档 id: {臂: (asr, acc[, diverged])}}。"""
    return {"reproduce": {"pass": rep, "reasons": [] if rep else ["edge 0 差 0.5"]},
            "baseline": {"diverged": False, "main": _summ(1.0, 0.85), "fixed": _summ(1.0, 0.85)},
            "budgets": {bid: {"budget": {"id": bid}, "arms": {a: _arm(*v) for a, v in arms.items()}}
                        for bid, arms in budgets.items()}}


def calib(f=None, acc=None, div=(), rep_fail=(), drop=(), cfg=CFG, sha=None):
    """f / acc：{(档 id, 轮): 值}，缺省 f = 0、acc = 0.85（精度降幅 0）。"""
    f, acc = f or {}, acc or {}
    seed = S.seed_of(cfg["calibrate"]["run"])
    rounds = {}
    for t in cfg["calibrate"]["rounds"]:
        b = {}
        for bid in GRID:
            if (bid, t) in drop:
                continue
            b[bid] = {"body": (_asr(f.get((bid, t), 0.0), seed), acc.get((bid, t), 0.85), (bid, t) in div)}
        rounds[str(t)] = _round(b, rep=t not in rep_fail)
    return {"schema": S.SCHEMA, "tool": "d3_pa2", "stage": "calibrate", "run_id": cfg["calibrate"]["run"],
            "config_sha": S.content_sha(cfg) if sha is None else sha, "rounds": rounds}


def frozen(bid="lr0.03-e5-probe", label="selected", sha=None):
    return {"label": label, "selected": bid if label == "selected" else None,
            "config_sha": S.content_sha(CFG) if sha is None else sha, "calibration_sha": "abc"}


def mains(bid="lr0.03-e5-probe", f4=0.8, fbody=0.9, f03=0.3, acc4=0.84, accb=0.84, div=(), rep_fail=(),
          per=None, selected=None, fsha="fz"):
    """per = {(run, 轮): {"stage4": f, "body": f, ...}} 覆盖缺省值。"""
    out = {}
    for run in CFG["main"]["runs"]:
        seed = S.seed_of(run)
        rounds = {}
        for t in CFG["main"]["rounds"]:
            o = (per or {}).get((run, t), {})
            arms = {"stage4": (_asr(o.get("stage4", f4), seed), acc4, (run, t, "stage4") in div),
                    "stem_stage3": (_asr(o.get("stem_stage3", f03), seed), 0.83),
                    "body": (_asr(o.get("body", fbody), seed), accb, (run, t, "body") in div)}
            rounds[str(t)] = _round({bid: arms}, rep=(run, t) not in rep_fail)
        out[run] = {"schema": S.SCHEMA, "tool": "d3_pa2", "stage": "main", "run_id": run,
                    "config_sha": S.content_sha(CFG), "frozen_selected": selected or bid, "frozen_sha": fsha,
                    "rounds": rounds}
    return out


# ── 阈值与预注册原文 ──────────────────────────────────────────────────────────
def test_thresholds_match_n016_and_the_config():
    txt = (EN / "FINDINGS.md").read_text(encoding="utf-8")
    sec = txt[txt.index("### N-016"):]
    assert (V.REMOVE, V.EARLY, V.ACC_GATE, V.TOL) == (0.5, 0.25, 0.05, 1e-9)
    assert "两轮都 f ≥ 0.5、两轮精度降幅都 ≤ 0.05" in sec
    assert "6 个 f（3 种子 × 2 轮）都 ≥ 0.5" in sec and "都 < 0.25 → `last_block_insufficient`" in sec
    assert "79d915ec7703b4ca" in sec and "N-016" in V.PREREG
    V.check_thresholds(CFG)
    bad = copy.deepcopy(CFG)
    bad["thresholds"]["remove"] = 0.6
    with pytest.raises(ValueError):
        V.check_thresholds(bad)


def test_f_value():
    assert V.f_value(1.0, 0.5, 0.0) == 0.5 and V.f_value(1.0, 1.0, 0.1) == 0.0
    assert V.f_value(0.1, 0.05, 0.1) is None and V.f_value(None, 0.5, 0.1) is None


# ── ① 选择 ───────────────────────────────────────────────────────────────────
def test_select_picks_the_smallest_eligible_budget():
    f = {(b, t): 0.9 for b in ("lr0.01-e20-reinit", "lr0.03-e20-probe") for t in (15, 60)}
    res = V.select(calib(f=f), CFG)
    assert res["label"] == "selected" and res["selected"] == "lr0.01-e20-reinit"   # 网格位置 5 先于 8
    assert len(res["table"]) == 12 and res["floor_e0"] == 0.0974


def test_select_needs_both_rounds():
    f = {("lr0.01-e5-probe", 15): 0.9, ("lr0.1-e20-probe", 15): 0.9, ("lr0.1-e20-probe", 60): 0.9}
    res = V.select(calib(f=f), CFG)
    assert res["selected"] == "lr0.1-e20-probe"
    row = next(r for r in res["table"] if r["id"] == "lr0.01-e5-probe")
    assert not row["eligible"] and any("第 60 轮 f" in w for w in row["why_not"])


def test_select_accuracy_gate_and_divergence_disqualify():
    f = {(b, t): 0.9 for b in ("lr0.01-e5-probe", "lr0.01-e5-reinit", "lr0.03-e5-probe") for t in (15, 60)}
    acc = {("lr0.01-e5-probe", 60): 0.79}                    # 精度降 0.06
    res = V.select(calib(f=f, acc=acc, div={("lr0.01-e5-reinit", 15)}), CFG)
    assert res["selected"] == "lr0.03-e5-probe"


def test_select_boundary_values_are_eligible():
    f = {("lr0.01-e5-probe", t): 0.5 for t in (15, 60)}
    acc = {("lr0.01-e5-probe", t): 0.80 for t in (15, 60)}   # 降 0.05（浮点 0.0500…）
    assert V.select(calib(f=f, acc=acc), CFG)["selected"] == "lr0.01-e5-probe"


def test_values_just_below_a_threshold_are_not_rounded_up():
    """f = 0.49996、精度降 0.05004：舍入到 4 位会变成 0.5 / 0.05 而过线 —— 判定必须用原值。"""
    f = {("lr0.01-e5-probe", t): 0.49996 for t in (15, 60)}
    f.update({("lr0.01-e5-reinit", t): 0.9 for t in (15, 60)})
    acc = {("lr0.01-e5-reinit", 60): 0.85 - 0.05004}
    f.update({("lr0.03-e5-probe", t): 0.9 for t in (15, 60)})
    res = V.select(calib(f=f, acc=acc), CFG)
    assert res["selected"] == "lr0.03-e5-probe"
    per = {("SNAP__collocated__s44", 60): {"body": 0.49996}}
    assert V.judge(CFG, frozen(), mains(per=per), frozen_sha="fz")["label"] == "uninformative"


def test_select_nothing_removes_is_not_removable():
    res = V.select(calib(), CFG)
    assert res["label"] == "not_removable" and res["selected"] is None


@pytest.mark.parametrize("kw,label", [
    ({"rep_fail": (60,)}, "invalid"),
    ({"sha": "0" * 16}, "invalid"),
    ({"drop": {("lr0.1-e20-reinit", 15)}}, "insufficient"),
])
def test_select_gates(kw, label):
    assert V.select(calib(**kw), CFG)["label"] == label


def test_select_without_a_file_is_insufficient():
    assert V.select(None, CFG)["label"] == "insufficient"


# ── ② 判定 ───────────────────────────────────────────────────────────────────
def test_judge_last_block_suffices():
    res = V.judge(CFG, frozen(), mains(), frozen_sha="fz")
    assert res["label"] == "last_block_suffices" and len(res["rows"]) == 6
    assert res["per_round"] == {"15": "last_block_suffices", "60": "last_block_suffices"}
    assert res["rows"][0]["arms"]["stem_stage3"]["f"] == pytest.approx(0.3, abs=1e-3)


def test_judge_last_block_insufficient_and_inconclusive():
    assert V.judge(CFG, frozen(), mains(f4=0.1), frozen_sha="fz")["label"] == "last_block_insufficient"
    per = {("SNAP__collocated__s44", 60): {"stage4": 0.4}}
    assert V.judge(CFG, frozen(), mains(per=per), frozen_sha="fz")["label"] == "inconclusive"


def test_judge_positive_control_failure_is_uninformative():
    per = {("SNAP__collocated__s43", 15): {"body": 0.3}}
    res = V.judge(CFG, frozen(), mains(per=per), frozen_sha="fz")
    assert res["label"] == "uninformative" and "阳性对照" in res["reasons"][0]


@pytest.mark.parametrize("kw,needle", [
    ({"acc4": 0.79}, "精度降"),
    ({"accb": 0.79}, "精度降"),
    ({"div": {("SNAP__collocated__s42", 15, "body")}}, "发散"),
    ({"rep_fail": {("SNAP__collocated__s44", 60)}}, "重现闸"),
    ({"selected": "lr0.1-e20-probe"}, "冻结清单选的"),
    ({"fsha": "other"}, "冻结清单 sha"),
])
def test_judge_gates(kw, needle):
    res = V.judge(CFG, frozen(), mains(**kw), frozen_sha="fz")
    assert res["label"] == "invalid" and any(needle in r for r in res["reasons"])


def test_judge_stem_stage3_accuracy_is_not_a_gate():
    """「只训前面几层」只描述：它的精度降多少都不让结果 invalid。"""
    m = mains()
    for j in m.values():
        for R in j["rounds"].values():
            R["budgets"]["lr0.03-e5-probe"]["arms"]["stem_stage3"]["main"]["acc"] = 0.5
    assert V.judge(CFG, frozen(), m, frozen_sha="fz")["label"] == "last_block_suffices"


def test_judge_frozen_states():
    assert V.judge(CFG, None, {})["label"] == "insufficient"
    assert V.judge(CFG, frozen(label="not_removable"), {})["label"] == "not_removable"
    assert V.judge(CFG, frozen(sha="0" * 16), mains(), frozen_sha="fz")["label"] == "invalid"
    m = mains()
    m["SNAP__collocated__s43"] = None
    assert V.judge(CFG, frozen(), m, frozen_sha="fz")["label"] == "insufficient"


# ── CLI ─────────────────────────────────────────────────────────────────────
def test_cli_select_then_judge(tmp_path, capsys):
    f = {("lr0.03-e5-probe", t): 0.9 for t in (15, 60)}
    cal = tmp_path / "SNAP__collocated__s42.calibrate.json"
    cal.write_text(json.dumps(calib(f=f)))
    fz = tmp_path / "frozen.json"
    assert V.main(["select", str(cal), "--out", str(fz)]) == 0
    sel = json.loads(fz.read_text())
    assert sel["label"] == "selected" and sel["selected"] == "lr0.03-e5-probe" and sel["calibration_sha"] == V.sha12(cal)
    d = tmp_path / "d3pa2"
    d.mkdir()
    for run, j in mains(fsha=V.sha12(fz)).items():
        (d / f"{run}.main.json").write_text(json.dumps(j))
    out = tmp_path / "v.json"
    assert V.main(["judge", "--d3pa2", str(d), "--frozen", str(fz), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["label"] == "last_block_suffices"
    assert "[d3pa2] judge: last_block_suffices" in capsys.readouterr().out


def test_verdict_does_not_import_tensorflow():
    src = (ROOT / "harness/d3pa2_verdict.py").read_text(encoding="utf-8")
    assert "tensorflow" not in src.split('"""', 2)[2]
