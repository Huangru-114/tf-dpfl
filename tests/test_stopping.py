"""
tests/test_stopping.py  —  自适应轮数（地板 + 按需延长）

## 这个模块防的是什么

把一个**数据依赖的决策放进了控制流** —— 停早了、停晚了，跑出来的 metrics.json
与正常 run 长得一模一样。所以每条判据、每种退化、以及自描述行的上下游，
都要有断言。

## 参数的来源（不是拍的）

判据 A `thresholds_crossed`：三层 ASR × θ∈{0.25,0.5,0.75} 全部越过，
连续 2 点 ≥ θ 才算（ASR 逐点 σ≈0.09，单点跨越是噪声）。
实测满足于 2edge round 20 / 10edge round 28 —— 都 ≤ 地板 30。

判据 B `pm_acc_plateau`：末 10 点 OLS 斜率 < 0.0010/轮。
残差 σ≈0.0023 → 斜率 SE≈0.00024；2edge 实测 +0.00036（判平，1.5×SE），
10edge +0.00229（判未平，8.8×SE），两边各约 2 倍余量。

**第 1 条测试就是拿这两格真实轨迹回打这些数字。参数错了它立刻红。**

纯 stdlib + numpy，不 import TF，本地秒级。
"""

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
sys.path.insert(0, str(ROOT / "harness"))

from server.stopping import (StoppingRule, ASR_KEYS,          # noqa: E402
                             REASON_CONVERGED, REASON_CAP, REASON_COARSE)

CALIB = ROOT / "experiments" / "calibration" / "results"
CFG = dict(criteria=["thresholds_crossed", "pm_acc_plateau"],
           floor_effective=150, cap_effective=300,
           thetas=[0.25, 0.5, 0.75], debounce=2,
           pm_window=10, pm_slope_tol=0.0010)


def _feed(path, cfg=None, edge_rounds=5, n_rounds=60, stop_at_first=True):
    """把一份真实 metrics.json 的轨迹逐轮喂进规则，返回 (停止轮, decision, rule)。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    asr = {x["round"]: x for x in d["rounds"]}
    pm = {x["round"]: x.get("pm_acc") for x in d["acc_rounds"]}
    rule = StoppingRule(cfg if cfg is not None else CFG, edge_rounds, n_rounds)
    last = None
    for r in sorted(set(asr) | set(pm)):
        a = asr.get(r, {})
        sig = {"pm_acc": pm.get(r),
               "global_asr": a.get("global_asr"),
               "edge_asr_mean": a.get("edge_asr"),
               "local_asr_benign_mean": a.get("local_benign_asr")}
        dec = rule.update(r, sig)
        last = (r, dec)
        if dec.stop and stop_at_first:
            return r, dec, rule
    return (last[0] if last else None), (last[1] if last else None), rule


# ══════════════════════════════════════════════════════════════════════════
# 1. 反向锚点：真实轨迹必须回打出标定的数字
# ══════════════════════════════════════════════════════════════════════════

REF = CALIB / "local_epochs5_seed42.metrics.json"          # 2edge  HHI=0.50
PROBE = CALIB / "ceiling_10edge_ep5_seed42.metrics.json"   # 10edge HHI=0.10


@pytest.mark.parametrize("path,want_round", [(REF, 20), (PROBE, 28)])
def test_thresholds_crossed_at_the_measured_round(path, want_round):
    """
    判据 A 单独用时，两格分别在 round 20 / 28 满足 —— 都 ≤ 地板 30。
    这两个数是 `experiments/calibration/RESULTS.md` 里记的实测值。
    """
    if not Path(path).exists():
        pytest.skip(f"缺 {Path(path).name}")
    cfg = dict(CFG, criteria=["thresholds_crossed"], floor_effective=0)
    r, dec, _ = _feed(path, cfg)
    assert dec.stop and dec.reason == REASON_CONVERGED
    assert r == want_round, f"在 round {r} 满足，实测应为 {want_round}"


def test_pm_plateau_matches_the_measured_slopes():
    """
    判据 B 的分界必须落在两格之间：2edge 判**平**，10edge 判**未平**。
    实测斜率 +0.00036 vs +0.00229，阈值 0.0010 在两者中间，各约 2 倍余量。
    """
    for path, want_flat, approx in ((REF, True, 0.00036), (PROBE, False, 0.00229)):
        if not Path(path).exists():
            pytest.skip(f"缺 {Path(path).name}")
        _, _, rule = _feed(path, dict(CFG, cap_effective=10 ** 9), stop_at_first=False)
        slope = rule.pm_slope()
        assert slope is not None
        assert (abs(slope) < CFG["pm_slope_tol"]) is want_flat, (
            f"{Path(path).stem}: 斜率 {slope:.5f}，阈值 {CFG['pm_slope_tol']}，"
            f"应判{'平' if want_flat else '未平'}")
        assert abs(slope - approx) < 0.0005, f"斜率 {slope:.5f} 与实测 {approx} 差太多"


def test_both_criteria_together_extend_the_slow_cell():
    """
    **这条是整个改动的目的**：快格停在地板，慢格延长。
    2edge（pm_acc 已平）停在地板 30；10edge（pm_acc 还在爬）必须**继续跑**。
    """
    for path in (REF, PROBE):
        if not Path(path).exists():
            pytest.skip("缺标定数据")
    r_ref, dec_ref, _ = _feed(REF)
    assert dec_ref.stop and r_ref == 30, f"2edge 应停在地板 30，实际 {r_ref}"
    r_p, dec_p, _ = _feed(PROBE)
    assert not (dec_p and dec_p.stop), "10edge 的 pm_acc 还在爬，不该在 30 轮停"


# ══════════════════════════════════════════════════════════════════════════
# 2. 地板 / 上限 / 三种退化
# ══════════════════════════════════════════════════════════════════════════

def _synth(rule, n, asr=0.9, pm=0.74, jitter=0.0):
    """喂 n 轮「早就满足判据」的观测。"""
    for r in range(1, n + 1):
        v = pm + (jitter if r % 2 else -jitter)
        rule.update(r, {"pm_acc": v, **{k: asr for k in ASR_KEYS}})
    return rule


def test_never_stops_before_the_floor():
    """只延长不早停：地板之前**永不**停，哪怕两条判据都满足。"""
    rule = StoppingRule(CFG, edge_rounds=5, n_rounds=60)
    for r in range(1, 30):                      # effective 5..145 < 150
        dec = rule.update(r, {"pm_acc": 0.74, **{k: 0.99 for k in ASR_KEYS}})
        assert not dec.stop, f"在 round {r}（effective {r*5}）就停了，地板是 150"
    assert rule.update(30, {"pm_acc": 0.74, **{k: 0.99 for k in ASR_KEYS}}).stop


def test_cap_reached_is_censored_not_converged():
    """到 cap 仍未满足 → `cap_reached`。**它是 censored，不能读成「收敛在 cap」。**"""
    rule = StoppingRule(CFG, edge_rounds=5, n_rounds=60)
    dec = None
    for r in range(1, 61):                      # ASR 永远越不过 0.25
        dec = rule.update(r, {"pm_acc": 0.5 + 0.01 * r,
                              **{k: 0.1 for k in ASR_KEYS}})
        if dec.stop:
            break
    assert dec.stop and dec.reason == REASON_CAP
    assert dec.reason != REASON_CONVERGED


def test_grid_too_coarse_is_its_own_reason():
    """点数凑不够 pm_window → `grid_too_coarse`，既不是收敛也不是崩溃。"""
    rule = StoppingRule(dict(CFG, floor_effective=160, cap_effective=320),
                        edge_rounds=40, n_rounds=8)
    dec = None
    for r in range(1, 9):                       # 最多 8 个点 < pm_window=10
        dec = rule.update(r, {"pm_acc": 0.74, **{k: 0.99 for k in ASR_KEYS}})
        if dec.stop:
            break
    assert dec.stop and dec.reason == REASON_COARSE


# ══════════════════════════════════════════════════════════════════════════
# 3. 去抖 / None / 越阈语义
# ══════════════════════════════════════════════════════════════════════════

def test_single_spike_does_not_count_as_crossing():
    """去抖：单个点跨过 θ 不算越过（ASR 逐点 σ≈0.09）。"""
    rule = StoppingRule(dict(CFG, thetas=[0.75], criteria=["thresholds_crossed"],
                             floor_effective=0), 5, 60)
    rule.update(1, {k: 0.80 for k in ASR_KEYS})          # 单点尖峰
    assert rule.n_crossed == 0, "单点就判越过了 —— 去抖没生效"
    rule.update(2, {k: 0.10 for k in ASR_KEYS})          # 掉回去
    assert rule.n_crossed == 0
    rule.update(3, {k: 0.80 for k in ASR_KEYS})
    rule.update(4, {k: 0.80 for k in ASR_KEYS})          # 连续 2 点
    assert rule.n_crossed == len(ASR_KEYS)


def test_crossing_is_sticky():
    """越过之后掉回去不撤销 —— 判据问的是「有没有达到过」。"""
    rule = StoppingRule(dict(CFG, thetas=[0.5], criteria=["thresholds_crossed"],
                             floor_effective=0), 5, 60)
    for _ in range(2):
        rule.update(1, {k: 0.9 for k in ASR_KEYS})
    assert rule.n_crossed == len(ASR_KEYS)
    rule.update(3, {k: 0.0 for k in ASR_KEYS})
    assert rule.n_crossed == len(ASR_KEYS), "掉回去就撤销了越阈状态"


def test_none_is_skipped_not_treated_as_zero():
    """
    陷阱 #13：未评估轮 / 无定义分组返回 `None`。当成 0 会
    （a）把越阈的连续计数打断，（b）把斜率算成一条陡降线。
    """
    rule = StoppingRule(dict(CFG, thetas=[0.5], criteria=["thresholds_crossed"],
                             floor_effective=0), 5, 60)
    rule.update(1, {k: 0.9 for k in ASR_KEYS})
    rule.update(2, {k: None for k in ASR_KEYS})          # 非评估轮
    rule.update(3, {k: 0.9 for k in ASR_KEYS})
    assert rule.n_crossed == len(ASR_KEYS), "None 打断了连续计数"

    rule2 = StoppingRule(dict(CFG, criteria=["pm_acc_plateau"], floor_effective=0), 5, 60)
    for r in range(1, 21):
        rule2.update(r, {"pm_acc": None if r % 2 else 0.74})
    s = rule2.pm_slope()
    assert s is None or abs(s) < 1e-6, f"None 参与了斜率拟合：slope={s}"


def test_disabled_when_no_stopping_config():
    """没配 `stopping` → 规则不介入，行为与改动前逐字相同（跑满 n_rounds）。"""
    rule = StoppingRule(None, edge_rounds=5, n_rounds=60)
    for r in range(1, 61):
        assert not rule.update(r, {"pm_acc": 0.74, **{k: 0.99 for k in ASR_KEYS}}).stop


# ══════════════════════════════════════════════════════════════════════════
# 4. 自描述：上下游分开测（陷阱 #10：解析器认得格式 ≠ 代码会打印它）
# ══════════════════════════════════════════════════════════════════════════

def test_server_run_actually_prints_the_stop_line():
    """上游：`CloudServer.run` 真的会打印 `[Stop]` 并 break。"""
    src = (ROOT / "fedavg" / "server" / "server.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    body = ast.unparse(fn)
    assert "stopper.update" in body, "run() 没有问停止规则"
    assert "log_line" in body, "run() 没有打印 [Stop] 行"
    assert "break" in body, "run() 判定停止后没有 break"


def test_collect_metrics_parses_the_stop_line():
    """下游：`collect_metrics` 真的解析得出来。"""
    from collect_metrics import collect
    rule = StoppingRule(CFG, 5, 60)
    line = rule.log_line(32, type("D", (), {
        "reason": REASON_CONVERGED,
        "detail": {"effective": 160, "crossed": 9, "crossed_total": 9,
                   "pm_slope": 0.00041}})())
    log = "\n".join([
        "[Config] loading experiments/attack/hfl-propagation/2edge_distributed.yaml",
        "[配置校验] 通过 | method=hier_fedrep | defense=none | attack=badpfl | 0 个警告",
        line,
        "[Round   1] Broadcasting to 2 edges...",
        "  [Cloud] GM=0.1 | EM=0.5 PM=0.6 | loss=2.0 | time=10.0s | comm=1.0MB (total=1MB)",
    ])
    run = collect(log)["run"]
    assert run["stopped_at_round"] == 32
    assert run["stopped_at_effective"] == 160
    assert run["stop_reason"] == REASON_CONVERGED
    assert run["stop_crossed"] == "9/9"


def test_old_log_without_stop_line_still_parses():
    """
    反向锚点：固定轮数的 run 与老日志没有 `[Stop]` 行 —— 原有字段必须照常解析，
    停止字段是 `None`（而不是让整个 run 块塌掉）。
    """
    from collect_metrics import collect
    log = "\n".join([
        "[Config] loading experiments/attack/hfl-propagation/2edge_distributed.yaml",
        "[配置校验] 通过 | method=hier_fedrep | defense=none | attack=badpfl | 0 个警告",
        "[Round   1] Broadcasting to 2 edges...",
        "  [Cloud] GM=0.1 | EM=0.5 PM=0.6 | loss=2.0 | time=10.0s | comm=1.0MB (total=1MB)",
    ])
    run = collect(log)["run"]
    assert run["method"] == "hier_fedrep"          # 老字段没塌
    assert run["stop_reason"] is None
    assert run["stopped_at_round"] is None
