#!/usr/bin/env python3
"""
harness/supp_verdict.py  —  补充实验第一批的预注册判定（N-015 / D-135）：D1 块数扫描、D2 威胁边界、D4 逐坐标中位数

规则全文：`experiments/thesis/PREREG-v2.md`（commit 25163a5，2026-11-22）。**本脚本写于规则提交之后、任何新数据回传之前**；
阈值只在下面的常量里定义一次，改它就是改预注册（守卫 tests/test_supp_verdict.py 有阈值反向锚点）。

    python3 harness/supp_verdict.py                      # 读 experiments/defense/edge-native/results/P2，三个判定都算
    python3 harness/supp_verdict.py --only d1 --out x.json

量（与 d0_power / g6_verdict 同一口径，末 10 个评估点）：
  V      受害边缘良性端攻击成功率：`per_edge_rounds[r][e].client_benign`，每台边缘末 10 点均值再平均
         （集中布点 = E1–E3；D2 的 E2、E3 另算；E1 单算）
  P      全部良性端池化攻击成功率：`rounds[].local_benign_asr`（D4 分散布点没有受害边缘时用它）
  MTA    `acc_rounds[].pm_acc`：全部 100 个客户端（含攻击者）的 fresh 个性化精度（D-071 口径，用户 11-22 确认）
  B0     E0 良性端攻击成功率；stale = `acc_rounds[].pm_acc_stale`（隔点记录，末 10 个评估点窗口）

三个种子（42 / 43 / 44）按种子配对；「三个种子都满足」才贴标签，否则 inconclusive。门槛比较带 1e-9 容差（D-133：0.9 − 0.8 = 0.0999…98
算作「等于 0.10」，即 ≥ 0.10 成立、< 0.10 不成立）。

有效性闸（任一不过 → invalid）：exit 0、client_failures / errors 为空、跑满 60 云轮、攻击者参与过、P2 模板、
run 块因素（块数、布点、防御名与作用层）与登记表一致；D1 另有闸 ②：TBS k1-snap 与 ERP / ERC main-t1-collocated 同种子
`instrumentation_check.compare` 逐位相同（checksum + 全部评估数值）。文件缺失 → insufficient（不是 invalid）。

纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d0_power import _edge_last10, last_k_mean, victim_jump    # noqa: E402
from instrumentation_check import compare                       # noqa: E402
from pilot_a4 import invalid_reasons                            # noqa: E402
from runs_table import window_mean                              # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "experiments/defense/edge-native/results/P2"
OUT = ROOT / "experiments/defense/edge-native/analysis/supp_verdict.json"
PREREG = "N-015（experiments/thesis/PREREG-v2.md @ 25163a5）；判定脚本写于规则之后、数据之前"

SEEDS = (42, 43, 44)
N_ROUNDS = 60
TOL = 1e-9
COLLOCATED, DISTRIBUTED, BOUNDARY = [10, 0, 0, 0], [3, 3, 2, 2], [9, 1, 0, 0]
VICTIMS = (1, 2, 3)

# ── 预注册阈值（PREREG-v2 §三 / §四 / §六；用户 11-22 全部照草案，D-135 ③）──
D1_MEANINGFUL = 0.02        # 三个种子都 D ≥ 0.02 → b1_meaningful
D1_FREE = 0.01              # 三个种子都 |D| < 0.01 → isolation_free
D1_BETTER = -0.01           # 三个种子都 D ≤ −0.01 → isolation_better
D2_CLEAN_PROTECTED = 0.30   # 三个种子都 Δ_clean ≥ 0.30
D2_CLEAN_UNPROTECTED = 0.10 # 三个种子都 Δ_clean ≤ 0.10
D2_E1_BREACHED = 0.20       # 三个种子都 Gap ≥ 0.20
D2_E1_PROTECTED = 0.10      # 三个种子都 |Gap| < 0.10
D4_MTA = 0.02               # ΔMTA 均值 ≤ 0.02 且逐种子 ≤ 0.02（D-071，同 g6_verdict）
D4_VS_B1 = 0.10             # ASR(M) − ASR(B=1) 三档


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return math.fsum(xs) / len(xs) if xs else None


# 带容差的比较：与门槛相差不到 TOL 的值算作「等于门槛」
def _ge(x, t):
    return x >= t - TOL


def _le(x, t):
    return x <= t + TOL


def _lt(x, t):
    return x < t - TOL


def _gt(x, t):
    return x > t + TOL


# ══════════════════════════════════════════════════════════════════════════
# 每个 run 的量
# ══════════════════════════════════════════════════════════════════════════

def edges_v(m: dict, edges) -> float | None:
    vals = [_edge_last10(m, e) for e in edges]
    return None if any(v is None for v in vals) else _mean(vals)


def pooled(m: dict) -> float | None:
    return last_k_mean([r.get("local_benign_asr") for r in m.get("rounds") or []])[0]


def mta(m: dict) -> float | None:
    return last_k_mean([r.get("pm_acc") for r in m.get("acc_rounds") or []])[0]


def stale_acc(m: dict) -> float | None:
    return window_mean(m.get("acc_rounds") or [], "pm_acc_stale", anchor="pm_acc")[0]


def cloud_fraction(m: dict) -> float | None:
    """上云参数比例（按 get_weights 元素数，含 BN 统计量）= 1 − 边缘段 / 全部；k = 0 → 1。"""
    run = m.get("run") or {}
    if int(run.get("edge_shared_blocks") or 0) == 0:
        return 1.0
    ts = run.get("tier_split") or {}
    if not ts.get("n_params"):
        return None
    return 1.0 - ts["n_edge_params"] / ts["n_params"]


def first_half_jump(m: dict) -> float | None:
    """受害边缘前半程（第 2–30 云轮）的平均灌入量（云聚合后点 − 上一云轮末；只对集中布点，同 d0_power.victim_jump）。"""
    js = [victim_jump(m, t) for t in range(1, N_ROUNDS // 2)]
    return _mean([j for j in js if j is not None]) if any(j is not None for j in js) else None


# ══════════════════════════════════════════════════════════════════════════
# 闸
# ══════════════════════════════════════════════════════════════════════════

def gate(m: dict, *, k: int | None = None, mpe: list | None = None, defense=None, layers=None) -> list:
    """一个 run 的有效性闸（PREREG §一 ① + ③）。defense 可给一个名字或一组允许的名字。"""
    reasons = list(invalid_reasons(m))
    run = m.get("run") or {}
    if m.get("exit_code") is None:
        reasons.append("exit_code 缺失")
    if m.get("errors"):
        reasons.append(f"errors 非空（{len(m['errors'])} 条）")
    if not (m.get("n_malicious_participations") or 0) > 0:
        reasons.append("攻击者一次都没参与")
    last = max((r.get("round") or 0 for r in m.get("acc_rounds") or []), default=0)
    if run.get("n_rounds") != N_ROUNDS or last != N_ROUNDS:
        reasons.append(f"没有跑满 {N_ROUNDS} 云轮（run.n_rounds={run.get('n_rounds')}，acc_rounds 最后一轮 {last}）")
    if (run.get("alignment") or {}).get("template") != "p2":
        reasons.append(f"不是 P2 模板（run.alignment.template={(run.get('alignment') or {}).get('template')!r}）")
    if k is not None and int(run.get("edge_shared_blocks") or 0) != k:
        reasons.append(f"run.edge_shared_blocks={run.get('edge_shared_blocks')!r} ≠ {k}")
    if mpe is not None and list(run.get("malicious_per_edge") or []) != list(mpe):
        reasons.append(f"run.malicious_per_edge={run.get('malicious_per_edge')!r} ≠ {list(mpe)}")
    if defense is not None:
        allowed = {defense} if isinstance(defense, str) else set(defense)
        if run.get("defense") not in allowed:
            reasons.append(f"run.defense={run.get('defense')!r} ∉ {sorted(allowed)}")
        if run.get("defense") == "edge_ref" and run.get("edge_ref_mode") != "observe":
            reasons.append(f"run.edge_ref_mode={run.get('edge_ref_mode')!r}：B=1 对照只能是 observe（只记录、全部接纳）")
    if layers is not None and run.get("defense_layers") != list(layers):
        reasons.append(f"run.defense_layers={run.get('defense_layers')!r} ≠ {list(layers)}（中位数作用层没对上）")
    return reasons


B1_DEFENSES = ("none", "edge_ref")       # B=1 的已有对照是 ERP / ERC（observe）；TBS k1-snap 是 none


def _gated(runs: dict, label: str, **kw) -> tuple:
    """{seed: m} → (缺哪些种子, {seed: 理由})。"""
    missing = [s for s in SEEDS if runs.get(s) is None]
    bad = {s: r for s in SEEDS if runs.get(s) is not None and (r := gate(runs[s], **kw))}
    return [f"{label} s{s}" for s in missing], {f"{label} s{s}": v for s, v in bad.items()}


def _status(missing: list, invalid: dict) -> str | None:
    if invalid:
        return "invalid"
    if missing:
        return "insufficient"
    return None


# ══════════════════════════════════════════════════════════════════════════
# D1 块数扫描
# ══════════════════════════════════════════════════════════════════════════

def classify_d1(ds: list) -> str:
    if all(_ge(d, D1_MEANINGFUL) for d in ds):
        return "b1_meaningful"
    if all(_lt(abs(d), D1_FREE) for d in ds):
        return "isolation_free"
    if all(_le(d, D1_BETTER) for d in ds):
        return "isolation_better"
    return "inconclusive"


def judge_d1(k1: dict, k4: dict, k2: dict | None = None, k0: dict | None = None,
             k1_ref: dict | None = None) -> dict:
    """k1 = TBS k1-snap、k4 / k2 = TBS、k0 = SNAP 集中、k1_ref = ERP / ERC main-t1-collocated（闸 ②），各为 {seed: m}。"""
    k2, k0, k1_ref = k2 or {}, k0 or {}, k1_ref or {}
    out = {"preregistered": PREREG, "experiment": "D1 块数扫描",
           "thresholds": {"meaningful": D1_MEANINGFUL, "free": D1_FREE, "better": D1_BETTER}}
    miss, bad = [], {}
    for label, runs, kk, dfn in (("k1-snap", k1, 1, "none"), ("k4", k4, 4, "none"),
                                 ("k1_ref", k1_ref, 1, B1_DEFENSES)):
        m_, b_ = _gated(runs, label, k=kk, mpe=COLLOCATED, defense=dfn)
        miss += m_
        bad.update(b_)
    # 闸 ②：B=1 存快照重跑（不设防）与已有 B=1（observe）逐位相同
    gate2 = {}
    for s in SEEDS:
        if k1.get(s) is None or k1_ref.get(s) is None:
            continue
        c = compare(k1_ref[s], k1[s])
        gate2[f"s{s}"] = {"n_checksums": c["n_checksums"], "n_fields": c["n_fields"],
                          "n_diffs": len(c["diffs"]), "n_missing": len(c["missing"]),
                          "first_diffs": [list(map(str, d)) for d in c["diffs"][:3]]}
        if c["diffs"] or c["missing"] or c["n_checksums"] < N_ROUNDS:
            bad[f"gate2 s{s}"] = [f"与已有 B=1 不逐位相同：{len(c['diffs'])} 处不同、{len(c['missing'])} 处缺失、"
                                  f"{c['n_checksums']} 个 checksum（闸 ②）"]
    out["gate2"] = gate2
    # 只报告的臂：k2、k0 缺失不影响判定，无效则不报它的数
    rep_bad = {}
    for label, runs, kk in (("k2", k2, 2), ("k0", k0, 0)):
        _, b_ = _gated(runs, label, k=kk, mpe=COLLOCATED, defense="none")
        rep_bad.update(b_)
    out["report_invalid"] = rep_bad
    report = {}
    for label, runs, kk in (("B=0", k0, 0), ("B=1", k1, 1), ("B=2", k2, 2), ("B=4", k4, 4)):
        tag = {0: "k0", 1: "k1-snap", 2: "k2", 4: "k4"}[kk]
        report[label] = {f"s{s}": {"V": _r(edges_v(m, VICTIMS)), "B0": _r(_edge_last10(m, 0)),
                                   "MTA": _r(mta(m)), "stale_acc": _r(stale_acc(m)),
                                   "cloud_fraction": _r(cloud_fraction(m), 6)}
                         for s in SEEDS if (m := runs.get(s)) is not None
                         and f"{tag} s{s}" not in bad and f"{tag} s{s}" not in rep_bad}
    out["report"] = report
    mono = {}
    for s in SEEDS:
        v = {kk: edges_v(r[s], VICTIMS) for kk, r in ((1, k1), (2, k2), (4, k4)) if r.get(s) is not None}
        if len(v) == 3:
            mono[f"s{s}"] = _le(v[4], v[2]) and _le(v[2], v[1])
    out["monotone_V_k4_le_k2_le_k1"] = mono           # 只标出，不是判定
    st = _status(miss, bad)
    if st:
        out.update({"label": st, "missing": miss, "invalid": bad})
        return out
    ds = [mta(k1[s]) - mta(k4[s]) for s in SEEDS]
    out.update({"label": classify_d1(ds), "D_per_seed": {f"s{s}": _r(d, 6) for s, d in zip(SEEDS, ds)},
                "D_mean": _r(_mean(ds), 6), "missing": [], "invalid": {}})
    return out


# ══════════════════════════════════════════════════════════════════════════
# D2 威胁边界 [9,1,0,0]
# ══════════════════════════════════════════════════════════════════════════

D2_READING = {
    ("clean_protected", "e1_breached"): "符合「信任域」解释：6.1 的机制从「和数据一致」升为「事先预测、被检验」",
    ("clean_protected", "e1_also_protected"): "一个攻击者不足以攻破：失效点在每边缘 1 个到 2–3 个攻击者之间，7.1 讨论",
}


def classify_d2(clean: list, gap: list) -> tuple:
    if all(_ge(x, D2_CLEAN_PROTECTED) for x in clean):
        c = "clean_protected"
    elif all(_le(x, D2_CLEAN_UNPROTECTED) for x in clean):
        c = "clean_unprotected"
    else:
        c = "inconclusive"
    if all(_ge(x, D2_E1_BREACHED) for x in gap):
        e = "e1_breached"
    elif all(_lt(abs(x), D2_E1_PROTECTED) for x in gap):
        e = "e1_also_protected"
    else:
        e = "inconclusive"
    return c, e


def d2_reading(c: str, e: str) -> str:
    if c == "clean_unprotected":
        return "与预测相反：6.1 的有效条件改写为「攻击者只在一个边缘」，机制重新审视"
    return D2_READING.get((c, e), "报数字，不报方向（至少一个判定 inconclusive）")


def judge_d2(k0: dict, k1: dict, colloc_k0: dict | None = None, colloc_k1: dict | None = None) -> dict:
    colloc_k0, colloc_k1 = colloc_k0 or {}, colloc_k1 or {}
    out = {"preregistered": PREREG, "experiment": "D2 威胁边界 [9,1,0,0]",
           "thresholds": {"clean_protected": D2_CLEAN_PROTECTED, "clean_unprotected": D2_CLEAN_UNPROTECTED,
                          "e1_breached": D2_E1_BREACHED, "e1_also_protected": D2_E1_PROTECTED}}
    miss, bad = [], {}
    for label, runs, kk in (("k0", k0, 0), ("k1", k1, 1)):
        m_, b_ = _gated(runs, label, k=kk, mpe=BOUNDARY, defense="none")
        miss += m_
        bad.update(b_)
    report = {}
    for label, runs in (("B=0", k0), ("B=1", k1)):
        report[label] = {f"s{s}": {"E0": _r(_edge_last10(m, 0)), "E1": _r(_edge_last10(m, 1)),
                                   "E2": _r(_edge_last10(m, 2)), "E3": _r(_edge_last10(m, 3)),
                                   "MTA": _r(mta(m)), "P": _r(pooled(m))}
                         for s in SEEDS if (m := runs.get(s)) is not None}
    report["集中布点对照"] = {f"s{s}": {"V_B0": _r(edges_v(colloc_k0[s], VICTIMS)) if colloc_k0.get(s) else None,
                                   "V_B1": _r(edges_v(colloc_k1[s], VICTIMS)) if colloc_k1.get(s) else None}
                         for s in SEEDS}
    out["report"] = report
    st = _status(miss, bad)
    if st:
        out.update({"label": st, "missing": miss, "invalid": bad})
        return out
    clean = [edges_v(k0[s], (2, 3)) - edges_v(k1[s], (2, 3)) for s in SEEDS]
    gap = [_edge_last10(k1[s], 1) - edges_v(k1[s], (2, 3)) for s in SEEDS]
    c, e = classify_d2(clean, gap)
    out.update({"label": {"clean": c, "e1": e}, "reading": d2_reading(c, e),
                "delta_clean_per_seed": {f"s{s}": _r(x, 6) for s, x in zip(SEEDS, clean)},
                "gap_per_seed": {f"s{s}": _r(x, 6) for s, x in zip(SEEDS, gap)},
                "missing": [], "invalid": {}})
    return out


# ══════════════════════════════════════════════════════════════════════════
# D4 逐坐标中位数
# ══════════════════════════════════════════════════════════════════════════

def asr_q(m: dict, place: str) -> float | None:
    """集中 → 受害边缘 V；分散没有受害边缘 → 全部良性端池化 P（PREREG §六 ✎）。"""
    return edges_v(m, VICTIMS) if place == "collocated" else pooled(m)


def classify_blocks(d_asr: list, d_mta: list) -> dict:
    asr_ok = all(_gt(d, 0.0) for d in d_asr)
    ok_mean, ok_every = _le(_mean(d_mta), D4_MTA), all(_le(d, D4_MTA) for d in d_mta)
    return {"label": "blocks" if (asr_ok and ok_mean and ok_every) else "not_blocks",
            "asr_all_positive": asr_ok, "mta_mean_ok": ok_mean, "mta_every_seed_ok": ok_every}


def classify_vs_b1(diff: list) -> str:
    if all(_ge(d, D4_VS_B1) for d in diff):
        return "weaker_than_b1"
    if all(_le(d, -D4_VS_B1) for d in diff):
        return "stronger_than_b1"
    if all(_lt(abs(d), D4_VS_B1) for d in diff):
        return "comparable_to_b1"
    return "inconclusive"


def judge_d4_cell(arm: dict, ctrl: dict, b1: dict, layer: str, place: str) -> dict:
    mpe = COLLOCATED if place == "collocated" else DISTRIBUTED
    out = {"layer": layer, "placement": place,
           "asr_quantity": "V（受害边缘）" if place == "collocated" else "P（全部良性端池化）"}
    miss, bad = [], {}
    for label, runs, kw in (("arm", arm, {"k": 0, "defense": "median", "layers": [layer]}),
                            ("control", ctrl, {"k": 0, "defense": "none"})):
        m_, b_ = _gated(runs, label, mpe=mpe, **kw)
        miss += m_
        bad.update(b_)
    out["report"] = {f"s{s}": {"ASR": _r(asr_q(m, place)), "MTA": _r(mta(m)), "stale_acc": _r(stale_acc(m)),
                               "B0": _r(_edge_last10(m, 0)) if place == "collocated" else None,
                               "jump_first_half": _r(first_half_jump(m)) if place == "collocated" else None}
                     for s in SEEDS if (m := arm.get(s)) is not None}
    st = _status(miss, bad)
    if st:
        out.update({"label": st, "missing": miss, "invalid": bad})
        return out
    d_asr = [asr_q(ctrl[s], place) - asr_q(arm[s], place) for s in SEEDS]
    d_mta = [mta(ctrl[s]) - mta(arm[s]) for s in SEEDS]
    out.update(classify_blocks(d_asr, d_mta))
    out.update({"d_asr_per_seed": {f"s{s}": _r(x, 6) for s, x in zip(SEEDS, d_asr)},
                "d_asr_mean": _r(_mean(d_asr), 6),
                "d_mta_per_seed": {f"s{s}": _r(x, 6) for s, x in zip(SEEDS, d_mta)},
                "d_mta_mean": _r(_mean(d_mta), 6), "missing": [], "invalid": {}})
    # 与 B=1 比（ERP / ERC main-t1 同布点同种子）：缺或无效只影响这一项
    b_miss, b_bad = _gated(b1, "b1", k=1, mpe=mpe, defense=B1_DEFENSES)
    if b_miss or b_bad:
        out["vs_b1"] = {"label": _status(b_miss, b_bad), "missing": b_miss, "invalid": b_bad}
    else:
        diff = [asr_q(arm[s], place) - asr_q(b1[s], place) for s in SEEDS]
        out["vs_b1"] = {"label": classify_vs_b1(diff),
                        "diff_per_seed": {f"s{s}": _r(x, 6) for s, x in zip(SEEDS, diff)}}
    return out


def judge_d4(arms: dict, ctrl: dict, b1: dict) -> dict:
    """arms = {(layer, place): {seed: m}}；ctrl / b1 = {place: {seed: m}}。"""
    cells = [judge_d4_cell(arms.get((layer, place)) or {}, ctrl.get(place) or {}, b1.get(place) or {}, layer, place)
             for layer in ("edge", "cloud") for place in ("collocated", "distributed")]
    return {"preregistered": PREREG, "experiment": "D4 逐坐标中位数",
            "thresholds": {"mta": D4_MTA, "vs_b1": D4_VS_B1}, "cells": cells}


# ══════════════════════════════════════════════════════════════════════════
# 载入（按 run_id 找文件）
# ══════════════════════════════════════════════════════════════════════════

def _load(results: Path, group: str, run_id: str):
    p = Path(results) / group / f"{run_id}.metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _by_seed(results, group, pattern) -> dict:
    return {s: _load(results, group, pattern.format(s=s)) for s in SEEDS}


def _b1_ref(results, place) -> dict:
    """B=1 的已有对照：s42 = ERP，s43 / s44 = ERC（都是 observe）。"""
    out = {42: _load(results, "ERP", f"ERP__main-t1-{place}__s42")}
    out.update({s: _load(results, "ERC", f"ERC__main-t1-{place}__s{s}") for s in (43, 44)})
    return out


def run_all(results: Path = RESULTS, only=None) -> dict:
    out = {"preregistered": PREREG}
    if only in (None, "d1"):
        out["d1"] = judge_d1(k1=_by_seed(results, "TBS", "TBS__k1-snap__s{s}"),
                             k4=_by_seed(results, "TBS", "TBS__k4__s{s}"),
                             k2=_by_seed(results, "TBS", "TBS__k2__s{s}"),
                             k0=_by_seed(results, "SNAP", "SNAP__collocated__s{s}"),
                             k1_ref=_b1_ref(results, "collocated"))
    if only in (None, "d2"):
        out["d2"] = judge_d2(k0=_by_seed(results, "TBH", "TBH__k0__s{s}"),
                             k1=_by_seed(results, "TBH", "TBH__k1__s{s}"),
                             colloc_k0=_by_seed(results, "SNAP", "SNAP__collocated__s{s}"),
                             colloc_k1=_b1_ref(results, "collocated"))
    if only in (None, "d4"):
        arms = {(layer, place): _by_seed(results, "MED", f"MED__{layer}-{place}__s{{s}}")
                for layer in ("edge", "cloud") for place in ("collocated", "distributed")}
        out["d4"] = judge_d4(arms, {p: _by_seed(results, "SNAP", f"SNAP__{p}__s{{s}}")
                                    for p in ("collocated", "distributed")},
                             {p: _b1_ref(results, p) for p in ("collocated", "distributed")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--only", choices=("d1", "d2", "d4"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    res = run_all(Path(a.results), a.only)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for key in ("d1", "d2"):
        if key in res:
            print(f"[{key}] {res[key]['label']}")
    if "d4" in res:
        for c in res["d4"]["cells"]:
            vs = (c.get("vs_b1") or {}).get("label")
            print(f"[d4] {c['layer']:5s} {c['placement']:11s} {c['label']}" + (f" | vs B=1: {vs}" if vs else ""))
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
