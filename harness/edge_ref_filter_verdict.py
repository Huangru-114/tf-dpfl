"""
harness/edge_ref_filter_verdict.py  —  方向 2 第二段的判定（FINDINGS N-014，D-132）：按参照分数丢弃上传，能不能保护良性端、代价多大

N-014 已经用户确认（2026-11-18）；本脚本先 commit，再交 ERF / ERC 的作业（同 D-088 的顺序）。

    python3 harness/edge_ref_filter_verdict.py \\
        --filter  <ERF 的 metrics.json …（24 个）> \\
        --control <SNAP__{collocated,distributed}__s42–44 + ERP 里 main-t0 以外的 6 格（s42）+ ERC（12 个）…> \\
        --out experiments/defense/edge-native/analysis/edge_ref_filter_verdict.json

格子 = 划分 {main, homo}（run.partition_profile：None → main，fixed → homo）× 三层个性化 {t0, t1}（run.edge_shared_blocks）
     × 布点 {collocated, distributed}（run.malicious_per_edge）。过滤臂 = run.edge_ref_mode == filter；对照 = observe 或不设防（SNAP）。
所有 run 按 run.seed 配对；每格要求过滤臂与对照都恰好有种子 42 / 43 / 44。
量（`d0_power.run_quantities`，末 10 个全量评估点）：P = 良性端攻击成功率（local_benign_asr），MTA = 个性化精度（pm_acc）。
  ΔASR = P(对照) − P(过滤)，ΔACC = MTA(对照) − MTA(过滤)，逐种子。
标签（每格）：ΔASR 均值 ≥ 0.15 且最小种子 ≥ 0.10 且 ΔACC 均值 ≤ 0.02 → protects；ASR 两条满足但 ΔACC 均值 > 0.02 → accuracy_bound；否则 no_effect。
组合（每个划分 × 布点；三种子均值）：A_both = 过滤 + t1，A_tier = t1 对照，A_filt = 过滤 + t0；
  A_both ≤ min(A_tier, A_filt) − 0.10 → additive；|A_both − min(A_tier, A_filt)| < 0.05 → redundant；其余 partial。
  三格中任一格 invalid → 组合也 invalid。
闸（任一不满足 → 该格 invalid）：每个 run exit_code 0、client_failures 为空、攻击者参与过；过滤臂 edge_ref_mode = filter 且 score / threshold 已记；
  对照的 edge_ref_mode ∈ {None, observe}；过滤臂与对照的 n_rounds / edge_rounds / n_edges / poison_ratio / partition / partition_profile /
  edge_shared_blocks / malicious_per_edge 相同；t0 对照的 P ≥ 0.5（否则 invalid_no_attack；t1 对照不设此闸）；种子集合 = {42, 43, 44}。
另报（不进判定）：在线的攻击者漏网率 / 良性误剔率（edge_ref_scores 的 admitted 列）、整轮全被剔的边缘轮数；集中布点下受害边缘 V 与攻击者所在边缘 B0 的良性 ASR。
纯标准库 + 已有 harness 模块，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d0_power import run_quantities                     # noqa: E402
from pilot_a4 import invalid_reasons                    # noqa: E402

DASR_MEAN, DASR_MIN, DACC_MAX = 0.15, 0.10, 0.02
ADD_MARGIN, RED_MARGIN, PLANT_MIN = 0.10, 0.05, 0.5
EPS = 1e-9          # 浮点容差：0.9 − 0.8 = 0.0999…98 不该因此判成「< 0.10」；只影响恰好压线的情形
SEEDS = (42, 43, 44)
PARTS, TIERS, PLACEMENTS = ("main", "homo"), ("t0", "t1"), ("collocated", "distributed")
SAME_KEYS = ("n_rounds", "edge_rounds", "n_edges", "poison_ratio", "partition", "partition_profile",
             "edge_shared_blocks", "malicious_per_edge")


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(xs):
    return math.fsum(xs) / len(xs) if xs else None


def cell_of(run: dict):
    part = "homo" if run.get("partition_profile") == "fixed" else "main"
    tier = "t1" if int(run.get("edge_shared_blocks") or 0) > 0 else "t0"
    mpe = [x for x in (run.get("malicious_per_edge") or []) if x]
    place = "collocated" if len(mpe) == 1 else "distributed"
    return part, tier, place


def online_rates(m: dict) -> dict:
    """admitted 列：攻击者漏网率（被接纳）/ 良性误剔率；整轮全被剔的 (云轮, edge, edge 轮) 数。"""
    t = m.get("edge_ref_scores") or {}
    cols = t.get("columns") or []
    rows = [dict(zip(cols, r)) for r in t.get("rows") or []]
    mal = set(int(i) for i in (m.get("run") or {}).get("malicious_ids") or [])
    a = [r["admitted"] for r in rows if r.get("admitted") is not None and int(r["cid"]) in mal]
    b = [r["admitted"] for r in rows if r.get("admitted") is not None and int(r["cid"]) not in mal]
    groups = {}
    for r in rows:
        if r.get("admitted") is not None:
            groups.setdefault((r.get("round"), r.get("edge_id"), r.get("edge_round")), []).append(r["admitted"])
    return {"attacker_pass": _mean(a), "benign_reject": None if not b else 1.0 - _mean(b),
            "n_attacker": len(a), "n_benign": len(b),
            "edge_rounds_all_rejected": sum(1 for v in groups.values() if not any(v))}


def _run_gate(m: dict, *, filt: bool) -> list:
    run = m.get("run") or {}
    out = list(invalid_reasons(m))
    if not (m.get("n_malicious_participations") or 0) > 0:
        out.append("攻击者一次也没参与")
    mode = run.get("edge_ref_mode")
    if filt:
        if mode != "filter":
            out.append(f"过滤臂的 edge_ref_mode = {mode!r}（应为 filter）")
        if run.get("edge_ref_score") is None or run.get("edge_ref_threshold") is None:
            out.append("过滤臂没有记下 score / threshold（[设定14]）")
    elif mode not in (None, "observe"):
        out.append(f"对照的 edge_ref_mode = {mode!r}（应为不设防或 observe）")
    return out


def judge_cell(filt: dict, ctrl: dict, tier: str) -> dict:
    """filt / ctrl：{seed: metrics}（同一格）。"""
    reasons = []
    if set(filt) != set(SEEDS):
        reasons.append(f"过滤臂的种子 {sorted(filt)} ≠ {list(SEEDS)}")
    if set(ctrl) != set(SEEDS):
        reasons.append(f"对照的种子 {sorted(ctrl)} ≠ {list(SEEDS)}")
    for s, m in sorted(filt.items()):
        reasons += [f"filter s{s}: {r}" for r in _run_gate(m, filt=True)]
    for s, m in sorted(ctrl.items()):
        reasons += [f"control s{s}: {r}" for r in _run_gate(m, filt=False)]
    for s in sorted(set(filt) & set(ctrl)):
        rf, rc = filt[s].get("run") or {}, ctrl[s].get("run") or {}
        diff = [k for k in SAME_KEYS if rf.get(k) != rc.get(k)]
        if diff:
            reasons.append(f"s{s}: 过滤臂与对照的 {diff} 不同")
    per_seed = {}
    for s in sorted(set(filt) & set(ctrl)):
        qf, qc = run_quantities(filt[s]), run_quantities(ctrl[s])
        per_seed[s] = {"P_control": _r(qc["P"]), "P_filter": _r(qf["P"]),
                       "MTA_control": _r(qc["MTA"]), "MTA_filter": _r(qf["MTA"]),
                       "V_control": _r(qc["V"]), "V_filter": _r(qf["V"]),
                       "B0_control": _r(qc["B0"]), "B0_filter": _r(qf["B0"]),
                       "dASR": None if qc["P"] is None or qf["P"] is None else qc["P"] - qf["P"],
                       "dACC": None if qc["MTA"] is None or qf["MTA"] is None else qc["MTA"] - qf["MTA"],
                       "online": {k: (_r(v) if isinstance(v, float) else v) for k, v in online_rates(filt[s]).items()}}
        if per_seed[s]["dASR"] is None or per_seed[s]["dACC"] is None:
            reasons.append(f"s{s}: P / MTA 缺失（末 10 个评估点不够）")
    out = {"per_seed": {str(s): {**v, "dASR": _r(v["dASR"]), "dACC": _r(v["dACC"])} for s, v in per_seed.items()}}
    if reasons:
        return {**out, "verdict": "invalid", "reasons": reasons}
    if tier == "t0":
        weak = [s for s in SEEDS if run_quantities(ctrl[s])["P"] < PLANT_MIN - EPS]
        if weak:
            return {**out, "verdict": "invalid_no_attack",
                    "reasons": [f"t0 对照的良性 ASR < {PLANT_MIN}：种子 {weak}（攻击没植入）"]}
    d_asr = [per_seed[s]["dASR"] for s in SEEDS]
    d_acc = [per_seed[s]["dACC"] for s in SEEDS]
    m_asr, n_asr, m_acc = _mean(d_asr), min(d_asr), _mean(d_acc)
    asr_ok = m_asr >= DASR_MEAN - EPS and n_asr >= DASR_MIN - EPS
    verdict = ("protects" if asr_ok and m_acc <= DACC_MAX + EPS else
               "accuracy_bound" if asr_ok else "no_effect")
    return {**out, "verdict": verdict, "dASR_mean": _r(m_asr), "dASR_min": _r(n_asr), "dACC_mean": _r(m_acc),
            # 不舍入：组合判定直接用它们比较（舍入到 4 位可能让压线的情形翻转）
            "P_filter_mean": _mean([run_quantities(filt[s])["P"] for s in SEEDS]),
            "P_control_mean": _mean([run_quantities(ctrl[s])["P"] for s in SEEDS])}


def judge_combo(cells: dict, part: str, place: str) -> dict:
    keys = {"both": (part, "t1", place), "filt": (part, "t0", place)}
    for k in keys.values():
        if cells[k]["verdict"] in ("invalid", "invalid_no_attack"):
            return {"verdict": "invalid", "reasons": [f"{'-'.join(k)} 是 {cells[k]['verdict']}"]}
    a_both = cells[keys["both"]]["P_filter_mean"]
    a_filt = cells[keys["filt"]]["P_filter_mean"]
    a_tier = cells[keys["both"]]["P_control_mean"]          # t1 的对照 = 只开三层个性化
    best = min(a_tier, a_filt)
    verdict = ("additive" if a_both <= best - ADD_MARGIN + EPS else
               "redundant" if abs(a_both - best) < RED_MARGIN - EPS else "partial")
    return {"verdict": verdict, "A_both": _r(a_both), "A_tier": _r(a_tier), "A_filt": _r(a_filt), "best_single": _r(best)}


def assemble(filter_runs: list, control_runs: list) -> dict:
    filt, ctrl, dup = {}, {}, []
    for bucket, runs in ((filt, filter_runs), (ctrl, control_runs)):
        for m in runs:
            run = m.get("run") or {}
            key, s = cell_of(run), run.get("seed")
            if s in bucket.setdefault(key, {}):
                dup.append(f"{'-'.join(key)} s{s}")
            bucket[key][s] = m
    out = {"rule": "N-014", "thresholds": {"dASR_mean": DASR_MEAN, "dASR_min": DASR_MIN, "dACC_max": DACC_MAX,
                                           "additive": ADD_MARGIN, "redundant": RED_MARGIN, "plant_min": PLANT_MIN},
           "cells": {}, "combination": {}}
    cells = {}
    for part in PARTS:
        for tier in TIERS:
            for place in PLACEMENTS:
                key = (part, tier, place)
                res = judge_cell(filt.get(key, {}), ctrl.get(key, {}), tier)
                if dup:
                    res = {**res, "verdict": "invalid", "reasons": [f"同一格同一种子有多个 run：{dup}"]}
                cells[key] = res
                out["cells"]["-".join(key)] = res
    for part in PARTS:
        for place in PLACEMENTS:
            out["combination"][f"{part}-{place}"] = judge_combo(cells, part, place)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="方向 2 第二段的判定（N-014）")
    ap.add_argument("--filter", nargs="+", required=True)
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    load = lambda ps: [json.loads(Path(p).read_text(encoding="utf-8")) for p in ps]   # noqa: E731
    res = assemble(load(a.filter), load(a.control))
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    for k, r in res["cells"].items():
        print(f"{k:24s} {r['verdict']:16s} ΔASR 均值 {r.get('dASR_mean')} 最小 {r.get('dASR_min')}  ΔACC {r.get('dACC_mean')}"
              + (f"  原因：{r.get('reasons')}" if "invalid" in r["verdict"] else ""))
    for k, r in res["combination"].items():
        print(f"组合 {k:20s} {r['verdict']}  " + ("" if r["verdict"] == "invalid" else
              f"A_both {r['A_both']} / A_tier {r['A_tier']} / A_filt {r['A_filt']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
