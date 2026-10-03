"""
harness/g1_verdict.py  —  G1 / G1R5 的预注册判定：3-D（edge 可观测性）与 3-C（云周期内的锯齿）（2026-10-03）

    python3 harness/g1_verdict.py [--json out.json] [--resamples 50]

**规则是预注册的，脚本不是**：规则 = FINDINGS **N-007**（2026-10-02 写于数据之前，用户 2026-10-03 确认）。
本脚本写于 G1 seed 42 / 43 的描述性读数（F-083，`g1_scores.py`）**之后**；seed 44 与 G1R5 已回传，
但脚本与测试**先 commit、再打开它们**（只用合成数据与 s42 / s43 测试；git 历史可查）。同 g6_verdict / g3_did 的情形。

3-D（每个主分数 × 每个周期长度 R 分别判；判读格 = C1·分散）
  分数（越大越可疑）：主 = norm_w / neg_cos_global_w / neg_cos_edge_w / s_ck；诊断（只报告）= norm / norm_s / neg_cos_global / neg_cos_edge。
  视角：`g1_scores.view_arrays`（窗口 20 有效轮；edge 视角 = 本窗口本 edge 全部更新的稳健 z；
        全局（等池）= 从本窗口全部更新里抽同样多个当参照、重复 50 次取平均）；ΔAUROC = edge − 全局（等池），一个 run 一个值。
  有效性闸（N-007；任一不过 → invalid，不判方向）：
    ① 同 R、同分数的 random·分散 零对照：三个 seed 的 |ΔAUROC| 均值 < 0.05；
    ② 六个 run（C1·分散 + random·分散 × 3 seed）：exit 0、client_failures 空、攻击者参与、因素与登记一致。
  ③ C1·分散 的全局（等池）AUROC ≥ 0.7，否则 `undetectable`。N-007 没写是「三个 seed 的均值」还是「每个 seed」
     → 两种读法都算：一致才用；不一致记 `gate3_reading_matters`，并附上两种读法下各自的结论（同 g6_verdict 的 mta_reading_matters）。
  方向：三个 seed 的 Δ 都 > 0.05 → `edge_better`；|Δ| 都 < 0.05 → `no_edge_gain`；其余 `mixed`。缺 run → `insufficient` / `missing`。
  次要读数（N-007 必报，不进判定）：FPR = 5% 时的 TPR（raw / edge / 全局整池 / 全局等池）；norm_w vs norm_s 的 AUROC；
  s_ck 的 k*（argmax 的类）是目标类 0 的比例（恶意 / 良性分开）；集中格只报告；0.05 用 seed 间 SD 复核（只报告，不回头改判定）。

3-C（集中布点；量 = 受害 edge E1–E3 良性端 fresh-PM ASR（`client_benign`）的三 edge 均值）
  Δ_jump(g) = 第 g 云轮的云聚合后评估点 − 第 g−1 云轮末的全量点；
  r_down(g) = (第 g 云轮的云聚合后评估点 − 第 g 云轮末的全量点) / R（每有效轮；正 = 周期内洗掉）；
  g = 2 … ⌊n_rounds / 2⌋（只取前半程周期）；一个 run 取这些 g 上的均值。
  三个 seed 的 r_down 都 ≥ 0.005 → `self_cleaning`；都 ≤ 0.001 → `no_cleaning`；其余 `user_decides`。
  另报 PLAN §3 的旧读法（bootstrap CI 下界 > 0）与每周期总洗掉量 r_down × R。
  G1R5（C1·集中·R5，R5 桥）也是集中布点格，用同一条规则判，标 `bridge`；它与 G8 不直接相减（总洗掉率 vs 净衰减，N-007）。

纯标准库 + numpy，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fedavg"))
from analysis.functional_score import tpr_at_fpr                    # noqa: E402  纯 numpy
from collect_metrics import LIGHT_EDGE_COLUMNS                      # noqa: E402
from decay_verdict import edge_by_round                             # noqa: E402
from g1_scores import aurocs_from_arrays, collect_updates, view_arrays   # noqa: E402
from pilot_a4 import invalid_reasons                                # noqa: E402
from verdicts import bootstrap_mean_ci                              # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
TARGET = 0                       # base.yaml backdoor.target_label（G1 / G1R5 的 set 不改它）
VICTIMS = (1, 2, 3)
PARTITIONS = {"C1": ("designed", "C1"), "random": ("equal_random", None)}   # 标签 → (run.partition, run.partition_condition)
PLACEMENTS = {"collocated": [10, 0, 0, 0], "distributed": [3, 3, 2, 2]}
G1_ROUNDS = {10: 30, 20: 15}     # R → n_rounds（固定 300 有效轮）
G1R5_R, G1R5_ROUNDS = 5, 60

# ── N-007 的参数（预注册；改了要在 DECISIONS 留记录）────────────────────────────
WINDOW = 20
RESAMPLES = 50
FPR = 0.05
MAIN_SCORES = ("norm_w", "neg_cos_global_w", "neg_cos_edge_w", "s_ck")
DIAG_SCORES = ("norm", "norm_s", "neg_cos_global", "neg_cos_edge")
D_EDGE = 0.05                    # 方向门槛（|ΔAUROC|）—— N-007：没有证据，取自单点噪声量级
NULL_MAX = 0.05                  # 闸 ①：random·分散 三个 seed 的 |Δ| 均值上限
DETECT_MIN = 0.7                 # 闸 ③：C1·分散 全局（等池）AUROC 下限
SELF_CLEAN = 0.005               # 3-C：三个 seed 的 r_down 都 ≥ → self_cleaning（每有效轮）
NO_CLEAN = 0.001                 # 3-C：三个 seed 的 r_down 都 ≤ → no_cleaning


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sd(vals):
    """样本标准差（ddof = 1）；少于 2 个值 → None。"""
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def g1_run_id(part: str, place: str, R: int, seed: int) -> str:
    return f"G1__{part}_{place}_R{R}__s{seed}"


def g1r5_run_id(seed: int) -> str:
    return f"G1R5__C1-collocated-R5__s{seed}"


# ══════════════════════════════════════════════════════════════════════════
# 有效性（闸 ②）
# ══════════════════════════════════════════════════════════════════════════

def run_reasons(m: dict | None, part: str, place: str, R: int, n_rounds: int, seed: int | None = None) -> list:
    """一个 run 能不能进判定：D-044 的 invalid_reasons + 因素与登记一致 + 攻击者参与 + 记录开关都开了。"""
    if m is None:
        return ["缺"]
    reasons = list(invalid_reasons(m))
    if m.get("exit_code") is None:                     # invalid_reasons 放过缺失 / null（pilot 的约定）；N-007 要求 exit 0
        reasons.append("exit_code 缺失（N-007 闸 ② 要求 exit 0）")
    run = m.get("run") or {}
    want_part, want_cond = PARTITIONS[part]
    if run.get("partition") != want_part or (want_cond is not None and run.get("partition_condition") != want_cond):
        reasons.append(f"run.partition={run.get('partition')!r}/{run.get('partition_condition')!r} ≠ {part}")
    if run.get("malicious_per_edge") != PLACEMENTS[place]:
        reasons.append(f"run.malicious_per_edge={run.get('malicious_per_edge')} ≠ {PLACEMENTS[place]}")
    if run.get("edge_rounds") != R or run.get("n_rounds") != n_rounds:
        reasons.append(f"run.edge_rounds / n_rounds = {run.get('edge_rounds')} / {run.get('n_rounds')} ≠ {R} / {n_rounds}")
    for sw in ("update_geometry", "post_agg_eval", "update_ck"):
        if run.get(sw) is not True:
            reasons.append(f"run.{sw}={run.get(sw)!r}（记录开关没生效 = 陷阱 #7 同类）")
    if not (m.get("n_malicious_participations") or 0) > 0:
        reasons.append("攻击者一次都没参与（n_malicious_participations = 0）")
    if seed is not None and run.get("seed") != seed:
        reasons.append(f"run.seed={run.get('seed')!r} ≠ 文件名里的 seed {seed}")
    reasons += table_reasons(m, R, n_rounds)
    return reasons


def table_reasons(m: dict, R: int, n_rounds: int, n_edges: int = 4) -> list:
    """3-D 的记录表是否完整（G1 对抗式审查：3-C 有缺点检查，3-D 原来没有）：
    update_geometry 每个 (云轮, edge, edge 轮) 一行；ck_before 每个评分点 × edge 一行；ck_scores 都能配上 ck_before。"""
    out = []
    geo = m.get("update_geometry") or {}
    gc = geo.get("columns") or []
    if not gc:
        return ["没有 update_geometry 表"]
    keys = {(r[gc.index("round")], r[gc.index("edge_id")], r[gc.index("edge_round")]) for r in geo.get("rows") or []}
    want = n_rounds * R * n_edges
    if len(keys) != want:
        out.append(f"update_geometry 只有 {len(keys)} / {want} 个 (云轮, edge, edge 轮)")
    run = m.get("run") or {}
    every = run.get("update_ck_every")
    cb, cs = m.get("ck_before") or {}, m.get("ck_scores") or {}
    if every:
        bcols = cb.get("columns") or []
        before = {(r[bcols.index("round")], r[bcols.index("edge_id")], r[bcols.index("edge_round")])
                  for r in cb.get("rows") or []} if bcols else set()
        want_ck = (n_rounds * R // int(every)) * n_edges
        if len(before) != want_ck:
            out.append(f"ck_before 只有 {len(before)} / {want_ck} 个评分点 × edge")
        scols = cs.get("columns") or []
        orphan = sum((r[scols.index("round")], r[scols.index("edge_id")], r[scols.index("edge_round")]) not in before
                     for r in cs.get("rows") or []) if scols else 0
        if orphan:
            out.append(f"ck_scores 有 {orphan} 行配不上 ck_before")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3-D：每个 run 的分数读数
# ══════════════════════════════════════════════════════════════════════════

def _tprs(a: dict | None) -> dict:
    out = {"tpr5_raw": None, "tpr5_edge": None, "tpr5_global_full": None, "tpr5_global_matched": None}
    if a is None:
        return out
    mal = a["mal"]
    out["tpr5_raw"] = tpr_at_fpr(a["x"][mal], a["x"][~mal], FPR)
    out["tpr5_edge"] = tpr_at_fpr(a["z_edge"][mal], a["z_edge"][~mal], FPR)
    out["tpr5_global_full"] = tpr_at_fpr(a["z_gfull"][mal], a["z_gfull"][~mal], FPR)
    vals = []
    for z in a["z_matched"]:
        ok = ~np.isnan(z)
        v = tpr_at_fpr(z[ok & mal], z[ok & ~mal], FPR)
        if v is not None:
            vals.append(v)
    out["tpr5_global_matched"] = float(np.mean(vals)) if vals else None
    return out


def kstar_stats(ups: list, target: int = TARGET) -> dict:
    """s_ck 的 k*（Δ_k 最大的类）是目标类的比例，恶意 / 良性分开（N-007 次要读数）。

    frac_target            = 并列时按**均分**计（每个并列类得 1/并列数）—— 报告用这一列；
    frac_target_first_idx  = np.argmax 取第一个并列者（commit 1c3119e 的读法；目标类 0 恰是最小编号，
                             并列时偏向它 → 偏高，G1 对抗式审查发现，F-085）。"""
    out = {}
    for name, flag in (("malicious", True), ("benign", False)):
        sel = [u for u in ups if u.get("s_ck") is not None and u["mal"] is flag and u.get("ck_kstar") is not None]
        n = len(sel)
        first = sum(u["ck_kstar"] == target for u in sel)
        split = sum((1.0 / len(u["ck_kstar_ties"])) if target in (u.get("ck_kstar_ties") or []) else 0.0
                    for u in sel)
        n_tied = sum(len(u.get("ck_kstar_ties") or []) > 1 for u in sel)
        out[name] = {"n": n, "frac_target": (split / n) if n else None,
                     "frac_target_first_idx": (first / n) if n else None, "n_tied": n_tied}
    return out


def score_readouts(m: dict, window: int = WINDOW, resamples: int = RESAMPLES) -> dict:
    """一个 run：每个分数的 AUROC 表（与 g1_scores.view_aurocs 同源）+ TPR@FPR=5% + k* 统计。"""
    ups = collect_updates(m)
    seed = (m.get("run") or {}).get("seed") or 0       # 与 g1_scores.readout 相同的等池种子
    out = {}
    for key in MAIN_SCORES + DIAG_SCORES:
        a = view_arrays(ups, key, window, resamples, seed)
        out[key] = {**aurocs_from_arrays(a), **_tprs(a)}
    return {"scores": out, "kstar": kstar_stats(ups), "n_updates": len(ups)}


# ══════════════════════════════════════════════════════════════════════════
# 3-D：判定
# ══════════════════════════════════════════════════════════════════════════

def direction(deltas: list) -> str:
    if all(d > D_EDGE for d in deltas):
        return "edge_better"
    if all(abs(d) < D_EDGE for d in deltas):
        return "no_edge_gain"
    return "mixed"


def judge_3d_one(c1: dict, rnd: dict, key: str) -> dict:
    """c1 / rnd：{seed: {"reasons": [...], "readout": {...} | None}}（C1·分散 / random·分散，同一个 R）。"""
    present = [s for s in SEEDS if c1.get(s) is not None] + [s for s in SEEDS if rnd.get(s) is not None]
    out = {"score": key, "verdict": None, "reasons": []}
    if not present:
        out["verdict"] = "missing"
        return out
    if len(present) < 2 * len(SEEDS):
        out["verdict"] = "insufficient"
        out["reasons"].append("缺 run：" + ", ".join(
            f"{lab}/s{s}" for lab, grp in (("C1", c1), ("random", rnd)) for s in SEEDS if grp.get(s) is None))
        return out
    bad = [f"{lab}/s{s}: {r}" for lab, grp in (("C1", c1), ("random", rnd)) for s in SEEDS
           for r in grp[s]["reasons"]]
    if bad:                                                            # 闸 ②
        out.update(verdict="invalid", reasons=bad)
        return out
    sc = {lab: {s: grp[s]["readout"]["scores"][key] for s in SEEDS} for lab, grp in (("C1", c1), ("random", rnd))}
    d_c1 = [sc["C1"][s]["delta"] for s in SEEDS]
    d_rnd = [sc["random"][s]["delta"] for s in SEEDS]
    gm = [sc["C1"][s]["global_matched"] for s in SEEDS]
    out.update({
        "delta_C1": {f"s{s}": _r(d) for s, d in zip(SEEDS, d_c1)},
        "delta_random": {f"s{s}": _r(d) for s, d in zip(SEEDS, d_rnd)},
        "global_matched_C1": {f"s{s}": _r(v) for s, v in zip(SEEDS, gm)},
        "edge_C1": {f"s{s}": _r(sc["C1"][s]["edge"]) for s in SEEDS},
        "raw_C1": {f"s{s}": _r(sc["C1"][s]["raw"]) for s in SEEDS},
    })
    if any(d is None for d in d_c1 + d_rnd) or any(v is None for v in gm):
        out.update(verdict="invalid", reasons=["某个 run 的 ΔAUROC / 全局 AUROC 无定义（池里没有恶意或良性）"])
        return out
    null_mean = sum(abs(d) for d in d_rnd) / len(d_rnd)
    mean, lo, hi = bootstrap_mean_ci(d_c1)
    out.update({"null_abs_delta_mean": _r(null_mean), "gate1_null_ok": null_mean < NULL_MAX,
                "delta_C1_mean": _r(mean), "delta_C1_ci": [_r(lo), _r(hi)],
                "global_matched_C1_mean": _r(sum(gm) / len(gm)), "global_matched_C1_min": _r(min(gm))})
    if not null_mean < NULL_MAX:                                       # 闸 ①
        out.update(verdict="invalid",
                   reasons=[f"闸 ①：random·分散 零对照 |Δ| 均值 {null_mean:.4f} ≥ {NULL_MAX}（视角构造本身有偏）"])
        return out
    ok_mean = sum(gm) / len(gm) >= DETECT_MIN
    ok_every = min(gm) >= DETECT_MIN
    direc = direction(d_c1)
    out.update({"gate3_mean_ok": ok_mean, "gate3_every_seed_ok": ok_every, "direction_if_detectable": direc})
    if not ok_mean and not ok_every:                                   # 闸 ③
        out["verdict"] = "undetectable"
    elif ok_mean and ok_every:
        out["verdict"] = direc
    else:
        out["verdict"] = "gate3_reading_matters"
        out["reasons"].append(f"闸 ③ 两种读法不一致：均值读法 → {direc if ok_mean else 'undetectable'}；"
                              f"逐 seed 读法 → {direc if ok_every else 'undetectable'}")
    return out


def judge_3d(readouts: dict) -> dict:
    """readouts：{(part, place, R, seed): {"reasons": [...], "readout": {...} | None} | None}。"""
    verdicts = {}
    for R in G1_ROUNDS:
        c1 = {s: readouts.get(("C1", "distributed", R, s)) for s in SEEDS}
        rnd = {s: readouts.get(("random", "distributed", R, s)) for s in SEEDS}
        verdicts[f"R{R}"] = {key: judge_3d_one(c1, rnd, key) for key in MAIN_SCORES}
    return verdicts


def threshold_review(readouts: dict) -> dict:
    """N-007：0.05 用 seed 间 SD 复核（只报告，不回头改判定）。"""
    out = {}
    for R in G1_ROUNDS:
        for part in PARTITIONS:
            for key in MAIN_SCORES + DIAG_SCORES:
                ds = []
                for s in SEEDS:
                    x = readouts.get((part, "distributed", R, s))
                    if x and x.get("readout"):
                        ds.append(x["readout"]["scores"][key]["delta"])
                out.setdefault(f"R{R}", {}).setdefault(part, {})[key] = {
                    "delta_sd": _r(_sd(ds)), "n_seeds": len([d for d in ds if d is not None])}
    return out


def cell_table(readouts: dict) -> dict:
    """全部 8 格 × seed × 分数的读数（集中格只在这里出现，不进判定）。"""
    out = {}
    for part in PARTITIONS:
        for place in PLACEMENTS:
            for R in G1_ROUNDS:
                cell = f"{part}_{place}_R{R}"
                for s in SEEDS:
                    x = readouts.get((part, place, R, s))
                    if not x or not x.get("readout"):
                        continue
                    rd = x["readout"]
                    out.setdefault(cell, {})[f"s{s}"] = {
                        "n_updates": rd["n_updates"],
                        "scores": {k: {kk: (_r(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                   for k, v in rd["scores"].items()},
                        "kstar": {k: {"n": v["n"], "frac_target": _r(v["frac_target"]),
                                      "frac_target_first_idx": _r(v["frac_target_first_idx"]), "n_tied": v["n_tied"]}
                                  for k, v in rd["kstar"].items()}}
    return out


def norm_split_summary(table: dict) -> dict:
    """norm_w vs norm_s（F-081 补注）：每格三个 seed 的 raw / 全局（等池）AUROC 均值。"""
    out = {}
    for cell, seeds in table.items():
        row = {}
        for key in ("norm", "norm_w", "norm_s"):
            for view in ("raw", "global_matched"):
                row[f"{key}_{view}_mean"] = _r(_mean([v["scores"][key][view] for v in seeds.values()]))
        out[cell] = row
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3-C：锯齿
# ══════════════════════════════════════════════════════════════════════════

def _victim_mean(rows_by_edge: dict) -> float | None:
    vals = [rows_by_edge.get(e) for e in VICTIMS]
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def victim_full(m: dict) -> dict:
    """{云轮 g: 受害 edge 良性 ASR 三 edge 均值}（云轮末全量点，per_edge_rounds）。"""
    per = {e: edge_by_round(m, e) for e in VICTIMS}
    rounds = sorted(set().union(*[set(v) for v in per.values()]))
    out = {}
    for g in rounds:
        v = _victim_mean({e: per[e].get(g) for e in VICTIMS})
        if v is not None:
            out[g] = v
    return out


def victim_post(m: dict) -> dict:
    """{云轮 g: 同上}（第 g 云轮的云聚合后评估点，per_edge_post_agg_rounds；列序 = LIGHT_EDGE_COLUMNS）。"""
    cols = list(LIGHT_EDGE_COLUMNS)
    ei, ci = cols.index("edge_id"), cols.index("client_benign")
    out = {}
    for g, rows in (m.get("per_edge_post_agg_rounds") or {}).items():
        by_edge = {int(r[ei]): r[ci] for r in rows or [] if r and r[ei] is not None}
        v = _victim_mean(by_edge)
        if v is not None:
            out[int(g)] = v
    return out


def sawtooth(m: dict) -> dict:
    """一个集中布点 run 的 3-C 量。缺点 → 该 g 不计，并在 n_g_used 里看得出。"""
    run = m.get("run") or {}
    R, n = int(run["edge_rounds"]), int(run["n_rounds"])
    full, post = victim_full(m), victim_post(m)
    gs = list(range(2, n // 2 + 1))
    per_g = []
    for g in gs:
        if g in post and g in full and (g - 1) in full:
            per_g.append({"g": g, "post": _r(post[g]), "full_prev": _r(full[g - 1]), "full": _r(full[g]),
                          "d_jump": post[g] - full[g - 1], "r_down": (post[g] - full[g]) / R})
    rd = _mean([p["r_down"] for p in per_g])
    dj = _mean([p["d_jump"] for p in per_g])
    for p in per_g:
        p["d_jump"], p["r_down"] = _r(p["d_jump"], 5), _r(p["r_down"], 6)
    return {"R": R, "n_rounds": n, "g_range": [2, n // 2], "n_g_expected": len(gs), "n_g_used": len(per_g),
            "r_down": rd, "d_jump": dj, "wash_per_cycle": None if rd is None else rd * R,
            "last_cycle": {"post": _r(post.get(n)), "full": _r(full.get(n))}, "per_g": per_g}


def judge_3c_cell(runs: dict, part: str, R: int, n_rounds: int, bridge: bool = False) -> dict:
    """runs：{seed: metrics | None}（集中布点、同一格）。"""
    out = {"cell": f"{part}_collocated_R{R}", "bridge": bridge, "verdict": None, "reasons": [], "per_seed": {}}
    present = [s for s in SEEDS if runs.get(s) is not None]
    if not present:
        out["verdict"] = "missing"
        return out
    vals = {}
    for s in SEEDS:
        m = runs.get(s)
        if m is None:
            continue
        bad = run_reasons(m, part, "collocated", R, n_rounds, seed=s)
        if not m.get("per_edge_post_agg_rounds"):
            bad.append("没有 per_edge_post_agg_rounds（云聚合后评估点）")
        cols = m.get("per_edge_light_columns")
        if cols is None:                                # 缺列名就没法核对按位置读的列序（G1 对抗式审查）
            bad.append("没有 per_edge_light_columns（无法核对云聚合后点的列序）")
        elif list(cols) != list(LIGHT_EDGE_COLUMNS):
            bad.append(f"per_edge_light_columns {cols} ≠ {list(LIGHT_EDGE_COLUMNS)}（列序变了）")
        q = sawtooth(m) if not bad else None
        if q is not None and q["n_g_used"] < q["n_g_expected"]:
            bad.append(f"前半程周期缺点：用到 {q['n_g_used']} / {q['n_g_expected']} 个 g")
        if bad:
            out["reasons"] += [f"s{s}: {r}" for r in bad]
            continue
        vals[s] = q
        out["per_seed"][f"s{s}"] = {"r_down": _r(q["r_down"], 5), "d_jump": _r(q["d_jump"]),
                                    "wash_per_cycle": _r(q["wash_per_cycle"]), "n_g_used": q["n_g_used"],
                                    "g_range": q["g_range"], "last_cycle": q["last_cycle"], "per_g": q["per_g"]}
    if out["reasons"]:
        out["verdict"] = "invalid"
        return out
    if len(vals) < len(SEEDS):
        out["verdict"] = "insufficient"
        out["reasons"].append("缺 run：" + ", ".join(f"s{s}" for s in SEEDS if s not in vals))
        return out
    rds = [vals[s]["r_down"] for s in SEEDS]
    mean, lo, hi = bootstrap_mean_ci(rds)
    if all(v >= SELF_CLEAN for v in rds):
        v = "self_cleaning"
    elif all(v <= NO_CLEAN for v in rds):
        v = "no_cleaning"
    else:
        v = "user_decides"
    out.update({"verdict": v, "r_down_mean": _r(mean, 5), "r_down_ci": [_r(lo, 5), _r(hi, 5)],
                "plan3_reading_ci_above_zero": lo is not None and lo > 0,
                "d_jump_mean": _r(_mean([vals[s]["d_jump"] for s in SEEDS])),
                "wash_per_cycle_mean": _r(_mean([vals[s]["wash_per_cycle"] for s in SEEDS]))})
    return out


def judge_3c(g1_runs: dict, g1r5_runs: dict) -> dict:
    """g1_runs：{(part, place, R, seed): metrics | None}；g1r5_runs：{seed: metrics | None}。"""
    out = {}
    for part in PARTITIONS:
        for R, n in G1_ROUNDS.items():
            cell = judge_3c_cell({s: g1_runs.get((part, "collocated", R, s)) for s in SEEDS}, part, R, n)
            out[cell["cell"]] = cell
    br = judge_3c_cell(g1r5_runs, "C1", G1R5_R, G1R5_ROUNDS, bridge=True)
    out[br["cell"]] = br
    return out


# ══════════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════════

def judge(g1_runs: dict, g1r5_runs: dict, window: int = WINDOW, resamples: int = RESAMPLES) -> dict:
    readouts = {}
    for part in PARTITIONS:
        for place in PLACEMENTS:
            for R, n in G1_ROUNDS.items():
                for s in SEEDS:
                    m = g1_runs.get((part, place, R, s))
                    if m is None:
                        continue
                    reasons = run_reasons(m, part, place, R, n, seed=s)
                    readouts[(part, place, R, s)] = {
                        "reasons": reasons,
                        "readout": score_readouts(m, window, resamples) if not reasons else None}
    v3d = judge_3d(readouts)
    v3c = judge_3c(g1_runs, g1r5_runs)
    table = cell_table(readouts)
    return {
        "rule": "FINDINGS N-007（预注册，用户 2026-10-03 确认）",
        "preregistered_rule": True, "script_written_after_data": True,
        "data_seen_before_script": "G1 seed 42 / 43 的描述性读数（F-083）；seed 44 与 G1R5 在本脚本 commit 之后才打开",
        "params": {"window": window, "resamples": resamples, "fpr": FPR, "d_edge": D_EDGE, "null_max": NULL_MAX,
                   "detect_min": DETECT_MIN, "self_clean": SELF_CLEAN, "no_clean": NO_CLEAN, "target": TARGET},
        "overall": {"3D": {R: {k: v["verdict"] for k, v in d.items()} for R, d in v3d.items()},
                    "3C": {c: v["verdict"] for c, v in v3c.items() if not v["bridge"]},
                    "3C_bridge": {c: v["verdict"] for c, v in v3c.items() if v["bridge"]}},
        "untested_confounds": [
            "3-D：y_t 富集的良性客户端也可能降低 c_{y_t}（N-007 原文；要 C2 设计测，G1 只有 C1 与 random，本轮没测）",
            "3-D：所有分数都是在非自适应攻击者下测的；norm_s 所在的 BN 统计量通道不进任何受害者的模型，攻击者可零代价伪造（代码路径证据）",
            "3-C：lr 按有效轮衰减与 ASR 的水平效应（前半程贴 floor / 后半程贴天花板）都会改变 r_down，没有分离",
            "3-C bridge：r_down 以每有效轮计，R5 的周期短，同样的每周期洗掉量在 R5 下更容易越过 0.005"],
        "3D": v3d, "3C": v3c,
        "threshold_review": threshold_review(readouts),
        "norm_split": norm_split_summary(table),
        "cells": table,
        "invalid_runs": {f"{p}_{pl}_R{R}__s{s}": x["reasons"] for (p, pl, R, s), x in sorted(readouts.items())
                         if x["reasons"]},
        "missing_runs": [g1_run_id(p, pl, R, s) for p in PARTITIONS for pl in PLACEMENTS for R in G1_ROUNDS
                         for s in SEEDS if g1_runs.get((p, pl, R, s)) is None]
                        + [g1r5_run_id(s) for s in SEEDS if g1r5_runs.get(s) is None],
    }


def load(results_dir: Path = RESULTS, seeds=SEEDS) -> tuple:
    def _l(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    g1 = {(p, pl, R, s): _l(results_dir / "G1" / f"{g1_run_id(p, pl, R, s)}.metrics.json")
          for p in PARTITIONS for pl in PLACEMENTS for R in G1_ROUNDS for s in seeds}
    g1r5 = {s: _l(results_dir / "G1R5" / f"{g1r5_run_id(s)}.metrics.json") for s in seeds}
    return g1, g1r5


def main(argv=None):
    ap = argparse.ArgumentParser(description="G1 / G1R5 的预注册判定（N-007；脚本写于 s42 / s43 读数之后）")
    ap.add_argument("--json")
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    a = ap.parse_args(argv)
    res = judge(*load(), resamples=a.resamples)
    print("== 3-D（判读格 C1·分散；ΔAUROC = edge − 全局等池）")
    for R, d in res["3D"].items():
        for key, v in d.items():
            print(f"  {R} {key:<18} Δ(C1) {v.get('delta_C1')}  零对照 |Δ|均值 {v.get('null_abs_delta_mean')}  "
                  f"全局等池(C1) {v.get('global_matched_C1')} → {v['verdict']}")
            for r in v["reasons"]:
                print(f"      {r}")
    print("== 3-C（集中布点，受害 edge E1–E3 良性 ASR；g = 2 … ⌊n/2⌋）")
    for c, v in res["3C"].items():
        rd = ", ".join(f"{s}: {x['r_down']}" for s, x in v["per_seed"].items())
        tag = " [bridge]" if v["bridge"] else ""
        print(f"  {c:<26}{tag} r_down {{{rd}}}  Δ_jump 均值 {v.get('d_jump_mean')}  "
              f"每周期洗掉 {v.get('wash_per_cycle_mean')} → {v['verdict']}")
        for r in v["reasons"]:
            print(f"      {r}")
    if res["invalid_runs"]:
        print(f"invalid runs: {res['invalid_runs']}")
    if res["missing_runs"]:
        print(f"missing runs: {res['missing_runs']}")
    print(f"判定：{json.dumps(res['overall'], ensure_ascii=False)}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[g1_verdict] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
