"""
harness/pilot_a4.py  —  A4 可行性 pilot 的预注册判定（先定后跑，改阈值要在 DECISIONS 留记录）

    python3 harness/pilot_a4.py experiments/attack/hfl-mechanism/pilot/registry.yaml [--json out.json]

读 pilot 登记表里 6 个 run 的 metrics.json（缺的记 missing，不猜），给出三个判定：

  D-029（范围经 D-038 扩大）可行性 —— 两格（flat / 2edge_distributed）都要满足：
      stop_reason == "converged"，且**陈旧 PM 列** pm_acc_stale 的末 10 点均值
      ≥ P1 同 seed 重复的最小值 − 0.006（flat ≥ 0.7367、2edge ≥ 0.7297）。
      pm_acc 门槛按陈旧列判，因为 P1 的 pm_acc 就是在陈旧 PM 上测的（D-033 / D-038）。
      ASR 只记录、不设门槛。
      pass → A08 改 deviate、A25 改 done；fail → 逐个开关消融、回审计。
  D-031 训练顺序（A26）—— head_first 臂 = D029 组的两个 run，body_first 臂 = A26 组：
      两格都满足 abs(Δ fresh pm_acc 末 10 点均值) ≤ 0.006 且 abs(Δ local_benign_asr 末 10 点均值) ≤ 0.07
      → same（维持 head_first，A26 改 deviate）；否则 different（带回审计由用户定）。
      单 seed，只分辨得出大差异（D-031 已写明）。
  A15 确定性（D-028）—— DET 组两次 run 的 [Checksum] 前 5 轮逐轮相同 → pass（A15 改 done）。

**有效性闸**（D-044，在上面三个判定之前、不改任何阈值）：exit_code ≠ 0 或 client_failures
非空 → 该 run 判 `invalid`。起因（FINDINGS F-043）：第一轮 pilot 的恶意端每次都在
on_round_start 抛异常、被 `_collect_updates_*` 吞掉并踢出聚合 → 整段训练没有攻击者；
若评估侧没崩，D-029（ASR 不设门槛）会判 pass、D-031 两臂 ASR 都≈0 会判 same。
崩溃也不再判 `fail`：崩溃不是可行性结论。

G2P（D-047，G2 一致性复测，2026-09-27 登记）—— 旧 P1 与 G2 重合的 4 格（seed42）：
      flat / 2edge-R5 取 D029 组，4edge-R5 / 10edge-R5 取 G2P 组。
      T50 = local_benign_asr 首次越过 0.5 的有效轮（`analyze_exp3.first_crossing`，线性插值）；
      比值 r = T50(HFL) / T50(flat)。旧 P1 参照从 hfl-propagation/results/*_seed42 现算，不手抄
      （2026-09-27 算得 0.815 / 0.398 / 1.489）。
      3 个比值都与旧 P1 在 1 的同一侧 → consistent；任一侧落在 [0.9, 1.1] 算「约等于 1」、不计为不一致。
      任一格 T50 删失（没越过 0.5）→ undetermined。单 seed：**不代替** 3-A 的 5 seed 判定。

纯标准库 + PyYAML，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from registry import Registry          # noqa: E402
from runs_table import last_k_mean, window_mean, SIDE_COLUMN_ANCHOR   # noqa: E402
from analyze_exp3 import first_crossing  # noqa: E402

# ── 预注册阈值（DECISIONS D-029 / D-031 / D-028；改动要在 DECISIONS 留记录）────
D029_PM_FLOOR = {"flat": 0.7367, "2edge_distributed": 0.7297}
D031_PM_TOL = 0.006
D031_ASR_TOL = 0.07
DET_ROUNDS = 5
CELLS = ("flat", "2edge_distributed")

# G2P（D-047）
G2P_THETA = 0.5
G2P_BAND = (0.9, 1.1)                  # 「约等于 1」，不计为不一致
G2P_HFL = ("2edge_distributed", "4edge_distributed", "10edge_distributed")
P1_RESULTS = HERE.parent / "experiments/attack/hfl-propagation/results"
P1_FILES = {"flat": "flat_baseline_seed42.metrics.json",
            **{c: f"{c}_seed42.metrics.json" for c in G2P_HFL}}


def _mean10(rows, key):
    # 副列（D-054 起陈旧 pm_acc 隔点算）按「末 10 个评估点窗口」取，不能先丢 None 再往回够
    if key in SIDE_COLUMN_ANCHOR:
        return window_mean(rows, key, anchor=SIDE_COLUMN_ANCHOR[key])
    mean, n = last_k_mean([r.get(key) for r in rows or []])
    return mean, n


def invalid_reasons(m: dict) -> list:
    """这个 run 能不能算数（前提，不是阈值；D-044）。空列表 = 有效。"""
    reasons = []
    if m.get("exit_code") not in (0, None):
        reasons.append(f"exit_code={m.get('exit_code')}（崩溃，看 errors[]）")
    fails = m.get("client_failures") or []
    if fails:
        ids = sorted({f["client_id"] for f in fails if "client_id" in f}
                     | {i for f in fails for i in f.get("dropped") or []})
        reasons.append(f"client_failures 非空（{len(fails)} 条，client_id={ids}）："
                       f"客户端异常被吞、该端被踢出聚合")
    return reasons


def _invalid(extra: dict, **runs) -> dict | None:
    """runs：{标签: metrics}；多于一个时每条原因前面标上是哪个 run 的。"""
    tag = len(runs) > 1
    reasons = [f"{name}: {r}" if tag else r
               for name, m in runs.items() for r in invalid_reasons(m)]
    return {**extra, "verdict": "invalid", "reasons": reasons} if reasons else None


def judge_d029(m: dict | None, cell: str) -> dict:
    if m is None:
        return {"cell": cell, "verdict": "missing"}
    bad = _invalid({"cell": cell}, run=m)
    if bad:
        return bad
    run = m.get("run") or {}
    pm, n = _mean10(m.get("acc_rounds"), "pm_acc_stale")
    asr, _ = _mean10(m.get("rounds"), "local_benign_asr")
    floor = D029_PM_FLOOR[cell]
    reasons = []
    if run.get("stop_reason") != "converged":
        reasons.append(f"stop_reason={run.get('stop_reason')!r}（要求 converged）")
    if pm is None:
        reasons.append("没有 pm_acc_stale（陈旧 PM 列缺失：evaluation.pm_model 不是 fresh？）")
    elif pm < floor:
        reasons.append(f"陈旧 PM 的 pm_acc 末 10 点 {pm:.4f} < 门槛 {floor}")
    return {"cell": cell, "verdict": "fail" if reasons else "pass", "reasons": reasons,
            "pm_acc_stale_last10": pm, "n_points": n, "floor": floor,
            "stop_reason": run.get("stop_reason"),
            "stopped_at_effective": run.get("stopped_at_effective"),
            "local_benign_asr_last10": asr}


def judge_d031(head_first: dict | None, body_first: dict | None, cell: str) -> dict:
    if head_first is None or body_first is None:
        return {"cell": cell, "verdict": "missing"}
    bad = _invalid({"cell": cell}, head_first=head_first, body_first=body_first)
    if bad:
        return bad
    pm_h, _ = _mean10(head_first.get("acc_rounds"), "pm_acc")
    pm_b, _ = _mean10(body_first.get("acc_rounds"), "pm_acc")
    a_h, _ = _mean10(head_first.get("rounds"), "local_benign_asr")
    a_b, _ = _mean10(body_first.get("rounds"), "local_benign_asr")
    if None in (pm_h, pm_b, a_h, a_b):
        return {"cell": cell, "verdict": "missing", "reasons": ["末 10 点均值算不出来"]}
    d_pm, d_asr = abs(pm_b - pm_h), abs(a_b - a_h)
    same = d_pm <= D031_PM_TOL and d_asr <= D031_ASR_TOL
    return {"cell": cell, "verdict": "same" if same else "different",
            "pm_acc_head_first": pm_h, "pm_acc_body_first": pm_b, "abs_d_pm_acc": d_pm,
            "asr_head_first": a_h, "asr_body_first": a_b, "abs_d_asr": d_asr,
            "tol_pm_acc": D031_PM_TOL, "tol_asr": D031_ASR_TOL}


def judge_det(a: dict | None, b: dict | None, k: int = DET_ROUNDS) -> dict:
    if a is None or b is None:
        return {"verdict": "missing"}
    bad = _invalid({}, rep1=a, rep2=b)
    if bad:
        return bad
    ca = {c["round"]: c["global"] for c in a.get("checksums") or []}
    cb = {c["round"]: c["global"] for c in b.get("checksums") or []}
    rounds = list(range(1, k + 1))
    if not all(r in ca and r in cb for r in rounds):
        return {"verdict": "missing", "reasons": [f"前 {k} 轮的 [Checksum] 不全"]}
    first_diff = next((r for r in rounds if ca[r] != cb[r]), None)
    return {"verdict": "pass" if first_diff is None else "fail",
            "first_diverging_round": first_diff,
            "checksums": [[r, ca[r], cb[r]] for r in rounds]}


def t50(m: dict, key: str = "local_benign_asr", theta: float = G2P_THETA):
    """有效轮上的首次越阈（插值）；没越过 / 点太少 → None。"""
    eff = (m.get("run") or {}).get("edge_rounds") or 1
    series = [(r["round"] * eff, r[key]) for r in m.get("rounds") or [] if r.get(key) is not None]
    return first_crossing(series, theta).t_theta


def _side(r: float) -> str:
    lo, hi = G2P_BAND
    return "≈1" if lo <= r <= hi else ("<1" if r < 1 else ">1")


def judge_g2p(new: dict, p1: dict) -> dict:
    """new / p1：{"flat" 与 G2P_HFL 里的格: metrics 或 None}。"""
    cells = ("flat",) + G2P_HFL
    miss = [c for c in cells if new.get(c) is None]
    if miss:
        return {"verdict": "missing", "reasons": [f"新结果缺 {miss}"]}
    miss = [c for c in cells if p1.get(c) is None]
    if miss:
        return {"verdict": "missing", "reasons": [f"旧 P1 参照缺 {miss}"]}
    bad = _invalid({}, **{c: new[c] for c in cells})
    if bad:
        return bad
    t_new = {c: t50(new[c]) for c in cells}
    t_p1 = {c: t50(p1[c]) for c in cells}
    cens = [f"新 {c}" for c, t in t_new.items() if t is None] + \
           [f"P1 {c}" for c, t in t_p1.items() if t is None]
    rows, agree = {}, True
    for c in G2P_HFL:
        if None in (t_new[c], t_new["flat"], t_p1[c], t_p1["flat"]):
            continue
        r_new, r_p1 = t_new[c] / t_new["flat"], t_p1[c] / t_p1["flat"]
        s_new, s_p1 = _side(r_new), _side(r_p1)
        ok = "≈1" in (s_new, s_p1) or s_new == s_p1
        agree &= ok
        g_new = [t50(new[x], "global_asr") for x in (c, "flat")]
        rows[c] = {"ratio_new": round(r_new, 3), "ratio_p1": round(r_p1, 3),
                   "side_new": s_new, "side_p1": s_p1, "agree": ok,
                   "ratio_new_global_asr": (round(g_new[0] / g_new[1], 3)
                                            if None not in g_new else None)}
    out = {"t50_new": {c: None if t is None else round(t, 2) for c, t in t_new.items()},
           "t50_p1": {c: None if t is None else round(t, 2) for c, t in t_p1.items()},
           "cells": rows, "band": list(G2P_BAND)}
    if cens:
        return {**out, "verdict": "undetermined", "reasons": [f"T50 删失（没越过 {G2P_THETA}）：{cens}"]}
    return {**out, "verdict": "consistent" if agree else "inconsistent", "reasons": []}


def load_p1_reference(results_dir=P1_RESULTS) -> dict:
    out = {}
    for c, name in P1_FILES.items():
        p = Path(results_dir) / name
        out[c] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    return out


def judge_all(metrics: dict, p1: dict | None = None) -> dict:
    """metrics：{(group, cell): metrics dict 或 None}；p1：旧 P1 参照（缺省从盘上读）。"""
    d029 = {c: judge_d029(metrics.get(("D029", c)), c) for c in CELLS}
    d031 = {c: judge_d031(metrics.get(("D029", c)), metrics.get(("A26", c)), c) for c in CELLS}
    det = judge_det(metrics.get(("DET", "rep1")), metrics.get(("DET", "rep2")))
    g2p_new = {"flat": metrics.get(("D029", "flat")),
               "2edge_distributed": metrics.get(("D029", "2edge_distributed")),
               "4edge_distributed": metrics.get(("G2P", "4edge_distributed")),
               "10edge_distributed": metrics.get(("G2P", "10edge_distributed"))}
    g2p = judge_g2p(g2p_new, load_p1_reference() if p1 is None else p1)

    def _overall(parts, ok):
        vs = [p["verdict"] for p in parts]
        if "invalid" in vs:
            return "invalid"
        if "missing" in vs:
            return "missing"
        return ok if all(v == ok for v in vs) else ("fail" if ok == "pass" else "different")

    return {
        "D-029": {"overall": _overall(d029.values(), "pass"), "cells": d029},
        "D-031": {"overall": _overall(d031.values(), "same"), "cells": d031},
        "A15-determinism": det,
        "G2P": g2p,
        "next": {
            "D-029 pass": "A08 → deviate，A25 → done（AUDIT 与 test_registry 同步改）",
            "D-029 fail": "逐个开关消融（模板 + set 改回一项，登记进本 pilot 表），回审计",
            "D-031 same": "维持 head_first，A26 → deviate",
            "D-031 different": "带回审计，由用户定",
            "A15 pass": "A15 → done；fail：看第一个分叉轮，回审计（D-028）",
            "invalid": "不是结论：看 reasons / errors[] / client_failures[]，修好后重跑（D-044）",
            "G2P consistent / inconsistent": "带回给用户定 G2 的规模（D-047）；不代替 3-A 的 5 seed 判定",
        },
    }


def load_pilot(registry_path) -> dict:
    reg = Registry(registry_path)
    out = {}
    for run in reg.runs():
        p = reg.metrics_path(run)
        out[(run["group"], run["cell"])] = (json.loads(p.read_text(encoding="utf-8"))
                                            if p.is_file() else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="A4 pilot 的预注册判定")
    ap.add_argument("registry", help="experiments/attack/hfl-mechanism/pilot/registry.yaml")
    ap.add_argument("--json", help="把判定写到这个文件")
    a = ap.parse_args(argv)
    res = judge_all(load_pilot(a.registry))
    for name in ("D-029", "D-031"):
        print(f"{name}: {res[name]['overall']}")
        for cell, r in res[name]["cells"].items():
            extra = {k: v for k, v in r.items() if k not in ("cell", "verdict")}
            print(f"  {cell:<18} {r['verdict']:<9} {json.dumps(extra, ensure_ascii=False)}")
    print(f"A15 确定性: {res['A15-determinism']['verdict']} "
          f"{json.dumps({k: v for k, v in res['A15-determinism'].items() if k != 'verdict'}, ensure_ascii=False)}")
    g = res["G2P"]
    print(f"G2P（G2 一致性复测）: {g['verdict']} {'; '.join(g.get('reasons', []))}")
    for cell, r in (g.get("cells") or {}).items():
        print(f"  {cell:<20} {json.dumps(r, ensure_ascii=False)}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
