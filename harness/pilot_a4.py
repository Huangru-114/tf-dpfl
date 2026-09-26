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
from runs_table import last_k_mean     # noqa: E402

# ── 预注册阈值（DECISIONS D-029 / D-031 / D-028；改动要在 DECISIONS 留记录）────
D029_PM_FLOOR = {"flat": 0.7367, "2edge_distributed": 0.7297}
D031_PM_TOL = 0.006
D031_ASR_TOL = 0.07
DET_ROUNDS = 5
CELLS = ("flat", "2edge_distributed")


def _mean10(rows, key):
    mean, n = last_k_mean([r.get(key) for r in rows or []])
    return mean, n


def judge_d029(m: dict | None, cell: str) -> dict:
    if m is None:
        return {"cell": cell, "verdict": "missing"}
    run = m.get("run") or {}
    pm, n = _mean10(m.get("acc_rounds"), "pm_acc_stale")
    asr, _ = _mean10(m.get("rounds"), "local_benign_asr")
    floor = D029_PM_FLOOR[cell]
    reasons = []
    if m.get("exit_code") not in (0, None):
        reasons.append(f"exit_code={m.get('exit_code')}")
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
    ca = {c["round"]: c["global"] for c in a.get("checksums") or []}
    cb = {c["round"]: c["global"] for c in b.get("checksums") or []}
    rounds = list(range(1, k + 1))
    if not all(r in ca and r in cb for r in rounds):
        return {"verdict": "missing", "reasons": [f"前 {k} 轮的 [Checksum] 不全"]}
    first_diff = next((r for r in rounds if ca[r] != cb[r]), None)
    return {"verdict": "pass" if first_diff is None else "fail",
            "first_diverging_round": first_diff,
            "checksums": [[r, ca[r], cb[r]] for r in rounds]}


def judge_all(metrics: dict) -> dict:
    """metrics：{(group, cell): metrics dict 或 None}。"""
    d029 = {c: judge_d029(metrics.get(("D029", c)), c) for c in CELLS}
    d031 = {c: judge_d031(metrics.get(("D029", c)), metrics.get(("A26", c)), c) for c in CELLS}
    det = judge_det(metrics.get(("DET", "rep1")), metrics.get(("DET", "rep2")))

    def _overall(parts, ok):
        vs = [p["verdict"] for p in parts]
        if "missing" in vs:
            return "missing"
        return ok if all(v == ok for v in vs) else ("fail" if ok == "pass" else "different")

    return {
        "D-029": {"overall": _overall(d029.values(), "pass"), "cells": d029},
        "D-031": {"overall": _overall(d031.values(), "same"), "cells": d031},
        "A15-determinism": det,
        "next": {
            "D-029 pass": "A08 → deviate，A25 → done（AUDIT 与 test_registry 同步改）",
            "D-029 fail": "逐个开关消融（模板 + set 改回一项，登记进本 pilot 表），回审计",
            "D-031 same": "维持 head_first，A26 → deviate",
            "D-031 different": "带回审计，由用户定",
            "A15 pass": "A15 → done；fail：看第一个分叉轮，回审计（D-028）",
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
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
