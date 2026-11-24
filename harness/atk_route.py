#!/usr/bin/env python3
"""
harness/atk_route.py  —  攻击基线第一阶段（探路）的分流（PREREG-v2 第七节 A1；N-015 / D-135；接线 D-141）

规则全文：`experiments/thesis/PREREG-v2.md`（commit 25163a5，2026-11-22）第七节「阶段 A1：探路与分流」。
**本脚本写于规则提交之后、任何静态攻击的数据回传之前**（D-141 接线会话，2026-11-24）。阈值只在下面的常量里定义一次。
这是**路由决定，不是论文结论**：它决定每种攻击下一步去哪（进第二阶段正式实验 / 补种子 43、44）。

    python3 harness/atk_route.py                    # 读 experiments/defense/edge-native/results/P2/ATK，写 analysis/atk_route.json
    python3 harness/atk_route.py --out x.json
    python3 harness/atk_route.py --probe            # 探路组 ATKP 的放行检查（不是判定）→ analysis/atkp_check.json

探路组 ATKP（4 云轮）的放行检查（--probe）：四格都过同一道闸（轮数改成 4）且有云聚合后评估点的攻击成功率；
badnet 与 badnet-rep（同配置）`instrumentation_check.compare` 逐位相同（checksum + 全部评估数值）。全过 → pass →
把 `atkp-ok` 加进登记表的 available、全量 materialize、交 ATK。

量（与 d0_power / supp_verdict 同一口径，末 10 个评估点；集中布点 [10,0,0,0]）：
  B0   被攻陷边缘（边缘 0）良性端的原始攻击成功率：`per_edge_rounds[r][0].client_benign` 末 10 点均值
  V    受害边缘（边缘 1–3）良性端的原始攻击成功率：每台边缘末 10 点均值再平均
  另报（不进分流）：P（全部良性端池化）、MTA（`acc_rounds[].pm_acc`，含攻击者的全部客户端 fresh 个性化精度）、
  参照列「干净样本被判成目标类的比例」（`[EvalDetail]` 的 yt_clean_benign：池化 = rounds[]，分边缘 =
  per_edge_detail_rounds）—— 静态触发器没有 Bad-PFL 那种「投毒率 0 的影子攻击者」底线，原始攻击成功率旁边附上它作参照
  （报告里约 0.011–0.020）。

分流（按顺序取第一条成立的；门槛比较带 1e-9 容差，同 D-133）：
  propagates   V  ≥ 0.30                  → 进第二阶段（{不设防 + 补 2 个种子, B=1, 上传前加固} × 集中 × 3 种子）
  local_only   B0 ≥ 0.30 且 V < 0.30      → 补种子 43、44（不设防），作问题一的证据
  no_implant   B0 < 0.30                  → 补种子 43、44（不设防），作问题一的证据
多个种子时：三个种子（42 / 43 / 44）都在同一分支才写成结论（`conclusion`），否则 `single_seed_observation`。

有效性闸（任一不过 → invalid，不分流，排查后重跑）：
  ① 同 supp_verdict.gate：exit 0、client_failures / errors 为空、跑满 60 云轮、攻击者参与过、P2 模板；
  ③ run 块因素：块数 0、布点 [10,0,0,0]、不设防、attack = static、static = true、trigger = 本格的触发器；
  另（D-141 新增，防「恶意端静默变成良性端」）：每个参与过的攻击者在 poison_counts 里都有记录、累计被投毒张数 > 0，
  且报告投毒的端都是攻击者。文件缺失 → missing（不是 invalid）。

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
from d0_power import last_k_mean, run_quantities               # noqa: E402
from instrumentation_check import compare                       # noqa: E402
from supp_verdict import _ge, gate as supp_gate                 # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "experiments/defense/edge-native/results/P2"
OUT = ROOT / "experiments/defense/edge-native/analysis/atk_route.json"
PROBE_OUT = ROOT / "experiments/defense/edge-native/analysis/atkp_check.json"
PROBE_GROUP, PROBE_ROUNDS = "ATKP", 4
PROBE_CELLS = ("badnet", "badnet-rep", "blended", "dba")
PREREG = ("N-015（experiments/thesis/PREREG-v2.md @ 25163a5，第七节 A1）；"
          "分流脚本写于规则之后、静态攻击数据之前（D-141）")

GROUP = "ATK"
TRIGGERS = ("badnet", "blended", "dba")
SEEDS = (42, 43, 44)
COLLOCATED = [10, 0, 0, 0]
VICTIMS = (1, 2, 3)
CONTROL = ("SNAP", "SNAP__collocated__s42")      # Bad-PFL 在同条件下的对照（只作参照行，不分流）

# ── 预注册阈值（PREREG-v2 第七节；用户 11-22 照草案，D-135 ③）──
ROUTE_THRESHOLD = 0.30
NEXT_STEP = {
    "propagates": "第二阶段：{不设防（补 2 个种子）, B=1, 上传前加固} × 集中 × 3 种子",
    "local_only": "补种子 43、44（不设防），作问题一的证据",
    "no_implant": "补种子 43、44（不设防），作问题一的证据",
}


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return math.fsum(xs) / len(xs) if xs else None


# ══════════════════════════════════════════════════════════════════════════
# 每个 run
# ══════════════════════════════════════════════════════════════════════════

def classify(b0, v) -> str | None:
    """分流规则本身（按表的顺序）。任一量缺失 → None。"""
    if b0 is None or v is None:
        return None
    if _ge(v, ROUTE_THRESHOLD):
        return "propagates"
    if _ge(b0, ROUTE_THRESHOLD):
        return "local_only"
    return "no_implant"


def _edge_detail_last10(m: dict, edge_id: int, key: str = "yt_clean_benign"):
    cols = list(m.get("per_edge_detail_columns") or [])
    if key not in cols or "edge_id" not in cols:
        return None
    ei, ki = cols.index("edge_id"), cols.index(key)
    series = []
    for rnd, rows in sorted((m.get("per_edge_detail_rounds") or {}).items(), key=lambda t: int(t[0])):
        row = next((r for r in rows or [] if r and int(r[ei]) == edge_id), None)
        if row is not None and row[ki] is not None:
            series.append(row[ki])
    return last_k_mean(series)[0]


def reference_columns(m: dict) -> dict:
    """干净样本被判成目标类的比例（参照列，不进分流）。"""
    victims = [_edge_detail_last10(m, e) for e in VICTIMS]
    return {
        "yt_clean_pooled": _r(last_k_mean([r.get("yt_clean_benign") for r in m.get("rounds") or []])[0]),
        "yt_clean_e0": _r(_edge_detail_last10(m, 0)),
        "yt_clean_victims": _r(None if any(v is None for v in victims) else _mean(victims)),
    }


def poison_reasons(m: dict) -> list:
    """D-141 的闸：参与过的攻击者都真的投了毒；报告投毒的都是攻击者。"""
    reasons = []
    counts = {int(r["client"]): r for r in m.get("poison_counts") or []}
    mal = {int(i) for i in ((m.get("run") or {}).get("malicious_ids") or [])}
    part = {int(c) for c, rounds in (m.get("malicious_participation_by_client") or {}).items() if rounds}
    if not counts:
        return ["poison_counts 为空：没有一行 [Poison] —— 恶意端可能静默变成了良性端"]
    silent = sorted(c for c in part if counts.get(c, {}).get("n_poisoned", 0) <= 0)
    if silent:
        reasons.append(f"参与过却一张都没投毒的攻击者：{silent}")
    stray = sorted(set(counts) - mal) if mal else []
    if stray:
        reasons.append(f"报告投毒的端不在 malicious_ids 里：{stray}")
    return reasons


def gate(m: dict, trigger: str, n_rounds: int | None = None) -> list:
    """n_rounds：缺省 = 正式组的 60 云轮（supp_verdict 的闸）；探路组传 4，把「跑满」那一条换成 4。"""
    reasons = supp_gate(m, k=0, mpe=COLLOCATED, defense="none")
    if n_rounds is not None:
        reasons = [r for r in reasons if not r.startswith("没有跑满")]
        run = m.get("run") or {}
        last = max((r.get("round") or 0 for r in m.get("acc_rounds") or []), default=0)
        if run.get("n_rounds") != n_rounds or last != n_rounds:
            reasons.append(f"没有跑满 {n_rounds} 云轮（run.n_rounds={run.get('n_rounds')}，acc_rounds 最后一轮 {last}）")
    run = m.get("run") or {}
    if run.get("attack") != "static":
        reasons.append(f"run.attack={run.get('attack')!r} ≠ 'static'")
    if run.get("static") is not True:
        reasons.append(f"run.static={run.get('static')!r}（缺 [设定15] 或不是 static）")
    if run.get("trigger") != trigger:
        reasons.append(f"run.trigger={run.get('trigger')!r} ≠ {trigger!r}")
    return reasons + poison_reasons(m)


def poison_summary(m: dict) -> dict:
    rows = m.get("poison_counts") or []
    n_p = sum(int(r.get("n_poisoned") or 0) for r in rows)
    n_s = sum(int(r.get("n_seen") or 0) for r in rows)
    return {"n_clients": len(rows), "n_poisoned": n_p, "n_seen": n_s,
            "rate": _r(n_p / n_s if n_s else None)}


def route_run(m: dict | None, trigger: str) -> dict:
    if m is None:
        return {"route": "missing"}
    bad = gate(m, trigger)
    q = run_quantities(m)
    out = {"B0": _r(q["B0"]), "V": _r(q["V"]), "P": _r(q["P"]), "MTA": _r(q["MTA"]),
           **reference_columns(m), "poison": poison_summary(m)}
    if bad:
        return {"route": "invalid", "reasons": bad, **out}
    return {"route": classify(q["B0"], q["V"]), **out}


def conclude(routes: dict) -> dict:
    """{seed: route_run(...)} → 三个种子都在同一分支才是结论；只有 s42 → 下一步按 s42 的分支走。"""
    got = {s: r["route"] for s, r in routes.items() if r["route"] not in ("missing", None)}
    if any(v == "invalid" for v in got.values()):
        return {"status": "invalid_present", "next": "排查 invalid 的 run 后重跑"}
    if 42 not in got:
        return {"status": "insufficient", "next": "等 s42 回传"}
    first = got[42]
    if all(s in got for s in SEEDS):
        same = len({got[s] for s in SEEDS}) == 1
        return {"status": "conclusion" if same else "single_seed_observation",
                "branch": first if same else None, "branches": {str(s): got[s] for s in SEEDS},
                "next": NEXT_STEP[first] if same else "写成单种子观察"}
    return {"status": "routed_on_s42", "branch": first, "next": NEXT_STEP[first]}


# ══════════════════════════════════════════════════════════════════════════
# 读文件 + 主程序
# ══════════════════════════════════════════════════════════════════════════

def _load(results: Path, group: str, run_id: str):
    p = results / group / f"{run_id}.metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run_all(results: Path = RESULTS) -> dict:
    out = {"preregistered": PREREG, "threshold": ROUTE_THRESHOLD, "attacks": {}}
    for trig in TRIGGERS:
        routes = {s: route_run(_load(results, GROUP, f"{GROUP}__{trig}__s{s}"), trig) for s in SEEDS}
        routes = {s: r for s, r in routes.items() if not (s != 42 and r["route"] == "missing")}
        out["attacks"][trig] = {"runs": {str(s): r for s, r in routes.items()}, **conclude(routes)}
    ctrl = _load(results, *CONTROL)
    if ctrl is not None:
        q = run_quantities(ctrl)
        out["control_badpfl"] = {"run_id": CONTROL[1], "B0": _r(q["B0"]), "V": _r(q["V"]),
                                 "P": _r(q["P"]), "MTA": _r(q["MTA"]),
                                 "route_if_applied": classify(q["B0"], q["V"]), **reference_columns(ctrl)}
    return out


def _last_edge_value(m: dict, edge_id: int, key: str = "client_benign"):
    for _, rows in sorted((m.get("per_edge_rounds") or {}).items(), key=lambda t: -int(t[0])):
        row = next((e for e in rows or [] if int(e.get("edge_id", -1)) == edge_id), None)
        if row is not None and row.get(key) is not None:
            return row[key]
    return None


def probe_check(results: Path = RESULTS) -> dict:
    """ATKP 的放行检查（不是判定）：闸 + 云聚合后评估点 + 同配置两次逐位相同。"""
    cells, reasons = {}, []
    runs = {c: _load(results, PROBE_GROUP, f"{PROBE_GROUP}__{c}__s42") for c in PROBE_CELLS}
    for cell, m in runs.items():
        if m is None:
            cells[cell] = {"status": "missing"}
            reasons.append(f"{cell}: 缺文件")
            continue
        bad = gate(m, cell.replace("-rep", ""), n_rounds=PROBE_ROUNDS)
        post = m.get("post_agg_rounds") or []
        if not post or any(p.get("local_benign_asr") is None for p in post):
            bad.append(f"云聚合后评估点缺失或攻击成功率为空（{len(post)} 个点）")
        cells[cell] = {"status": "ok" if not bad else "invalid", "reasons": bad,
                       "poison": poison_summary(m), "n_post_agg": len(post),
                       "B0_last": _r(_last_edge_value(m, 0))}      # 只是给人看的：4 云轮里 E0 有没有开始涨
        reasons += [f"{cell}: {x}" for x in bad]
    rep = None
    if runs["badnet"] is not None and runs["badnet-rep"] is not None:
        c = compare(runs["badnet"], runs["badnet-rep"])
        rep = {"n_checksums": c["n_checksums"], "n_fields": c["n_fields"],
               "n_diffs": len(c["diffs"]), "n_missing": len(c["missing"]), "first_diffs": c["diffs"][:5]}
        if c["diffs"] or c["missing"] or c["n_checksums"] != PROBE_ROUNDS:
            reasons.append(f"badnet vs badnet-rep 不逐位相同：{rep}")
    return {"check": "ATKP 放行（D-141；不是预注册判定）", "verdict": "pass" if not reasons else "fail",
            "reasons": reasons, "cells": cells, "repeat": rep}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--probe", action="store_true", help="探路组 ATKP 的放行检查")
    a = ap.parse_args(argv)
    if a.probe:
        res = probe_check(a.results)
        out = a.out or PROBE_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        for cell, d in res["cells"].items():
            print(f"{cell:11s} {d['status']:8s} {d.get('poison')} post_agg={d.get('n_post_agg')} {d.get('reasons') or ''}")
        print(f"重复格：{res['repeat']}")
        print(f"ATKP → {res['verdict']}  {res['reasons'] or ''}\n→ {out}")
        return 0 if res["verdict"] == "pass" else 1
    a.out = a.out or OUT
    res = run_all(a.results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for trig, d in res["attacks"].items():
        r42 = d["runs"].get("42", {})
        print(f"{trig:8s} s42: {r42.get('route')!s:12s} B0={r42.get('B0')} V={r42.get('V')} "
              f"yt_clean={r42.get('yt_clean_pooled')} | {d['status']} → {d.get('next')}")
    if "control_badpfl" in res:
        c = res["control_badpfl"]
        print(f"对照 Bad-PFL {c['run_id']}: B0={c['B0']} V={c['V']} → {c['route_if_applied']}")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
