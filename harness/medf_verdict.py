#!/usr/bin/env python3
"""
harness/medf_verdict.py  —  公平的边缘内中位数（MEDF，预注册 N-017 / D-142）的判定

规则全文：`experiments/defense/edge-native/FINDINGS.md` 的 N-017。**本脚本写于规则之后、任何 MEDF 数据回传之前**；
阈值只在下面的常量里定义一次，改它就是改预注册（守卫 tests/test_medf_verdict.py 有阈值反向锚点）。

    python3 harness/medf_verdict.py                    # 读 experiments/defense/edge-native/results/P2/MEDF
    python3 harness/medf_verdict.py --out x.json

两个 run（种子 42，单种子 = 探路）：
  不设防  MEDF__ctrl-t1-distributed__s42
  中位数  MEDF__edge-t1-distributed__s42
共同设定：参与率 0.2（每台边缘每个边缘轮 5 个上传）、分散布点 [3,3,2,2]、三层个性化 B=1、30 云轮（150 有效轮）。

量（第 21–30 云轮的全量评估点均值；R5 + 网格 5 → 每个云轮末都是全量点，窗口必须 10 个点齐全）：
  P    全部良性端池化攻击成功率 `rounds[].local_benign_asr`
  MTA  `acc_rounds[].pm_acc`（全部 100 个客户端的 fresh 个性化精度，D-071 口径）
  ΔP = P(不设防) − P(中位数)；ΔMTA = MTA(不设防) − MTA(中位数)

有效性闸（任一不过 → invalid）：exit 0、client_failures / errors 为空、攻击者参与过、跑满 30 云轮、P2 模板、
run 块因素（参与率 0.2、B=1、布点、防御名与作用层）与登记表一致；中位数臂的每一次边缘聚合都恰好 5 个上传
（`admitted[].total`，确认参与率真的生效 —— 陷阱 #7 / #19 同类）。不设防臂 P < 0.8 → invalid_no_attack（后门没起来，比较无意义）。
文件缺失 → insufficient。

标签（带 1e-9 容差，同 D-133）：
  accuracy_bound  ΔMTA > 0.02
  useful          ΔMTA ≤ 0.02 且 ΔP ≥ 0.30           → 才补种子 43 / 44
  weak            ΔMTA ≤ 0.02 且 0 < ΔP < 0.30
  no_effect       ΔMTA ≤ 0.02 且 ΔP ≤ 0

只报告（不进标签）：逐边缘良性端攻击成功率（同窗口）、「每次聚合攻击者过半」的理论概率（超几何分布；
中位数的 [Decision] 行不带客户端编号，实际比例算不出来 —— N-017 写明的偏离）。

纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d0_power import edge_by_round                              # noqa: E402
from pilot_a4 import invalid_reasons                            # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "experiments/defense/edge-native/results/P2"
OUT = ROOT / "experiments/defense/edge-native/analysis/medf_verdict.json"
PREREG = "N-017（experiments/defense/edge-native/FINDINGS.md）；判定脚本写于规则之后、数据之前"

GROUP = "MEDF"
SEED = 42
CTRL = f"{GROUP}__ctrl-t1-distributed__s{SEED}"
ARM = f"{GROUP}__edge-t1-distributed__s{SEED}"

# ── 设定（与登记表 MEDF 一致；闸逐项核对）──
N_ROUNDS = 30
CLIENT_FRACTION = 0.2
UPLOADS_PER_AGG = 5                 # 0.2 × 25 端
DISTRIBUTED = [3, 3, 2, 2]
CLIENTS_PER_EDGE = 25
WINDOW = tuple(range(21, 31))       # 第 21–30 云轮
TOL = 1e-9

# ── 预注册阈值（N-017）──
CTRL_MIN_P = 0.80                   # 不设防臂 P ≥ 0.8，否则 invalid_no_attack
USEFUL_DP = 0.30                    # ΔP ≥ 0.30
MTA_MAX = 0.02                      # ΔMTA ≤ 0.02（D-071）


def _ge(x, t):
    return x >= t - TOL


def _le(x, t):
    return x <= t + TOL


def _gt(x, t):
    return x > t + TOL


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


# ══════════════════════════════════════════════════════════════════════════
# 量
# ══════════════════════════════════════════════════════════════════════════

def window_mean(rows: list, key: str, window=WINDOW):
    """rows 里 round ∈ window 的 key 的均值；任一轮缺失或为 None → None（窗口必须齐全）。"""
    by = {r.get("round"): r.get(key) for r in rows or []}
    vals = [by.get(t) for t in window]
    if any(v is None for v in vals):
        return None
    return math.fsum(vals) / len(vals)


def pooled(m: dict):
    return window_mean(m.get("rounds"), "local_benign_asr")


def mta(m: dict):
    return window_mean(m.get("acc_rounds"), "pm_acc")


def per_edge(m: dict, n_edges: int = 4) -> dict:
    out = {}
    for e in range(n_edges):
        by = edge_by_round(m, e)
        vals = [by.get(t) for t in WINDOW]
        out[e] = None if any(v is None for v in vals) else _r(math.fsum(vals) / len(vals))
    return out


def p_majority(n: int, a: int, m: int) -> float:
    """n 端里 a 个攻击者、无放回选 m 个，攻击者 ≥ 一半（偶数 m 取 m/2，奇数取 (m+1)/2）的概率。"""
    k0 = (m + 1) // 2 if m % 2 else m // 2
    return math.fsum(comb(a, k) * comb(n - a, m - k) for k in range(k0, min(a, m) + 1)) / comb(n, m)


# ══════════════════════════════════════════════════════════════════════════
# 闸
# ══════════════════════════════════════════════════════════════════════════

def gate(m: dict, *, defense: str, layers=None) -> list:
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
    cf = run.get("client_fraction")
    if cf is None or abs(float(cf) - CLIENT_FRACTION) > TOL:
        reasons.append(f"run.client_fraction={cf!r} ≠ {CLIENT_FRACTION}")
    if int(run.get("edge_shared_blocks") or 0) != 1:
        reasons.append(f"run.edge_shared_blocks={run.get('edge_shared_blocks')!r} ≠ 1")
    if list(run.get("malicious_per_edge") or []) != DISTRIBUTED:
        reasons.append(f"run.malicious_per_edge={run.get('malicious_per_edge')!r} ≠ {DISTRIBUTED}")
    if run.get("defense") != defense:
        reasons.append(f"run.defense={run.get('defense')!r} ≠ {defense!r}")
    if layers is not None:
        if run.get("defense_layers") != list(layers):
            reasons.append(f"run.defense_layers={run.get('defense_layers')!r} ≠ {list(layers)}")
        edge_aggs = [a for a in m.get("admitted") or [] if str(a.get("layer", "")).startswith("edge")]
        bad = sorted({a.get("total") for a in edge_aggs if a.get("total") != UPLOADS_PER_AGG}, key=str)
        if not edge_aggs:
            reasons.append("admitted 里没有边缘聚合的 [Decision] 行（中位数没有生效？）")
        elif bad:
            reasons.append(f"有边缘聚合的上传数 ≠ {UPLOADS_PER_AGG}：{bad}（参与率没生效）")
    return reasons


# ══════════════════════════════════════════════════════════════════════════
# 判定
# ══════════════════════════════════════════════════════════════════════════

def classify(d_p: float, d_mta: float) -> str:
    if _gt(d_mta, MTA_MAX):
        return "accuracy_bound"
    if _ge(d_p, USEFUL_DP):
        return "useful"
    if _gt(d_p, 0.0):
        return "weak"
    return "no_effect"


def judge(ctrl: dict | None, arm: dict | None) -> dict:
    out = {"prereg": PREREG, "runs": {"ctrl": CTRL, "median": ARM},
           "thresholds": {"ctrl_min_p": CTRL_MIN_P, "useful_dp": USEFUL_DP, "mta_max": MTA_MAX, "tol": TOL},
           "window_rounds": [WINDOW[0], WINDOW[-1]],
           "p_majority_theory": {f"{a}/{CLIENTS_PER_EDGE}": _r(p_majority(CLIENTS_PER_EDGE, a, UPLOADS_PER_AGG), 6)
                                 for a in sorted(set(DISTRIBUTED))}}
    missing = [rid for rid, m in ((CTRL, ctrl), (ARM, arm)) if m is None]
    if missing:
        return {**out, "verdict": "insufficient", "missing": missing}
    invalid = {}
    for rid, m, kw in ((CTRL, ctrl, {"defense": "none"}), (ARM, arm, {"defense": "median", "layers": ["edge"]})):
        if r := gate(m, **kw):
            invalid[rid] = r
    if invalid:
        return {**out, "verdict": "invalid", "reasons": invalid}
    q = {"ctrl": {"P": pooled(ctrl), "MTA": mta(ctrl)}, "median": {"P": pooled(arm), "MTA": mta(arm)}}
    if any(v is None for d in q.values() for v in d.values()):
        return {**out, "verdict": "invalid", "reasons": {"window": f"第 {WINDOW[0]}–{WINDOW[-1]} 云轮的全量评估点不齐全"},
                "quantities": {k: {kk: _r(vv) for kk, vv in d.items()} for k, d in q.items()}}
    report = {"quantities": {k: {kk: _r(vv) for kk, vv in d.items()} for k, d in q.items()},
              "per_edge_benign": {"ctrl": per_edge(ctrl), "median": per_edge(arm)}}
    if not _ge(q["ctrl"]["P"], CTRL_MIN_P):
        return {**out, **report, "verdict": "invalid_no_attack",
                "reasons": {CTRL: f"不设防臂 P = {q['ctrl']['P']:.4f} < {CTRL_MIN_P}"}}
    d_p = q["ctrl"]["P"] - q["median"]["P"]
    d_mta = q["ctrl"]["MTA"] - q["median"]["MTA"]
    return {**out, **report, "delta_P": _r(d_p), "delta_MTA": _r(d_mta), "verdict": classify(d_p, d_mta)}


def _load(results: Path, run_id: str):
    p = results / GROUP / f"{run_id}.metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run_all(results: Path = RESULTS) -> dict:
    return judge(_load(results, CTRL), _load(results, ARM))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = run_all(a.results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "delta_P", "delta_MTA", "missing", "reasons")
                      if k in res}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
