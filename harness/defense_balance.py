"""
harness/defense_balance.py  —  论文整合：阶段三的全部防御臂放进同一把尺子（2026-11-19）

    python3 harness/defense_balance.py [--json OUT.json] [--md OUT.md]

**只报告、不判定**：描述性分析，不是预注册判定（`experiments/thesis/INTEGRATION-PLAN.md` 第二节）。
各臂的正式标签仍以各自的判定文件为准（h1 / p1 / edge_ref_filter_verdict）。

为什么：阶段二测出「每次云聚合把后门灌进干净边缘，两次云聚合之间边缘内训练洗掉一部分；
净增量 = 灌入 − 洗掉 > 0 就走向饱和」（hfl-mechanism REPORT §5.12 / §10.1）。这把尺子能把阶段三的
每一种防御放进同一个框架：减少灌入（三层个性化）、加快洗掉（上传前加固）、减少攻击者的贡献（参照过滤）。
阶段三的全部实验都开着云聚合后评估（`evaluation.post_agg_eval`），所以不需要新训练。

臂按 run 块的**实际**因素归类（不看文件名，同 runs_table 的原则）：
  划分     run.partition_profile == "fixed" → homo（组内更同质），否则 main（现主线）
  防御     run.edge_harden 不是 None / "none" → harden（上传前加固）；run.edge_ref_mode == "filter" → filter（参照过滤）；
           其余 → none（不设防；observe 只记录，已证与不设防逐位相同，F-126）
  三层个性化 run.edge_shared_blocks（0 = 关）
  布点     恰好第 0 台边缘有攻击者 → collocated（集中），否则 distributed（分散）
同一 (划分, 布点, 臂, 种子) 出现多次（SNAP 与 ERP main-t0 是同一个实验）→ 按 GROUPS 的顺序取第一个，其余记进 duplicates。

量（定义与 g1_verdict.sawtooth / d0_power 相同）：
  终值      V（受害边缘 E1–E3 良性端）/ B0（E0 良性端）/ P（全部良性端池化）/ MTA（个性化精度），末 10 个评估点（d0_power.run_quantities）
  灌入      jump_g = 第 g 云轮的云聚合后评估点 − 第 g−1 云轮末全量点
  洗掉      wash_g = 第 g 云轮的云聚合后评估点 − 第 g 云轮末全量点（一个周期洗掉的总量，= g1_verdict 的 r_down × R）
  净增量    net_g  = 第 g 云轮末 − 第 g−1 云轮末（= jump_g − wash_g）
  前半程 g = 2 … ⌊n/2⌋（同 g1_verdict / F-124），后半程 g = ⌊n/2⌋+1 … n；只对集中布点算（分散布点没有干净的受害边缘）。
  受害边缘取三台的均值；ASR 版（client_benign）与 margin 版（margin_p50，饱和时更有动态范围）各算一份；E0 只算 ASR 版。
  Δ = 同划分、同布点、同种子的不设防对照 − 本臂（正 = 本臂更低，同 edge_ref_filter_verdict）。
  参照过滤臂另报在线放行 / 误剔率与「整轮全被剔」的边缘轮占比（≥ 0.99 记为冷启动死锁，F-128：不作证据）。

纯标准库，不 import TF / numpy。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect_metrics import LIGHT_EDGE_COLUMNS     # noqa: E402
from d0_power import run_quantities                # noqa: E402
from decay_verdict import edge_by_round             # noqa: E402
from edge_ref_filter_verdict import online_rates   # noqa: E402
from pilot_a4 import invalid_reasons                # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "experiments/defense/edge-native/results/P2"
OUT = ROOT / "experiments/thesis/analysis/defense_balance.json"
GROUPS = ("SNAP", "P1", "H1", "ERP", "ERC", "ERF")   # 去重优先级：SNAP 在前（main-t0 对照取 SNAP）
VICTIMS, E0 = (1, 2, 3), 0
DEADLOCK_FRAC = 0.99
QUANTS = ("V", "B0", "P", "MTA")

ARM_LABELS = {
    ("none", 0): "不设防",
    ("none", 1): "三层个性化",
    ("harden", 0): "上传前加固",
    ("filter", 0): "参照过滤",
    ("filter", 1): "参照过滤 + 三层个性化",
}
ARM_ORDER = list(ARM_LABELS.values())


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return math.fsum(xs) / len(xs) if xs else None


# ══════════════════════════════════════════════════════════════════════════
# 归类
# ══════════════════════════════════════════════════════════════════════════

def arm_of(run: dict) -> dict:
    """run 块 → {partition, defense, tier, placement, arm}。"""
    harden = run.get("edge_harden")
    if harden not in (None, "none"):
        defense = "harden"
    elif run.get("edge_ref_mode") == "filter":
        defense = "filter"
    else:
        defense = "none"
    tier = int(run.get("edge_shared_blocks") or 0)
    mpe = list(run.get("malicious_per_edge") or [])
    nz = [i for i, x in enumerate(mpe) if x]
    placement = "collocated" if nz == [E0] else "distributed"
    return {"partition": "homo" if run.get("partition_profile") == "fixed" else "main",
            "defense": defense, "tier": tier, "placement": placement,
            "arm": ARM_LABELS.get((defense, tier), f"{defense}+k{tier}")}


# ══════════════════════════════════════════════════════════════════════════
# 序列
# ══════════════════════════════════════════════════════════════════════════

def full_series(m: dict, edges, col: str = "client_benign") -> dict:
    """{云轮 g: edges 的均值}（云轮末全量点）。client_benign 读 per_edge_rounds；margin_p50 读 per_edge_detail_rounds。
    某轮缺任何一台 edge → 该轮不计。"""
    if col == "client_benign":
        per = {e: edge_by_round(m, e) for e in edges}
    else:
        cols = list(m.get("per_edge_detail_columns") or [])
        if col not in cols or "edge_id" not in cols:
            return {}
        ei, ci = cols.index("edge_id"), cols.index(col)
        per = {e: {} for e in edges}
        for rnd, rows in (m.get("per_edge_detail_rounds") or {}).items():
            for row in rows or []:
                if row and row[ei] is not None and int(row[ei]) in per and row[ci] is not None:
                    per[int(row[ei])][int(rnd)] = row[ci]
    rounds = sorted(set().union(*[set(v) for v in per.values()])) if per else []
    out = {}
    for g in rounds:
        vals = [per[e].get(g) for e in edges]
        if all(v is not None for v in vals):
            out[g] = math.fsum(vals) / len(vals)
    return out


def post_series(m: dict, edges, col: str = "client_benign") -> dict | None:
    """{云轮 g: edges 的均值}（第 g 云轮的云聚合后评估点）。列序与 LIGHT_EDGE_COLUMNS 不符 → None（按位置读会读错列）。"""
    cols = m.get("per_edge_light_columns")
    if cols is not None and list(cols) != list(LIGHT_EDGE_COLUMNS):
        return None
    lc = list(LIGHT_EDGE_COLUMNS)
    ei, ci = lc.index("edge_id"), lc.index(col)
    out = {}
    for g, rows in (m.get("per_edge_post_agg_rounds") or {}).items():
        by_edge = {int(r[ei]): r[ci] for r in rows or [] if r and r[ei] is not None}
        vals = [by_edge.get(e) for e in edges]
        if all(v is not None for v in vals):
            out[int(g)] = math.fsum(vals) / len(vals)
    return out


def balance(full: dict, post: dict, lo: int, hi: int) -> dict:
    """g ∈ [lo, hi] 上灌入 / 洗掉 / 净增量的均值；缺点的 g 不计（n_used < n_expected 看得出）。"""
    jumps, washes, nets = [], [], []
    for g in range(lo, hi + 1):
        if g in post and g in full and (g - 1) in full:
            jumps.append(post[g] - full[g - 1])
            washes.append(post[g] - full[g])
            nets.append(full[g] - full[g - 1])
    return {"g_range": [lo, hi], "n_expected": max(0, hi - lo + 1), "n_used": len(jumps),
            "jump": _mean(jumps), "wash": _mean(washes), "net": _mean(nets)}


def halves(n_rounds: int):
    return (2, n_rounds // 2), (n_rounds // 2 + 1, n_rounds)


def reject_profile(m: dict) -> dict | None:
    """参照过滤的在线读数（edge_ref_filter_verdict.online_rates）+ 整轮全被剔的边缘轮占比。没有 admitted 列 → None。"""
    t = m.get("edge_ref_scores") or {}
    cols = t.get("columns") or []
    if "admitted" not in cols:
        return None
    rows = [dict(zip(cols, r)) for r in t.get("rows") or []]
    groups = {(r.get("round"), r.get("edge_id"), r.get("edge_round")) for r in rows if r.get("admitted") is not None}
    o = online_rates(m)
    frac = None if not groups else o["edge_rounds_all_rejected"] / len(groups)
    return {"attacker_pass": _r(o["attacker_pass"]), "benign_reject": _r(o["benign_reject"]),
            "edge_rounds_all_rejected": o["edge_rounds_all_rejected"], "edge_rounds": len(groups),
            "all_rejected_frac": _r(frac), "deadlock": frac is not None and frac >= DEADLOCK_FRAC}


# ══════════════════════════════════════════════════════════════════════════
# 每个 run
# ══════════════════════════════════════════════════════════════════════════

def run_row(m: dict) -> dict:
    run = m.get("run") or {}
    a = arm_of(run)
    reasons = list(invalid_reasons(m))
    if not (m.get("n_malicious_participations") or 0) > 0:
        reasons.append("攻击者一次也没参与")
    row = {**a, "seed": run.get("seed"), "invalid": reasons,
           "quantities": {k: _r(v) for k, v in run_quantities(m).items() if k in QUANTS},
           "balance": None, "online": reject_profile(m) if a["defense"] == "filter" else None}
    if a["placement"] != "collocated" or reasons:
        return row
    n = int(run.get("n_rounds") or 0)
    post_v, post_e0 = post_series(m, VICTIMS), post_series(m, (E0,))
    post_vm = post_series(m, VICTIMS, "margin_p50")
    if post_v is None or post_e0 is None or post_vm is None:
        row["invalid"].append(f"per_edge_light_columns ≠ {list(LIGHT_EDGE_COLUMNS)}（无法按位置读云聚合后点）")
        return row
    if not post_v:
        row["invalid"].append("没有云聚合后评估点（per_edge_post_agg_rounds）")
        return row
    full_v, full_e0 = full_series(m, VICTIMS), full_series(m, (E0,))
    full_vm = full_series(m, VICTIMS, "margin_p50")
    h1, h2 = halves(n)
    row["balance"] = {
        "victim_asr": {"first": balance(full_v, post_v, *h1), "second": balance(full_v, post_v, *h2)},
        "victim_margin": {"first": balance(full_vm, post_vm, *h1), "second": balance(full_vm, post_vm, *h2)},
        "e0_asr": {"first": balance(full_e0, post_e0, *h1), "second": balance(full_e0, post_e0, *h2)},
    }
    return row


# ══════════════════════════════════════════════════════════════════════════
# 载入、去重、汇总
# ══════════════════════════════════════════════════════════════════════════

def load(results_dir: Path = RESULTS, groups=GROUPS):
    """→ (rows, duplicates)。rows 每个 (划分, 布点, 臂, 种子) 只留一个（GROUPS 顺序优先）。不投毒的开发实验不收。"""
    rows, dups, seen = [], [], {}
    for g in groups:
        for p in sorted((Path(results_dir) / g).glob("*.metrics.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            run = m.get("run") or {}
            if not (run.get("poison_ratio") or 0) > 0:
                continue
            row = run_row(m)
            row["source"] = f"{g}/{p.name[:-len('.metrics.json')]}"
            key = (row["partition"], row["placement"], row["arm"], row["seed"])
            if key in seen:
                dups.append({"key": list(key), "kept": seen[key], "dropped": row["source"]})
                continue
            seen[key] = row["source"]
            rows.append(row)
    return rows, dups


def _flat(row: dict) -> dict:
    """一个 run 的标量，供均值与 Δ 用。"""
    out = dict(row["quantities"])
    b = row.get("balance")
    if b:
        for series, halves_ in b.items():
            for half, d in halves_.items():
                for k in ("jump", "wash", "net"):
                    out[f"{series}.{half}.{k}"] = d[k]
    return out


def assemble(rows: list) -> dict:
    valid = [r for r in rows if not r["invalid"]]
    cells = {}
    for r in valid:
        cells.setdefault((r["partition"], r["placement"], r["arm"]), []).append(r)
    ctrl = {(r["partition"], r["placement"], r["seed"]): _flat(r) for r in valid if r["arm"] == ARM_LABELS[("none", 0)]}
    out = []
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    for (part, place, arm), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 99), kv[0][2])):
        rs = sorted(rs, key=lambda r: r["seed"])
        flats = {r["seed"]: _flat(r) for r in rs}
        keys = sorted(set().union(*[set(f) for f in flats.values()]))
        mean = {k: _r(_mean([f.get(k) for f in flats.values()])) for k in keys}
        delta = {}
        if arm != ARM_LABELS[("none", 0)]:
            for k in QUANTS:
                per = {s: (None if ctrl.get((part, place, s), {}).get(k) is None or f.get(k) is None
                           else ctrl[(part, place, s)][k] - f[k]) for s, f in flats.items()}
                delta[k] = {"per_seed": {s: _r(v) for s, v in per.items()}, "mean": _r(_mean(list(per.values())))}
        online = [r["online"] for r in rs if r.get("online")]
        out.append({"partition": part, "placement": place, "arm": arm, "seeds": [r["seed"] for r in rs],
                    "sources": [r["source"] for r in rs],
                    "per_seed": {r["seed"]: {k: _r(v) for k, v in flats[r["seed"]].items()} for r in rs},
                    "mean": mean, "delta_vs_control": delta,
                    "online": {r["seed"]: r["online"] for r in rs if r.get("online")} or None,
                    "deadlock": bool(online) and all(o["deadlock"] for o in online)})
    return {"cells": out,
            "invalid": [{"source": r["source"], "reasons": r["invalid"]} for r in rows if r["invalid"]]}


def analyse(results_dir: Path = RESULTS) -> dict:
    rows, dups = load(results_dir)
    res = assemble(rows)
    res["duplicates"] = dups
    res["note"] = ("描述性分析，不是预注册判定；各臂的正式标签以各自的判定文件为准。"
                   "Δ = 同划分、同布点、同种子的不设防对照 − 本臂。灌入 / 洗掉 / 净增量的定义见脚本头注释。")
    return res


# ══════════════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════════════

def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _seeds(c: dict, key: str, nd=3) -> str:
    return " / ".join(_f(c["per_seed"][s].get(key), nd) for s in c["seeds"])


def markdown(res: dict) -> str:
    L = []
    for part, title in (("main", "现主线划分"), ("homo", "组内更同质划分")):
        cs = [c for c in res["cells"] if c["partition"] == part]
        if not cs:
            continue
        L.append(f"## {title}\n")
        col = [c for c in cs if c["placement"] == "collocated"]
        if col:
            L.append("### 集中布点：终值（种子均值）与相对不设防的下降\n")
            L.append("| 臂 | 种子 | 受害边缘 | 攻击者所在边缘的良性端 | 全部良性端 | 个性化精度 | 受害边缘下降 | 精度下降 |")
            L.append("|---|---|---|---|---|---|---|---|")
            for c in col:
                m, d = c["mean"], c["delta_vs_control"]
                note = "（冷启动死锁，不作证据）" if c["deadlock"] else ""
                L.append(f"| {c['arm']}{note} | {len(c['seeds'])} | {_f(m.get('V'))} | {_f(m.get('B0'))} | {_f(m.get('P'))} "
                         f"| {_f(m.get('MTA'))} | {_f((d.get('V') or {}).get('mean'))} | {_f((d.get('MTA') or {}).get('mean'), 4)} |")
            L.append("")
            L.append("### 集中布点：受害边缘每个云周期的灌入 / 洗掉 / 净增量（逐种子）\n")
            L.append("| 臂 | 前半程灌入 | 前半程洗掉 | 前半程净增量 | 后半程灌入 | 后半程净增量 | 前半程灌入（margin） | 后半程净增量（margin） |")
            L.append("|---|---|---|---|---|---|---|---|")
            for c in col:
                L.append(f"| {c['arm']} | {_seeds(c, 'victim_asr.first.jump')} | {_seeds(c, 'victim_asr.first.wash')} "
                         f"| {_seeds(c, 'victim_asr.first.net')} | {_seeds(c, 'victim_asr.second.jump')} "
                         f"| {_seeds(c, 'victim_asr.second.net')} | {_seeds(c, 'victim_margin.first.jump', 2)} "
                         f"| {_seeds(c, 'victim_margin.second.net', 2)} |")
            L.append("")
        dist = [c for c in cs if c["placement"] == "distributed"]
        if dist:
            L.append("### 分散布点：终值（种子均值）与相对不设防的下降\n")
            L.append("| 臂 | 种子 | 全部良性端 | 个性化精度 | 良性端下降 | 精度下降 |")
            L.append("|---|---|---|---|---|---|")
            for c in dist:
                m, d = c["mean"], c["delta_vs_control"]
                note = "（冷启动死锁，不作证据）" if c["deadlock"] else ""
                L.append(f"| {c['arm']}{note} | {len(c['seeds'])} | {_f(m.get('P'))} | {_f(m.get('MTA'))} "
                         f"| {_f((d.get('P') or {}).get('mean'))} | {_f((d.get('MTA') or {}).get('mean'), 4)} |")
            L.append("")
    if res["invalid"]:
        L.append("### 无效、未计入的实验\n")
        L += [f"- {x['source']}：{'；'.join(x['reasons'])}" for x in res["invalid"]]
        L.append("")
    L.append(f"> {res['note']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="论文整合：阶段三各防御臂的统一对比（只报告）")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--json", default=str(OUT))
    ap.add_argument("--md", default=None, help="另存一份 markdown 表")
    args = ap.parse_args(argv)
    res = analyse(Path(args.results))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = markdown(res)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md, encoding="utf-8")
    print(md)
    print(f"[defense_balance] {len(res['cells'])} 格；重复 {len(res['duplicates'])}；无效 {len(res['invalid'])} → {args.json}")


if __name__ == "__main__":
    main()
