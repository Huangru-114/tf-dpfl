"""
harness/d0_power.py  —  阶段三 D0：功效分析（门槛从已有数据的 seed 间噪声定）（2026-10-09）

    python3 harness/d0_power.py [--json out.json] [--draws 20000]

为什么（`experiments/defense/edge-native/PLAN.md` §3.5 / §8 第 1 条）：H1 / H3 / P1 / CCSF 的门槛
（ΔV 0.15 / 最小 seed 0.10 / no_effect 0.05、ΔMTA 0.02、Δmargin 2 logit、P0 的 R_H 0.5 / 0.2 与 jump < 0.05 排除）
是在计划里按阶段二的效应量**估**的。本脚本用已回传的 P2 数据量出：

  ① 每个主量在**同一配置**下的 seed 间 SD（配置 = G1R5 / G0-C1 / G1 的 C1 格 / G6 三臂）；
  ② 按 seed 配对的「两臂之差」的 SD（真实干预：G6 b−a、c−a；结构变化：G1 R10−R20、集中−分散；攻击−floor：G1R5 − G0-C1）。
     P2 下同配置同 seed 逐位相同（F-045 / F-084）→ 配对差的噪声**只来自**「干预 × seed」交互，这就是判定规则面对的噪声；
  ③ P0 的分母：G1R5 在快照轮 t = 6 / 15 的 Δ_jump（第 t+1 云轮的云聚合后评估点 − 第 t 云轮末的全量点）—— 小于 0.05 的快照按 PLAN 不计；
  ④ 3 seed 判定规则（H1：均值 ≥ 0.15 且最小 seed ≥ 0.10 → protects；三个 |Δ| 都 < 0.05 → no_effect；其余 partial）
     在「真效应 μ、配对 SD σ」下的通过概率（正态 Monte Carlo，固定 RNG）；精度规则（均值 ≤ 0.02 且每个 seed ≤ 0.025）同样。

量的定义与 g6_verdict / g1_verdict 相同（末 10 个评估点）：
  V           受害 edge E1–E3 的 fresh-PM 良性 ASR（`per_edge_rounds[r][e].client_benign`），每 edge 末 10 点均值再三 edge 平均（只对集中布点）
  B0          E0 良性端（同上，edge 0）
  P           全部良性端池化 ASR（`rounds[].local_benign_asr`）
  margin_p50  良性端触发样本 margin 中位数（`rounds[].margin_p50`，池化；S9 之后才有）
  margin_v    受害 edge 的 margin_p50（`per_edge_detail_rounds`，三 edge 平均；只对集中布点）
  MTA         fresh pm_acc（`acc_rounds[].pm_acc`）

**只报告，不判定**：门槛由用户拍板（PLAN §8 第 1 条），结论写进 FINDINGS。纯标准库，不 import TF / numpy。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect_metrics import LIGHT_EDGE_COLUMNS     # noqa: E402
from decay_verdict import edge_by_round             # noqa: E402
from pilot_a4 import invalid_reasons                # noqa: E402
from runs_table import LAST_K                       # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
OUT = HERE.parent / "experiments/defense/edge-native/analysis/d0_power.json"
SEEDS = (42, 43, 44)
VICTIMS = (1, 2, 3)
QUANTS = ("V", "B0", "P", "margin_p50", "margin_v", "MTA")

# 配置（seed 间 SD）：标签 → 组/run_id 模板
CONFIGS = {
    "G1R5 (C1·集中·R5, ρ=0.2)": "G1R5/G1R5__C1-collocated-R5__s{s}",
    "G0-C1 (C1·集中·R5, ρ=0)": "G0/G0__C1__s{s}",
    "G1 C1·集中·R10": "G1/G1__C1_collocated_R10__s{s}",
    "G1 C1·集中·R20": "G1/G1__C1_collocated_R20__s{s}",
    "G1 C1·分散·R10": "G1/G1__C1_distributed_R10__s{s}",
    "G1 C1·分散·R20": "G1/G1__C1_distributed_R20__s{s}",
    "G6 a (旧划分·集中·R5)": "G6/G6__a__s{s}",
    "G6 b": "G6/G6__b__s{s}",
    "G6 c": "G6/G6__c__s{s}",
}
# 配对差（seed 配对）：标签 → (参照, 处理, 类别)。Δ = 参照 − 处理（正 = 处理更低，同 g6_verdict）
PAIRS = {
    "G6 a − b（3-E k=1，真实干预）": ("G6 a (旧划分·集中·R5)", "G6 b", "intervention"),
    "G6 a − c（3-E k=2，真实干预）": ("G6 a (旧划分·集中·R5)", "G6 c", "intervention"),
    "G1 集中 R10 − R20": ("G1 C1·集中·R10", "G1 C1·集中·R20", "structural"),
    "G1 分散 R10 − R20": ("G1 C1·分散·R10", "G1 C1·分散·R20", "structural"),
    "G1 R10 分散 − 集中": ("G1 C1·分散·R10", "G1 C1·集中·R10", "structural"),
    "G1R5 − G0-C1（攻击 − floor）": ("G1R5 (C1·集中·R5, ρ=0.2)", "G0-C1 (C1·集中·R5, ρ=0)", "attack_vs_floor"),
}
SNAPSHOT_ROUNDS = (6, 15)         # PLAN §4 SNAP / P0
JUMP_MIN = 0.05                   # PLAN §4 P0：基线 jump < 0.05 的快照不计

# PLAN §4 H1 的规则（草案；本脚本只量它的操作特性）
PROTECT_MEAN, PROTECT_MIN, NO_EFFECT = 0.15, 0.10, 0.05
NO_EFFECT_MAX = 0.10              # D-101：no_effect 改为「|均值| < 0.05 且最大 |Δ| < 0.10」（原「三个 |Δ| 都 < 0.05」= rule="v1"）
MTA_MEAN, MTA_EACH = 0.02, 0.025
MU_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
MTA_MU_GRID = (0.0, 0.01, 0.02, 0.03)
MC_SEED = 20261009


def _r(v, nd=4):
    return None if v is None else round(v, nd)


# 求和一律 math.fsum：内置 sum 的浮点结果随 Python 版本变（3.12 起补偿求和，F-089）。
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return math.fsum(vals) / len(vals) if vals else None


def _sd(vals):
    """样本 SD（n−1）；少于 2 个点 → None。"""
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = math.fsum(vals) / len(vals)
    return math.sqrt(math.fsum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def last_k_mean(values, k: int = LAST_K):
    """同 runs_table.last_k_mean（末 k 个有定义的点），只是用 fsum。"""
    vals = [v for v in values if v is not None][-k:]
    return (math.fsum(vals) / len(vals), len(vals)) if vals else (None, 0)


# ══════════════════════════════════════════════════════════════════════════
# 每个 run 的量
# ══════════════════════════════════════════════════════════════════════════

def _edge_last10(m: dict, e: int):
    return last_k_mean([v for _, v in sorted(edge_by_round(m, e).items())])[0]


def _victim_margin(m: dict):
    cols = list(m.get("per_edge_detail_columns") or [])
    if "margin_p50" not in cols:
        return None
    ei, mi = cols.index("edge_id"), cols.index("margin_p50")
    per = {e: {} for e in VICTIMS}
    for rnd, rows in (m.get("per_edge_detail_rounds") or {}).items():
        for row in rows or []:
            if row and int(row[ei]) in per and row[mi] is not None:
                per[int(row[ei])][int(rnd)] = row[mi]
    vals = [last_k_mean([v for _, v in sorted(per[e].items())])[0] for e in VICTIMS]
    return None if any(v is None for v in vals) else _mean(vals)


def run_quantities(m: dict) -> dict:
    run = m.get("run") or {}
    collocated = list(run.get("malicious_per_edge") or []) == [10, 0, 0, 0]
    rounds, acc = m.get("rounds") or [], m.get("acc_rounds") or []
    vict = [_edge_last10(m, e) for e in VICTIMS] if collocated else [None]
    return {
        "V": None if (not collocated or any(v is None for v in vict)) else _mean(vict),
        "B0": _edge_last10(m, 0) if collocated else None,
        "P": last_k_mean([r.get("local_benign_asr") for r in rounds])[0],
        "margin_p50": last_k_mean([r.get("margin_p50") for r in rounds])[0],
        "margin_v": _victim_margin(m) if collocated else None,
        "MTA": last_k_mean([r.get("pm_acc") for r in acc])[0],
    }


def victim_jump(m: dict, t: int):
    """快照轮 t 的灌入量：第 t+1 云轮的云聚合后评估点 − 第 t 云轮末的全量点（受害 edge 三均值；同 g1_verdict.sawtooth）。"""
    cols = list(LIGHT_EDGE_COLUMNS)
    ei, ci = cols.index("edge_id"), cols.index("client_benign")
    rows = (m.get("per_edge_post_agg_rounds") or {}).get(str(t + 1)) or []
    post = {int(r[ei]): r[ci] for r in rows if r and r[ei] is not None}
    full = {e: edge_by_round(m, e).get(t) for e in VICTIMS}
    if any(post.get(e) is None or full[e] is None for e in VICTIMS):
        return None
    return _mean([post[e] for e in VICTIMS]) - _mean([full[e] for e in VICTIMS])


def all_jumps(m: dict) -> list:
    n = int((m.get("run") or {}).get("n_rounds") or 0)
    return [j for j in (victim_jump(m, t) for t in range(1, n)) if j is not None]


# ══════════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════════

def seed_table(runs: dict) -> dict:
    """runs：{配置标签: {seed: metrics | None}} → 每配置每量的逐 seed 值、均值、SD。"""
    out = {}
    for label, by_seed in runs.items():
        per, invalid, missing = {}, [], []
        for s in SEEDS:
            m = by_seed.get(s)
            if m is None:
                missing.append(s)
                continue
            bad = invalid_reasons(m)
            if bad:
                invalid.append({"seed": s, "reasons": bad})
                continue
            per[s] = run_quantities(m)
        q = {}
        for k in QUANTS:
            vals = {s: per[s][k] for s in per if per[s][k] is not None}
            if not vals:
                continue
            q[k] = {"per_seed": {f"s{s}": _r(v) for s, v in vals.items()},
                    "mean": _r(_mean(list(vals.values()))), "sd": _r(_sd(list(vals.values())))}
        out[label] = {"quantities": q, "invalid": invalid, "missing": missing, "_per": per}
    return out


def pair_table(seeds: dict) -> dict:
    out = {}
    for label, (ref, trt, kind) in PAIRS.items():
        a, b = seeds.get(ref, {}).get("_per", {}), seeds.get(trt, {}).get("_per", {})
        q = {}
        for k in QUANTS:
            d = {s: a[s][k] - b[s][k] for s in SEEDS
                 if s in a and s in b and a[s][k] is not None and b[s][k] is not None}
            if len(d) < 2:
                continue
            q[k] = {"per_seed": {f"s{s}": _r(v) for s, v in d.items()},
                    "mean": _r(_mean(list(d.values()))), "sd": _r(_sd(list(d.values())))}
        out[label] = {"ref": ref, "treatment": trt, "kind": kind, "quantities": q}
    return out


def jump_table(g1r5: dict) -> dict:
    per_seed, pooled = {}, []
    for s in SEEDS:
        m = g1r5.get(s)
        if m is None or invalid_reasons(m):
            continue
        js = all_jumps(m)
        pooled += js
        per_seed[f"s{s}"] = {f"t{t}": _r(victim_jump(m, t)) for t in SNAPSHOT_ROUNDS}
    snap_vals = [v for d in per_seed.values() for v in d.values() if v is not None]
    return {"per_seed": per_seed,
            "snapshots_below_min": sum(1 for v in snap_vals if v < JUMP_MIN),
            "n_snapshots": len(snap_vals), "jump_min": JUMP_MIN,
            "all_g": {"n": len(pooled), "mean": _r(_mean(pooled)), "sd": _r(_sd(pooled)),
                      "frac_below_min": _r(sum(1 for v in pooled if v < JUMP_MIN) / len(pooled)) if pooled else None}}


# ── 判定规则的操作特性（正态 Monte Carlo）────────────────────────────────────

def classify_h1(d: list, rule: str = "d101") -> str:
    """H1 / H3 的 3 seed ΔV 标签（不含精度条件）。rule="v1"：PLAN 定稿时的 no_effect（三个 |Δ| 都 < 0.05）；
    rule="d101"：用户 2026-10-10 确认的 no_effect（|均值| < 0.05 且最大 |Δ| < 0.10）。protects 两版相同。"""
    mean = math.fsum(d) / len(d)
    if mean >= PROTECT_MEAN and min(d) >= PROTECT_MIN:
        return "protects"
    if rule == "v1":
        null = all(abs(x) < NO_EFFECT for x in d)
    elif rule == "d101":
        null = abs(mean) < NO_EFFECT and max(abs(x) for x in d) < NO_EFFECT_MAX
    else:
        raise ValueError(f"未知规则 {rule!r}")
    return "no_effect" if null else "partial"


def oc_asr_rule(mu: float, sd: float, n: int = 3, draws: int = 20000, seed: int = MC_SEED,
                rule: str = "v1") -> dict:
    """真效应 mu、配对 SD sd 下，H1 规则三个标签的概率。sd = 0 → 退化为确定性。"""
    rng = random.Random(seed)
    cnt = {"protects": 0, "no_effect": 0, "partial": 0}
    for _ in range(draws):
        cnt[classify_h1([rng.gauss(mu, sd) for _ in range(n)], rule)] += 1
    return {k: _r(v / draws, 3) for k, v in cnt.items()}


def oc_mta_rule(mu: float, sd: float, n: int = 3, draws: int = 20000, seed: int = MC_SEED) -> float:
    """真精度损失 mu、配对 SD sd 下，精度条件（均值 ≤ 0.02 且每个 seed ≤ 0.025）成立的概率。"""
    rng = random.Random(seed)
    ok = 0
    for _ in range(draws):
        d = [rng.gauss(mu, sd) for _ in range(n)]
        if math.fsum(d) / n <= MTA_MEAN and max(d) <= MTA_EACH:
            ok += 1
    return _r(ok / draws, 3)


def operating_characteristics(pairs: dict, draws: int) -> dict:
    """σ 取真实干预（G6）的配对 SD；另加 0.05 / 0.10 两个假设值作敏感性。"""
    def sds(k):
        vals = sorted({v["quantities"][k]["sd"] for v in pairs.values()
                       if v["kind"] == "intervention" and k in v["quantities"]
                       and v["quantities"][k]["sd"] is not None})
        return vals
    asr_sds = sorted(set(sds("V")) | {0.05, 0.10})
    mta_sds = sorted(set(sds("MTA")) | {0.005, 0.01})
    return {
        "asr_rule": {f"sd={sd}": {f"mu={mu}": oc_asr_rule(mu, sd, draws=draws) for mu in MU_GRID}
                     for sd in asr_sds},
        "asr_rule_d101": {f"sd={sd}": {f"mu={mu}": oc_asr_rule(mu, sd, draws=draws, rule="d101")
                                       for mu in MU_GRID} for sd in asr_sds},
        "mta_rule": {f"sd={sd}": {f"mu={mu}": oc_mta_rule(mu, sd, draws=draws) for mu in MTA_MU_GRID}
                     for sd in mta_sds},
        "sd_source": "intervention 配对（G6 a−b / a−c）的 SD + 假设值；正态近似、n=3",
    }


def analyse(runs: dict, draws: int = 20000) -> dict:
    seeds = seed_table(runs)
    pairs = pair_table(seeds)
    jumps = jump_table(runs.get("G1R5 (C1·集中·R5, ρ=0.2)", {}))
    oc = operating_characteristics(pairs, draws)
    for v in seeds.values():
        v.pop("_per")
    return {"configs": seeds, "pairs": pairs, "p0_jump": jumps, "operating_characteristics": oc,
            "rule": {"protect_mean": PROTECT_MEAN, "protect_min": PROTECT_MIN, "no_effect": NO_EFFECT,
                     "no_effect_max_d101": NO_EFFECT_MAX,
                     "mta_mean": MTA_MEAN, "mta_each": MTA_EACH, "draws": draws, "mc_seed": MC_SEED},
            "note": "只报告：门槛由用户拍板（PLAN §8 第 1 条）。配对差 Δ = 参照 − 处理。"}


def load(results_dir: Path = RESULTS) -> dict:
    out = {}
    for label, tmpl in CONFIGS.items():
        out[label] = {}
        for s in SEEDS:
            p = results_dir / f"{tmpl.format(s=s)}.metrics.json"
            out[label][s] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 D0 功效分析（只报告）")
    ap.add_argument("--json", default=str(OUT))
    ap.add_argument("--draws", type=int, default=20000)
    args = ap.parse_args(argv)
    res = analyse(load(), draws=args.draws)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("── seed 间 SD（同一配置）──")
    for label, v in res["configs"].items():
        q = v["quantities"]
        print(f"  {label:<28} " + "  ".join(f"{k} {q[k]['mean']}±{q[k]['sd']}" for k in QUANTS if k in q)
              + (f"  invalid={v['invalid']}" if v["invalid"] else ""))
    print("── 配对差 Δ = 参照 − 处理（均值 ± SD）──")
    for label, v in res["pairs"].items():
        q = v["quantities"]
        print(f"  {label:<30} " + "  ".join(f"{k} {q[k]['mean']}±{q[k]['sd']}" for k in QUANTS if k in q))
    j = res["p0_jump"]
    print(f"── P0 分母 Δ_jump（G1R5）：{j['per_seed']}；< {JUMP_MIN} 的快照 {j['snapshots_below_min']}/{j['n_snapshots']}；"
          f"全部云轮 {j['all_g']}")
    for key, title in (("asr_rule", "H1 规则的操作特性（n=3，原 no_effect）"),
                       ("asr_rule_d101", "H1 规则的操作特性（n=3，D-101 的 no_effect）")):
        print(f"── {title} ──")
        for sd, row in res["operating_characteristics"][key].items():
            print(f"  {sd:<10} " + "  ".join(f"{mu}: P{v['protects']}/N{v['no_effect']}" for mu, v in row.items()))
    for sd, row in res["operating_characteristics"]["mta_rule"].items():
        print(f"  MTA {sd:<8} " + "  ".join(f"{mu}: ok {v}" for mu, v in row.items()))
    print(f"[d0_power] → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
