"""
harness/g3_did.py  —  3-B（目标类分布）：G3 的差中差 + 三张分解表 + 探索性读数（2026-09-29）

    python3 harness/g3_did.py [--json out.json]

**规则是预注册的，脚本不是**：判定规则写在 PLAN §3 的 3-B 行（D-062，数据回来之前定）；
本脚本在 G3 / G0 回传**之后**才写，实现的是同一条规则，复现 FINDINGS F-066（原始）与 F-073（excess）的数字。
T50 与 hdir 两段是**探索性**的（看过数据之后才看的量），不参与判定。

量（主列 = 逐 edge 良性端 fresh-PM ASR，`per_edge_rounds[r][e].client_benign`）：
  窗口     每个 G3 run 的末 10 个「E1 / E2 / E3 都有值」的云轮（各 run 停轮不同 → 窗口落在不同的有效轮上）
  raw_e    G3 在窗口内的均值
  floor_e  同 seed、同划分的 G0（ρ=0 影子攻击者，固定 60 轮）在**同一批云轮**的均值
  excess_e raw_e − floor_e
  对比     c(x) = x_E3 − (x_E1 + x_E2) / 2        （E3 在 C3 里几乎没有目标类：y_t 占比 0.005）
  差中差   DiD(x) = c(x)_C3 − c(x)_C1             （C1 = 各 edge y_t 都是 0.10 的基准）
  恒等式   DiD(excess) = DiD(raw) − DiD(floor)    —— 天花板下 DiD(raw) ≈ 0，于是 DiD(excess) ≈ −DiD(floor)

判定（PLAN §3，按 seed 配对、`verdicts.bootstrap_mean_ci`；3 seed 时 CI 下界 = 最小的 seed 值）：
  natural_features   CI 上界 < 0 → 迁移依赖目标类的自然特征（预注册的第一支）
  no_difference      CI 含 0（预注册的第二支）
  opposite           CI 下界 > 0 → 预注册没有这一支，单独报
  另报 ceiling 标注：C1 / C3 受害 edge 的 raw 均值 ≥ CEILING（0.9）→ 原始差被天花板压缩，excess 的差来自 floor

纯标准库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decay_verdict import edge_by_round          # noqa: E402
from g5_verdict import spearman                  # noqa: E402
from pilot_a4 import invalid_reasons             # noqa: E402
from verdicts import bootstrap_mean_ci           # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
CELLS = ("C1", "C2", "C3", "C4")
HDIR = ("hdir-a0.1", "hdir-a0.3", "hdir-a1", "hdir-a10")
EDGES = (0, 1, 2, 3)
WINDOW = 10
CEILING = 0.9
THETA = 0.5


def _r(v, nd=4):
    return None if v is None else round(v, nd)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def contrast(x: dict):
    """c(x) = E3 − (E1 + E2)/2；任一缺 → None。"""
    if any(x.get(e) is None for e in (1, 2, 3)):
        return None
    return x[3] - (x[1] + x[2]) / 2


def window_rounds(m: dict, k: int = WINDOW) -> list:
    common = set(edge_by_round(m, 1)) & set(edge_by_round(m, 2)) & set(edge_by_round(m, 3))
    return sorted(common)[-k:]


def first_crossing(series: dict, theta: float = THETA):
    """首次 ≥ θ 的云轮，与前一点线性插值；第一个点就 ≥ θ → 该点（左删失按首点记）；从未越过 → None。"""
    prev = None
    for r in sorted(series):
        v = series[r]
        if v >= theta:
            if prev is None:
                return float(r)
            pr, pv = prev
            return pr + (theta - pv) / (v - pv) * (r - pr)
        prev = (r, v)
    return None


def cell_run(g3: dict, g0: dict | None) -> dict:
    """一个 (格子, seed)：窗口、逐 edge raw / floor / excess、三个对比、逐 edge T50（云轮）。"""
    rounds = window_rounds(g3)
    ser = {e: edge_by_round(g3, e) for e in EDGES}
    raw = {e: _mean([ser[e].get(r) for r in rounds]) for e in EDGES}
    floor = excess = None
    missing_floor = []
    if g0 is not None:
        fser = {e: edge_by_round(g0, e) for e in EDGES}
        missing_floor = [r for r in rounds if any(r not in fser[e] for e in EDGES)]
        floor = {e: _mean([fser[e].get(r) for r in rounds]) for e in EDGES}
        excess = {e: None if raw[e] is None or floor[e] is None else raw[e] - floor[e] for e in EDGES}
    t50 = {e: first_crossing(ser[e]) for e in EDGES}
    yt = {int(p["edge_id"]): p.get("yt_share") for p in ((g3.get("run") or {}).get("data") or {}).get("per_edge") or []}
    return {
        "window": [rounds[0], rounds[-1]] if rounds else None, "n_window": len(rounds),
        "stopped_at_round": (g3.get("run") or {}).get("stopped_at_round"),
        "yt_share": yt, "raw": raw, "floor": floor, "excess": excess,
        "c_raw": contrast(raw), "c_floor": contrast(floor or {}), "c_excess": contrast(excess or {}),
        "victim_raw_mean": _mean([raw[e] for e in (1, 2, 3)]),
        "t50": t50,
        "c_t50": None if any(t50[e] is None for e in (1, 2, 3)) else t50[3] - (t50[1] + t50[2]) / 2,
        "missing_floor_rounds": missing_floor,
    }


def run_reasons(g3: dict | None, g0: dict | None) -> list:
    if g3 is None:
        return ["G3 缺"]
    reasons = list(invalid_reasons(g3))
    if g0 is None:
        return reasons + ["G0 缺"]
    reasons += [f"G0: {x}" for x in invalid_reasons(g0)]
    if (g0.get("run") or {}).get("poison_ratio") not in (0, 0.0):
        reasons.append(f"G0 run.poison_ratio={(g0.get('run') or {}).get('poison_ratio')} ≠ 0（不是影子攻击者）")
    return reasons


def judge(g3_runs: dict, g0_runs: dict) -> dict:
    """g3_runs / g0_runs：{(cell, seed): metrics}。"""
    per, invalid = {}, {}
    for c in CELLS:
        for s in SEEDS:
            g3, g0 = g3_runs.get((c, s)), g0_runs.get((c, s))
            if g3 is None or g0 is None:                 # 缺的格子 → insufficient，不是 invalid
                continue
            bad = run_reasons(g3, g0)
            if bad:
                invalid[f"{c}/s{s}"] = bad
                continue
            q = cell_run(g3, g0)
            if q["missing_floor_rounds"]:
                invalid[f"{c}/s{s}"] = [f"G0 缺云轮 {q['missing_floor_rounds']}（floor 要同一批轮次）"]
                continue
            per[(c, s)] = q

    did = {}
    for s in SEEDS:
        a, b = per.get(("C3", s)), per.get(("C1", s))
        if a is None or b is None:
            continue
        did[s] = {k: _r(a[k] - b[k]) if a[k] is not None and b[k] is not None else None
                  for k in ("c_raw", "c_floor", "c_excess", "c_t50")}
    vals = [did[s]["c_excess"] for s in SEEDS if s in did and did[s]["c_excess"] is not None]
    mean, lo, hi = bootstrap_mean_ci(vals) if len(vals) == len(SEEDS) else (None, None, None)
    ceiling = [f"{c}/s{s}" for (c, s), q in per.items()
               if c in ("C1", "C3") and q["victim_raw_mean"] is not None and q["victim_raw_mean"] >= CEILING]

    if invalid:
        overall = "invalid"
    elif not per:
        overall = "missing"
    elif len(vals) < len(SEEDS):
        overall = "insufficient"
    elif hi < 0:
        overall = "natural_features"
    elif lo > 0:
        overall = "opposite"
    else:
        overall = "no_difference"
    return {
        "overall": overall, "invalid": invalid,
        "did_excess_mean": _r(mean), "did_excess_ci": [_r(lo), _r(hi)],
        "did_per_seed": {str(s): v for s, v in did.items()},
        "ceiling_runs": sorted(ceiling),
        "per_run": [{"cell": c, "seed": s, **{k: (_rd(v) if isinstance(v, dict) else _r(v) if isinstance(v, float) else v)
                                              for k, v in q.items()}}
                    for (c, s), q in sorted(per.items())],
    }


def _rd(d):
    return {str(k): _r(v) for k, v in d.items()}


def explore_hdir(hdir_runs: dict) -> dict:
    """探索性：hdir 四档没有 floor（G0 不含 hdir）→ 只看原始 ASR 与实测 y_t 占比。
    edge 之间不独立（同一 run 共享 body），秩相关只作描述。"""
    rows = []
    for (cell, s), m in sorted(hdir_runs.items()):
        if invalid_reasons(m):
            continue
        q = cell_run(m, None)
        rows.append({"cell": cell, "seed": s, "stopped_at_round": q["stopped_at_round"],
                     "yt_share": _rd(q["yt_share"]), "raw": _rd(q["raw"])})
    return {"runs": rows}


def explore_yt(per_run: list, hdir: dict) -> dict:
    """探索性：全部 24 个 G3 run 上，攻击者 edge（E0）的 y_t 占比 vs 受害 edge 原始 ASR 均值；受害 edge 自己的 y_t vs 它的原始 ASR。"""
    items = [(r["yt_share"], r["raw"]) for r in per_run] + [(r["yt_share"], r["raw"]) for r in hdir["runs"]]
    x0 = [yt["0"] for yt, _ in items]
    y0 = [_mean([raw[str(e)] for e in (1, 2, 3)]) for _, raw in items]
    xv = [yt[str(e)] for yt, _ in items for e in (1, 2, 3)]
    yv = [raw[str(e)] for _, raw in items for e in (1, 2, 3)]
    return {"n_runs": len(items), "rho_e0_yt_vs_victim_raw": _r(spearman(x0, y0), 3),
            "n_victim_edges": len(xv), "rho_victim_yt_vs_victim_raw": _r(spearman(xv, yv), 3)}


def _load(results_dir: Path, group: str, cell: str, seed: int):
    p = results_dir / group / f"{group}__{cell}__s{seed}.metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load(results_dir: Path = RESULTS):
    g3 = {(c, s): _load(results_dir, "G3", c, s) for c in CELLS for s in SEEDS}
    g0 = {(c, s): _load(results_dir, "G0", c, s) for c in CELLS for s in SEEDS}
    hd = {(c, s): _load(results_dir, "G3", c, s) for c in HDIR for s in SEEDS}
    drop = lambda d: {k: v for k, v in d.items() if v is not None}
    return drop(g3), drop(g0), drop(hd)


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-B：G3 的差中差（PLAN §3 规则）+ 探索性读数")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    g3, g0, hd = load()
    res = judge(g3, g0)
    hdir = explore_hdir(hd)
    res["explore_hdir"] = hdir
    res["explore_yt"] = explore_yt(res["per_run"], hdir) if res["per_run"] else None

    print(f"{'格/seed':10s} {'窗口':>9s} {'c_raw':>8s} {'c_floor':>8s} {'c_excess':>9s} {'受害 raw':>8s} {'c_T50':>7s}")
    fmt = lambda v, w, p: f"{v:{w}.{p}f}" if v is not None else f"{'na':>{w}}"
    for r in res["per_run"]:
        w = r["window"] or ["na", "na"]
        print(f"{r['cell']}/s{r['seed']:<5d} {w[0]:>4}–{w[1]:<4} {fmt(r['c_raw'], 8, 4)} {fmt(r['c_floor'], 8, 4)} "
              f"{fmt(r['c_excess'], 9, 4)} {fmt(r['victim_raw_mean'], 8, 3)} {fmt(r['c_t50'], 7, 2)}")
    for s, d in res["did_per_seed"].items():
        print(f"  DiD(C3 − C1) s{s}: raw {d['c_raw']} − floor {d['c_floor']} = excess {d['c_excess']}；T50 {d['c_t50']} 云轮（探索性）")
    print(f"  天花板（受害 raw ≥ {CEILING}）：{res['ceiling_runs']}")
    if res["explore_yt"]:
        print(f"  探索性秩相关：{res['explore_yt']}")
    if res["invalid"]:
        print(f"  invalid：{res['invalid']}")
    print(f"判定：{res['overall']}（DiD excess 均值 {res['did_excess_mean']}，CI {res['did_excess_ci']}）")
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
