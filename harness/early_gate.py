"""
harness/early_gate.py  —  检查 2：早期轨迹（前 60 有效轮）有没有「收敛门控」（**探索性，看过数据之后**）

    python3 harness/early_gate.py [--json out.json] [--plot out.png] [--horizon 60]

问题：报告 §5.10 / F-076 解释「实验 1 看到的门控门槛在 P2 下已被越过」。这里直接看从第 0 轮起就投毒的 P2 run
在前 60 个有效轮内的轨迹：ASR 是「从第一个评估点就上升」，还是「贴 floor 一段之后才起飞」。

run（全部从第 1 个 cloud 轮起投毒，ρ = 0.2）与同轮 floor（ρ = 0，同 seed）：
  G3-C1            ↔ G0-C1           （同划分、同 R5，floor 精确对应）
  G6-a             ↔ FLR-g6a         （同上）
  G1R5（C1·集中·R5）↔ G0-C1           （同上）
  G1 {random, C1} × {集中, 分散} × R{10, 20} ↔ G0-{random, C1}（**R 不同**：G0 是 R5，同有效轮的 floor 只是近似）
  G8F（flat）       ↔ 无（没有 flat 的 ρ = 0 run → floor 记 None）
  G3 没有 random 格；random 划分从第 0 轮起投毒的只有 G1-random。

量（每个评估点；有效轮 = 云轮 × R，G1 另有网格 5 的轻评估点）：
  atk_benign   攻击者所在 edge 的良性端 fresh-PM ASR（集中布点 = E0 的 client_benign；
               分散布点 / flat 每个 edge 都有攻击者 → 池化的 local_benign_asr）
  victim       受害 edge E1–E3 的 client_benign 三 edge 均值（只有集中布点有定义）
  pm_acc       fresh pm_acc（全量点 acc_rounds；G1 的轻评估点 light_rounds）
  floor_*      floor run 在**同一有效轮**的同一个量（没有该点 → None）

读法（事后写定，只描述，不判定）：
  first_cross  atk_benign 首次 > 0.5 的有效轮，及当时的 pm_acc
  shape        crossed_at_first_point  首个评估点就 > 0.5 → 门槛若存在，也在首个评估点之前，本分辨率看不出
               floor_then_takeoff      越过之前至少 2 个点、且都贴 floor（atk_benign − floor ≤ 0.10）
               rising_from_first_point 其余越过的情形（越过之前已有点高出 floor > 0.10）
               crossed_later_no_floor  越过之前的点没有 floor（G8F）→ 只报 before_cross.max，不分类
               not_crossed             前 horizon 有效轮内没越过

纯标准库（--plot 要 matplotlib）。不改任何结果文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
VICTIMS = (1, 2, 3)
THETA = 0.5
NEAR_FLOOR = 0.10
LIGHT_COLS_DEFAULT = ("edge_id", "edge_asr", "client_benign", "client_malicious", "pm_acc",
                      "em_acc", "margin_p50", "benign_asr_p90")


def specs() -> list:
    """[(cell, group, run 文件名模板, floor 文件名模板 或 None, floor 是否精确)]；模板里的 {s} = seed。"""
    out = [("G3-C1", "G3", "G3/G3__C1__s{s}", "G0/G0__C1__s{s}", True),
           ("G6-a", "G6", "G6/G6__a__s{s}", "FLR/FLR__g6a__s{s}", True),
           ("G1R5-C1-coll", "G1R5", "G1R5/G1R5__C1-collocated-R5__s{s}", "G0/G0__C1__s{s}", True)]
    for part in ("C1", "random"):
        for place in ("collocated", "distributed"):
            for R in (10, 20):
                out.append((f"G1-{part}-{place[:4]}-R{R}", "G1", f"G1/G1__{part}_{place}_R{R}__s{{s}}",
                            f"G0/G0__{part}__s{{s}}", False))
    out.append(("G8F-flat", "G8F", "G8F/G8F__std__s{s}", None, False))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 取数
# ══════════════════════════════════════════════════════════════════════════

def _R(m: dict) -> int:
    return int((m.get("run") or {}).get("edge_rounds") or 1)


def collocated(m: dict) -> bool:
    mpe = (m.get("run") or {}).get("malicious_per_edge") or []
    return len(mpe) > 1 and mpe[0] > 0 and all(v == 0 for v in mpe[1:])


def edge_points(m: dict, edge_id: int, key: str = "client_benign") -> dict:
    """{有效轮: 值}：全量点（per_edge_rounds，有效轮 = 云轮 × R）+ 轻评估点（per_edge_light_rounds）。"""
    R, out = _R(m), {}
    for g, rows in (m.get("per_edge_rounds") or {}).items():
        for r in rows or []:
            if int(r.get("edge_id", -1)) == edge_id and r.get(key) is not None:
                out[int(g) * R] = r[key]
    cols = list(m.get("per_edge_light_columns") or LIGHT_COLS_DEFAULT)
    if key in cols:
        ei, ki = cols.index("edge_id"), cols.index(key)
        for eff, rows in (m.get("per_edge_light_rounds") or {}).items():
            for r in rows or []:
                if r and r[ei] is not None and int(r[ei]) == edge_id and r[ki] is not None:
                    out.setdefault(int(eff), r[ki])
    return out


def pooled_points(m: dict, key: str) -> dict:
    """{有效轮: 值}：rounds[] / acc_rounds[] 的全量点 + light_rounds[] 的轻评估点。"""
    R, out = _R(m), {}
    src = m.get("acc_rounds") if key in ("pm_acc", "em_acc", "gm_acc") else m.get("rounds")
    for r in src or []:
        if r.get(key) is not None:
            out[int(r["round"]) * R] = r[key]
    for r in m.get("light_rounds") or []:
        if r.get(key) is not None:
            out.setdefault(int(r["effective_round"]), r[key])
    return out


def atk_benign(m: dict, as_collocated: bool | None = None) -> dict:
    """as_collocated：按哪种视角取（floor run 要跟随被比的 run 的视角；它自己的 malicious_per_edge 只是名义布点）。"""
    coll = collocated(m) if as_collocated is None else as_collocated
    return edge_points(m, 0) if coll else pooled_points(m, "local_benign_asr")


def victim(m: dict) -> dict:
    if not collocated(m):
        return {}
    per = [edge_points(m, e) for e in VICTIMS]
    common = set(per[0]).intersection(*per[1:])
    return {t: sum(p[t] for p in per) / len(per) for t in common}


def trajectory(m: dict, floor: dict | None, horizon: int) -> list:
    """[{eff, atk_benign, victim, pm_acc, floor_atk, floor_victim}]，有效轮 ≤ horizon，按有效轮排序。"""
    a, v, acc = atk_benign(m), victim(m), pooled_points(m, "pm_acc")
    fa = atk_benign(floor, collocated(m)) if floor else {}
    fv = victim(floor) if floor else {}
    effs = sorted(t for t in set(a) | set(acc) if t <= horizon)
    return [{"eff": t, "atk_benign": a.get(t), "victim": v.get(t), "pm_acc": acc.get(t),
             "floor_atk": fa.get(t), "floor_victim": fv.get(t)} for t in effs]


def _nearest_acc(traj: list, eff: int):
    """eff 处的 pm_acc；该点没有 → 之前最近的一个有值的点（flat 的 pm_acc 每 5 轮一个）。"""
    best = None
    for p in traj:
        if p["eff"] <= eff and p["pm_acc"] is not None:
            best = (p["eff"], p["pm_acc"])
    return best


def describe(traj: list, theta: float = THETA, near: float = NEAR_FLOOR) -> dict:
    asr_pts = [p for p in traj if p["atk_benign"] is not None]
    effs = [p["eff"] for p in asr_pts]
    gaps = sorted({b - a for a, b in zip(effs, effs[1:])})
    out = {"eval_effs": effs, "gaps": gaps, "n_points": len(effs),
           "first_point": None, "first_cross": None, "shape": "not_crossed"}
    if not asr_pts:
        out["shape"] = "no_data"
        return out
    f = asr_pts[0]
    out["first_point"] = {"eff": f["eff"], "atk_benign": f["atk_benign"], "floor_atk": f["floor_atk"],
                          "victim": f["victim"], "pm_acc": f["pm_acc"]}
    k = next((i for i, p in enumerate(asr_pts) if p["atk_benign"] > theta), None)
    if k is None:
        return out
    c = asr_pts[k]
    acc = _nearest_acc(traj, c["eff"])
    out["first_cross"] = {"eff": c["eff"], "atk_benign": c["atk_benign"], "floor_atk": c["floor_atk"],
                          "victim": c["victim"], "pm_acc": None if acc is None else acc[1],
                          "pm_acc_eff": None if acc is None else acc[0]}
    before = asr_pts[:k]
    out["before_cross"] = {"n": len(before),
                           "max": max((p["atk_benign"] for p in before), default=None),
                           "max_excess": max((p["atk_benign"] - p["floor_atk"] for p in before
                                              if p["floor_atk"] is not None), default=None)}
    if k == 0:
        out["shape"] = "crossed_at_first_point"
    elif any(p["floor_atk"] is None for p in before):
        out["shape"] = "crossed_later_no_floor"
    elif len(before) >= 2 and all(p["atk_benign"] - p["floor_atk"] <= near for p in before):
        out["shape"] = "floor_then_takeoff"
    else:
        out["shape"] = "rising_from_first_point"
    return out


# ══════════════════════════════════════════════════════════════════════════

def _load(stem: str | None, results: Path):
    if stem is None:
        return None
    p = results / f"{stem}.metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _r(v, nd=3):
    return None if v is None else round(v, nd)


def analyse(results: Path = RESULTS, horizon: int = 60, seeds=SEEDS) -> dict:
    cells = {}
    for cell, group, stem, fstem, exact in specs():
        runs = {}
        for s in seeds:
            m = _load(stem.format(s=s), results)
            if m is None:
                runs[str(s)] = {"missing": True}
                continue
            fl = _load(fstem.format(s=s), results) if fstem else None
            tr = trajectory(m, fl, horizon)
            d = describe(tr)
            runs[str(s)] = {"R": _R(m), "collocated": collocated(m), **d,
                            "trajectory": [{k: (_r(v) if k != "eff" else v) for k, v in p.items()} for p in tr]}
        cells[cell] = {"group": group, "floor_run": fstem, "floor_exact": exact, "runs": runs}
    return {"purpose": "检查 2（探索性，看过数据之后）：前 %d 有效轮的早期轨迹与收敛门控" % horizon,
            "theta": THETA, "near_floor": NEAR_FLOOR, "horizon": horizon, "cells": cells}


def summary_rows(res: dict) -> list:
    rows = []
    for cell, c in res["cells"].items():
        for s, r in c["runs"].items():
            if r.get("missing"):
                continue
            fp, fc = r["first_point"] or {}, r["first_cross"] or {}
            rows.append({"cell": cell, "seed": int(s), "R": r["R"], "gaps": r["gaps"], "n": r["n_points"],
                         "first_eff": fp.get("eff"), "first_asr": _r(fp.get("atk_benign")),
                         "first_floor": _r(fp.get("floor_atk")), "first_victim": _r(fp.get("victim")),
                         "cross_eff": fc.get("eff"), "cross_acc": _r(fc.get("pm_acc")), "shape": r["shape"]})
    return rows


def plot(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("G3-C1", ["G3-C1"]), ("G6-a", ["G6-a"]), ("G1R5 (C1·coll·R5)", ["G1R5-C1-coll"]),
              ("G1 collocated", [c for c in res["cells"] if c.startswith("G1-") and "-coll-" in c]),
              ("G1 distributed", [c for c in res["cells"] if c.startswith("G1-") and "-dist-" in c]),
              ("G8F flat", ["G8F-flat"])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, (title, cells) in zip(axes.flat, panels):
        for cell in cells:
            for s, r in res["cells"][cell]["runs"].items():
                if r.get("missing"):
                    continue
                tr = r["trajectory"]
                xs = [p["eff"] for p in tr if p["atk_benign"] is not None]
                ax.plot(xs, [p["atk_benign"] for p in tr if p["atk_benign"] is not None],
                        color="#c2410c", lw=1.2, alpha=.8, marker="o", ms=2.5)
                vx = [p["eff"] for p in tr if p["victim"] is not None]
                if vx:
                    ax.plot(vx, [p["victim"] for p in tr if p["victim"] is not None],
                            color="#1d4ed8", lw=1.0, alpha=.7)
                ax_ = [p for p in tr if p["pm_acc"] is not None]
                ax.plot([p["eff"] for p in ax_], [p["pm_acc"] for p in ax_], color="#15803d", lw=.9, ls=":")
                fx = [p for p in tr if p["floor_atk"] is not None]
                ax.plot([p["eff"] for p in fx], [p["floor_atk"] for p in fx], color="#6b7280", lw=.9, ls="--")
        ax.axhline(THETA, color="#9ca3af", lw=.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, res["horizon"] + 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("effective round")
    for ax in axes[:, 0]:
        ax.set_ylabel("ASR / accuracy")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="#c2410c", marker="o", ms=3), Line2D([], [], color="#1d4ed8"),
                Line2D([], [], color="#15803d", ls=":"), Line2D([], [], color="#6b7280", ls="--")],
               ["attacker-edge benign ASR (distributed / flat: pooled benign)", "victim edges E1–E3",
                "fresh pm_acc", "floor (ρ=0, same seed, same eff. round)"],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Check 2 (exploratory): first %d effective rounds, poisoning from round 1 — one line per seed"
                 % res["horizon"], fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--json")
    ap.add_argument("--plot")
    a = ap.parse_args(argv)
    res = analyse(Path(a.results), a.horizon)
    print(f"{'cell':<22}{'seed':>5}{'R':>4}  {'gaps':<8}{'n':>3}{'1st eff':>8}{'1st ASR':>8}{'floor':>7}"
          f"{'victim':>7}{'>0.5 @':>7}{'acc':>7}  shape")
    for r in summary_rows(res):
        print(f"{r['cell']:<22}{r['seed']:>5}{r['R']:>4}  {str(r['gaps']):<8}{r['n']:>3}{str(r['first_eff']):>8}"
              f"{str(r['first_asr']):>8}{str(r['first_floor']):>7}{str(r['first_victim']):>7}"
              f"{str(r['cross_eff']):>7}{str(r['cross_acc']):>7}  {r['shape']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[early_gate] {a.json}")
    if a.plot:
        plot(res, Path(a.plot))
        print(f"[early_gate] {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
