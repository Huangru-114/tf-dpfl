"""
harness/report_figures.py  —  REPORT.md 的结果图（Experiment 3 改版，P2 已回传的各组）

    python3 harness/report_figures.py [--out experiments/attack/hfl-mechanism/figures/final] [--only G5 G8 …]

每张图 = 一个「数据整理」函数（纯标准库，可单测）+ 一个「画图」函数（函数里才 import matplotlib）。
**数都从各组已有的判定脚本里来**（`g5_verdict` / `g5ab_verdict` / `flr_verdict` / `decay_verdict` /
`g3_did` / `g7_posthoc` / `pilot_a4`）：图上的点与判定输出是同一份，不另写一套口径。
风格与工具函数全部来自 `figures.py`（固定顺序的已校验调色板、floor = 中性灰虚线、横轴 = 有效轮、
图内只写英文 —— 集群容器里的 matplotlib 没有中文字体）；中文说明写在 REPORT.md。

| 键 | 文件 | 子实验 | 来源 |
|---|---|---|---|
| G5   | F7_G5_convergence_gating.png | 3.3 | g5_verdict |
| G5AB | G5AB_generator_semantics.png | S4  | g5ab_verdict |
| F1   | F1_floor_G0_FLR.png          | 3.1 | G0 + flr_verdict |
| G3   | F4_3B_G3.png                 | 3-B | g3_did |
| G6   | 3E_G6_G6D.png                | 3-E | G6 / G6D（S7 的判定代码还没写 → 只画读数） |
| G8   | F3_decay_G8_G8F.png          | 3-C | decay_verdict（含 --flat 分支） |
| G7   | G7_preprocessing.png         | 背景 | g7_posthoc |
| G2P  | G2P_T50_ratio.png            | 背景（pilot，单 seed） | pilot_a4 |
F0（划分散点）由现有的 `partition_preview.py --plot` 出，不在这里。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decay_verdict as DV                                         # noqa: E402
import flr_verdict as FV                                           # noqa: E402
import g3_did as G3                                                # noqa: E402
import g5_verdict as G5                                            # noqa: E402
import g5ab_verdict as G5AB                                        # noqa: E402
import g7_posthoc as G7                                            # noqa: E402
import pilot_a4 as PA                                              # noqa: E402
from figures import FLOOR, GRID, INK, INK_2, SERIES, SURFACE, _style, band, spread_labels  # noqa: E402,F401
from runs_table import last_k_mean                                 # noqa: E402

ROOT = HERE.parent
STUDY = ROOT / "experiments/attack/hfl-mechanism"
RESULTS = STUDY / "results/P2"
OUT = STUDY / "figures/final"
SEEDS = (42, 43, 44)
MARKERS = {42: "o", 43: "s", 44: "^"}          # seed 用形状区分（颜色留给实体）


# ══════════════════════════════════════════════════════════════════════════
# 通用小工具（纯标准库）
# ══════════════════════════════════════════════════════════════════════════

def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _eff(m: dict) -> int:
    return int((m.get("run") or {}).get("edge_rounds") or 1)


def pooled_curve(m: dict, key: str = "local_benign_asr", shift: float = 0.0) -> list:
    """rounds[] 的一列 → [(有效轮 − shift, 值)]，跳过 None。"""
    er = _eff(m)
    return [(r["round"] * er - shift, r[key]) for r in m.get("rounds") or [] if r.get(key) is not None]


def acc_curve(m: dict, key: str = "pm_acc", shift: float = 0.0) -> list:
    er = _eff(m)
    return [(r["round"] * er - shift, r[key]) for r in m.get("acc_rounds") or [] if r.get(key) is not None]


def edge_group_curve(m: dict, edges, key: str = "client_benign", shift: float = 0.0) -> list:
    """若干 edge 的 per_edge_rounds 值按轮取平均（该轮任一 edge 缺 → 不出点）。"""
    er = _eff(m)
    per = [DV.edge_by_round(m, e, key) for e in edges]
    common = sorted(set.intersection(*(set(p) for p in per))) if per else []
    return [(r * er - shift, sum(p[r] for p in per) / len(per)) for r in common]


def benign_weights(m: dict) -> tuple:
    """(攻击者所在 edge 的良性端数, 其余 edge 的良性端数)，取自 per_edge_rounds 的第一行。"""
    per = m.get("per_edge_rounds") or {}
    rows = per[min(per, key=int)] if per else []
    same = sum(int(e.get("n_benign") or 0) for e in rows if e.get("has_malicious"))
    diff = sum(int(e.get("n_benign") or 0) for e in rows if not e.get("has_malicious"))
    return same, diff


# ══════════════════════════════════════════════════════════════════════════
# 数据整理（每张图一个；纯标准库）
# ══════════════════════════════════════════════════════════════════════════

def data_g5(results_dir: Path = RESULTS) -> dict | None:
    """3.3：判定（原样取 g5_verdict.judge）+ 对齐到窗口起点的轨迹 + 同 edge / 受害 edge 拆解 + 收敛程度。"""
    g5, g0, g5ab = G5.load(results_dir)
    if not g5 or any(g0.get(s) is None for s in SEEDS):
        return None
    verdict = G5.judge(g5, g0, g5ab)
    traj, floors, split, pm_start = {}, {}, {}, {}
    for (t0, s), m in sorted(g5.items()):
        c = G5.CELLS[t0]
        traj.setdefault(t0, {})[s] = pooled_curve(m, shift=t0)
        f = [(x, v) for x, v in pooled_curve(g0[s], shift=t0) if x <= c["n_rounds"] * G5.R_EDGE - t0]
        floors.setdefault(t0, {})[s] = f
        rows = {int(r["round"]): r for r in m.get("rounds") or []}
        frows = {int(r["round"]): r for r in g0[s].get("rounds") or []}
        q = {}
        for key, name in (("same_edge_asr", "attacker_edge"), ("diff_edge_asr", "victim_edges"),
                          ("local_benign_asr", "pooled")):
            win = [rows[r][key] for r in c["window"] if r in rows and rows[r].get(key) is not None]
            fl = _mean(frows[r].get(key) for r in c["window"] if r in frows)
            q[name] = None if not win or fl is None else max(win) - fl
        split[(t0, s)] = q
    for s in SEEDS:
        acc = {int(r["round"]): r.get("pm_acc") for r in g0[s].get("acc_rounds") or []}
        for t0 in G5.T0S:
            pm_start[(t0, s)] = acc.get(t0 // G5.R_EDGE)       # 窗口开始前一轮末 = 有效轮 t0
    return {"verdict": verdict, "traj": traj, "floors": floors, "split": split,
            "pm_curves": {s: acc_curve(g0[s]) for s in SEEDS}, "pm_start": pm_start,
            "weights": {k: benign_weights(m) for k, m in g5.items()}}


def data_g5ab(results_dir: Path = RESULTS) -> dict | None:
    runs = G5AB.load(results_dir)
    if not any(runs.values()):
        return None
    v = G5AB.judge(runs)
    rows = []
    for key, d in v["deltas_B_minus_A"].items():
        cell, q, s = key.split("/")
        rows.append({"cell": cell, "quantity": q, "seed": int(s[1:]), "delta": d})
    return {"verdict": v, "rows": rows}


PARTS = ("random", "C1", "C2", "C3", "C4")
FLOOR_WIN = (51, 60)                               # G0 的 floor 窗口（F-073）


def data_floor(results_dir: Path = RESULTS) -> dict | None:
    g0 = {(p, s): _load(results_dir / "G0" / f"G0__{p}__s{s}.metrics.json") for p in PARTS for s in SEEDS}
    if not any(g0.values()):
        return None
    curves = {p: [pooled_curve(g0[(p, s)]) for s in SEEDS if g0[(p, s)]] for p in PARTS}
    scatter = []
    for (p, s), m in sorted(g0.items()):
        if m is None:
            continue
        yt = {int(e["edge_id"]): e.get("yt_share")
              for e in ((m.get("run") or {}).get("data") or {}).get("per_edge") or []}
        for e in range(4):
            v = DV.window_mean(m, e, FLOOR_WIN)
            if v is not None and yt.get(e) is not None:
                scatter.append({"partition": p, "seed": s, "edge": e, "yt_share": yt[e], "floor": v})
    flr = FV.judge(FV.load(results_dir))
    return {"curves": curves, "scatter": scatter, "flr": flr}


def data_g3(results_dir: Path = RESULTS) -> dict | None:
    g3, g0, _ = G3.load(results_dir)
    if not g3:
        return None
    return {"verdict": G3.judge(g3, g0)}


ARMS = ("a", "b", "c")
ARM_LABEL = {"a": "a: FedRep baseline (k=0)", "b": "b: last block edge-shared (k=1)",
             "c": "c: last two blocks edge-shared (k=2)"}


def data_g6(results_dir: Path = RESULTS) -> dict | None:
    g6 = {(a, s): _load(results_dir / "G6" / f"G6__{a}__s{s}.metrics.json") for a in ARMS for s in SEEDS}
    g6d = {(a, 42): _load(results_dir / "G6D" / f"G6D__{a}__s42.metrics.json") for a in ARMS}
    if not any(g6.values()):
        return None
    curves = {a: {"victims": [edge_group_curve(g6[(a, s)], (1, 2, 3)) for s in SEEDS if g6[(a, s)]],
                  "e0": [edge_group_curve(g6[(a, s)], (0,)) for s in SEEDS if g6[(a, s)]]} for a in ARMS}
    end = []
    for (a, s), m in sorted(g6.items()):
        if m is None:
            continue
        vict = _mean(last_k_mean([v for _, v in edge_group_curve(m, (e,))])[0] for e in (1, 2, 3))
        end.append({"arm": a, "seed": s, "group": "G6", "asr": vict,
                    "pm_acc": last_k_mean([v for _, v in acc_curve(m)])[0]})
    for (a, s), m in sorted(g6d.items()):
        if m is None:
            continue
        end.append({"arm": a, "seed": s, "group": "G6D",
                    "asr": last_k_mean([v for _, v in pooled_curve(m)])[0],
                    "pm_acc": last_k_mean([v for _, v in acc_curve(m)])[0]})
    return {"curves": curves, "end": end}


def data_decay(results_dir: Path = RESULTS) -> dict | None:
    trip = DV.load(results_dir)
    if not any(t[0] for t in trip.values()):
        return None
    hfl = DV.judge(trip)
    flat_pairs = DV.load_flat(results_dir)
    flat = DV.judge_flat(flat_pairs)
    stop_eff = (DV.STOP_ROUND - 1) * 5                 # G8：第 30 云轮末 = 有效轮 150
    out = {"hfl_verdict": hfl, "flat_verdict": flat, "stop_eff": stop_eff,
           "g8_victims": [], "g8_e0": [], "flr_victims": [], "g6a_victims": [],
           "flat_pooled": [], "hfl_pooled": []}
    for s in SEEDS:
        g8, flr, g6a = trip[s]
        if g8:
            out["g8_victims"].append(edge_group_curve(g8, DV.VICTIMS))
            out["g8_e0"].append(edge_group_curve(g8, (0,)))
            out["hfl_pooled"].append(pooled_curve(g8, shift=stop_eff))
        if flr:
            out["flr_victims"].append(edge_group_curve(flr, DV.VICTIMS))
        if g6a:
            out["g6a_victims"].append(edge_group_curve(g6a, DV.VICTIMS))
        g8f = flat_pairs[s][0]
        if g8f:
            out["flat_pooled"].append(pooled_curve(g8f, shift=(DV.FLAT_STOP_ROUND - 1)))
    return out


def data_g7(results_dir: Path = RESULTS) -> dict | None:
    pairs = G7.load(results_dir / "G7")
    if not any(p[0] for p in pairs.values()):
        return None
    v = G7.judge(pairs)
    pm = {s: {arm: (G7._last10(m["acc_rounds"], "pm_acc") if m else None)
              for arm, m in zip(("std", "official"), pairs[s])} for s in SEEDS}
    first = {s: (pooled_curve(pairs[s][1])[0][0] if pairs[s][1] else None) for s in SEEDS}
    return {"verdict": v, "pm_acc": pm, "first_eval_official": first}


def data_g2p(registry: Path = STUDY / "pilot/registry.yaml") -> dict | None:
    try:
        res = PA.judge_all(PA.load_pilot(registry))
    except (FileNotFoundError, OSError):
        return None
    return {"g2p": res["G2P"]}


# ══════════════════════════════════════════════════════════════════════════
# 画图
# ══════════════════════════════════════════════════════════════════════════

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig(plt, nrows, ncols, size):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _band_line(ax, curves, color, label, lw=2.0, alpha=0.16, ls="-"):
    pts = band([c for c in curves if c])
    if not pts:
        return None
    xs, mu, lo, hi = zip(*pts)
    ax.fill_between(xs, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(xs, mu, color=color, linewidth=lw, linestyle=ls, label=label)
    return xs[-1], mu[-1]


def _floor_line(ax, curves, label):
    pts = band([c for c in curves if c])
    if pts:
        xs, mu, _, _ = zip(*pts)
        ax.plot(xs, mu, color=FLOOR, linewidth=1.5, linestyle="--", label=label)


def _panel_title(ax, text):
    ax.set_title(text, color=INK, fontsize=9, loc="left")


def _legend(ax, **kw):
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, **kw)


def _direct_labels(ax, ends, gap_frac=0.05):
    """线尾直接标注（≤ 4 个系列时与图例并用）。ends：[(x, y, text)]。"""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    ys = spread_labels([e[1] for e in ends], gap=gap_frac * (hi - lo))
    for (x, _, t), y in zip(ends, ys):
        ax.annotate(t, (x, y), xytext=(5, 0), textcoords="offset points", color=INK,
                    fontsize=7, va="center", annotation_clip=False)


def _save(fig, out, title):
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def plot_g5(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 2, 2, (11.5, 8.2))
    v = d["verdict"]
    t0s = sorted(d["traj"])
    color = {t0: SERIES[i] for i, t0 in enumerate(t0s)}

    # ① 轨迹，对齐到窗口起点
    ax = axes[0][0]
    _style(ax, "effective rounds relative to window start (r_eff - t0)", "benign ASR (pooled, fresh-PM)")
    ax.axvspan(0, 20, color=GRID, alpha=0.7, linewidth=0)
    ax.text(10, 0.97, "poison window\n(20 eff. rounds)", ha="center", va="top", fontsize=7, color=INK_2)
    for t0 in t0s:
        _band_line(ax, list(d["traj"][t0].values()), color[t0], f"t0 = {t0}", lw=1.6, alpha=0.10)
    _floor_line(ax, [c for t0 in t0s for c in d["floors"][t0].values()],
                "floor: G0-random, rho=0 (mean over t0)")
    ax.set_xlim(-15, 76)
    ax.set_ylim(0, 1)
    _legend(ax, loc="upper right")
    _panel_title(ax, "(a) Same shape at every t0: rises inside the window, falls back toward floor after it")

    # ② 预注册量：峰值 excess vs t0
    ax = axes[0][1]
    _style(ax, "t0 (effective round at which poisoning starts)", "excess over floor")
    per = {(p["t0"], p["seed"]): p for p in v["per_run"]}
    for s in SEEDS:
        xs = [t for t in t0s if (t, s) in per]
        ax.plot(xs, [per[(t, s)]["peak_excess"] for t in xs], color=INK_2, linewidth=1,
                marker=MARKERS[s], markersize=6, label=f"peak excess, seed {s} "
                f"(rho = {v['per_seed'].get(f's{s}', {}).get('rho_peak')})")
        ax.plot(xs, [per[(t, s)]["dilution_excess"] for t in xs], color=FLOOR, linewidth=0,
                marker=MARKERS[s], markersize=6, markerfacecolor="none")
    means = [_mean(per[(t, s)]["peak_excess"] for s in SEEDS if (t, s) in per) for t in t0s]
    ax.plot(t0s, means, color=INK, linewidth=2.4, label="peak excess, mean of 3 seeds")
    ax.plot([], [], color=FLOOR, linewidth=0, marker="o", markerfacecolor="none",
            label="dilution excess (t0+65..75, report only)")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    lo, hi = v["rho_peak_ci"]
    ax.set_xticks(t0s)
    ax.set_ylim(-0.1, 0.62)
    _legend(ax, loc="upper left")
    _panel_title(ax, f"(b) Pre-registered verdict (D-081): {v['overall']}\n     Spearman rho(t0, peak excess): mean {v['rho_peak_mean']:.2f},"
                     f" 95% CI [{lo}, {hi}]\n     ('gated' would need every seed's rho > 0)")

    # ③ 拆解：攻击者所在 edge vs 受害 edge
    ax = axes[1][0]
    _style(ax, "t0 (effective round at which poisoning starts)", "peak excess over floor (window max)")
    cols = {"attacker_edge": SERIES[5], "victim_edges": SERIES[6], "pooled": INK}
    names = {"attacker_edge": "benign clients in the attacker's edge E0 (1/6 of pool)",
             "victim_edges": "benign clients in victim edges E1-E3 (5/6 of pool)",
             "pooled": "pooled (= verdict quantity)"}
    ends = []
    for k in ("attacker_edge", "victim_edges", "pooled"):
        vals = [[d["split"][(t, s)][k] for s in SEEDS if (t, s) in d["split"]] for t in t0s]
        mu, lo_, hi_ = [_mean(v) for v in vals], [min(v) for v in vals], [max(v) for v in vals]
        ax.fill_between(t0s, lo_, hi_, color=cols[k], alpha=0.14, linewidth=0)
        ax.plot(t0s, mu, color=cols[k], linewidth=2, marker="o", markersize=5, label=names[k])
        ends.append((t0s[-1], mu[-1], {"attacker_edge": "E0", "victim_edges": "E1-E3",
                                       "pooled": "pooled"}[k]))
    ax.set_xticks(t0s)
    ax.set_ylim(0, 1)
    _direct_labels(ax, ends)
    _legend(ax, loc="center left", bbox_to_anchor=(0.0, 0.62))
    _panel_title(ax, "(c) The attacker's edge is saturated at every t0; variation is cross-edge spread")

    # ④ t0 覆盖的收敛区间
    ax = axes[1][1]
    _style(ax, "effective rounds", "clean accuracy (pm_acc, G0-random)")
    _band_line(ax, list(d["pm_curves"].values()), INK_2, "G0-random pm_acc (3 seeds)", lw=1.8)
    for t0 in t0s:
        vals = [d["pm_start"][(t0, s)] for s in SEEDS if d["pm_start"].get((t0, s)) is not None]
        if not vals:
            continue
        ax.axvline(t0, color=color[t0], linewidth=1.4, linestyle=":")
        ax.plot([t0], [_mean(vals)], marker="o", markersize=8, color=color[t0],
                markeredgecolor=SURFACE, markeredgewidth=2, linewidth=0)
        ax.annotate(f"t0={t0}\n{_mean(vals):.2f}", (t0, _mean(vals)), xytext=(4, -22),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlim(0, 300)
    ax.set_ylim(0.3, 0.95)
    _legend(ax, loc="lower right")
    ends = [_mean(d["pm_start"][(t, s)] for s in SEEDS) for t in (t0s[0], t0s[-1])]
    _panel_title(ax, f"(d) The windows start from under-trained ({ends[0]:.2f}) to near-converged ({ends[1]:.2f})")
    return _save(fig, out, "G5 (3.3): is backdoor implantation gated by convergence?  "
                           "4 edges, attackers [10,0,0,0], R_edge=5, rho=0.2, 3 seeds")


def plot_g5ab(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 1, (7.2, 3.8))
    ax = axes[0][0]
    _style(ax, "B - A (generator always trained  minus  frozen outside window)", "")
    th = d["verdict"]["threshold"]
    ax.axvspan(-th, th, color=GRID, alpha=0.7, linewidth=0)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    rows = sorted(d["rows"], key=lambda r: (r["quantity"], r["cell"], r["seed"]))
    cols = {"peak": SERIES[0], "dilution": SERIES[1]}
    for i, r in enumerate(rows):
        ax.plot([r["delta"]], [i], marker=MARKERS[r["seed"]], markersize=8, color=cols[r["quantity"]],
                linewidth=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['quantity']}  {r['cell']}  seed {r['seed']}" for r in rows], fontsize=7)
    for q, c in cols.items():
        ax.plot([], [], marker="o", color=c, linewidth=0, label=q)
    ax.text(0, len(rows) - 0.4, f"|B - A| < {th}: 'insensitive' zone", ha="center", fontsize=7, color=INK_2)
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    _legend(ax, loc="lower right")
    _panel_title(ax, f"verdict: {d['verdict']['overall']} -> G5 uses A (window)")
    return _save(fig, out, "G5AB: does keeping the generator trained outside the window matter?")


def plot_floor(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 3, (15, 4.4))
    col = {p: SERIES[i] for i, p in enumerate(PARTS)}

    ax = axes[0][0]
    _style(ax, "effective rounds", "benign ASR with rho=0 (floor, pooled)")
    for p in PARTS:
        _band_line(ax, d["curves"][p], col[p], p, lw=1.6, alpha=0.12)
    ax.axhline(0.3, color=INK, linewidth=1, linestyle="-.")
    ax.text(5, 0.31, "3.1 criterion: floor_gen >= 0.3 would mean 'plateau = adversarial fragility'",
            fontsize=7, color=INK, va="bottom")
    ax.set_ylim(0, 0.6)
    _legend(ax, loc="upper right", title="partition", title_fontsize=7)
    _panel_title(ax, "(a) G0: floor stays well below 0.3 (3.1 rejected)")

    ax = axes[0][1]
    _style(ax, "share of target class y_t in this edge (log)", "floor of this edge (rounds 51-60)")
    ax.set_xscale("log")
    for p in PARTS:
        pts = [r for r in d["scatter"] if r["partition"] == p]
        ax.scatter([r["yt_share"] for r in pts], [r["floor"] for r in pts], s=26, color=col[p],
                   edgecolors=SURFACE, linewidths=0.8, label=p, zorder=3)
    ticks = (0.005, 0.01, 0.03, 0.1, 0.3)             # C1–C4 的名义占比（random 实测散在 0.1 附近）
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 0.25)
    _legend(ax, loc="upper left", title="partition", title_fontsize=7)
    _panel_title(ax, "(b) G0: an edge's floor follows its y_t share")

    ax = axes[0][2]
    _style(ax, "edge", "floor (last 10 eval points)")
    flr = d["flr"]
    for i, p in enumerate(flr["per_seed"]):
        if p.get("verdict") != "ok":
            continue
        ax.plot(range(4), p["floor_edge"], color=INK_2, linewidth=1, marker=MARKERS[p["seed"]],
                markersize=7, label=f"seed {p['seed']}")
    th = flr["thresholds"]
    ax.axhline(th["high"], color=INK, linewidth=1, linestyle="-.", label=f">= {th['high']} anywhere: not negligible")
    ax.axhline(th["low"], color=INK_2, linewidth=1, linestyle=":", label=f"<= {th['low']} everywhere: negligible")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["E0 (attackers)", "E1", "E2", "E3"])
    ax.set_ylim(0, 0.2)
    _legend(ax, loc="lower left", ncol=2)
    _panel_title(ax, f"(c) FLR (G6(a) config, rho=0): verdict {flr['overall']}")
    return _save(fig, out, "Floor (3.1): benign ASR of the same trigger when the attacker poisons nothing (rho=0)")


def plot_g3(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 3, (15, 4.4))
    v = d["verdict"]
    per = {(r["cell"], r["seed"]): r for r in v["per_run"]}
    for j, cell in enumerate(("C1", "C3")):
        ax = axes[0][j]
        _style(ax, "edge", "benign ASR (window mean)")
        for k, (key, c, lab) in enumerate((("raw", SERIES[0], "attack run (G3)"),
                                           ("floor", FLOOR, "floor, rho=0 (G0, same rounds)"))):
            for e in range(4):
                vals = [per[(cell, s)][key][str(e)] for s in SEEDS if (cell, s) in per]
                x = e + (k - 0.5) * 0.36
                ax.bar(x, _mean(vals), width=0.34, color=c, alpha=0.85 if key == "raw" else 0.6,
                       label=lab if e == 0 else None)
                for s in SEEDS:
                    if (cell, s) in per:
                        ax.plot([x], [per[(cell, s)][key][str(e)]], marker=MARKERS[s], color=INK,
                                markersize=4, linewidth=0)
        yt = per.get((cell, 42), {}).get("yt_share") or {}
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"E{e}{' (attackers)' if e == 0 else ''}\ny_t={yt.get(str(e), yt.get(e))}"
                            for e in range(4)], fontsize=7)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        _legend(ax, loc="upper center", ncol=2)
        _panel_title(ax, f"({'ab'[j]}) {cell}: attack is at the ceiling; floors differ")

    ax = axes[0][2]
    _style(ax, "", "C3 - C1 difference-in-differences")
    comps = (("c_raw", SERIES[0], "DiD raw ASR"), ("c_floor", FLOOR, "DiD floor"),
             ("c_excess", SERIES[1], "DiD excess = raw - floor (pre-registered)"))
    for i, s in enumerate(SEEDS):
        dd = v["did_per_seed"].get(str(s)) or {}
        for k, (key, c, lab) in enumerate(comps):
            ax.bar(i + (k - 1) * 0.26, dd.get(key) or 0, width=0.24, color=c,
                   label=lab if i == 0 else None)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylim(-0.12, 0.12)
    _legend(ax, loc="lower left")
    lo, hi = v["did_excess_ci"]
    _panel_title(ax, f"(c) verdict '{v['overall']}' (CI [{lo:.3f}, {hi:.3f}]):\n     DiD excess is almost exactly minus DiD floor")
    return _save(fig, out, "3-B (G3): does the victim edge's share of the target class change transfer?  "
                           "C3 = E3 has almost no y_t")


def plot_g6(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 3, (15, 4.4))
    col = {a: SERIES[i] for i, a in enumerate(ARMS)}
    for j, (key, what) in enumerate((("victims", "victim edges E1-E3"), ("e0", "attacker's edge E0 (benign)"))):
        ax = axes[0][j]
        _style(ax, "effective rounds", f"benign ASR, {what}")
        ends = []
        for a in ARMS:
            e = _band_line(ax, d["curves"][a][key], col[a], ARM_LABEL[a], lw=1.8, alpha=0.14)
            if e:
                ends.append((e[0], e[1], a))
        ax.set_ylim(0, 1.02)
        if key == "victims":
            _direct_labels(ax, ends)
        _legend(ax, loc="upper left" if key == "victims" else "lower center")
        _panel_title(ax, "(a) edge-shared layers block the spread to other edges" if key == "victims"
                     else "(b) ...but do not protect benign clients next to the attackers")
    ax = axes[0][2]
    _style(ax, "fresh clean accuracy pm_acc (last 10)", "benign ASR (last 10)")
    for r in d["end"]:
        hollow = r["group"] == "G6D"
        ax.plot([r["pm_acc"]], [r["asr"]], marker=MARKERS[r["seed"]], markersize=8, linewidth=0,
                color=col[r["arm"]], markerfacecolor="none" if hollow else col[r["arm"]],
                markeredgewidth=1.6)
    for a in ARMS:
        ax.plot([], [], marker="o", color=col[a], linewidth=0, label=f"arm {a}")
    ax.plot([], [], marker="o", color=INK_2, linewidth=0, label="filled: G6, victim edges ([10,0,0,0])")
    ax.plot([], [], marker="o", color=INK_2, markerfacecolor="none", linewidth=0,
            label="hollow: G6D, all benign ([3,3,2,2], s42)")
    ax.plot([], [], linewidth=0, label="circle / square / triangle = seed 42 / 43 / 44")
    ax.set_ylim(0, 1.05)
    _legend(ax, loc="center left")
    _panel_title(ax, "(c) cost: <= 0.02 accuracy; no effect when attackers are everywhere")
    return _save(fig, out, "3-E (G6 / G6D): three-tier personalization (edge-shared middle layers)")


def plot_decay(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 2, (13, 4.6))
    ax = axes[0][0]
    _style(ax, "effective rounds", "benign ASR")
    stop = d["stop_eff"]
    ax.axvspan(0, stop, color=GRID, alpha=0.6, linewidth=0, label="attackers poison (cloud rounds 1-30)")
    e1 = _band_line(ax, d["g8_e0"], SERIES[1], "G8: attacker's edge E0 (benign)", lw=1.8)
    e2 = _band_line(ax, d["g8_victims"], SERIES[0], "G8: victim edges E1-E3", lw=1.8)
    _band_line(ax, d["g6a_victims"], SERIES[0], "G6(a): victim edges, attack never stops",
               lw=1.2, alpha=0.0, ls=":")
    _floor_line(ax, d["flr_victims"], "floor: FLR victim edges (rho=0)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 350)
    _direct_labels(ax, [(e1[0], e1[1], "E0"), (e2[0], e2[1], "E1-E3")] if e1 and e2 else [])
    _legend(ax, loc="center right", bbox_to_anchor=(1.0, 0.62))
    per = [p for p in d["hfl_verdict"]["per_seed"] if p.get("verdict") == "ok"]
    exc = " / ".join(f"{p['victims']['excess']:.3f}" for p in per)
    _panel_title(ax, f"(a) HFL: after the attackers leave, the backdoor fades toward the floor\n"
                     f"     victim excess over floor, rounds 51-60: {exc} (verdict {d['hfl_verdict']['overall']})")

    ax = axes[0][1]
    _style(ax, "effective rounds since the attackers stopped", "benign ASR (pooled)")
    f1 = _band_line(ax, d["flat_pooled"], SERIES[2], "G8F: flat (1 edge)", lw=1.8)
    f2 = _band_line(ax, d["hfl_pooled"], SERIES[3], "G8: HFL (4 edges)", lw=1.8)
    lo_, hi_ = d["flat_verdict"]["thresholds"]["low"], d["flat_verdict"]["thresholds"]["high"]
    w = d["flat_verdict"]["windows_effective"]["main"]
    ax.axvspan(w[0] - 150, w[1] - 150, color=GRID, alpha=0.6, linewidth=0, label="D-077 judging window")
    ax.axhline(hi_, color=INK_2, linewidth=0.8, linestyle=":")
    ax.axhline(lo_, color=INK_2, linewidth=0.8, linestyle=":")
    ax.text(w[1] - 150 + 2, hi_, f"{hi_} 'plateau'", fontsize=7, color=INK_2, va="bottom")
    ax.text(w[1] - 150 + 2, lo_, f"{lo_} 'decays'", fontsize=7, color=INK_2, va="top")
    ax.plot([200], [0.35], marker="D", markersize=8, color=INK, linewidth=0,
            label="external 1B-2 (official code, flat)")
    ax.annotate("~0.35 after\n200 rounds", (200, 0.35), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=7, color=INK)
    ax.set_xlim(-150, 215)
    ax.set_ylim(0, 1.02)
    _direct_labels(ax, [(f1[0], f1[1], "flat"), (f2[0], f2[1], "HFL")] if f1 and f2 else [])
    _legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    _panel_title(ax, f"(b) flat decays too (verdict {d['flat_verdict']['overall']}):\n"
                     f"     the decay is not caused by the HFL structure")
    return _save(fig, out, "3-C (G8 / G8F): what happens after the attackers stop?")


def plot_g7(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 3, (14, 3.8))
    per = {p["seed"]: p for p in d["verdict"]["per_seed"]}
    arms = (("std", SERIES[0], "standard (normalize + augment)"),
            ("official", SERIES[1], "official Bad-PFL preprocessing"))
    specs = (("T50 of benign ASR (effective rounds)", "t50"), ("benign ASR (last 10)", "asr10"),
             ("fresh pm_acc (last 10)", "pm"))
    for j, (lab, key) in enumerate(specs):
        ax = axes[0][j]
        _style(ax, lab, "")
        for i, s in enumerate(SEEDS):
            p = per.get(s) or {}
            xs = []
            for arm, c, name in arms:
                if key == "pm":
                    x = d["pm_acc"][s][arm]
                else:
                    x = p.get(f"{key}_{arm}")
                cens = x == "<首点"                    # g7_posthoc 的左删失记法（首个评估点就越过）
                if cens:
                    x = d["first_eval_official"][s]
                if x is None:
                    continue
                xs.append(x)
                ax.plot([x], [i], marker="<" if cens else "o", markersize=8, color=c, linewidth=0,
                        label=name if i == 0 else None)
            if len(xs) == 2:
                ax.plot(xs, [i, i], color=INK_2, linewidth=1, zorder=0)
        ax.set_yticks(range(len(SEEDS)))
        ax.set_yticklabels([f"seed {s}" for s in SEEDS])
        ax.set_ylim(-0.6, len(SEEDS) - 0.4)
        if j == 0:
            _legend(ax, loc="lower right")
            ax.text(0.99, 0.99, "'<' = already above 0.5 at the first eval point", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=INK_2)
    _panel_title(axes[0][0], "(a) attack succeeds earlier")
    _panel_title(axes[0][1], "(b) and a bit higher")
    _panel_title(axes[0][2], "(c) but the model itself is ~0.10 weaker (confound)")
    return _save(fig, out, f"G7 (background, post-hoc): official preprocessing -> "
                           f"'{d['verdict']['overall']}' (2 edges, R_edge=5)")


def plot_g2p(d: dict, out: Path):
    plt = _plt()
    fig, axes = _fig(plt, 1, 1, (7.2, 3.4))
    ax = axes[0][0]
    g = d["g2p"]
    _style(ax, "T50(HFL) / T50(flat)   (log scale; < 1 = HFL implants faster)", "")
    ax.set_xscale("log")
    lo, hi = g.get("band", (0.9, 1.1))
    ax.axvspan(lo, hi, color=GRID, alpha=0.8, linewidth=0)
    ax.axvline(1, color=INK_2, linewidth=0.8)
    cells = list((g.get("cells") or {}).items())
    for i, (cell, r) in enumerate(cells):
        ax.plot([r["ratio_p1"], r["ratio_new"]], [i, i], color=INK_2, linewidth=1, zorder=0)
        ax.plot([r["ratio_p1"]], [i], marker="o", markersize=8, color=SERIES[0], linewidth=0,
                label="P1 (old protocol)" if i == 0 else None)
        ax.plot([r["ratio_new"]], [i], marker="o", markersize=8, color=SERIES[1], linewidth=0,
                label="P2 (current protocol)" if i == 0 else None)
        ax.annotate(f"{r['ratio_new']:.2f}", (r["ratio_new"], i), xytext=(0, 7),
                    textcoords="offset points", fontsize=7, color=INK, ha="center")
    ticks = (0.4, 0.5, 0.75, 1, 1.5, 2, 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([c.replace("_distributed", ", distributed") for c, _ in cells], fontsize=8)
    ax.set_ylim(-0.6, len(cells) - 0.3)
    _legend(ax, loc="lower right")
    _panel_title(ax, f"pilot, seed 42 only: verdict '{g.get('verdict')}' (same side of 1 as P1)")
    return _save(fig, out, "G2P (3-A pilot): speed of implantation, HFL vs flat")


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════

FIGURES = {
    "G5":   ("F7_G5_convergence_gating.png", data_g5, plot_g5),
    "G5AB": ("G5AB_generator_semantics.png", data_g5ab, plot_g5ab),
    "F1":   ("F1_floor_G0_FLR.png", data_floor, plot_floor),
    "G3":   ("F4_3B_G3.png", data_g3, plot_g3),
    "G6":   ("3E_G6_G6D.png", data_g6, plot_g6),
    "G8":   ("F3_decay_G8_G8F.png", data_decay, plot_decay),
    "G7":   ("G7_preprocessing.png", data_g7, plot_g7),
    "G2P":  ("G2P_T50_ratio.png", data_g2p, plot_g2p),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="REPORT.md 的结果图（数据取自各组的判定脚本）")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in a.only or FIGURES:
        name, prep, draw = FIGURES[key]
        data = prep()
        if data is None:
            print(f"[report_figures] {key}: 结果不在盘上，跳过")
            continue
        draw(data, out / name)
        print(f"[report_figures] {key} → {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
