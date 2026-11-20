"""
harness/thesis_figures.py  —  论文图（2026-11-20，`experiments/thesis/FIGURES.md`）

    python3 harness/thesis_figures.py [--only fig5_3 …] [--out experiments/thesis/figures]

结构同 harness/report_figures.py：每张图 = 一个「数据整理」函数（不画图，可单测）+ 一个「画图」函数
（返回 matplotlib Figure，由 main 存成 PDF + PNG）。数据尽量直接调用已有函数（report_figures 的 data_*、
defense_balance、partition_preview、判定脚本），图上的数与判定输出是同一份。
外观全部来自 thesis_style（尺寸、字号、实体颜色 / 线型 / 标记）；图内只写英文论文用语，不写组名、规则编号、内部变量名
（tests/test_thesis_figures.py 扫描渲染出的全部文字）。

| 键 | 节 | 内容 |
|---|---|---|
| fig4_1 | 4.5 | 数据划分的异质性：边缘间 vs 边缘内；同边缘两端 vs 跨边缘两端 |
| fig4_2 | 4.3 | 灌入—洗掉分解示意（不设防、集中布点、种子 42 的受害边缘；攻击成功率 / margin 两行） |
| fig5_1 | 5.1 | 底线：不投毒时同一触发器的成功率；各边缘底线随目标类占比 |
| fig5_2 | 5.2 | 植入不受收敛门控：5 个投毒起点的轨迹；峰值超出底线的量（被攻陷边缘 vs 受害边缘） |
| fig5_3 | 5.3 | 锯齿：云周期 10 的轨迹；周期内相位剖面（攻击成功率 / margin） |
| fig5_4 | 5.4 | 攻击者走后的衰减：受害 / 被攻陷边缘；分层 vs 不分层（攻击成功率 / margin） |
| fig5_5 | 5.5 | 可观测性：各检测量在边缘视角与全局视角下的 AUROC；权重更新范数的 ROC |
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import defense_balance as DB                                      # noqa: E402
import report_figures as R                                        # noqa: E402
import thesis_style as TS                                         # noqa: E402

ROOT = HERE.parent
P2_MECH = ROOT / "experiments/attack/hfl-mechanism/results/P2"
P2_EDGE = ROOT / "experiments/defense/edge-native/results/P2"
G1_VERDICT_JSON = ROOT / "experiments/attack/hfl-mechanism/analysis/g1_verdict.json"
SEEDS = (42, 43, 44)
VICTIMS = (1, 2, 3)

ATTACK_STEPS = "effective rounds"
ASR_LABEL = "attack success rate"
MARGIN_LABEL = "trigger margin (median)"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ══════════════════════════════════════════════════════════════════════════
# 4.1 数据划分的异质性
# ══════════════════════════════════════════════════════════════════════════

PARTITION_CASES = (   # (partition_preview 的 case 名, 图上标签, 是否是实验里用到的划分, {面板: (dx, dy, ha)})
    ("random", "random assignment", False, {"a": (5, 0, "left"), "b": (5, 0, "left")}),
    ("C1", "main", True, {"a": (-5, 3, "right"), "b": (5, 0, "left")}),
    ("hdir-a0.3", "Dirichlet 0.3", False, {"a": (5, 0, "left"), "b": (5, 3, "left")}),
    ("hdir-a0.1", "Dirichlet 0.1", False, {"a": (0, 9, "center"), "b": (0, 9, "center")}),
    ("C1-fixed0.6", "\u03bb = 0.6", False, {"a": (5, -5, "left"), "b": (0, -9, "center")}),
    ("C1-fixed0.4", "\u03bb = 0.4\n(more homogeneous)", True, {"a": (5, 0, "left"), "b": (5, 0, "left")}),
    ("C1-fixed0.2", "\u03bb = 0.2", False, {"a": (5, 0, "left"), "b": (5, 0, "left")}),
)


def data_partition(seeds=SEEDS) -> dict:
    import partition_preview as PP
    rows = PP.preview(seeds)
    out = []
    for case, label, used, offs in PARTITION_CASES:
        rs = [r for r in rows if r["case"] == case]
        out.append({"case": case, "label": label, "used": used, "offsets": offs, "n_seeds": len(rs),
                    **{k: _mean([r[k] for r in rs]) for k in ("h_inter", "h_intra", "h_pair_same", "h_pair_cross")}})
    return {"cases": out}


def plot_partition(d: dict):
    fig, axes = TS.figure(1, 2, height=2.5)
    for ax, key, (xk, yk, xl, yl) in zip(axes[0], "ab", (
            ("h_inter", "h_intra", "between-edge heterogeneity", "client vs. its own edge"),
            ("h_pair_cross", "h_pair_same", "client pairs across edges", "client pairs within an edge"))):
        for c in d["cases"]:
            kw = ({"color": TS.NEUTRAL["ink"], "markerfacecolor": TS.NEUTRAL["ink"]} if c["used"]
                  else {"color": TS.NEUTRAL["floor"], "markerfacecolor": "none"})
            ax.plot([c[xk]], [c[yk]], marker="o", markersize=5, linewidth=0, **kw)
            dx, dy, ha = c["offsets"][key]
            ax.annotate(c["label"], (c[xk], c[yk]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=TS.FS_SMALL,
                        color=TS.NEUTRAL["ink"] if c["used"] else TS.NEUTRAL["ink2"])
        ax.set_xlabel(f"{xl} (mean TV)")
        ax.set_ylabel(f"{yl} (mean TV)")
    ax = axes[0][1]
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN, linestyle=":")
    ax.text(0.715, 0.745, "equal", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"], ha="left", va="top")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 0.75)
    axes[0][0].set_xlim(0.0, 1.0)
    axes[0][0].set_ylim(0.0, 0.5)
    TS.panel(axes[0][0], "a")
    TS.panel(axes[0][1], "b")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 4.2 灌入—洗掉分解示意
# ══════════════════════════════════════════════════════════════════════════

DECOMP_RUN = P2_EDGE / "SNAP/SNAP__collocated__s42.metrics.json"
DECOMP_MAX_CLOUD = 15
DECOMP_CYCLE = 11         # 标注的云周期（第 11 云轮：云聚合在有效轮 50；灌入 > 洗掉，净增量为正的典型周期）


def data_decomposition(path: Path = DECOMP_RUN) -> dict | None:
    m = _load(path)
    if m is None:
        return None
    R_ = int((m.get("run") or {}).get("edge_rounds") or 5)
    xmax = DECOMP_MAX_CLOUD * R_
    out = {"edge_rounds": R_, "cycle": DECOMP_CYCLE, "seed": (m.get("run") or {}).get("seed")}
    for key, col in (("asr", "client_benign"), ("margin", "margin_p50")):
        out[key] = [(x, y) for x, y in R.g1_timeline(m, VICTIMS, col) if x <= xmax]
        full, post = DB.full_series(m, VICTIMS, col), DB.post_series(m, VICTIMS, col)
        g = DECOMP_CYCLE
        out[f"{key}_cycle"] = {"before": full.get(g - 1), "after_agg": post.get(g), "end": full.get(g),
                               "jump": post[g] - full[g - 1], "wash": post[g] - full[g], "net": full[g] - full[g - 1]}
    return out


def plot_decomposition(d: dict):
    fig, axes = TS.figure(2, 1, height=3.3, sharex=True)
    R_, g = d["edge_rounds"], d["cycle"]
    x0, x1 = (g - 1) * R_, g * R_
    col = TS.ROLE["victim"]["color"]
    for ax, key, ylab in ((axes[0][0], "asr", ASR_LABEL), (axes[1][0], "margin", MARGIN_LABEL)):
        pts = d[key]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=TS.LW)
        ends = [p for p in pts if abs(p[0] - round(p[0])) < 1e-6]
        ax.plot([p[0] for p in ends], [p[1] for p in ends], linewidth=0, marker="o", markersize=2.5, color=col)
        c = d[f"{key}_cycle"]
        ax.axvspan(x0, x1, color=TS.NEUTRAL["grid"], alpha=0.6, linewidth=0)
        ax.annotate("", xy=(x0 + 0.3, c["after_agg"]), xytext=(x0 + 0.3, c["before"]),
                    arrowprops={"arrowstyle": "-|>", "color": TS.NEUTRAL["ink"], "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
        ax.annotate("", xy=(x1 - 0.3, c["end"]), xytext=(x1 - 0.3, c["after_agg"]),
                    arrowprops={"arrowstyle": "-|>", "color": TS.NEUTRAL["ink2"], "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
        f = "{:+.3f}" if key == "asr" else "{:+.2f}"      # 位数够多，图上「灌入 − 洗掉 = 净增量」在舍入后仍对得上
        fmt = lambda v, f=f: f.format(v).replace("-", "\u2212")                         # noqa: E731
        ax.annotate(f"injection\n{fmt(c['jump'])}", (x0, c["before"]), xytext=(-3, -3), textcoords="offset points",
                    fontsize=TS.FS_SMALL, ha="right", va="top", color=TS.NEUTRAL["ink"])
        ax.annotate(f"washing {fmt(-c['wash'])}\nnet {fmt(c['net'])}", (x1 - 0.3, c["end"]),
                    xytext=(8, -5), textcoords="offset points", ha="center", va="top",
                    fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"])
        ax.set_ylabel(ylab)
    axes[0][0].set_ylim(0, 1)
    axes[1][0].axhline(0, color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN)
    axes[1][0].set_xlabel(ATTACK_STEPS)
    axes[1][0].set_xlim(0, DECOMP_MAX_CLOUD * R_ + 1)
    TS.panel(axes[0][0], "a")
    TS.panel(axes[1][0], "b")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5.1 底线
# ══════════════════════════════════════════════════════════════════════════

def data_floor_thesis(results_dir: Path = P2_MECH) -> dict | None:
    d = R.data_floor(results_dir)
    if d is None:
        return None
    attacked = [R.pooled_curve(m) for s in SEEDS
                if (m := _load(results_dir / f"G1R5/G1R5__C1-collocated-R5__s{s}.metrics.json"))]
    return {"floor_main": d["curves"]["C1"], "attacked_main": attacked, "scatter": d["scatter"],
            "criterion": 0.3}


def plot_floor_thesis(d: dict):
    fig, axes = TS.figure(1, 2, height=2.3)
    ax = axes[0][0]
    TS.band(ax, d["attacked_main"], TS.NEUTRAL["ink"], label="with poisoning")
    TS.floor_line(ax, d["floor_main"], label="no poisoning (trigger floor)")
    TS.threshold(ax, d["criterion"])
    ax.annotate(f"criterion {d['criterion']}", (1.0, d["criterion"]), xycoords=("axes fraction", "data"),
                xytext=(-2, -3), textcoords="offset points", ha="right", va="top",
                fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 300)
    ax.set_xlabel(ATTACK_STEPS)
    ax.set_ylabel(f"{ASR_LABEL} (benign clients)")
    ax.legend(loc="center right")
    ax = axes[0][1]
    pts = d["scatter"]
    ax.plot([p["yt_share"] for p in pts], [p["floor"] for p in pts], linewidth=0, marker="o", markersize=3.5,
            color=TS.NEUTRAL["ink"], markerfacecolor="none")
    ax.set_xscale("log")
    ticks = (0.005, 0.01, 0.03, 0.1, 0.3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("share of the target class in the edge (log)")
    ax.set_ylabel("trigger floor of the edge")
    TS.panel(axes[0][0], "a")
    TS.panel(axes[0][1], "b")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5.2 植入不受收敛门控
# ══════════════════════════════════════════════════════════════════════════

def data_gating(results_dir: Path = P2_MECH) -> dict | None:
    d = R.data_g5(results_dir)
    if d is None:
        return None
    t0s = sorted(d["traj"])
    split = {k: {t0: [d["split"][(t0, s)][k] for s in SEEDS if (t0, s) in d["split"]] for t0 in t0s}
             for k in ("attacker_edge", "victim_edges", "pooled")}
    return {"t0s": t0s, "traj": d["traj"], "floors": d["floors"], "split": split,
            "verdict": d["verdict"]["overall"], "window": 20}


def plot_gating(d: dict):
    fig, axes = TS.figure(1, 2, height=2.4)
    ax = axes[0][0]
    cols = TS.ramp("gray", len(d["t0s"]))
    ax.axvspan(0, d["window"], color=TS.NEUTRAL["grid"], alpha=0.7, linewidth=0)
    ax.text(d["window"] / 2, 0.98, "poisoning\nwindow", ha="center", va="top", fontsize=TS.FS_SMALL,
            color=TS.NEUTRAL["ink2"])
    ends = []
    for t0, c in zip(d["t0s"], cols):
        e = TS.band(ax, list(d["traj"][t0].values()), c, label=f"start at round {t0}", alpha=0.08)
    TS.floor_line(ax, [c for t0 in d["t0s"] for c in d["floors"][t0].values()])
    ax.set_xlim(-10, 75)
    ax.set_ylim(0, 1)
    ax.set_xlabel("effective rounds since poisoning started")
    ax.set_ylabel(f"{ASR_LABEL} (benign clients)")
    ax.legend(loc="center right", fontsize=TS.FS_SMALL - 1)
    ax = axes[0][1]
    ends = []
    for k, ent in (("attacker_edge", TS.ROLE["compromised"]), ("victim_edges", TS.ROLE["victim"])):
        mu = [_mean(d["split"][k][t]) for t in d["t0s"]]
        lo = [min(d["split"][k][t]) for t in d["t0s"]]
        hi = [max(d["split"][k][t]) for t in d["t0s"]]
        ax.fill_between(d["t0s"], lo, hi, color=ent["color"], alpha=TS.BAND_ALPHA, linewidth=0)
        ax.plot(d["t0s"], mu, color=ent["color"], marker="o", markersize=3.5, label=ent["label"])
        ends.append((d["t0s"][-1], mu[-1], "compromised edge" if k == "attacker_edge" else "victim edges"))
    ax.set_xticks(d["t0s"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("effective round at which poisoning starts")
    ax.set_ylabel("peak rise above the floor")
    ax.legend(loc="center right")
    TS.panel(axes[0][0], "a")
    TS.panel(axes[0][1], "b")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5.3 锯齿
# ══════════════════════════════════════════════════════════════════════════

def data_sawtooth(results_dir: Path = P2_MECH) -> dict | None:
    d = R.data_g1_sawtooth(results_dir)
    if d is None:
        return None
    return {"series": d["series"], "phase": d["phase"], "main_R": 10,
            "r_down": {R_: d["verdict"][c]["r_down_mean"] for R_, c in
                       ((5, "C1_collocated_R5"), (10, "C1_collocated_R10"), (20, "C1_collocated_R20"))}}


def plot_sawtooth(d: dict):
    plt = TS.setup()
    fig = plt.figure(figsize=(TS.TEXTWIDTH, 3.9), layout="constrained")
    ax = fig.subplot_mosaic([["a", "a"], ["b", "c"]], height_ratios=[1.15, 1])
    R_ = d["main_R"]
    s = d["series"][R_]
    TS.band(ax["a"], s["e0"], TS.ROLE["compromised"]["color"], label=TS.ROLE["compromised"]["label"], lw=TS.LW_THIN)
    TS.band(ax["a"], s["victims"], TS.ROLE["victim"]["color"], label=TS.ROLE["victim"]["label"], lw=TS.LW)
    ax["a"].set_xlim(0, 300)
    ax["a"].set_ylim(0, 1)
    ax["a"].set_xlabel(f"{ATTACK_STEPS} (one cloud aggregation every {R_})")
    ax["a"].set_ylabel(ASR_LABEL)
    ax["a"].legend(loc="lower right")
    Rs = sorted(d["phase"])
    cols = TS.ramp("blue", len(Rs))
    for key, lab in (("b", ASR_LABEL), ("c", MARGIN_LABEL)):
        ends = []
        for R2, c in zip(Rs, cols):
            pts = d["phase"][R2]["asr" if key == "b" else "margin"]
            ax[key].plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker="o", markersize=3)
            ends.append((pts[-1][0], pts[-1][1], f"cycle {R2}"))
        ax[key].set_xlabel("position within the cloud cycle")
        ax[key].set_ylabel(lab)
        ax[key].set_xlim(-0.05, 1.32)
        ax[key].set_xticks([0, 0.5, 1])
        ax[key].set_xticklabels(["after\naggregation", "", "end of\ncycle"])
        TS.end_labels(ax[key], ends)
    ax["c"].axhline(0, color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN)
    for k in ("a", "b", "c"):
        TS.panel(ax[k], k)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5.4 衰减
# ══════════════════════════════════════════════════════════════════════════

def data_decay_thesis(results_dir: Path = P2_MECH) -> dict | None:
    d, mg = R.data_decay(results_dir), R.data_g8_margin(results_dir)
    if d is None or mg is None:
        return None
    stop = d["stop_eff"]
    shift = lambda cs: [[(x - stop, y) for x, y in c] for c in cs]      # noqa: E731
    return {"victims": shift(d["g8_victims"]), "e0": shift(d["g8_e0"]), "floor": shift(d["flr_victims"]),
            "hfl_pooled": d["hfl_pooled"], "flat_pooled": d["flat_pooled"],
            "hfl_margin": mg["hfl"]["margin_p50"], "flat_margin": mg["flat"]["margin_p50"],
            "hfl_verdict": d["hfl_verdict"]["overall"], "flat_verdict": d["flat_verdict"]["overall"]}


def plot_decay_thesis(d: dict):
    fig, axes = TS.figure(1, 3, height=2.3)
    ax = axes[0][0]
    TS.band(ax, d["e0"], TS.ROLE["compromised"]["color"], label="compromised edge", lw=TS.LW_THIN)
    TS.band(ax, d["victims"], TS.ROLE["victim"]["color"], label="victim edges")
    TS.floor_line(ax, d["floor"], label="floor")
    ax.set_ylabel(ASR_LABEL)
    ax.set_ylim(0, 1)
    for ax, key, lab in ((axes[0][1], "pooled", ASR_LABEL), (axes[0][2], "margin", MARGIN_LABEL)):
        for k in ("hfl", "flat"):
            st = TS.STRUCT[k]
            TS.band(ax, d[f"{k}_{key}"], st["color"], label=st["label"], ls=st["ls"], alpha=0.12)
        ax.set_ylabel(lab)
    axes[0][1].set_ylim(0, 1)
    axes[0][2].axhline(0, color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN)
    for i, ax in enumerate(axes[0]):
        ax.axvline(0, color=TS.NEUTRAL["ink2"], linewidth=TS.LW_THIN)
        ax.set_xlim(-60, 200)
        TS.panel(ax, "abc"[i])
    fig.supxlabel("effective rounds since the attackers stopped", fontsize=TS.FS)
    handles, labels = [], []
    for ax in (axes[0][0], axes[0][1]):          # 曲线一开始贴着上沿 → 图例放到面板上方一行
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=TS.FS_SMALL,
               handlelength=1.8, columnspacing=1.2)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 5.5 可观测性
# ══════════════════════════════════════════════════════════════════════════

SCORE_LABEL = {"norm_w": "weight-update norm", "neg_cos_global_w": "cosine to all other updates",
               "neg_cos_edge_w": "cosine to same-edge updates", "s_ck": "functional score"}


def data_views(path: Path = G1_VERDICT_JSON, results_dir: Path = P2_MECH) -> dict | None:
    v = _load(path)
    if v is None:
        return None
    d3 = v["3D"]["R10"]
    rows = []
    for k in SCORE_LABEL:
        q = d3[k]
        rows.append({"score": k, "label": SCORE_LABEL[k], "verdict": q["verdict"],
                     "edge": [q["edge_C1"][f"s{s}"] for s in SEEDS],
                     "global": [q["global_matched_C1"][f"s{s}"] for s in SEEDS]})
    bn = v["norm_split"]["C1_distributed_R10"]["norm_s_global_matched_mean"]
    g = R.data_g1_views(path, results_dir)
    roc = g["roc"]["norm_w"] if g else None
    return {"rows": rows, "bn_stats_global_mean": bn, "roc": roc, "grid": g["grid"] if g else None,
            "detect_min": v["params"]["detect_min"]}


def plot_views(d: dict):
    fig, axes = TS.figure(1, 2, height=2.5, gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0][0]
    n = len(d["rows"])
    for i, r in enumerate(d["rows"]):
        y0 = n - i
        for j, s in enumerate(SEEDS):
            y = y0 + (j - 1) * 0.26
            ax.plot([r["global"][j], r["edge"][j]], [y, y], color=TS.NEUTRAL["floor"], linewidth=0.6, zorder=1)
            ax.plot([r["edge"][j]], [y], marker=TS.SEED_MARKER[s], markersize=3.2, linewidth=0, zorder=2,
                    color=TS.VIEW["edge"]["color"])
            ax.plot([r["global"][j]], [y], marker=TS.SEED_MARKER[s], markersize=6, linewidth=0, zorder=3,
                    color=TS.VIEW["global"]["color"], markerfacecolor="none", markeredgewidth=0.8)
    ax.plot([d["bn_stats_global_mean"]], [0], marker="D", markersize=4.5, linewidth=0,
            color=TS.VIEW["global"]["color"], markerfacecolor="none")
    ax.set_yticks([n - i for i in range(n)] + [0])
    ax.set_yticklabels([r["label"] for r in d["rows"]] + ["batch-norm statistics\n(global view, mean)"])
    TS.threshold(ax, d["detect_min"], horizontal=False)
    ax.text(d["detect_min"], n + 0.55, f"{d['detect_min']} ", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"],
            ha="right", va="top")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.6, n + 0.6)
    ax.set_xlabel("AUROC (attackers vs. benign clients)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="o", linewidth=0, color=TS.VIEW["edge"]["color"], label=TS.VIEW["edge"]["label"])
    ax.plot([], [], marker="o", markersize=6, linewidth=0, color=TS.VIEW["global"]["color"],
            markerfacecolor="none", label=TS.VIEW["global"]["label"])
    ax.legend(loc="lower right", fontsize=TS.FS_SMALL - 1)
    ax = axes[0][1]
    if d["roc"]:
        for k, view in (("edge", "edge"), ("global_matched", "global")):
            st = TS.VIEW[view]
            ax.plot(d["grid"], d["roc"][k], color=st["color"], linestyle=st["ls"], label=f"{view} view",
                    drawstyle="steps-post")
    ax.plot([0, 1], [0, 1], color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.legend(loc="lower right", fontsize=TS.FS_SMALL - 1)
    TS.panel(axes[0][0], "a")
    TS.panel(axes[0][1], "b")
    return fig


# ══════════════════════════════════════════════════════════════════════════

FIGURES = {
    "fig4_1": ("fig4_1_partition", data_partition, plot_partition),
    "fig4_2": ("fig4_2_decomposition", data_decomposition, plot_decomposition),
    "fig5_1": ("fig5_1_floor", data_floor_thesis, plot_floor_thesis),
    "fig5_2": ("fig5_2_gating", data_gating, plot_gating),
    "fig5_3": ("fig5_3_sawtooth", data_sawtooth, plot_sawtooth),
    "fig5_4": ("fig5_4_decay", data_decay_thesis, plot_decay_thesis),
    "fig5_5": ("fig5_5_observability", data_views, plot_views),
}


def render(key: str):
    """→ (文件名, Figure)；数据不在盘上 → (文件名, None)。"""
    name, prep, plot = FIGURES[key]
    d = prep()
    return name, (None if d is None else plot(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description="论文图（只读已有结果）")
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(FIGURES))
    ap.add_argument("--out", default=str(TS.OUT))
    args = ap.parse_args(argv)
    for key in args.only or list(FIGURES):
        name, fig = render(key)
        if fig is None:
            print(f"[thesis_figures] {key}: 数据不在盘上，跳过")
            continue
        paths = TS.save(fig, name, Path(args.out))
        print(f"[thesis_figures] {key} → {', '.join(str(p) for p in paths)}")


if __name__ == "__main__":
    main()
