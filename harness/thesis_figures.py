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
| fig6_1 | 6.1 | 边缘级共享块：受害边缘 / 被攻陷边缘良性端的轨迹，两种划分 |
| fig6_4 | 6.2 | 上传前加固：离线筛选（削减率 vs 精度代价）；在线轨迹 |
| fig6_5 | 6.3 | 不分层的统计量聚类：组件拆分与缓存池（逐种子的攻击成功率下降 vs 精度下降） |
| fig6_6 | 6.3 | 参照打分的可分性：各候选分数在四格里的 AUROC（只打分阶段） |
| fig6_7 | 6.3 | 在线过滤：轨迹；攻击者 vs 良性分数；反馈循环（上一云轮后门强度 vs 本轮攻击者分数，种子 42） |
| fig6_8 | 6.3 | 良性端误判分布：每个良性端被丢弃上传的比例（经验分布） |
| fig6_9 | 6.4 | 综合比较：受害边缘 / 被攻陷边缘 / 分散布点 / 精度下降 / 防御计算量 |
| fig6_10 | 6.4 | 灌入—洗掉按防御分解（前半程；攻击成功率与 margin 两行） |
第 6 章的数据由 digests() 一次读完阶段三全部结果、每个文件压成摘要后即丢（同时只占一个 metrics.json 的内存）。
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
# 第 6 章：阶段三的防御臂（数据一次读完压成「摘要」，每个 metrics.json 读完即丢，避免同时占内存）
# ══════════════════════════════════════════════════════════════════════════

ARM_KEY = {("none", 0): "none", ("none", 1): "tier", ("harden", 0): "harden",
           ("filter", 0): "filter", ("filter", 1): "filter+tier"}
ARM_ORDER = ("none", "tier", "harden", "filter", "filter+tier")
EDGE_REF_PROBE = ROOT / "experiments/defense/edge-native/analysis/edge_ref_probe.json"
P0_SCREEN = ROOT / "experiments/defense/edge-native/analysis/p0/SNAP__collocated__s42.screen.json"
CCSD5 = ROOT / "experiments/defense/edge-native/analysis/ccsd5_verdict.json"
CCSDP = ROOT / "experiments/defense/edge-native/analysis/ccsdp_verdict.json"
HARDEN_CHOSEN = "sau-e8-train-high"          # 在线加固用的配置（离线筛选里代价最小又削得动的）


def _score_of(run: dict, probe: dict) -> str:
    """参照过滤用的分数：过滤臂记在 run 块；只打分 / 不设防的对照用同划分选中的分数（N-013 的选择）。"""
    part = "homo" if run.get("partition_profile") == "fixed" else "main"
    return run.get("edge_ref_score") or probe[part]["selected_score"]


def _digest(m: dict, probe: dict) -> dict:
    run = m.get("run") or {}
    a = DB.arm_of(run)
    R_ = int(run.get("edge_rounds") or 5)
    vm = DB.full_series(m, VICTIMS, "margin_p50")
    out = {"victims": R.edge_group_curve(m, VICTIMS), "e0": R.edge_group_curve(m, (0,)),
           "pooled": R.pooled_curve(m), "victims_margin": [(g * R_, v) for g, v in sorted(vm.items())],
           "compute_h": 0.0}
    if a["defense"] == "harden":
        out["compute_h"] = sum(h.get("seconds") or 0.0 for h in m.get("harden") or []) / 3600
    elif a["defense"] == "filter":
        out["compute_h"] = ((m.get("timing_summary") or {}).get("edge_ref_total_s") or 0.0) / 3600
    t = m.get("edge_ref_scores") or {}
    cols = t.get("columns") or []
    if cols:
        rows = [dict(zip(cols, r)) for r in t.get("rows") or []]
        mal = {int(i) for i in run.get("malicious_ids") or []}
        score = _score_of(run, probe)
        if "admitted" in cols:                                   # 每个良性端被剔的比例
            per = {}
            for r in rows:
                if r.get("admitted") is not None and int(r["cid"]) not in mal:
                    n = per.setdefault(int(r["cid"]), [int(r["edge_id"]), 0, 0])
                    n[1] += 1
                    n[2] += int(not r["admitted"])
            out["benign_reject"] = [{"cid": c, "edge": v[0], "n": v[1], "rate": v[2] / v[1]} for c, v in sorted(per.items())]
        if a["placement"] == "collocated":                       # 攻击者所在边缘每个云轮的分数中位数
            e0 = R.edge_group_curve(m, (0,))
            e0_by_g = {int(x) // R_: y for x, y in e0}
            att, ben = {}, {}
            for r in rows:
                if int(r["edge_id"]) == 0 and r.get(score) is not None:
                    (att if int(r["cid"]) in mal else ben).setdefault(int(r["round"]), []).append(r[score])
            med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2  # noqa: E731
            out["score_rounds"] = {"score": score, "attacker": {g: med(v) for g, v in sorted(att.items())},
                                   "benign": {g: med(v) for g, v in sorted(ben.items())},
                                   "e0_prev": {g: e0_by_g.get(g - 1) for g in sorted(att)}}
            out["threshold"] = run.get("edge_ref_threshold")
    return out


_DIGESTS = None
_ANALYSE = None


def _analyse():
    """defense_balance.analyse() 只算一次（6.9 与 6.10 共用；它也要读全部阶段三结果）。"""
    global _ANALYSE
    if _ANALYSE is None:
        _ANALYSE = DB.analyse()
    return _ANALYSE


def digests(results_dir: Path = P2_EDGE) -> dict:
    """{(划分, 布点, 臂, 种子): 摘要}；去重同 defense_balance.load（GROUPS 顺序优先）、不投毒的开发实验不收。"""
    global _DIGESTS
    if _DIGESTS is not None and results_dir == P2_EDGE:
        return _DIGESTS
    probe = _load(EDGE_REF_PROBE) or {"main": {"selected_score": "mix_res"}, "homo": {"selected_score": "kl_all"}}
    out = {}
    for g in DB.GROUPS:
        for p in sorted((Path(results_dir) / g).glob("*.metrics.json")):
            m = _load(p)
            run = m.get("run") or {}
            if not (run.get("poison_ratio") or 0) > 0:
                continue
            a = DB.arm_of(run)
            key = (a["partition"], a["placement"], ARM_KEY.get((a["defense"], a["tier"]), a["arm"]), run.get("seed"))
            if key not in out:
                out[key] = {"source": f"{g}/{p.stem.replace('.metrics', '')}", **_digest(m, probe)}
            del m
    if results_dir == P2_EDGE:
        _DIGESTS = out
    return out


def _curves(D, part, place, arm, key):
    return [D[k][key] for k in sorted(D) if k[:3] == (part, place, arm) and D[k].get(key)]


def _arm_band(ax, D, part, place, arm, key, **kw):
    st = TS.ARM[arm]
    return TS.band(ax, _curves(D, part, place, arm, key), st["color"], label=st["label"], ls=st["ls"], **kw)


def _fig_legend(fig, arms, ncol=None):
    import matplotlib.lines as mlines
    hs = [mlines.Line2D([], [], color=TS.ARM[a]["color"], linestyle=TS.ARM[a]["ls"], linewidth=TS.LW,
                        label=TS.ARM[a]["label"]) for a in arms]
    fig.legend(handles=hs, loc="outside upper center", ncol=ncol or len(hs), fontsize=TS.FS_SMALL,
               handlelength=2.6, columnspacing=1.4)


# ── 6.1 边缘级共享块：轨迹 ─────────────────────────────────────────────────

def data_tier() -> dict | None:
    D = digests()
    need = [(p, "collocated", a) for p in ("main", "homo") for a in ("none", "tier")]
    if any(not _curves(D, *n, "victims") for n in need):
        return None
    return {"D": D, "n_seeds": {n: len(_curves(D, *n, "victims")) for n in need}}


def plot_tier(d: dict):
    fig, axes = TS.figure(2, 2, height=3.9, sharex=True, sharey=True)
    D = d["D"]
    for i, (part, plab) in enumerate((("main", "main partition"), ("homo", "more homogeneous partition"))):
        for j, (key, clab) in enumerate((("victims", "victim edges"), ("e0", "compromised edge, benign clients"))):
            ax = axes[i][j]
            for arm in ("none", "tier"):
                _arm_band(ax, D, part, "collocated", arm, key)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 300)
            if i == 0:
                ax.set_title(clab, fontsize=TS.FS_SMALL, loc="left", color=TS.NEUTRAL["ink"])
            TS.panel(ax, "abcd"[2 * i + j])
        axes[i][0].set_ylabel(f"{ASR_LABEL}\n({plab})")
    for ax in axes[1]:
        ax.set_xlabel(ATTACK_STEPS)
    _fig_legend(fig, ("none", "tier"))
    return fig


# ── 6.4 上传前加固 ─────────────────────────────────────────────────────────

HARDEN_FAMILY = (("pgd", "adversarial training (PGD)", "o"), ("trades", "TRADES", "s"), ("sau", "SAU", "D"),
                 ("tgt", "targeted adversarial training", "^"), ("cft", "clean fine-tuning", "v"))


def data_hardening() -> dict | None:
    import p0_verdict as PV
    res = _load(P0_SCREEN)
    D = digests()
    if res is None or not _curves(D, "main", "collocated", "harden", "victims"):
        return None
    pts = []
    for cid in res.get("configs") or []:
        for t, r in PV.snapshot_rows(res, cid).items():
            if r["usable"] and r["d_pool"] is not None:
                pts.append({"cfg": cid, "snapshot": t, "r_h": r["r_h"], "d_pool": r["d_pool"], "acc_ok": r["acc_ok"],
                            "family": cid.split("-")[0]})
    return {"points": pts, "D": D, "chosen": HARDEN_CHOSEN, "r_go": 0.5, "acc_max": 0.02}


def plot_hardening(d: dict):
    fig, axes = TS.figure(1, 3, height=2.45, gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0][0]
    for fam, lab, mk in HARDEN_FAMILY:
        pts = [p for p in d["points"] if p["family"] == fam]
        ax.plot([p["d_pool"] for p in pts], [p["r_h"] for p in pts], linewidth=0, marker=mk, markersize=3.5,
                color=TS.NEUTRAL["ink2"], markerfacecolor="none", label=lab)
    ch = [p for p in d["points"] if p["cfg"] == d["chosen"]]
    ax.plot([p["d_pool"] for p in ch], [p["r_h"] for p in ch], linewidth=0, marker="D", markersize=4.5,
            color=TS.ARM["harden"]["color"], label="configuration used online")
    TS.threshold(ax, d["r_go"])
    ax.annotate(f"criterion {d['r_go']}", (0.6, d["r_go"]), xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"])
    TS.threshold(ax, d["acc_max"], horizontal=False)
    ax.text(d["acc_max"], 1.98, f" {d['acc_max']}", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"], va="top")
    ax.set_xlim(-0.02, 0.4)
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("drop in clean accuracy")
    ax.set_ylabel("reduction of the injection")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=TS.FS_SMALL - 1, handletextpad=0.3,
               columnspacing=1.2)
    D = d["D"]
    for ax, key, lab in ((axes[0][1], "victims", "victim edges"), (axes[0][2], "e0", "compromised edge, benign")):
        for arm in ("none", "harden"):
            _arm_band(ax, D, "main", "collocated", arm, key)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 300)
        ax.set_xlabel(ATTACK_STEPS)
        ax.set_title(lab, fontsize=TS.FS_SMALL, loc="left", color=TS.NEUTRAL["ink"])
    axes[0][1].set_ylabel(ASR_LABEL)
    _fig_legend(fig, ("none", "harden"))
    for i, ax in enumerate(axes[0]):
        TS.panel(ax, "abc"[i])
    return fig


# ── 6.5 不分层的统计量聚类（CCS 组件拆分与缓存池）────────────────────────

def data_flat_clustering() -> dict | None:
    a, b = _load(CCSD5), _load(CCSDP)
    if a is None or b is None:
        return None
    runs = a["runs"]
    pts = []
    for variant in ("cluster", "at", "both"):
        for s, q in runs[variant].items():
            off = runs["off"][s]
            pts.append({"variant": variant, "seed": int(s), "dP": off["P"] - q["P"], "dMTA": off["MTA"] - q["MTA"]})
    for s, q in b["delta"]["cache"].items():
        pts.append({"variant": "cache", "seed": int(s), "dP": q["dP"], "dMTA": q["dMTA"]})
    return {"points": pts, "mean_cluster_dP": a["headline"]["mean_delta_P"], "dP_min": 0.5, "dMTA_max": 0.02}


FLAT_VARIANTS = (("cluster", "clustering only", "o"), ("cache", "clustering + cache pool", "D"),
                 ("at", "adversarial training only", "^"), ("both", "both components", "s"))


def plot_flat_clustering(d: dict):
    fig, axes = TS.figure(1, 2, height=2.5, sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    pts = {(p["variant"], p["seed"]): p for p in d["points"]}
    for k, (ax, xlim) in enumerate(((axes[0][0], (-0.02, 0.55)), (axes[0][1], (0.0, 0.035)))):
        for v, lab, mk in FLAT_VARIANTS:
            ps = [p for p in d["points"] if p["variant"] == v and xlim[0] <= p["dMTA"] <= xlim[1]]
            filled = v in ("cluster", "cache")
            ax.plot([p["dMTA"] for p in ps], [p["dP"] for p in ps], linewidth=0, marker=mk, markersize=4.5,
                    color=TS.ARM["filter"]["color"] if filled else TS.NEUTRAL["ink2"],
                    markerfacecolor=None if filled else "none", label=lab)
        if k == 1:
            for s in (43, 46):                                 # 缓存池在两个失守种子上的修复（及代价）
                if ("cluster", s) in pts and ("cache", s) in pts:
                    p0, p1 = pts[("cluster", s)], pts[("cache", s)]
                    ax.annotate("", xy=(p1["dMTA"], p1["dP"]), xytext=(p0["dMTA"], p0["dP"]),
                                arrowprops={"arrowstyle": "->", "color": TS.NEUTRAL["ink2"], "lw": 0.6,
                                            "shrinkA": 3, "shrinkB": 3})
        else:
            ax.axvspan(0.0, 0.035, color=TS.NEUTRAL["grid"], alpha=0.6, linewidth=0, zorder=0)   # 放大区域
        TS.threshold(ax, d["dP_min"], f"criterion {d['dP_min']}" if k == 1 else None)
        TS.threshold(ax, d["dMTA_max"], horizontal=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.1, 1.0)
    axes[0][1].set_xticks([0, 0.01, 0.02, 0.03])
    axes[0][0].set_ylabel("drop in attack success rate")
    axes[0][0].legend(loc="lower right", fontsize=TS.FS_SMALL - 1)
    axes[0][1].set_title("zoom: accuracy drop below 0.035", fontsize=TS.FS_SMALL, loc="left", color=TS.NEUTRAL["ink"])
    fig.supxlabel("drop in clean accuracy", fontsize=TS.FS)
    TS.panel(axes[0][0], "a")
    TS.panel(axes[0][1], "b")
    return fig


# ── 6.6 参照打分的可分性（只打分阶段）─────────────────────────────────────

SCORE_NAMES = {"kl_all": "reference divergence, all layers", "kl_stem": "reference divergence, input layer",
               "kl_stage1": "reference divergence, block 1", "kl_stage2": "reference divergence, block 2",
               "kl_stage3": "reference divergence, block 3", "kl_stage4": "reference divergence, block 4",
               "mix_res": "class-mixture residual", "mix_res_base": "class-share residual (no fit)",
               "ccs_cos_med": "cosine to cached uploads (majority)"}
CELLS = (("t0-collocated", "none", True), ("t1-collocated", "tier", True),
         ("t0-distributed", "none", False), ("t1-distributed", "tier", False))


def data_scoring() -> dict | None:
    p = _load(EDGE_REF_PROBE)
    if p is None:
        return None
    out = {"auroc_min": p["auroc_min"]}
    for part in ("main", "homo"):
        g = p[part]["auroc_grid"]
        out[part] = {"selected": p[part]["selected_score"], "scores": list(g),
                     "auroc": {s: {c: g[s][c]["auroc"] for c in g[s]} for s in g}}
    return out


def plot_scoring(d: dict):
    fig, axes = TS.figure(1, 2, height=2.9, sharey=True)
    scores = d["main"]["scores"]
    n = len(scores)
    for ax, part, lab in ((axes[0][0], "main", "main partition"), (axes[0][1], "homo", "more homogeneous partition")):
        for i, s in enumerate(scores):
            y = n - 1 - i
            for cell, arm, conc in CELLS:
                v = d[part]["auroc"].get(s, {}).get(cell)
                if v is None:
                    continue
                ax.plot([v], [y], linewidth=0, marker="o" if conc else "s", markersize=4,
                        color=TS.ARM[arm]["color"], markerfacecolor=None if conc else "none", markeredgewidth=0.9)
        TS.threshold(ax, d["auroc_min"], horizontal=False)
        ax.text(d["auroc_min"], -0.75, f"{d['auroc_min']} ", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"],
                ha="right", va="bottom")
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(-0.8, n - 0.4)
        ax.set_title(lab, fontsize=TS.FS_SMALL, loc="left", color=TS.NEUTRAL["ink"])
        ax.grid(axis="y", visible=False)
        sel = n - 1 - scores.index(d[part]["selected"])
        ax.axhspan(sel - 0.4, sel + 0.4, color=TS.NEUTRAL["grid"], alpha=0.7, linewidth=0, zorder=0)
    axes[0][0].set_yticks(range(n))
    axes[0][0].set_yticklabels([SCORE_NAMES.get(s, s) for s in reversed(scores)])
    import matplotlib.lines as mlines
    hs = [mlines.Line2D([], [], linewidth=0, marker="o" if conc else "s", markersize=4, color=TS.ARM[arm]["color"],
                        markerfacecolor=None if conc else "none",
                        label=f"{'concentrated' if conc else 'dispersed'}, {'no defense' if arm == 'none' else 'edge-shared block'}")
          for _, arm, conc in CELLS]
    fig.legend(handles=hs, loc="outside upper center", ncol=4, fontsize=TS.FS_SMALL - 1, columnspacing=1.0,
               handletextpad=0.2)
    fig.supxlabel("AUROC (attackers' uploads vs. benign uploads)", fontsize=TS.FS)
    TS.panel(axes[0][0], "a")
    TS.panel(axes[0][1], "b")
    return fig


# ── 6.7 在线过滤与反馈循环 ─────────────────────────────────────────────────

def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None


def data_filtering(seed: int = 42) -> dict | None:
    D = digests()
    obs, fil = D.get(("main", "collocated", "none", seed)), D.get(("main", "collocated", "filter", seed))
    obs_src = D.get(("main", "collocated", "none", seed), {}).get("source", "")
    if fil is None or "score_rounds" not in fil:
        return None
    # 不设防的种子 42 来自 SNAP（没打分）→ 打分读数取同种子的只打分实验（已证与不设防逐位相同）
    if obs is None or "score_rounds" not in obs:
        m = _load(P2_EDGE / f"ERP/ERP__main-t0-collocated__s{seed}.metrics.json")
        probe = _load(EDGE_REF_PROBE)
        obs = _digest(m, probe) if m else None
    out = {"D": D, "seed": seed, "threshold": fil["threshold"], "score": fil["score_rounds"]["score"], "scatter": {}}
    for name, dg in (("observe", obs), ("filter", fil)):
        sr = dg["score_rounds"]
        pairs = [(sr["e0_prev"][g], v) for g, v in sr["attacker"].items() if sr["e0_prev"].get(g) is not None]
        out["scatter"][name] = {"pairs": pairs, "r": _pearson([p[0] for p in pairs], [p[1] for p in pairs])}
    out["rounds"] = {k: fil["score_rounds"][k] for k in ("attacker", "benign")}
    out["obs_source"] = obs_src
    return out


def plot_filtering(d: dict):
    fig, axes = TS.figure(2, 2, height=4.3)
    D = d["D"]
    arms = ("none", "tier", "filter", "filter+tier")
    for ax, key, lab in ((axes[0][0], "victims", "victim edges"), (axes[0][1], "e0", "compromised edge, benign")):
        for arm in arms:
            _arm_band(ax, D, "main", "collocated", arm, key, alpha=0.10)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 300)
        ax.set_xlabel(ATTACK_STEPS)
        ax.set_title(lab, fontsize=TS.FS_SMALL, loc="left", color=TS.NEUTRAL["ink"])
    axes[0][0].set_ylabel(ASR_LABEL)
    ax = axes[1][0]
    st = TS.ARM["filter"]
    for k, lab, c, ls in (("attacker", "attackers' uploads", st["color"], "-"),
                          ("benign", "benign uploads, same edge", TS.NEUTRAL["ink2"], "--")):
        xs = sorted(d["rounds"][k])
        ax.plot([g * 5 for g in xs], [d["rounds"][k][g] for g in xs], color=c, linestyle=ls, linewidth=TS.LW_THIN,
                label=lab)
    ax.axhline(d["threshold"], color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN, linestyle=":")
    ax.annotate("threshold", (0.0, d["threshold"]), xycoords=("axes fraction", "data"), xytext=(2, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"])
    ax.set_xlim(0, 300)
    ax.set_xlabel(ATTACK_STEPS)
    ax.set_ylabel("suspicion score")
    ax.legend(loc="lower right", fontsize=TS.FS_SMALL - 1)
    ax = axes[1][1]
    for name, arm, lab in (("observe", "none", "scoring only"), ("filter", "filter", "filtering on")):
        s = d["scatter"][name]
        ax.plot([p[0] for p in s["pairs"]], [p[1] for p in s["pairs"]], linewidth=0, marker="o", markersize=3,
                color=TS.ARM[arm]["color"], markerfacecolor=None if name == "filter" else "none",
                label=f"{lab} (r = {s['r']:.2f})")
    ax.axhline(d["threshold"], color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN, linestyle=":")
    ax.set_xlabel("compromised edge's attack success rate,\nprevious cloud round")
    ax.set_ylabel("attackers' score (median)")
    ax.legend(loc="upper left", fontsize=TS.FS_SMALL - 1)
    _fig_legend(fig, arms)
    for i, ax in enumerate([axes[0][0], axes[0][1], axes[1][0], axes[1][1]]):
        TS.panel(ax, "abcd"[i])
    return fig


# ── 6.8 良性端误判分布 ─────────────────────────────────────────────────────

def data_misjudgement() -> dict | None:
    D = digests()
    out = {}
    for place in ("collocated", "distributed"):
        for arm in ("filter", "filter+tier"):
            rows = [r for k in sorted(D) if k[:3] == ("main", place, arm) for r in D[k].get("benign_reject") or []]
            if not rows:
                return None
            if place == "collocated":
                out[(place, arm, "victim")] = [r["rate"] for r in rows if r["edge"] != 0]
                out[(place, arm, "compromised")] = [r["rate"] for r in rows if r["edge"] == 0]
            else:
                out[(place, arm, "all")] = [r["rate"] for r in rows]
    return {"rates": out, "nominal": 0.10}


def _ecdf(vals):
    xs = sorted(vals)
    n = len(xs)
    return xs, [(i + 1) / n for i in range(n)]


def plot_misjudgement(d: dict):
    fig, axes = TS.figure(1, 3, height=2.3, sharey=True)
    panels = (("collocated", "victim", "victim edges"),
              ("collocated", "compromised", "compromised edge"),
              ("distributed", "all", "dispersed attackers"))
    for ax, (place, role, lab) in zip(axes[0], panels):
        for arm in ("filter", "filter+tier"):
            xs, ys = _ecdf(d["rates"][(place, arm, role)])
            st = TS.ARM[arm]
            ax.step(xs, ys, where="post", color=st["color"], linestyle=st["ls"], linewidth=TS.LW, label=st["label"])
        TS.threshold(ax, d["nominal"], horizontal=False)
        ax.text(d["nominal"], 0.02, f" {d['nominal']:.2f}", fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink2"])
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 1.02)
        ax.set_title(lab, fontsize=TS.FS_SMALL, loc="left", color=TS.NEUTRAL["ink"])
    axes[0][0].set_ylabel("share of benign clients")
    fig.supxlabel("share of a benign client's uploads that were discarded", fontsize=TS.FS)
    _fig_legend(fig, ("filter", "filter+tier"))
    for i, ax in enumerate(axes[0]):
        TS.panel(ax, "abc"[i])
    return fig


# ── 6.9 综合比较 ───────────────────────────────────────────────────────────

def data_comparison() -> dict | None:
    res = _analyse()
    D = digests()
    cells = {(c["partition"], c["placement"], c["arm"]): c for c in res["cells"]}
    lab2key = {DB.ARM_LABELS[k]: v for k, v in ARM_KEY.items()}
    out = {"cols": {}, "arms": list(ARM_ORDER)}
    for arm_label, key in lab2key.items():
        c = cells.get(("main", "collocated", arm_label))
        dd = cells.get(("main", "distributed", arm_label))
        out["cols"].setdefault("victim", {})[key] = {s: q.get("V") for s, q in (c or {}).get("per_seed", {}).items()}
        out["cols"].setdefault("compromised", {})[key] = {s: q.get("B0") for s, q in (c or {}).get("per_seed", {}).items()}
        out["cols"].setdefault("dispersed", {})[key] = {s: q.get("P") for s, q in (dd or {}).get("per_seed", {}).items()}
        out["cols"].setdefault("acc_drop", {})[key] = ((c or {}).get("delta_vs_control", {}).get("MTA") or {}).get("per_seed") \
            or ({s: 0.0 for s in (c or {}).get("per_seed", {})} if key == "none" else {})
        out["cols"].setdefault("compute", {})[key] = {k[3]: v["compute_h"] for k, v in D.items()
                                                      if k[:3] == ("main", "collocated", key)}
    if not out["cols"]["victim"].get("tier"):
        return None
    return out


COMPARE_COLS = (("victim", "victim\nedges", (0, 1)), ("compromised", "compromised\nedge", (0, 1)),
                ("dispersed", "dispersed\nattackers", (0, 1)), ("acc_drop", "accuracy\ndrop", (-0.01, 0.035)),
                ("compute", "GPU-hours\nper run", (0, 4.2)))


def _dot_rows(ax, values: dict, arms, xlim, threshold=None):
    n = len(arms)
    for i, arm in enumerate(arms):
        y = n - 1 - i
        vals = values.get(arm) or {}
        st = TS.ARM[arm]
        for s, v in sorted(vals.items()):
            if v is None:
                continue
            ax.plot([v], [y], linewidth=0, marker=TS.SEED_MARKER.get(int(s), "o"), markersize=3.5, color=st["color"],
                    markerfacecolor="none", markeredgewidth=0.8)
        vv = [v for v in vals.values() if v is not None]
        if vv:
            mu = sum(vv) / len(vv)
            ax.plot([mu, mu], [y - 0.28, y + 0.28], color=st["color"], linewidth=1.6)
    if threshold is not None:
        TS.threshold(ax, threshold, horizontal=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)


def plot_comparison(d: dict):
    fig, axes = TS.figure(1, 5, height=2.2, sharey=True)
    arms = d["arms"]
    for ax, (key, lab, xlim) in zip(axes[0], COMPARE_COLS):
        _dot_rows(ax, d["cols"][key], arms, xlim, threshold=0.02 if key == "acc_drop" else None)
        ax.set_title(lab, fontsize=TS.FS_SMALL, color=TS.NEUTRAL["ink"])
        ax.tick_params(axis="x", labelsize=TS.FS_SMALL - 1)
    axes[0][0].set_yticks(range(len(arms)))
    axes[0][0].set_yticklabels([TS.ARM[a]["label"] for a in reversed(arms)])
    for i, ax in enumerate(axes[0]):
        TS.panel(ax, "abcde"[i])
    return fig


# ── 6.10 灌入—洗掉按防御分解 ───────────────────────────────────────────────

def data_balance() -> dict | None:
    res = _analyse()
    lab2key = {DB.ARM_LABELS[k]: v for k, v in ARM_KEY.items()}
    out = {}
    for c in res["cells"]:
        if (c["partition"], c["placement"]) != ("main", "collocated") or c["arm"] not in lab2key:
            continue
        key = lab2key[c["arm"]]
        for scale in ("victim_asr", "victim_margin"):
            for q in ("jump", "wash", "net"):
                out.setdefault((scale, q), {})[key] = {s: v.get(f"{scale}.first.{q}") for s, v in c["per_seed"].items()}
    return {"vals": out, "arms": list(ARM_ORDER)} if out else None


def plot_balance(d: dict):
    fig, axes = TS.figure(2, 3, height=3.6, sharey=True)
    arms = d["arms"]
    lims = {"victim_asr": {"jump": (-0.01, 0.25), "wash": (-0.01, 0.25), "net": (-0.01, 0.045)},
            "victim_margin": {"jump": (0, 2.6), "wash": (0, 2.6), "net": (-0.3, 0.7)}}
    for i, (scale, slab) in enumerate((("victim_asr", "attack success rate"), ("victim_margin", "trigger margin"))):
        for j, (q, qlab) in enumerate((("jump", "injection per cycle"), ("wash", "washing per cycle"),
                                       ("net", "net change per cycle"))):
            ax = axes[i][j]
            _dot_rows(ax, d["vals"][(scale, q)], arms, lims[scale][q])
            if q == "net":
                ax.axvline(0, color=TS.NEUTRAL["floor"], linewidth=TS.LW_THIN)
            ax.set_xlabel(f"{qlab}\n({slab})", fontsize=TS.FS_SMALL)
            TS.panel(ax, "abcdef"[3 * i + j])
        axes[i][0].set_yticks(range(len(arms)))
        axes[i][0].set_yticklabels([TS.ARM[a]["label"] for a in reversed(arms)])
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
    "fig6_1": ("fig6_1_edge_shared_block", data_tier, plot_tier),
    "fig6_4": ("fig6_4_hardening", data_hardening, plot_hardening),
    "fig6_5": ("fig6_5_flat_clustering", data_flat_clustering, plot_flat_clustering),
    "fig6_6": ("fig6_6_reference_scoring", data_scoring, plot_scoring),
    "fig6_7": ("fig6_7_filtering_feedback", data_filtering, plot_filtering),
    "fig6_8": ("fig6_8_benign_misjudgement", data_misjudgement, plot_misjudgement),
    "fig6_9": ("fig6_9_comparison", data_comparison, plot_comparison),
    "fig6_10": ("fig6_10_injection_washing", data_balance, plot_balance),
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
