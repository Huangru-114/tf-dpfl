"""
harness/thesis_style.py  —  论文图的统一规范（2026-11-20，`experiments/thesis/FIGURES.md` 第三节）

    import thesis_style as TS
    plt = TS.setup()
    fig, axes = TS.figure(1, 2)            # 全宽、两个面板
    ...
    TS.save(fig, "fig5_3_sawtooth")        # → experiments/thesis/figures/fig5_3_sawtooth.{pdf,png}

规则（与 harness/figures.py 的报告图同一套调色板，只是尺寸、字号、文字面向印刷）：
  · 宽度 = 论文正文宽度 TEXTWIDTH（拿到模板里的 \\the\\textwidth 后只改这一个常数）；字号 8 / 7 磅；
    图上不写总标题，只写面板字母，说明全部放图注；
  · 颜色跟实体走、全书不变：边缘角色（受害边缘 / 被攻陷边缘）、防御（不设防 / 边缘级共享块 / 上传前加固 /
    参照过滤 / 过滤 + 共享块）、分层 vs 不分层；每个防御另有固定线型与标记（色盲与黑白打印也分得开）；
  · 种子用标记形状、种子间范围用浅色带；底线 = 灰色虚线；预注册门槛 = 灰色点线；
  · 有序类别（攻击起点、云周期长度）用单色阶，不占类别色；
  · 一个面板一个纵轴：攻击成功率与 margin 永远分两个面板。
画图代码里不直接写颜色值，一律经本模块的表取（tests/test_thesis_figures.py 守着）。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figures import FLOOR, GRID, INK, INK_2, SERIES, SURFACE   # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "experiments/thesis/figures"

TEXTWIDTH = 5.9          # 英寸（≈ 15 cm）：⚠ 用论文模板的 \the\textwidth / 72.27 替换
HALFWIDTH = 2.85
FS, FS_SMALL = 8, 7      # 轴标签 / 刻度、图例、标注
LW, LW_THIN = 1.3, 0.8
BAND_ALPHA = 0.18

# ── 实体 → 外观（全书不变）────────────────────────────────────────────────
ROLE = {
    "victim": {"color": SERIES[0], "label": "victim edges"},
    "compromised": {"color": SERIES[1], "label": "benign clients of the compromised edge"},
}
ARM = {   # 绿 / 琥珀 / 紫三色通过 dataviz 校验（含色盲分离）；粉与绿在红绿色盲下接近 → 靠线型与标记区分
    "none": {"color": INK_2, "ls": "-", "marker": "o", "label": "no defense"},
    "tier": {"color": SERIES[2], "ls": "--", "marker": "s", "label": "edge-shared block"},
    "harden": {"color": SERIES[3], "ls": "-.", "marker": "^", "label": "pre-upload hardening"},
    "filter": {"color": SERIES[6], "ls": ":", "marker": "D", "label": "edge-reference filtering"},
    "filter+tier": {"color": SERIES[4], "ls": (0, (5, 1, 1, 1)), "marker": "v",
                    "label": "filtering + edge-shared block"},
}
STRUCT = {
    "hfl": {"color": INK, "ls": "-", "label": "hierarchical (4 edges)"},
    "flat": {"color": INK_2, "ls": "--", "label": "flat (1 server)"},
}
VIEW = {   # 第 5.5 节：检测时用的参照池
    "edge": {"color": INK, "ls": "-", "marker": "o", "label": "edge view"},
    "global": {"color": INK_2, "ls": "--", "marker": "o", "label": "global view (size-matched pool)"},
}
NEUTRAL = {"ink": INK, "ink2": INK_2, "floor": FLOOR, "grid": GRID, "surface": SURFACE}
SEED_MARKER = {42: "o", 43: "s", 44: "^", 45: "D", 46: "v"}
# 有序类别：单色阶（dataviz 参考调色板）。蓝阶只用于「受害边缘」的有序变体（同一实体）；其余有序类别用灰阶。
BLUE_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
GRAY_RAMP = ("#acaba6", "#8f8e89", "#73726d", "#575652", "#3a3936")
# 校验（dataviz validate_palette.js --ordinal，2026-11-20）：两条色阶都 PASS（单色相、明度单调、相邻 ΔL ≥ 0.06、最浅一档对背景 ≥ 2:1）。
# 防御色（绿 / 琥珀 / 紫）--pairs all 全部 PASS；加上粉后粉—绿红绿色盲 ΔE 6.1 → 必须配线型与标记（ARM 表已配）。

# 已知防御的对照基线（逐坐标中位数，补充实验 D4 / F-132）用中性灰阶，不占类别色：颜色只留给本文的边缘侧防御，
# 基线与「不设防」同属灰色一族、靠明度 + 标记 + 行标签区分（6.9 的每一行都有直接标签，身份不只靠颜色）。
# 对背景对比度（2026-11-23 实测）：不设防 #52514e 7.7:1、云层 #73726d 4.7:1、边缘层 #8f8e89 3.2:1，都 ≥ 3:1（图形元素门槛）。
ARM["median_edge"] = {"color": GRAY_RAMP[1], "ls": (0, (1, 1)), "marker": "P", "label": "coordinate-wise median at edges"}
ARM["median_cloud"] = {"color": GRAY_RAMP[2], "ls": (0, (3, 1)), "marker": "X", "label": "coordinate-wise median at the cloud"}
# 6.2 共享深度权衡的两个划分：本文的机构式划分用边缘级共享块的绿（同一个实体：隔离多少块）；旧的逐类 Dirichlet 划分作复现序列、灰色虚线。
PARTITION = {
    "main": {"color": SERIES[2], "ls": "-", "marker": "s", "label": "institution-style partition"},
    "old": {"color": GRAY_RAMP[2], "ls": "--", "marker": "o", "label": "Dirichlet partition (earlier protocol)"},
}
# 图 6.7b（统计量偏移 vs 后门强度，D3 / F-136）：分散布点的边缘（每台都有攻击者）既不是受害边缘也不是被攻陷边缘 → 中性灰，不占类别色。
# 用浅一档的灰（GRAY_RAMP[1]），与图例里训练轮的黑色形状分开。
# 校验（dataviz validate_palette.js "#2a78d6,#eb6834,#8f8e89" --pairs all，2026-11-24）：CVD 最差 ΔE 10.9、正常视觉 17.3、对比度都 ≥ 3:1；
# 唯一的 FAIL 是灰色「色度低于下限」—— 它本来就是刻意的中性色，不当类别色用。
ROLE["dispersed"] = {"color": GRAY_RAMP[1], "label": "edges with attackers (dispersed placement)"}
# 训练进度（快照所在的云轮）用**实心**标记形状区分；全书「空心标记 = 种子」的约定不受影响（图注写明）。
ROUND_MARKER = {6: "o", 15: "D", 60: "s"}


def ramp(kind: str, n: int) -> list:
    """n 个有序类别的颜色（浅 → 深）；n ≤ 5。"""
    r = BLUE_RAMP if kind == "blue" else GRAY_RAMP
    if n > len(r):
        raise ValueError(f"有序类别最多 {len(r)} 个（{n}）")
    idx = [round(i * (len(r) - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [len(r) // 2]
    return [r[i] for i in idx]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": LW, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 2.4,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    return plt


def figure(nrows: int, ncols: int, height: float = 2.3, width: float = TEXTWIDTH, **kw):
    """全宽（或指定宽度）的图；height 是整张图的高度（英寸）。"""
    plt = setup()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False,
                             layout="constrained", **kw)
    return fig, axes


def panel(ax, letter: str):
    """面板字母 (a)(b)…，放在坐标轴左上角外侧。"""
    ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS, fontweight="bold", color=INK)


def band(ax, curves, color, label=None, ls="-", lw=LW, alpha=BAND_ALPHA, marker=None, markevery=None):
    """多条 (x, y) 曲线 → 只在所有曲线都有的 x 上画均值线 + [min, max] 带；返回 (末端 x, 末端均值)。"""
    from figures import band as _band
    pts = _band([c for c in curves if c])
    if not pts:
        return None
    xs, mu, lo, hi = zip(*pts)
    ax.fill_between(xs, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(xs, mu, color=color, linestyle=ls, linewidth=lw, label=label, marker=marker, markevery=markevery)
    return xs[-1], mu[-1]


def floor_line(ax, curves, label="trigger floor (no poisoning)"):
    from figures import band as _band
    pts = _band([c for c in curves if c])
    if pts:
        xs, mu, _, _ = zip(*pts)
        ax.plot(xs, mu, color=FLOOR, linewidth=LW_THIN, linestyle="--", label=label)


def threshold(ax, y, text=None, horizontal=True):
    """预注册门槛：灰色点线 + 数值标注。"""
    if horizontal:
        ax.axhline(y, color=FLOOR, linewidth=LW_THIN, linestyle=":")
        if text:
            ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right", va="bottom", fontsize=FS_SMALL, color=INK_2)
    else:
        ax.axvline(y, color=FLOOR, linewidth=LW_THIN, linestyle=":")


def end_labels(ax, ends, dx=3):
    """线尾直接标注；ends = [(x, y, text)]，纵向自动错开。"""
    from figures import spread_labels
    if not ends:
        return
    lo, hi = ax.get_ylim()
    ys = spread_labels([e[1] for e in ends], gap=0.07 * (hi - lo))
    for (x, _, t), y in zip(ends, ys):
        ax.annotate(t, (x, y), xytext=(dx, 0), textcoords="offset points", va="center",
                    fontsize=FS_SMALL, color=INK, annotation_clip=False)


def save(fig, name: str, out_dir: Path = OUT) -> list:
    """矢量 PDF（字体嵌入）+ 300 dpi PNG；宽度保持 figure() 给定的值（不用 bbox_inches=tight 改尺寸）。"""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None})   # 不写生成时间 → 重画不改数据时 PDF 逐字节相同（git 里不冒假改动）
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths
