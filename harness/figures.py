"""
harness/figures.py  —  由因素驱动的画图基元（读 runs_table.py 的两张表，不读文件名）

    python3 harness/figures.py trajectory --tables <dir> --metric local_benign_asr \\
        --group-by n_edges,edge_rounds --filter poison_ratio=0.2 \\
        --floor poison_ratio=0 --out fig.png
    python3 harness/figures.py per-edge --tables <dir> --run <run> \\
        --metric edge.client_benign --out fig.png

旧 plot_exp3.py 按文件名认格子（ORDER_3AB / ^3c_R / FLAT_CELL），新运行组一来就要改代码。
这里分组依据是 runs.csv 里的**实际**因素列（--group-by），所以登记表加组不用改画图。

出图规则（PLAN §6 / README）：
  · 横轴一律有效轮 r_eff = cloud_round × edge_rounds；
  · 跨 seed（与 replicate）→ 均值实线 + [min, max] 带；图例写明每组 n 个 run；
  · 只在所有 run 都有点的 r_eff 上画均值（否则某些点的「均值」只来自一部分 run）；
  · --floor 给出的下限组画成中性灰虚线（不是一种类别色：它不是参与比较的系列）；
  · ≥2 个系列必有图例；≤4 个系列另在线尾直接标注；单轴，不画双 y 轴；
  · 类别色按固定顺序分配（下方 SERIES，dataviz 参考调色板 light，已跑过
    validate_palette.js：全部 PASS；有 3 个色相对背景对比度 < 3:1 → 靠图例 + 直接标注 +
    runs.csv/series.csv 这张表来补足）；超过 7 组不生成新色相，改用 per-edge 那种小多图。
  · 图里的文字全用英文（集群容器里的 matplotlib 没有中文字体，会画成方块）。

matplotlib 只在画图时需要（容器里有）；本模块的数据整理部分是纯标准库。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# dataviz 参考调色板（light），固定顺序，不循环。
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4df"
FLOOR = "#8a8984"          # 中性灰：下限参考线


# ══════════════════════════════════════════════════════════════════════════
# 数据整理（纯标准库，可单测）
# ══════════════════════════════════════════════════════════════════════════

def load_tables(d):
    d = Path(d)
    with (d / "runs.csv").open(newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    with (d / "series.csv").open(newline="", encoding="utf-8") as f:
        series = [(r["run"], float(r["r_eff"]), int(r["cloud_round"]), int(r["edge_id"]),
                   r["metric"], float(r["value"])) for r in csv.DictReader(f)]
    return runs, series


def parse_filters(items):
    out = []
    for it in items or []:
        k, v = it.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def select(runs, filters):
    return [r for r in runs if all(_num_eq(r.get(k), v) for k, v in filters)]


def curve_of(series, run, metric, edge_id=-1):
    pts = sorted((x, v) for (rn, x, _, e, m, v) in series
                 if rn == run and m == metric and e == edge_id)
    return pts


def band(curves):
    """多条 (x, y) 曲线 → 只保留所有曲线都有的 x，返回 [(x, mean, min, max)]。"""
    if not curves:
        return []
    maps = [dict(c) for c in curves]
    xs = sorted(set.intersection(*(set(m) for m in maps)))
    out = []
    for x in xs:
        ys = [m[x] for m in maps]
        out.append((x, sum(ys) / len(ys), min(ys), max(ys)))
    return out


def _sort_token(x):
    try:
        return (0, float(x), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(x))


def group_runs(runs, group_by):
    groups = {}
    for r in runs:
        key = tuple(r.get(g) for g in group_by)
        groups.setdefault(key, []).append(r)
    return dict(sorted(groups.items(), key=lambda kv: [_sort_token(x) for x in kv[0]]))


# runs.csv 里描述「这是哪一格」的列（与 runs_table.FACTOR_KEYS 一致，外加口径版本）。
FACTOR_COLUMNS = ("protocol", "method", "attack", "defense", "n_clients", "n_edges",
                  "edge_rounds", "client_fraction", "poison_ratio", "malicious_per_edge",
                  "malicious_placement", "edge_assignment", "local_epochs", "plocal_epochs",
                  "attack_stop_round")


def heterogeneous_factors(groups) -> dict:
    """{组: [组内取值不唯一的因素列]}。seed / replicate 不算（它们本来就该变）。

    一条带子如果混了不同的格子（例：按 n_edges 分组却没固定攻击者比例），
    min–max 带画的就不是种子噪声 —— 旧 HHI 回归把 ρ / def / 3C 格子混在一起就是这种错（F-010）。
    """
    out = {}
    for key, rs in groups.items():
        bad = [c for c in FACTOR_COLUMNS if len({r.get(c) for r in rs}) > 1]
        if bad:
            out[key] = bad
    return out


def spread_labels(ys, gap):
    """线尾直接标注的纵坐标：保持原有上下顺序，相邻至少隔 gap（向上推开）。
    返回与输入同序的新纵坐标。"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    prev = None
    for i in order:
        y = ys[i] if prev is None else max(ys[i], prev + gap)
        out[i] = y
        prev = y
    return out


def _label(group_by, key):
    return ", ".join(f"{g}={v}" for g, v in zip(group_by, key))


# ══════════════════════════════════════════════════════════════════════════
# 画图
# ══════════════════════════════════════════════════════════════════════════

def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)


def trajectory(runs, series, *, metric, group_by, out, filters=(), floor=(), title=None,
               allow_mixed=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = select(runs, filters)
    groups = group_runs(chosen, group_by)
    if not groups:
        raise SystemExit("没有 run 符合过滤条件")
    mixed = heterogeneous_factors(groups)
    if mixed and not allow_mixed:
        lines = "\n".join(f"  {_label(group_by, k)}: {', '.join(v)}" for k, v in mixed.items())
        raise SystemExit("这些组内部还有不止一个取值的因素，带子会把不同格子混在一起：\n"
                         f"{lines}\n→ 把它们加进 --group-by 或 --filter（确实要混时加 --allow-mixed）")
    if len(groups) > len(SERIES):
        raise SystemExit(f"{len(groups)} 组超过 {len(SERIES)} 个固定色相 —— 拆成小多图或收窄 --filter")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax, "effective rounds (cloud round x edge_rounds)", metric)
    direct = len(groups) <= 4
    ends = []                                   # (x, y, 标注文字)：画完再统一排开，避免重叠
    for i, (key, rs) in enumerate(groups.items()):
        pts = band([curve_of(series, r["run"], metric) for r in rs])
        if not pts:
            continue
        xs, mu, lo, hi = zip(*pts)
        c = SERIES[i]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.16, linewidth=0)
        ax.plot(xs, mu, color=c, linewidth=2, label=f"{_label(group_by, key)}  (n={len(rs)})")
        if direct:
            ends.append((xs[-1], mu[-1], " / ".join(str(v) for v in key)))
    if floor:
        frs = select(runs, floor)
        pts = band([curve_of(series, r["run"], metric) for r in frs])
        if pts:
            xs, mu, _, _ = zip(*pts)
            ax.plot(xs, mu, color=FLOOR, linewidth=1.5, linestyle="--",
                    label=f"floor: {', '.join(f'{k}={v}' for k, v in floor)}  (n={len(frs)})")
    if metric.endswith("asr") or metric.endswith("acc") or metric.startswith("edge."):
        ax.set_ylim(0, 1)
    ymin, ymax = ax.get_ylim()
    for (x, _, text), y in zip(ends, spread_labels([e[1] for e in ends],
                                                   gap=0.045 * (ymax - ymin))):
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points",
                    color=INK, fontsize=7, va="center", annotation_clip=False)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def per_edge(runs, series, *, run, metric, out, title=None):
    """一个 run 的逐 edge 小多图（每个 edge 一格、共享 y 轴）—— 10 个 edge 也不用生成第 8 个色相。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = sorted({e for (rn, _, _, e, m, _) in series if rn == run and m == metric and e >= 0})
    if not edges:
        raise SystemExit(f"{run} 没有逐 edge 的 {metric}")
    ncol = min(5, len(edges))
    nrow = (len(edges) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 1.9 * nrow), dpi=150,
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, e in zip(axes.flat, edges):
        pts = curve_of(series, run, metric, e)
        _style(ax, "", "")
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[0], linewidth=2)
        ax.set_title(f"edge {e}", color=INK, fontsize=8, loc="left")
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(edges):]:
        ax.set_visible(False)
    fig.supxlabel("effective rounds", color=INK_2, fontsize=8)
    fig.supylabel(metric, color=INK_2, fontsize=8)
    fig.suptitle(title or run, color=INK, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="从 runs.csv / series.csv 出图")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trajectory")
    t.add_argument("--tables", required=True)
    t.add_argument("--metric", required=True)
    t.add_argument("--group-by", required=True, help="逗号分隔的 runs.csv 列名")
    t.add_argument("--filter", action="append", default=[], help="key=value，可重复")
    t.add_argument("--floor", action="append", default=[], help="下限组 key=value，可重复")
    t.add_argument("--title")
    t.add_argument("--allow-mixed", action="store_true", help="允许组内因素不唯一（不推荐）")
    t.add_argument("--out", required=True)
    p = sub.add_parser("per-edge")
    p.add_argument("--tables", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--metric", default="edge.client_benign")
    p.add_argument("--title")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runs, series = load_tables(args.tables)
    if args.cmd == "trajectory":
        trajectory(runs, series, metric=args.metric, group_by=args.group_by.split(","),
                   out=args.out, filters=parse_filters(args.filter),
                   floor=parse_filters(args.floor), title=args.title,
                   allow_mixed=args.allow_mixed)
    else:
        per_edge(runs, series, run=args.run, metric=args.metric, out=args.out, title=args.title)
    print(f"[figures] → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
