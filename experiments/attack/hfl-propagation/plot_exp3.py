#!/usr/bin/env python3
"""
plot_exp3.py  —  Experiment 3A/3B/3C 结果可视化。

读 results/<cell>_seed<seed>.metrics.json，跨 seed 聚合成 **均值 + min–max 界**，出图：
  fig_topology_summary.png   3A/3B：各拓扑的 global / benign(local) / PM-acc 柱（min–max 须）
  fig_per_edge_propagation.png 3A：逐 edge 的 edge_asr vs client_benign（min–max 带）
  fig_3c_frequency.png       3C：ASR/acc 随 edge_rounds 的曲线（min–max 带）

设计要点（按需求）：
  - 跨 seed 同设定 → 均值实线 + [min, max] 半透明带/误差须（不是 ±std）。
  - legend 颜色高对比、**不同色系**：用 Okabe–Ito 色盲安全定性色板，每个量一个独立色相。

用法（有 matplotlib 即可，不需要 GPU/TF）：
  python3 experiments/attack/hfl-propagation/plot_exp3.py
  python3 experiments/attack/hfl-propagation/plot_exp3.py --results <dir> --out <dir>
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt

# ── Okabe–Ito：色盲安全、色相彼此拉开（满足「高对比、不同色系」）───────────────
C = {
    "global":    "#0072B2",  # 蓝
    "edge":      "#D55E00",  # 朱红
    "benign":    "#009E73",  # 绿
    "malicious": "#CC79A7",  # 紫红
    "pm":        "#E69F00",  # 橙
}
CELL_RE = re.compile(r"^(?P<cell>.+)_seed(?P<seed>\d+)$")

# 3A/3B 展示顺序
ORDER_3AB = [
    "2edge_collocated", "2edge_distributed",
    "4edge_collocated", "4edge_distributed", "4edge_mixed",
    "10edge_collocated", "10edge_distributed", "10edge_mixed",
]


def load_results(results_dir: Path) -> dict:
    """{cell: [metrics_dict per seed]}，只收成功（无则尽力）的 json。"""
    runs = {}
    for p in sorted(results_dir.glob("*.metrics.json")):
        stem = p.name[:-len(".metrics.json")]   # 去掉双扩展名（.stem 只去 .json）
        m = CELL_RE.match(stem)
        if not m:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"[plot] 跳过无法解析的 {p.name}: {e}")
            continue
        if data.get("final") is None:
            print(f"[plot] 跳过无 final 的 {p.name}（run 可能没跑出后门评估）")
            continue
        runs.setdefault(m.group("cell"), []).append(data)
    return runs


def _agg(vals):
    """→ (mean, lo, hi)；空 → (nan,nan,nan)。"""
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(a.mean()), float(a.min()), float(a.max()))


def _final_scalar(runs_of_cell, key):
    return _agg([r["final"].get(key) for r in runs_of_cell])


def _pm(runs_of_cell):
    out = []
    for r in runs_of_cell:
        fa = r.get("final_acc") or {}
        out.append(fa.get("pm_acc") if fa.get("pm_acc") is not None
                   else fa.get("final_pm_weighted"))
    return _agg(out)


def _whisker(ax, x, agg, color, label=None, width=0.0):
    """画一个 mean 点/柱 + [lo,hi] 误差须。"""
    mean, lo, hi = agg
    if np.isnan(mean):
        return
    yerr = [[mean - lo], [hi - mean]]
    ax.errorbar(x, mean, yerr=yerr, fmt="o", color=color, ecolor=color,
                elinewidth=1.6, capsize=4, markersize=6, label=label)


# ══════════════════════════════════════════════════════════════════════════
# Fig A：3A/3B 拓扑汇总（global / benign / PM-acc，均值 + min–max 须）
# ══════════════════════════════════════════════════════════════════════════
def fig_topology_summary(runs, out):
    cells = [c for c in ORDER_3AB if c in runs]
    if not cells:
        return
    series = [("global_asr", "GM ASR (global)", C["global"], _final_scalar),
              ("local_benign_asr", "benign ASR (local)", C["benign"], _final_scalar),
              ("__pm__", "PM acc (MTA)", C["pm"], None)]
    x = np.arange(len(cells))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(cells)), 5))
    for i, (key, lbl, color, fn) in enumerate(series):
        means, los, his = [], [], []
        for c in cells:
            agg = _pm(runs[c]) if key == "__pm__" else _final_scalar(runs[c], key)
            means.append(agg[0]); los.append(agg[1]); his.append(agg[2])
        means, los, his = np.array(means), np.array(los), np.array(his)
        xpos = x + (i - 1) * w
        yerr = np.nan_to_num(np.clip(np.vstack([means - los, his - means]), 0, None))
        ax.bar(xpos, np.nan_to_num(means), width=w, color=color, label=lbl, alpha=0.9)
        ax.errorbar(xpos, np.nan_to_num(means), yerr=yerr, fmt="none", ecolor="#222222",
                    elinewidth=1.2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ASR / Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Exp 3A/3B — final metrics per topology (bar = mean, whisker = min–max over seeds)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out / "fig_topology_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig B：3A 逐 edge 传播（edge_asr vs client_benign，均值线 + min–max 带）
# ══════════════════════════════════════════════════════════════════════════
def _per_edge_agg(runs_of_cell):
    """→ {edge_id: {"edge_asr":(m,lo,hi), "client_benign":(...), "has_malicious":bool}}"""
    by_edge = {}
    for r in runs_of_cell:
        for pe in r.get("per_edge_final", []):
            e = int(pe["edge_id"])
            by_edge.setdefault(e, {"edge_asr": [], "client_benign": [], "has_malicious": False})
            by_edge[e]["edge_asr"].append(pe.get("edge_asr"))
            by_edge[e]["client_benign"].append(pe.get("client_benign"))
            by_edge[e]["has_malicious"] |= bool(pe.get("has_malicious"))
    return {e: {"edge_asr": _agg(d["edge_asr"]),
                "client_benign": _agg(d["client_benign"]),
                "has_malicious": d["has_malicious"]}
            for e, d in sorted(by_edge.items())}


def fig_per_edge(runs, out):
    cells = [c for c in ORDER_3AB if c in runs and runs[c] and runs[c][0].get("per_edge_final")]
    if not cells:
        return
    ncol = min(3, len(cells))
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for idx, c in enumerate(cells):
        ax = axes[idx // ncol][idx % ncol]
        pe = _per_edge_agg(runs[c])
        eids = sorted(pe.keys())
        for key, color, lbl in [("edge_asr", C["edge"], "edge model ASR"),
                                ("client_benign", C["benign"], "benign client ASR")]:
            mean = np.array([pe[e][key][0] for e in eids])
            lo   = np.array([pe[e][key][1] for e in eids])
            hi   = np.array([pe[e][key][2] for e in eids])
            ax.plot(eids, mean, "-o", color=color, label=lbl, markersize=4)
            ax.fill_between(eids, lo, hi, color=color, alpha=0.18)
        # 标出含恶意端的 edge
        for e in eids:
            if pe[e]["has_malicious"]:
                ax.axvspan(e - 0.5, e + 0.5, color="#999999", alpha=0.10)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("edge id"); ax.set_ylabel("ASR")
        ax.set_ylim(0, 1.02); ax.set_xticks(eids); ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(frameon=False, fontsize=8)
    for j in range(len(cells), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Exp 3A — per-edge propagation (grey band = edge with malicious; shaded = min–max over seeds)", y=1.0)
    fig.tight_layout()
    p = out / "fig_per_edge_propagation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {p}")


# ══════════════════════════════════════════════════════════════════════════
# Fig C：3C 聚合频率曲线（ASR/acc vs edge_rounds，均值线 + min–max 带）
# ══════════════════════════════════════════════════════════════════════════
def fig_3c(runs, out):
    cells = {}
    for c in runs:
        m = re.match(r"^3c_R(\d+)$", c)
        if m:
            cells[int(m.group(1))] = c
    if not cells:
        return
    Rs = sorted(cells.keys())

    def _drift_agg(R, key):
        return _agg([(r.get("drift_final") or {}).get(key) for r in runs[cells[R]]])

    has_drift = any((runs[cells[R]][0].get("drift_final")) for R in Rs)

    def _plot_series(ax, key, lbl, color, aggfn):
        mean, lo, hi = [], [], []
        for R in Rs:
            agg = aggfn(R, key)
            mean.append(agg[0]); lo.append(agg[1]); hi.append(agg[2])
        mean, lo, hi = np.array(mean), np.array(lo), np.array(hi)
        ax.plot(Rs, mean, "-o", color=color, label=lbl, markersize=5)
        ax.fill_between(Rs, lo, hi, color=color, alpha=0.18)

    nrow = 2 if has_drift else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(7.5, 4.6 * nrow), sharex=True, squeeze=False)
    top = axes[0][0]

    # ── 上panel：三层 ASR + MTA ─────────────────────────────────────────────
    def _asr_agg(R, key):
        return _pm(runs[cells[R]]) if key == "__pm__" else _final_scalar(runs[cells[R]], key)
    for key, lbl, color in [("global_asr", "GM ASR (global)", C["global"]),
                            ("edge_asr", "EM ASR (edge)", C["edge"]),
                            ("local_benign_asr", "benign ASR (local)", C["benign"]),
                            ("local_malicious_asr", "malicious ASR (local)", C["malicious"]),
                            ("__pm__", "PM acc (MTA)", C["pm"])]:
        _plot_series(top, key, lbl, color, _asr_agg)
    top.set_ylabel("ASR / Accuracy"); top.set_ylim(0, 1.02)
    top.set_title("Exp 3C — aggregation frequency (shaded = min–max over seeds)")
    top.legend(frameon=False, ncol=2); top.grid(alpha=0.25)

    # ── 下panel：漂移。相对量在左轴，绝对量 param_abs 在右孪生轴 ────────────
    if has_drift:
        bot = axes[1][0]
        for key, lbl, color in [("repr_mean", "repr shift (mean)", "#009E73"),
                                 ("repr_median", "repr shift (median)", "#56B4E9"),
                                 ("param_rel", "param drift (rel)", "#CC79A7")]:
            _plot_series(bot, key, lbl, color, _drift_agg)
        bot.set_ylabel("relative drift  (1-cos  /  ||d||/||theta||)")
        bot.grid(alpha=0.25)
        botr = bot.twinx()
        _plot_series(botr, "param_abs", "param drift (abs ||d||)", "#000000", _drift_agg)
        botr.set_ylabel("absolute ||d|| (backbone L2)")
        h1, l1 = bot.get_legend_handles_labels()
        h2, l2 = botr.get_legend_handles_labels()
        bot.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc="best")

    axb = axes[-1][0]
    axb.set_xscale("log", base=2)
    axb.set_xticks(Rs); axb.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axb.set_xlabel("edge_rounds  R_edge   (R_edge x R_cloud = 400 fixed;  R=1 ~ flat)")
    fig.tight_layout()
    p = out / "fig_3c_frequency.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {p}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here / "results"), help="metrics.json 目录")
    ap.add_argument("--out", default=str(here / "results" / "figures"), help="图输出目录")
    args = ap.parse_args()

    results_dir = Path(args.results)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = load_results(results_dir)
    if not runs:
        print(f"[plot] {results_dir} 下没有可用的 *.metrics.json —— 先跑实验（run_exp3.sh）。")
        return
    n_seeds = {c: len(v) for c, v in runs.items()}
    print(f"[plot] 载入 {len(runs)} 个设定；每设定 seed 数：{n_seeds}")
    for c, n in n_seeds.items():
        if n < 2:
            print(f"[plot] ⚠ {c} 只有 {n} 个 seed，min–max 界退化为一个点（补 seed 更有意义）")

    fig_topology_summary(runs, out)
    fig_per_edge(runs, out)
    fig_3c(runs, out)
    print(f"[plot] 完成，图在 {out}")


if __name__ == "__main__":
    main()
