"""
harness/partition_preview.py  —  S3 划分的离线预览（= 图 F0 的数据），不需要 TF、不需要 GPU

    python3 harness/partition_preview.py [--seeds 42 43 44] [--csv out.csv] [--plot F0.png]

对 C1–C4（固定设计）、层级 Dirichlet α_e ∈ {10, 1, 0.3, 0.1}、等大小 random 各跑一遍
`fedavg/data/designed_partition.py`（与训练时同一个函数、同一个 seed 流），报实测 (H_inter, H_intra)、
名义 H_inter（投影之前）、最紧的一类用量、各 edge 的 airplane 占比。

划分只依赖类计数（每类 6000），不依赖图片本身 → 用合成标签得到的索引集合与真实数据的**计数**完全相同。
开跑前先看 α_e 四档实测 H_inter 能不能分开（F-057：名义 p_e 90–100% 超供给，投影会把它压平）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from data import designed_partition as D   # noqa: E402

CASES = ([("designed", c, {"condition": c, "strength": 0.25}) for c in sorted(D.CONDITIONS)]
         + [("hdir", f"hdir-a{a:g}", {"alpha_edge": a}) for a in (10.0, 1.0, 0.3, 0.1)]
         + [("equal_random", "random", {})])


def config(kind, design, n_edges=4, n_clients=100):
    return {"seed": 42, "data": {"dataset": "cifar10", "per_client_test_ratio": 0.25},
            "federation": {"partition": kind, "n_clients": n_clients, "n_edges": n_edges,
                           "design": {"n_per_client": 500, "clean_per_edge": 500,
                                      "alpha_client": 0.5, **design}},
            "backdoor": {"target_label": D.TARGET}}


def preview(seeds=(42, 43, 44)) -> list:
    labels = np.repeat(np.arange(D.K), D.SUPPLY_PER_CLASS)
    rows = []
    for kind, name, design in CASES:
        for seed in seeds:
            sp = D.designed_partition(labels, config(kind, design), seed=seed)
            info = sp["info"]
            rows.append({"case": name, "partition": kind, "seed": seed,
                         "h_inter": round(info["h_inter"], 4), "h_intra": round(info["h_intra"], 4),
                         "h_inter_nominal": (None if info["h_inter_nominal"] is None
                                             else round(info["h_inter_nominal"], 4)),
                         "max_class_use": info["max_class_use"],
                         "yt_share": [round(e["yt_share"], 3) for e in sp["per_edge"]],
                         "index_sha": info["index_sha"]})
    return rows


def summary(rows) -> list:
    out = []
    for name in dict.fromkeys(r["case"] for r in rows):
        rs = [r for r in rows if r["case"] == name]
        hi = [r["h_inter"] for r in rs]
        nom = [r["h_inter_nominal"] for r in rs if r["h_inter_nominal"] is not None]
        out.append({"case": name, "n": len(rs),
                    "h_inter": round(float(np.mean(hi)), 3), "h_inter_min": min(hi), "h_inter_max": max(hi),
                    "h_intra": round(float(np.mean([r["h_intra"] for r in rs])), 3),
                    "h_inter_nominal": round(float(np.mean(nom)), 3) if nom else None,
                    "max_class_use": max(r["max_class_use"] for r in rs),
                    "yt_share_first_seed": rs[0]["yt_share"]})
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in dict.fromkeys(r["case"] for r in rows):
        rs = [r for r in rows if r["case"] == name]
        ax.scatter([r["h_inter"] for r in rs], [r["h_intra"] for r in rs], label=f"{name} (n={len(rs)})")
    ax.set_xlabel("H_inter (mean pairwise TV between edges)")
    ax.set_ylabel("H_intra (mean TV client vs its edge)")
    ax.set_title("F0: measured heterogeneity of S3 partitions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description="S3 划分离线预览（F0 数据）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--csv")
    ap.add_argument("--plot")
    a = ap.parse_args(argv)
    rows = preview(tuple(a.seeds))
    print(f"{'case':<12} {'H_inter':>8} {'(min–max)':>15} {'nominal':>8} {'H_intra':>8} {'max use':>8}  airplane / edge（首个 seed）")
    for s in summary(rows):
        nom = "—" if s["h_inter_nominal"] is None else f"{s['h_inter_nominal']:.3f}"
        print(f"{s['case']:<12} {s['h_inter']:>8.3f} {s['h_inter_min']:>7.3f}–{s['h_inter_max']:<7.3f} "
              f"{nom:>8} {s['h_intra']:>8.3f} {s['max_class_use']:>8}  {s['yt_share_first_seed']}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if a.plot:
        plot(rows, a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
