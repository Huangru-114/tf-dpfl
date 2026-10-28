"""
harness/p0_tables.py  —  阶段三 P0：把一个 screen 结果画成逐方法对比表（Markdown），只读回传的 JSON，不做任何判定

    python3 harness/p0_tables.py experiments/defense/edge-native/analysis/p0/SNAP__collocated__s42.screen.json \\
        --metrics experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json \\
        --out experiments/defense/edge-native/analysis/p0_s42_screen_tables.md

每个快照一张表。行 = 两条基线（中心平均前 = 原训练第 t 大轮末的记录；中心平均后、不加防御 = 离线重算的 G）+ 36 个方案 + 两个只报告的参照。
列 = 各区域良性客户端的攻击成功率（ASR）与正常准确率（ACC）、受害区域均值、挡掉比例、准确率损失、是否达标、触发样本 margin、计算量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fedavg"))
from hardening import spec      # noqa: E402

OBJ = {"pgd": "普通对抗训练（PGD）", "trades": "TRADES", "sau": "SAU", "tgt": "定向对抗训练", "cft": "只做干净微调（对照）"}
ORDER = ["sau", "pgd", "trades", "tgt", "cft"]


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _pct(v):
    return "—" if v is None else f"{100 * v:.0f}%"


def _eps(c):
    if c["eps_px"] is None:
        return ""
    e = c["eps_px"] * 255
    return "0.2（≈51/255）" if abs(e - 51) < 0.5 else f"{e:.0f}/255"


def _label(c):
    if c["objective"] == "cft":
        return "只做干净微调（SAU 的对照：Adam、不增强）" if c["id"].startswith("cft-sau") else "只做干净微调（对照：SGD）"
    name = OBJ[c["objective"]]
    if c["objective"] == "trades":
        name += f" β={c['beta']:g}"
    return name


def table(res, metrics, t, grid):
    S = res["snapshots"][str(t)]
    base = S["base"]
    J = S["J"]
    victims = res["victims"]
    edges = sorted(int(e) for e in base["per_edge"])
    full = {int(d["edge_id"]): d for d in (metrics.get("per_edge_rounds") or {}).get(str(t)) or []}
    facc = {int(d["edge_id"]): d for d in (metrics.get("per_edge_acc_rounds") or {}).get(str(t)) or []}
    per_cloud = res["fb_clients_per_cloud_round"]          # 一个大轮里全部客户端训练的前向反向次数（驱动写的）
    per_edge_cloud = per_cloud / len(edges)

    head = ("| 方法 | 扰动上限 | BN 层 | 洗几遍 | 攻击成功率 " + " / ".join(f"区{e}" for e in edges)
            + " | 受害区均值 | 挡掉比例 | 准确率 " + " / ".join(f"区{e}" for e in edges)
            + " | 池化准确率 | 池化降 | 单区最大降 | 达标 | 受害区 margin | 计算量 |")
    sep = "|" + "---|" * (head.count("|") - 1)
    rows = [head, sep]

    def asr_cells(pe):
        pe = {int(k): v for k, v in pe.items()}
        return " / ".join(_f(pe[e]["client_benign"], 2) for e in edges)

    def acc_cells(pe):
        pe = {int(k): v for k, v in pe.items()}
        return " / ".join(_f(pe[e]["pm_acc"], 2) for e in edges)

    vm = sum(full[e]["client_benign"] for e in victims) / len(victims) if full else None
    rows.append(f"| **中心平均前**（原训练第 {t} 大轮末记录） | | | | "
                + " / ".join(_f(full[e]["client_benign"], 2) for e in edges) + f" | {_f(vm)} | | "
                + " / ".join(_f(facc[e]["pm_acc"], 2) for e in edges) + " | | | | | | |")
    rows.append(f"| **中心平均后、不加防御**（= 洗前） | | | | {asr_cells(base['per_edge'])} | {_f(base['victim_asr'])} | 0% | "
                f"{acc_cells(base['per_edge'])} | {_f(base['pm_acc'])} | 0 | 0 | | {_f(base['margin_v'], 2)} | 0 |")
    by = {c["id"]: c for c in grid}
    ids = sorted(S["configs"], key=lambda i: (ORDER.index(by[i]["objective"]), i.startswith("cft-sau"),
                                               by[i]["eps_px"] or 0, by[i]["beta"] or 0, by[i]["bn"], by[i]["budget"]))
    for cid in ids:
        c, r = by[cid], S["configs"][cid]
        bn = "更新" if c["bn"] == "train" else "固定"
        passes = "1" if c["budget"] == "low" else "5"
        cost = f"{spec.fb_harden(c, 500, 32) / per_edge_cloud:.0%}"
        lead = f"| {_label(c)} | {_eps(c)} | {bn} | {passes} | "
        if r.get("diverged"):
            rows.append(lead + "**数值发散，未评估** | | | | | | | 否 | | " + cost + " |")
            continue
        ev = r["eval"]
        rh = spec.r_h(base["victim_asr"], ev["victim_asr"], J)
        acc = spec.accuracy_ok(base, ev)
        rows.append(lead + f"{asr_cells(ev['per_edge'])} | {_f(ev['victim_asr'])} | {_pct(rh)} | {acc_cells(ev['per_edge'])} | "
                    f"{_f(ev['pm_acc'])} | {_f(acc['d_pool'])} | {_f(acc['d_edge_max'])} | {'**是**' if acc['ok'] else '否'} | "
                    f"{_f(ev['margin_v'], 2)} | {cost} |")
    for key, name in (("damping_0.5", "参照：每区只吸收一半中心更新"), ("damping_0.25", "参照：每区只吸收 1/4 中心更新"),
                      ("oracle_isolation", "参照：中心平均时直接去掉区0（上帝视角）")):
        ev = (S.get("aux") or {}).get(key)
        if not ev:
            continue
        rh = spec.r_h(base["victim_asr"], ev["victim_asr"], J)
        acc = spec.accuracy_ok(base, ev)
        rows.append(f"| *{name}* | | | | {asr_cells(ev['per_edge'])} | {_f(ev['victim_asr'])} | {_pct(rh)} | "
                    f"{acc_cells(ev['per_edge'])} | {_f(ev['pm_acc'])} | {_f(acc['d_pool'])} | {_f(acc['d_edge_max'])} | — | "
                    f"{_f(ev['margin_v'], 2)} | 0 |")
    title = (f"### 快照：第 {t} 大轮（SNAP，seed 42，10 个攻击者全在区0，区1–3 是受害区域）\n\n"
             f"这次中心平均让受害区均值攻击成功率涨了 **{J:.3f}**（{_f(vm)} → {_f(base['victim_asr'])}）；"
             f"挡掉比例 = (洗前 − 洗后) / {J:.3f}。\n")
    return title + "\n" + "\n".join(rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("screen")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--grid", default=str(HERE.parent / "experiments/defense/edge-native/p0_configs.yaml"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import yaml
    res = json.loads(Path(a.screen).read_text(encoding="utf-8"))
    metrics = json.loads(Path(a.metrics).read_text(encoding="utf-8"))
    grid = spec.expand_grid(yaml.safe_load(Path(a.grid).read_text(encoding="utf-8"))["grid"])
    md = "\n".join(table(res, metrics, int(t), grid) for t in sorted(res["snapshots"], key=int))
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
