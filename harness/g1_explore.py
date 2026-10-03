"""
harness/g1_explore.py  —  G1 / G1R5 的**探索性**读数（2026-10-03；看过数据之后才写，**不进任何判定**）

    python3 harness/g1_explore.py [--json out.json]

判定在 `g1_verdict.py`（N-007，预注册）。这里回答的是判定之外、写报告和设计防御时想知道的几件事，
每一项都是描述性的，阈值一律没有，结论要写成「看到了什么」而不是「证明了什么」：

  ① 锯齿的周期内相位剖面（集中布点）：受害 edge E1–E3 均值与攻击者 edge E0 的良性端 ASR 和 margin 中位数，
    按「周期内第几个 edge 轮」排开 —— 云聚合后（er = 0）、轻评估点（er = 5, 10, 15 < R）、云轮末（er = R）；
    前半程（g = 2 … ⌊n/2⌋）与后半程（g > ⌊n/2⌋）分开平均。
    margin = log p_t − max_{k≠t} log p_k 的中位数（D-076 原定的 3-C 主量；ASR 饱和时 margin 仍能动）。
  ② lr 混杂的量级：r_down 前半程 vs 后半程（lr 按有效轮衰减，后半程每个 edge 轮动得少 —— 只看量级，不做因果拆分）；
  ③ 攻击者 edge 的反向锯齿：云聚合把 E0 稀释掉多少（drop = 云聚合后 − 上一云轮末）、周期内多快爬回去（r_up）；
  ④ 一个周期的净变化：Δ_jump − 每周期洗掉量（> 0 = 受害 edge 一个周期比一个周期高 → 走向饱和）；
  ⑤ 3-D 的逐窗口 AUROC（分散布点，norm_w / norm_s / s_ck）：可检测性是否随攻击进入饱和而变。

纯标准库 + numpy，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fedavg"))
from analysis.functional_score import auroc                         # noqa: E402
from collect_metrics import LIGHT_EDGE_COLUMNS                      # noqa: E402
from g1_scores import collect_updates, view_arrays                  # noqa: E402
from g1_verdict import (G1_ROUNDS, PARTITIONS, PLACEMENTS, SEEDS, VICTIMS,   # noqa: E402
                        load)

EDGE_COLS = list(LIGHT_EDGE_COLUMNS)


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rows_to_edge(rows, cols, col):
    ei, ci = cols.index("edge_id"), cols.index(col)
    return {int(r[ei]): r[ci] for r in rows or [] if r and r[ei] is not None}


def point_tables(m: dict, col: str) -> dict:
    """{(g, er): {edge: 值}}：er = 0 云聚合后、轻评估点、er = R 云轮末。col ∈ {client_benign, margin_p50}。"""
    run = m["run"]
    R = int(run["edge_rounds"])
    out = {}
    for g, rows in (m.get("per_edge_post_agg_rounds") or {}).items():
        out[(int(g), 0)] = _rows_to_edge(rows, EDGE_COLS, col)
    lc = list(m.get("per_edge_light_columns") or EDGE_COLS)
    for eff, rows in (m.get("per_edge_light_rounds") or {}).items():
        eff = int(eff)
        g, er = (eff - 1) // R + 1, (eff - 1) % R + 1
        out[(g, er)] = _rows_to_edge(rows, lc, col)
    if col == "client_benign":
        for g, rows in (m.get("per_edge_rounds") or {}).items():
            out[(int(g), R)] = {int(e["edge_id"]): e.get("client_benign") for e in rows or []}
    else:
        dc = list(m.get("per_edge_detail_columns") or [])
        if col in dc:
            for g, rows in (m.get("per_edge_detail_rounds") or {}).items():
                out[(int(g), R)] = _rows_to_edge(rows, dc, col)
    return out


def _victim(vals: dict):
    v = [vals.get(e) for e in VICTIMS]
    return None if any(x is None for x in v) else sum(v) / len(v)


def cycle_profile(m: dict, col: str) -> dict:
    """前 / 后半程各自按 er 平均的剖面：{half: {er: {victims, e0}}}。"""
    run = m["run"]
    n = int(run["n_rounds"])
    pts = point_tables(m, col)
    prof = {"first": {}, "second": {}}
    for (g, er), vals in pts.items():
        if g < 2:
            continue
        half = "first" if g <= n // 2 else "second"
        d = prof[half].setdefault(er, {"victims": [], "e0": []})
        d["victims"].append(_victim(vals))
        d["e0"].append(vals.get(0))
    return {h: {er: {k: _r(_mean(v)) for k, v in d.items()} for er, d in sorted(p.items())}
            for h, p in prof.items()}


def rates(m: dict) -> dict:
    """受害 edge 的 r_down / Δ_jump（ASR 与 margin）与 E0 的 drop / r_up，前 / 后半程分开。"""
    run = m["run"]
    R, n = int(run["edge_rounds"]), int(run["n_rounds"])
    out = {}
    for col in ("client_benign", "margin_p50"):
        pts = point_tables(m, col)
        for half, gs in (("first", range(2, n // 2 + 1)), ("second", range(n // 2 + 1, n + 1))):
            rd, dj, drop, rup = [], [], [], []
            for g in gs:
                post, full, prev = pts.get((g, 0)), pts.get((g, R)), pts.get((g - 1, R))
                if not (post and full and prev):
                    continue
                vp, vf, vq = _victim(post), _victim(full), _victim(prev)
                if None not in (vp, vf, vq):
                    rd.append((vp - vf) / R)
                    dj.append(vp - vq)
                if None not in (post.get(0), full.get(0), prev.get(0)):
                    drop.append(post[0] - prev[0])
                    rup.append((full[0] - post[0]) / R)
            out[f"{col}:{half}"] = {"victim_r_down": _r(_mean(rd), 5), "victim_d_jump": _r(_mean(dj)),
                                    "victim_net_per_cycle": _r(None if not dj else _mean(dj) - _mean(rd) * R),
                                    "e0_drop_at_agg": _r(_mean(drop)), "e0_r_up": _r(_mean(rup), 5),
                                    "n_cycles": len(rd)}
    return out


def window_aurocs(m: dict, keys=("norm_w", "norm_s", "s_ck"), window: int = 20) -> dict:
    """逐窗口（20 有效轮）的 raw 与 edge 视角 AUROC（不重抽样：只看随时间的走向）。"""
    ups = collect_updates(m)
    out = {}
    for key in keys:
        a = view_arrays(ups, key, window, resamples=0)
        if a is None:
            out[key] = []
            continue
        effs = np.array([u["eff"] for u in ups if u.get(key) is not None])
        blocks = (effs - 1) // window
        rows = []
        for b in sorted(set(blocks.tolist())):
            sel = blocks == b
            mal = a["mal"][sel]
            rows.append({"window": int(b), "eff_from": int(b * window + 1), "n": int(sel.sum()),
                         "n_mal": int(mal.sum()),
                         "raw": _r(auroc(a["x"][sel][mal], a["x"][sel][~mal])),
                         "edge": _r(auroc(a["z_edge"][sel][mal], a["z_edge"][sel][~mal]))})
        out[key] = rows
    return out


def _avg_dicts(ds: list) -> dict:
    """同结构字典逐叶子求均值（None 不计）。"""
    if not ds:
        return {}
    if isinstance(ds[0], dict):
        return {k: _avg_dicts([d.get(k) for d in ds if isinstance(d, dict)]) for k in ds[0]}
    return _r(_mean(ds), 5)


def explore(g1: dict, g1r5: dict) -> dict:
    res = {"note": "探索性：脚本写于看过 G1 / G1R5 全部数据之后；不进任何判定（判定见 g1_verdict.py / N-007）",
           "collocated": {}, "distributed_window_auroc": {}}
    cells = [(p, R, {s: g1.get((p, "collocated", R, s)) for s in SEEDS}) for p in PARTITIONS for R in G1_ROUNDS]
    cells.append(("C1", 5, g1r5))
    for part, R, runs in cells:
        per_seed = {}
        for s, m in runs.items():
            if m is None:
                continue
            per_seed[f"s{s}"] = {"profile_asr": cycle_profile(m, "client_benign"),
                                 "profile_margin": cycle_profile(m, "margin_p50"), "rates": rates(m)}
        res["collocated"][f"{part}_collocated_R{R}"] = {
            "per_seed": per_seed,
            "mean": {"rates": _avg_dicts([v["rates"] for v in per_seed.values()])}}
    for part in PARTITIONS:
        for R in G1_ROUNDS:
            for s in SEEDS:
                m = g1.get((part, "distributed", R, s))
                if m is not None:
                    res["distributed_window_auroc"][f"{part}_distributed_R{R}__s{s}"] = window_aurocs(m)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="G1 / G1R5 的探索性读数（不进判定）")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = explore(*load())
    print("== 集中布点：受害 edge（E1–E3 均值）与攻击者 edge E0，前 / 后半程（三个 seed 的均值）")
    for cell, v in res["collocated"].items():
        rt = v["mean"]["rates"]
        for half in ("first", "second"):
            a_, m_ = rt.get(f"client_benign:{half}", {}), rt.get(f"margin_p50:{half}", {})
            print(f"  {cell:<24} {half:<6} ASR r_down {a_.get('victim_r_down')}  Δ_jump {a_.get('victim_d_jump')}  "
                  f"净/周期 {a_.get('victim_net_per_cycle')} | margin r_down {m_.get('victim_r_down')}  "
                  f"Δ_jump {m_.get('victim_d_jump')} | E0 drop {a_.get('e0_drop_at_agg')}  r_up {a_.get('e0_r_up')}")
    print("== 分散布点：norm_w 逐窗口 AUROC（raw / edge）")
    for k, v in res["distributed_window_auroc"].items():
        print(f"  {k:<32} " + "  ".join(f"[{r['eff_from']}] {r['raw']}/{r['edge']}" for r in v["norm_w"]))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[g1_explore] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
