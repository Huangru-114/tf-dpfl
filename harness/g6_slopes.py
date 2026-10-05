"""
harness/g6_slopes.py  —  检查 3：3-E 是「阻断」还是「延迟」（**探索性，看过数据之后**；不改 G6 的预注册判定 F-077）

    python3 harness/g6_slopes.py [--json out.json]

问题：G6 的 b / c 臂把受害 edge 的 ASR 压低了（`blocks`），但如果末段 ASR 仍在稳定上升，那只是延迟。
量（每个 run；受害 edge E1–E3 的 fresh-PM 良性 ASR `per_edge_rounds[g][e].client_benign` 的三 edge 均值）：
  slope_50 / slope_100  最后 50 / 100 个有效轮内（G6：R5、60 云轮 → 第 51–60 / 41–60 云轮，10 / 20 个点）
                        对有效轮的 OLS 斜率（每有效轮）；se_* = 该 run 内的 OLS 标准误（残差 σ / √Sxx）
  end                   末 10 个评估点均值（= g6_verdict 的 victim_asr）
  end_minus_floor       end − 同 seed FLR 受害 edge floor（FLR 第 51–60 云轮均值；decay_verdict 的同一个量）
跨 seed：均值 + bootstrap 95% CI（`verdicts.bootstrap_mean_ci`；3 seed 时 CI = [最小, 最大] seed）；
臂 − a 臂的斜率差按 seed 配对。

纯标准库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decay_verdict import edge_by_round, group_mean, ols_slope   # noqa: E402
from verdicts import bootstrap_mean_ci                            # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
ARMS = ("a", "b", "c")
VICTIMS = (1, 2, 3)
FLOOR_WIN = (51, 60)
WINDOWS_EFF = (50, 100)


def victim_series(m: dict) -> dict:
    """{云轮: E1–E3 均值}；任一 edge 缺这一轮 → 该轮不计。"""
    per = [edge_by_round(m, e) for e in VICTIMS]
    common = set(per[0]).intersection(*per[1:])
    return {g: sum(p[g] for p in per) / len(per) for g in sorted(common)}


def slope_se(pts):
    """OLS 斜率的标准误（残差 σ / √Sxx，自由度 n − 2）；点数 < 3 → None。"""
    if len(pts) < 3:
        return None
    b = ols_slope(pts)
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    rss = sum((y - (my + b * (x - mx))) ** 2 for x, y in pts)
    return (rss / (len(pts) - 2) / sxx) ** 0.5


def slope_last(series: dict, R: int, n_rounds: int, eff_window: int):
    """最后 eff_window 个有效轮（云轮 g 满足 g·R > n_rounds·R − eff_window）的 OLS 斜率（每有效轮）、点数、标准误。"""
    lo = n_rounds * R - eff_window
    pts = [(g * R, v) for g, v in series.items() if g * R > lo]
    return ols_slope(pts), len(pts), slope_se(pts)


def run_quantities(m: dict, flr: dict | None) -> dict:
    run = m["run"]
    R, n = int(run["edge_rounds"]), int(run["n_rounds"])
    s = victim_series(m)
    last = [s[g] for g in sorted(s)][-10:]
    end = sum(last) / len(last) if last else None
    floor = group_mean(flr, VICTIMS, FLOOR_WIN) if flr else None
    out = {"R": R, "n_rounds": n, "end": end, "floor": floor,
           "end_minus_floor": None if end is None or floor is None else end - floor}
    for w in WINDOWS_EFF:
        out[f"slope_{w}"], out[f"n_{w}"], out[f"se_{w}"] = slope_last(s, R, n, w)
    return out


def _ci(vals):
    vals = [v for v in vals if v is not None]
    m, lo, hi = bootstrap_mean_ci(vals)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}


def analyse(runs: dict, flrs: dict) -> dict:
    """runs: {(arm, seed): metrics}；flrs: {seed: metrics}。"""
    per = {arm: {s: run_quantities(runs[(arm, s)], flrs.get(s)) for s in SEEDS if (arm, s) in runs}
           for arm in ARMS}
    arms = {}
    for arm in ARMS:
        q = per[arm]
        d = {f"slope_{w}": _ci([q[s][f"slope_{w}"] for s in q]) for w in WINDOWS_EFF}
        d["end"] = _ci([q[s]["end"] for s in q])
        d["end_minus_floor"] = _ci([q[s]["end_minus_floor"] for s in q])
        if arm != "a":
            for w in WINDOWS_EFF:
                k = f"slope_{w}"
                d[f"{k}_minus_a"] = _ci([q[s][k] - per["a"][s][k] for s in q
                                         if s in per["a"] and q[s][k] is not None and per["a"][s][k] is not None])
        arms[arm] = d
    return {"purpose": "检查 3（探索性，看过数据之后）：G6 受害 edge ASR 末段斜率 —— 阻断还是延迟",
            "per_seed": {arm: {str(s): v for s, v in per[arm].items()} for arm in ARMS}, "arms": arms}


def load(results: Path = RESULTS):
    runs, flrs = {}, {}
    for s in SEEDS:
        for arm in ARMS:
            p = results / f"G6/G6__{arm}__s{s}.metrics.json"
            if p.exists():
                runs[(arm, s)] = json.loads(p.read_text(encoding="utf-8"))
        p = results / f"FLR/FLR__g6a__s{s}.metrics.json"
        if p.exists():
            flrs[s] = json.loads(p.read_text(encoding="utf-8"))
    return runs, flrs


def _f(v, nd=5):
    return "None" if v is None else f"{v:+.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = analyse(*load(Path(a.results)))
    for arm in ARMS:
        for s, q in res["per_seed"][arm].items():
            print(f"{arm} s{s}  slope50 {_f(q['slope_50'])} ±{q['se_50']:.5f} (n={q['n_50']})  "
                  f"slope100 {_f(q['slope_100'])} ±{q['se_100']:.5f} (n={q['n_100']})  end {q['end']:.3f}  floor {q['floor']:.3f}  end−floor {_f(q['end_minus_floor'], 3)}")
    for arm, d in res["arms"].items():
        print(f"[{arm}] " + "  ".join(f"{k} {_f(v['mean'])} [{_f(v['lo'])}, {_f(v['hi'])}]" for k, v in d.items()))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[g6_slopes] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
