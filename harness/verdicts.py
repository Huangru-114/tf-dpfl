"""
harness/verdicts.py  —  执行 PLAN.md §3 的预注册判定

    python3 harness/verdicts.py <runs.csv> --out verdicts.json

规矩（PLAN §3，数据回来之前定下，改阈值要在 DECISIONS.md 留记录）：
  · 终值一律是末 10 点均值（runs_table.py 已算好）；
  · 「显著」= 按 seed **配对**的 bootstrap 95% CI 不含零（10000 次重采样，固定种子）；
  · seed 数低于要求 → 判定为 `insufficient`，**不报方向**；
  · 删失（跑完也没越过阈值）单独计数，不当数值参与平均。

现在实现的是框架 + 3-A（flat vs HFL 的 T50 比值）。其余子实验的判定随各自的功能会话
（S3–S8）加进来，用同一套 bootstrap 与 seed 门槛。

纯标准库，不 import numpy / TF。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

N_BOOT = 10000
BOOT_SEED = 20260924
ALPHA = 0.05

# 3-A 的预注册参数（PLAN §3；等效区间 ±10% 标了「⚠待确认」）
T50_METRIC = "local_benign_asr"
T50_THETA = 0.5
T50_MIN_SEEDS = 5
T50_EQUIV_MARGIN = 0.10


# ══════════════════════════════════════════════════════════════════════════
# 统计原语
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_mean_ci(values, *, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """均值的百分位 bootstrap CI。n<2 → (mean, None, None)：一个点给不出区间。"""
    vals = [float(v) for v in values]
    if not vals:
        return None, None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None, None
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_boot))
    lo = boots[int(math.floor(alpha / 2 * n_boot))]
    hi = boots[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return mean, lo, hi


def _f(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════
# 3-A：T50(HFL) / T50(flat)，按 seed 配对
# ══════════════════════════════════════════════════════════════════════════

def per_seed_t(rows, metric=T50_METRIC, theta=T50_THETA) -> dict:
    """{seed: t 或 'censored'}。同一 seed 的多份 replicate 取几何平均（它们是同一格的重复）。
    任一 replicate 删失 → 这个 seed 记删失（不能把删失当成一个数去平均）。"""
    col, state = f"t{theta}_{metric}", f"t{theta}_{metric}_state"
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    out = {}
    for seed, rs in by_seed.items():
        ts = [_f(r.get(col)) for r in rs]
        if any(r.get(state) != "crossed" or t is None or t <= 0 for r, t in zip(rs, ts)):
            out[seed] = "censored"
        else:
            out[seed] = math.exp(sum(math.log(t) for t in ts) / len(ts))
    return out


def verdict_t50_ratio(hfl_rows, flat_rows, *, min_seeds=T50_MIN_SEEDS,
                      margin=T50_EQUIV_MARGIN, metric=T50_METRIC, theta=T50_THETA) -> dict:
    """3-A 判定。返回 {verdict, n_paired, n_censored, log_ratio{mean,lo,hi}, ratio{…}, per_seed}。

    verdict ∈
      structural_delay  log r 的 CI 全 > 0（HFL 更慢，而且参与量已匹配）
      hfl_faster        log r 的 CI 全 < 0（原计划没预料到的方向，P1 数据提示可能出现）
      approx_equal      CI 落在 [log(1-margin), log(1+margin)] 内
      inconclusive      其余
      insufficient      配对的 seed 数 < min_seeds（不报方向）
    """
    th, tf = per_seed_t(hfl_rows, metric, theta), per_seed_t(flat_rows, metric, theta)
    paired, censored, per_seed = [], 0, {}
    for seed in sorted(set(th) & set(tf), key=str):
        a, b = th[seed], tf[seed]
        if a == "censored" or b == "censored":
            censored += 1
            per_seed[str(seed)] = {"hfl": a, "flat": b, "log_ratio": None}
            continue
        lr = math.log(a / b)
        paired.append(lr)
        per_seed[str(seed)] = {"hfl": a, "flat": b, "log_ratio": lr}
    out = {"metric": metric, "theta": theta, "n_paired": len(paired),
           "n_censored": censored, "min_seeds": min_seeds, "margin": margin,
           "per_seed": per_seed}
    if len(paired) < min_seeds:
        out.update(verdict="insufficient", log_ratio=None, ratio=None)
        return out
    mean, lo, hi = bootstrap_mean_ci(paired)
    out["log_ratio"] = {"mean": mean, "lo": lo, "hi": hi}
    out["ratio"] = {"mean": math.exp(mean), "lo": math.exp(lo), "hi": math.exp(hi)}
    if lo > 0:
        out["verdict"] = "structural_delay"
    elif hi < 0:
        out["verdict"] = "hfl_faster"
    elif math.log(1 - margin) <= lo and hi <= math.log(1 + margin):
        out["verdict"] = "approx_equal"
    else:
        out["verdict"] = "inconclusive"
    return out


# ══════════════════════════════════════════════════════════════════════════
# 从 runs.csv 组织 3-A 的比较
# ══════════════════════════════════════════════════════════════════════════

# 3-A 要比较的是「拓扑」本身；其余因素必须相同，否则不算同一组对照。
_SAME_EXCEPT_TOPOLOGY = ("method", "attack", "defense", "n_clients", "client_fraction",
                         "poison_ratio", "local_epochs", "plocal_epochs", "attack_stop_round",
                         "protocol")


def _n_malicious_total(r):
    """恶意端总数（从 malicious_per_edge 算）：拓扑可以变，攻击者数量不能变。
    否则「2% 攻击者的 2edge」会被拿去和「10% 攻击者的 flat」比。"""
    try:
        return sum(json.loads(r.get("malicious_per_edge") or "null") or [])
    except (TypeError, ValueError):
        return None


def _ctx(r):
    return (*(r.get(k) for k in _SAME_EXCEPT_TOPOLOGY), _n_malicious_total(r))


def plan_3a(rows) -> list:
    """每个 HFL 拓扑 (n_edges, edge_rounds, malicious_per_edge) 对它同上下文的 flat 比一次。"""
    flats, hfls = {}, {}
    for r in rows:
        if str(r.get("exit_code")) not in ("0", "0.0"):
            continue
        key = _ctx(r)
        if str(r.get("n_edges")) == "1":
            flats.setdefault(key, []).append(r)
        else:
            topo = (r.get("n_edges"), r.get("edge_rounds"), r.get("malicious_per_edge"))
            hfls.setdefault((key, topo), []).append(r)
    out = []
    for (key, topo), hrows in sorted(hfls.items(), key=lambda kv: str(kv[0])):
        frows = flats.get(key)
        if not frows:
            continue
        v = verdict_t50_ratio(hrows, frows)
        v["hfl"] = {"n_edges": topo[0], "edge_rounds": topo[1], "malicious_per_edge": topo[2]}
        out.append(v)
    return out


def load_runs_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(description="执行 PLAN §3 的预注册判定")
    ap.add_argument("runs_csv")
    ap.add_argument("--out", help="verdicts.json 输出路径")
    args = ap.parse_args(argv)
    rows = load_runs_csv(args.runs_csv)
    protocols = sorted({r.get("protocol") for r in rows})
    report = {"runs_csv": str(args.runs_csv), "protocols": protocols,
              "rules": {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "alpha": ALPHA,
                        "t50_metric": T50_METRIC, "t50_theta": T50_THETA,
                        "t50_min_seeds": T50_MIN_SEEDS, "t50_equiv_margin": T50_EQUIV_MARGIN},
              "3A": plan_3a(rows)}
    for v in report["3A"]:
        h = v["hfl"]
        r = v["ratio"]
        rtxt = "—" if r is None else f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
        print(f"[3-A] n_edges={h['n_edges']} R={h['edge_rounds']} mpe={h['malicious_per_edge']}: "
              f"{v['verdict']}  (paired={v['n_paired']}, censored={v['n_censored']}, "
              f"ratio={rtxt})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
