"""
harness/edge_ref_probe.py  —  方向 2 第一段的判定（FINDINGS N-013，D-131）：参照打分能不能分开攻击者与良性端

N-013 已经用户确认（2026-11-17）；本脚本先 commit，再交 ERP / ERPB 的作业（同 D-088 的顺序）。

    python3 harness/edge_ref_probe.py \\
        --attack <ERP 的 metrics.json …（8 个：{main, homo} × {t0, t1} × {collocated, distributed}，s42）> \\
        --dev    <ERPB 的 metrics.json …（4 个：{main, homo} × {t0, t1}，s41，投毒比例 0）> \\
        --out experiments/defense/edge-native/analysis/edge_ref_probe.json

规则（**按划分分别执行**；划分 = run.partition_profile：None → main（客户端 Dirichlet），fixed → homo）：
  1. 每个候选分数 s、每个有攻击的格子 c（三层个性化 {t0, t1} × 布点 {集中, 分散}）：
     AUROC(s, c) = 攻击者上传 vs 良性上传（全部边缘、全部边缘轮池化；身份取自 run.malicious_ids，**分数本身不知道身份**）。
  2. 选分数 s* = argmax_s min_c AUROC(s, c)（四格取最小值、再取最大；并列取候选表里靠前的）。
  3. 阈值 τ_t = 同划分、同三层个性化 t 的开发 run 里 s* 的 90% 分位点（全部上传，不用身份）。
  4. 判定：min_c AUROC(s*, c) ≥ 0.75 → feasible；否则 infeasible（该划分下「参照过滤不可行」，第二段不做）。
另报（不进判定）：全部候选 × 全部格子的 AUROC 与 FPR 5% 处的 TPR；按 τ_t 回放的攻击者漏网率 / 良性误剔率（逐格）；
  假客户端（零分布）在 τ_t 之上的比例；缓存池多数原则（ccs_rej）的命中率 / 误伤率；两个划分之间的对比。
闸（任一不满足 → 该划分 invalid）：每个 run 有效（exit_code 0、client_failures 为空）、run.edge_ref 为真、
  有攻击的格子 poison_ratio > 0、开发 run poison_ratio == 0、每格恰好一个 run、分数表非空、攻击者至少上传过一次。
纯标准库 + numpy（+ fedavg/analysis/functional_score），不 import TF。
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
from analysis.functional_score import auroc, tpr_at_fpr   # noqa: E402
from pilot_a4 import invalid_reasons                      # noqa: E402

AUROC_MIN, DEV_Q = 0.75, 0.90
# 候选分数：都是「越大越可疑」。顺序 = 并列时的优先级（预注册，N-013）。
CANDIDATES = ("kl_all", "kl_stem", "kl_stage1", "kl_stage2", "kl_stage3", "kl_stage4",
              "mix_res", "mix_res_base", "ccs_cos_med")
NULL_SCORES = ("kl_all", "kl_stage4", "mix_res", "mix_res_base")
TIERS, PLACEMENTS = ("t0", "t1"), ("collocated", "distributed")


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def partition_of(run: dict) -> str:
    return "homo" if run.get("partition_profile") == "fixed" else "main"


def tier_of(run: dict) -> str:
    return "t1" if int(run.get("edge_shared_blocks") or 0) > 0 else "t0"


def placement_of(run: dict) -> str | None:
    mpe = run.get("malicious_per_edge")
    if not mpe:
        return None
    nz = [x for x in mpe if x]
    return "collocated" if len(nz) == 1 else "distributed"


def table(m: dict) -> list:
    """edge_ref_scores → [{列: 值}]。"""
    t = m.get("edge_ref_scores") or {}
    cols = t.get("columns") or []
    return [dict(zip(cols, row)) for row in t.get("rows") or []]


def split_scores(m: dict, s: str):
    """(攻击者分数, 良性分数)；None（无定义，如缓存池里只有它自己）丢掉。"""
    mal = set(int(i) for i in (m.get("run") or {}).get("malicious_ids") or [])
    pos, neg = [], []
    for r in table(m):
        v = r.get(s)
        if v is None:
            continue
        (pos if int(r["cid"]) in mal else neg).append(float(v))
    return pos, neg


def all_scores(m: dict, s: str) -> list:
    return [float(r[s]) for r in table(m) if r.get(s) is not None]


def quantile(x, q: float):
    return None if not x else float(np.quantile(np.asarray(x, np.float64), q))


def _gate(m: dict, *, dev: bool) -> list:
    run = m.get("run") or {}
    reasons = list(invalid_reasons(m))
    if not run.get("edge_ref"):
        reasons.append("run.edge_ref 不为真（[设定14] 缺行或没开）")
    pr = run.get("poison_ratio")
    if dev and pr not in (0, 0.0):
        reasons.append(f"开发 run 的 poison_ratio = {pr!r}（应为 0）")
    if not dev and not (pr or 0) > 0:
        reasons.append(f"有攻击的 run poison_ratio = {pr!r}")
    if not table(m):
        reasons.append("edge_ref_scores 为空")
    if not dev:
        mal = set(int(i) for i in run.get("malicious_ids") or [])
        if not any(int(r["cid"]) in mal for r in table(m)):
            reasons.append("攻击者一次也没上传过（分数表里没有恶意端）")
    return reasons


def judge_partition(attack: dict, dev: dict) -> dict:
    """attack：{(tier, placement): metrics}；dev：{tier: metrics}（同一划分）。"""
    want_a = [(t, p) for t in TIERS for p in PLACEMENTS]
    reasons = []
    for k in want_a:
        if k not in attack:
            reasons.append(f"缺有攻击的格子 {k}")
    for t in TIERS:
        if t not in dev:
            reasons.append(f"缺开发 run {t}")
    for k, m in attack.items():
        reasons += [f"{k}: {r}" for r in _gate(m, dev=False)]
    for t, m in dev.items():
        reasons += [f"dev-{t}: {r}" for r in _gate(m, dev=True)]
    if reasons:
        return {"verdict": "invalid", "reasons": reasons}

    grid = {}
    for s in CANDIDATES:
        grid[s] = {}
        for (t, p) in want_a:
            pos, neg = split_scores(attack[(t, p)], s)
            grid[s][f"{t}-{p}"] = {"auroc": _r(auroc(pos, neg)), "tpr_at_fpr05": _r(tpr_at_fpr(pos, neg, 0.05)),
                                   "n_pos": len(pos), "n_neg": len(neg)}
    # 某格 AUROC 无定义（那一格这个分数全是 None）→ 这个分数的「四格最小值」无定义，不参与选择
    mins = {}
    for s in CANDIDATES:
        vals = [v["auroc"] for v in grid[s].values()]
        mins[s] = None if any(v is None for v in vals) else min(vals)
    best = max(CANDIDATES, key=lambda s: (-1.0 if mins[s] is None else mins[s], -CANDIDATES.index(s)))
    best_min = mins[best]
    tau = {t: _r(quantile(all_scores(dev[t], best), DEV_Q), 6) for t in TIERS}

    replay = {}
    for (t, p) in want_a:
        pos, neg = split_scores(attack[(t, p)], best)
        th = tau[t]
        replay[f"{t}-{p}"] = {
            "attacker_pass_rate": _r(np.mean([v <= th for v in pos])) if pos and th is not None else None,
            "benign_reject_rate": _r(np.mean([v > th for v in neg])) if neg and th is not None else None,
        }
    null_above = {}
    if best in NULL_SCORES:
        for t in TIERS:
            vals = []
            for (tt, p) in want_a:
                if tt != t:
                    continue
                nt = attack[(tt, p)].get("edge_ref_null") or {}
                cols = nt.get("columns") or []
                for row in nt.get("rows") or []:
                    vals += [float(v) for v in (dict(zip(cols, row)).get(best) or []) if v is not None]
            null_above[t] = _r(np.mean([v > tau[t] for v in vals])) if vals and tau[t] is not None else None
    ccs = {}
    for (t, p) in want_a:
        pos, neg = split_scores(attack[(t, p)], "ccs_rej")
        ccs[f"{t}-{p}"] = {"hit_rate": _r(np.mean(pos)) if pos else None, "false_reject_rate": _r(np.mean(neg)) if neg else None}

    return {"verdict": ("feasible" if best_min is not None and best_min >= AUROC_MIN else "infeasible"),
            "selected_score": best, "selected_min_auroc": best_min, "threshold_dev_q90": tau,
            "auroc_grid": grid, "min_auroc_by_score": {s: mins[s] for s in CANDIDATES},
            "descriptive": {"replay_at_threshold": replay, "null_fraction_above_threshold": null_above,
                            "ccs_cache_majority": ccs}}


def assemble(attack_runs: list, dev_runs: list) -> dict:
    by_part = {"main": ({}, {}), "homo": ({}, {})}
    dup = []
    for m in attack_runs:
        run = m.get("run") or {}
        a, _ = by_part[partition_of(run)]
        k = (tier_of(run), placement_of(run))
        if k in a:
            dup.append(f"{partition_of(run)} {k}")
        a[k] = m
    for m in dev_runs:
        run = m.get("run") or {}
        _, d = by_part[partition_of(run)]
        k = tier_of(run)
        if k in d:
            dup.append(f"{partition_of(run)} dev {k}")
        d[k] = m
    out = {"rule": "N-013", "auroc_min": AUROC_MIN, "dev_quantile": DEV_Q, "candidates": list(CANDIDATES)}
    for part, (a, d) in by_part.items():
        res = judge_partition(a, d)
        if dup:
            res = {"verdict": "invalid", "reasons": [f"同一格有多个 run：{dup}"]}
        out[part] = res
    # 两个划分之间（描述，不进判定）
    if all(out[p].get("verdict") in ("feasible", "infeasible") for p in ("main", "homo")):
        out["contrast"] = {s: {"main": out["main"]["min_auroc_by_score"][s], "homo": out["homo"]["min_auroc_by_score"][s]}
                           for s in CANDIDATES}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="方向 2 第一段的判定（N-013）")
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    load = lambda ps: [json.loads(Path(p).read_text(encoding="utf-8")) for p in ps]   # noqa: E731
    res = assemble(load(a.attack), load(a.dev))
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    for part in ("main", "homo"):
        r = res[part]
        print(f"{part}: {r['verdict']}  s*={r.get('selected_score')}  min AUROC={r.get('selected_min_auroc')}  "
              f"τ={r.get('threshold_dev_q90')}" + (f"  原因：{r.get('reasons')}" if r["verdict"] == "invalid" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
