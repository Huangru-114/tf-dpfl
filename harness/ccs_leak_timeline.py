"""
harness/ccs_leak_timeline.py  —  CCS 聚类「放过了哪些攻击者上传、什么时候放过的」+ 训练稳定性读数（FINDINGS F-118，D-125）
**只读已有的 metrics.json，描述性读数，不是判定。**

    python3 harness/ccs_leak_timeline.py <metrics.json> [<metrics.json> …] [--block 25]

每个 run 输出一个 JSON 对象：
  leak（只对开了聚类的 run，即有 `ccs_decisions`）：
    admitted / selected       攻击者上传：被放进聚合的个数 / 被选中参加的个数
    admitted_per_block        每 block 个云轮放进来的攻击者上传数
    rounds_ge2                一轮放进 ≥ 2 个攻击者上传的云轮
    by_attacker               每个攻击者 "放进 / 选中"
    benign_rejection          良性端逐端被剔比例的中位数 / 最大值 / 被剔 ≥ 一半的端数
    history_replay            反事实：客户端过去出现 ≥ min_seen 次且被剔比例 ≥ frac → 这次直接剔除；
                              输出放进来的攻击者上传数与多剔掉的良性上传数（**只改剔除决定，不重新训练**）
  global_asr_every10          共享模型（`rounds[].global_asr`）每 10 轮一个点
  stability                   共享模型精度（`acc_rounds[].em_acc`）第一次 < 0.3 的轮次、< 0.3 的评估点个数 / 总数、末 10 点均值
纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

LOW_EM = 0.3


def _mal(m):
    return set(int(i) for i in ((m.get("run") or {}).get("malicious_ids") or []))


def leak(m: dict, block: int = 25) -> dict | None:
    cd = m.get("ccs_decisions") or []
    if not cd:
        return None
    mal = _mal(m)
    adm, sel = [], []
    by_sel, by_adm, b_sel, b_rej = Counter(), Counter(), Counter(), Counter()
    for d in cd:
        cur, rej = set(int(i) for i in d.get("cur") or []), set(int(i) for i in d.get("rejected") or [])
        a = (cur & mal) - rej
        adm.append(len(a))
        sel.append(len(cur & mal))
        for c in cur & mal:
            by_sel[c] += 1
            by_adm[c] += c in a
        for c in cur - mal:
            b_sel[c] += 1
            b_rej[c] += c in rej
    rates = sorted(b_rej[c] / b_sel[c] for c in b_sel)
    return {
        "admitted": sum(adm), "selected": sum(sel),
        "admitted_per_block": [sum(adm[i:i + block]) for i in range(0, len(adm), block)],
        "rounds_ge2": [i + 1 for i, a in enumerate(adm) if a >= 2],
        "by_attacker": {str(c): f"{by_adm[c]}/{by_sel[c]}" for c in sorted(mal) if by_sel[c]},
        "benign_rejection": {"median": round(rates[len(rates) // 2], 4) if rates else None,
                             "max": round(rates[-1], 4) if rates else None,
                             "n_ge_half": sum(r >= 0.5 for r in rates), "n_benign": len(rates)},
        "history_replay": history_replay(m),
    }


def history_replay(m: dict, min_seen: int = 2, frac: float = 0.5) -> dict:
    """反事实：按 [CCS] 行的顺序逐轮走，客户端此前出现 ≥ min_seen 次且被剔比例 ≥ frac → 这一轮视为剔除。
    「出现」与「被剔」都按原始的剔除决定计（历史不因回放而改变）。"""
    mal = _mal(m)
    hist: dict[int, list[int]] = {}
    mal_admitted = extra_benign = n_benign = 0
    for d in m.get("ccs_decisions") or []:
        cur, rej = set(int(i) for i in d.get("cur") or []), set(int(i) for i in d.get("rejected") or [])
        for c in sorted(cur):
            r, n = hist.get(c, [0, 0])
            flagged = n >= min_seen and r / n >= frac
            if c in mal:
                mal_admitted += (c not in rej) and not flagged
            else:
                n_benign += 1
                extra_benign += (c not in rej) and flagged
            hist[c] = [r + (c in rej), n + 1]
    return {"min_seen": min_seen, "frac": frac, "attacker_admitted": mal_admitted,
            "extra_benign_rejected": extra_benign, "benign_uploads": n_benign}


def stability(m: dict) -> dict:
    em = [(a["round"], a["em_acc"]) for a in m.get("acc_rounds") or [] if a.get("em_acc") is not None]
    low = [r for r, v in em if v < LOW_EM]
    return {"first_below": low[0] if low else None, "n_below": len(low), "n_points": len(em),
            "last10_mean": round(statistics.mean(v for _, v in em[-10:]), 4) if em else None}


def describe(m: dict, block: int = 25) -> dict:
    run = m.get("run") or {}
    g = [(r["round"], r["global_asr"]) for r in m.get("rounds") or [] if r.get("global_asr") is not None]
    return {"run_id": (run.get("provenance") or {}).get("run_id"), "seed": run.get("seed"),
            "ccs_at": run.get("ccs_at"), "ccs_cluster": run.get("ccs_cluster"),
            "leak": leak(m, block),
            "global_asr_every10": [[r, round(v, 4)] for r, v in g if r % 10 == 0],
            "stability": stability(m)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="CCS 聚类漏网时间表 + 训练稳定性（描述性，F-118）")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--block", type=int, default=25)
    a = ap.parse_args(argv)
    for p in a.paths:
        print(json.dumps(describe(json.loads(Path(p).read_text(encoding="utf-8")), a.block), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
