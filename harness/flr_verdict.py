"""
harness/flr_verdict.py  —  floor 验证 pilot（FLR，D-061）的**预注册**判定

    python3 harness/flr_verdict.py [--json out.json]

问题：ρ=0 影子攻击者下的良性 ASR（= floor：从未被投毒的模型被同一个触发器生成器 + ξ 打到多少）
有多大、在 edge 之间差多少 → 决定 G0 要不要按「划分 × edge」逐格测（PLAN §4）。

配对：FLR__g6a__s{seed} ↔ G6__a__s{seed}。两者的配置只差 backdoor.poison_ratio（0 vs 0.2）；
划分、恶意端 id、评估用的固定攻击者都只由 seed 决定 → 同 seed 下相同（下面会核对恶意端 id）。

量：逐 edge 的 floor_e = `per_edge_rounds[*][e].client_benign`（主列：fresh-PM 良性 ASR）末 10 个评估点均值。
E0 的 10 个影子攻击者不计入（client_benign 只数良性端）。

判定（阈值 ⚠ 待用户确认，D-061；改阈值要在 DECISIONS 留一条，不能悄悄改这里）：
  negligible      3 seed × 4 edge 的 floor 全部 ≤ LOW
                  → G0 取消逐划分 floor；3-B / 3.3 用原始 ASR，注明「floor ≤ LOW（FLR 实测）」
  not_negligible  任一 floor ≥ HIGH，或任一 seed 的 |floor_E0 − mean(floor_E1..E3)| ≥ GAP
                  → G0 按原计划逐划分测
  user_decides    其余
  insufficient    有效 seed 少于 3；missing / invalid 另报（invalid 用 D-044 的闸 + 「不是 ρ=0」）
另报（不参与判定）：总体 local_benign_asr 与 global_asr 的 floor；与 G6(a) 的 excess = 攻击 − floor 及
floor / 攻击 比值；floor 的首 10 点 vs 末 10 点（floor 是否随训练上升）。G6(a) 缺时 excess 记 None。

纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_a4 import invalid_reasons       # noqa: E402
from runs_table import last_k_mean         # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
N_EDGES = 4
LOW, HIGH, GAP = 0.05, 0.10, 0.05          # ⚠ 待用户确认（D-061）
EDGE_KEY = "client_benign"


def edge_series(m: dict, edge_id: int, key: str = EDGE_KEY) -> list:
    """per_edge_rounds（键是云轮号的字符串）→ 按轮排序的该 edge 的 key 序列（None 保留）。"""
    per = m.get("per_edge_rounds") or {}
    out = []
    for rnd in sorted(per, key=int):
        row = next((e for e in per[rnd] or [] if int(e.get("edge_id", -1)) == edge_id), None)
        out.append(None if row is None else row.get(key))
    return out


def _mean(vals, first=False):
    vals = [v for v in vals if v is not None]
    if first:
        vals = vals[:10]
    return last_k_mean(vals)[0]


def floor_reasons(m: dict) -> list:
    """FLR 的 run 能不能当 floor：D-044 的闸 + 真的是 ρ=0 + 4 个 edge。"""
    reasons = list(invalid_reasons(m))
    run = m.get("run") or {}
    if run.get("poison_ratio") not in (0, 0.0):
        reasons.append(f"run.poison_ratio={run.get('poison_ratio')}（不是 ρ=0 的影子攻击者）")
    if run.get("n_edges") not in (None, N_EDGES):
        reasons.append(f"run.n_edges={run.get('n_edges')} ≠ {N_EDGES}")
    if not m.get("per_edge_rounds"):
        reasons.append("没有 per_edge_rounds（逐 edge ASR 没解析出来）")
    return reasons


def judge_seed(flr: dict | None, atk: dict | None, seed: int) -> dict:
    if flr is None:
        return {"seed": seed, "verdict": "missing"}
    bad = floor_reasons(flr)
    if bad:
        return {"seed": seed, "verdict": "invalid", "reasons": bad}
    floors = [_mean(edge_series(flr, e)) for e in range(N_EDGES)]
    if any(f is None for f in floors):
        return {"seed": seed, "verdict": "invalid",
                "reasons": [f"有 edge 没有 {EDGE_KEY} 的值：{floors}"]}
    victims = floors[1:]
    gap = floors[0] - sum(victims) / len(victims)
    out = {"seed": seed, "verdict": "ok",
           "floor_edge": [round(f, 4) for f in floors],
           "gap_e0_vs_victims": round(gap, 4),
           "floor_edge_first10": [None if (v := _mean(edge_series(flr, e), first=True)) is None
                                  else round(v, 4) for e in range(N_EDGES)],
           "floor_local_benign": _r(_mean([r.get("local_benign_asr") for r in flr.get("rounds") or []])),
           "floor_global": _r(_mean([r.get("global_asr") for r in flr.get("rounds") or []]))}
    # ── 与攻击臂 G6(a) 配对（只报告，不参与判定）──────────────────────────
    if atk is None or invalid_reasons(atk):
        out["attack"] = None if atk is None else {"invalid": invalid_reasons(atk)}
        return out
    same_mal = sorted((flr.get("run") or {}).get("malicious_ids") or []) == \
        sorted((atk.get("run") or {}).get("malicious_ids") or [])
    atk_edge = [_mean(edge_series(atk, e)) for e in range(N_EDGES)]
    out["attack"] = {
        "same_malicious_ids": same_mal,
        "attack_edge": [_r(a) for a in atk_edge],
        "excess_edge": [None if a is None else round(a - f, 4) for a, f in zip(atk_edge, floors)],
        "floor_over_attack": [None if not a else round(f / a, 4) for a, f in zip(atk_edge, floors)],
    }
    return out


def _r(v):
    return None if v is None else round(v, 4)


def judge(pairs: dict, low: float = LOW, high: float = HIGH, gap: float = GAP) -> dict:
    """pairs：{seed: (flr_metrics, g6a_metrics)}。"""
    per = [judge_seed(*pairs.get(s, (None, None)), s) for s in SEEDS]
    ok = [p for p in per if p["verdict"] == "ok"]
    vs = {p["verdict"] for p in per}
    if "invalid" in vs:
        overall = "invalid"
    elif len(ok) < len(SEEDS):
        overall = "insufficient" if ok else "missing"
    else:
        allf = [f for p in ok for f in p["floor_edge"]]
        if max(allf) >= high or any(abs(p["gap_e0_vs_victims"]) >= gap for p in ok):
            overall = "not_negligible"
        elif max(allf) <= low:
            overall = "negligible"
        else:
            overall = "user_decides"
    text = {
        "negligible": f"floor 可忽略（全部 ≤ {low}）→ G0 取消逐划分 floor；3-B / 3.3 用原始 ASR 并注明上界",
        "not_negligible": f"floor 不可忽略（某 edge ≥ {high}，或 E0 与受害 edge 相差 ≥ {gap}）→ G0 按原计划逐划分测",
        "user_decides": f"floor 介于 {low} 与 {high} 之间 → 用户定 G0 规模",
        "insufficient": "有效 seed 不足 3 个（不报方向）",
        "missing": "FLR 结果还没回来",
        "invalid": "有 run 无效（D-044 的闸 / 不是 ρ=0）",
    }[overall]
    return {"overall": overall, "summary": text, "thresholds": {"low": low, "high": high, "gap": gap},
            "per_seed": per, "preregistered": True}


def load(results_dir: Path = RESULTS) -> dict:
    def _l(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    return {s: (_l(results_dir / "FLR" / f"FLR__g6a__s{s}.metrics.json"),
                _l(results_dir / "G6" / f"G6__a__s{s}.metrics.json")) for s in SEEDS}


def main(argv=None):
    ap = argparse.ArgumentParser(description="FLR（floor 验证 pilot）的预注册判定，D-061")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = judge(load())
    for p in res["per_seed"]:
        print(f"  seed {p['seed']}: {p['verdict']:<8} "
              f"{json.dumps({k: v for k, v in p.items() if k not in ('seed', 'verdict')}, ensure_ascii=False)}")
    print(res["summary"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
