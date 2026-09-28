"""
harness/decay_verdict.py  —  G8（3-C 攻击停止版 = 1B-2 的 HFL 复现，D-075）的**预注册**判定

    python3 harness/decay_verdict.py [--json out.json]

问题：P2 的 HFL（4 edge、攻击者全在 E0）下，攻击者在第 30 个 cloud 轮（有效轮 150）之后停手，
后门是衰减到 floor，还是像 1B-2（flat、官方代码：200 轮干净训练后仍约 0.45）那样停在平台上？
防御含义：衰减到 floor → 「检测并踢出攻击者」就够；平台高于 floor → 必须主动清除（上传前加固）。

配对（同 seed；划分、恶意端 id、评估攻击者都只由 seed 决定）：
  G8__a__s{seed}   = G6(a) + attack_stop_round 31 + n_rounds 70（+ 两个存盘开关）
  FLR__g6a__s{seed} = G6(a) × ρ=0 → floor（第 51–60 轮 = 它的末 10 点）
  G6__a__s{seed}   = 攻击不停的参照（只报告）

量（受害 edge = E1–E3 的逐 edge 良性 ASR 主列 `per_edge_rounds[*][e].client_benign`，逐 edge 取窗口均值再平均）：
  A_T    第 28–30 轮（攻击者停手前的水平）
  A_main 第 51–60 轮 —— **主判定窗口**：与 FLR 的 floor 窗口是同一批有效轮（同 lr；S9 设计复核：
         lr 按 0.992^有效轮 衰减，拿第 61–70 轮去比 FLR 的 51–60 轮会混进 lr 差）
  floor  FLR 的第 51–60 轮
  excess = A_main − floor
另报（不参与判定）：A_end（第 61–70 轮）与 excess_end、retention = (A_main − floor) / (A_T − floor)、
E0 的同样几个量、良性端池化 margin 中位数在第 31–70 轮的 OLS 斜率（schema 7）、G6(a) 同窗口的值。

判定（阈值沿用 FLR 已确认的 0.05 / 0.10，D-068；改阈值要在 DECISIONS 留一条）：
  persists         3 个 seed 的受害 edge excess 全部 ≥ HIGH
  decays_to_floor  全部 ≤ LOW
  user_decides     其余
  insufficient / missing / invalid 同 flr_verdict（有效性闸 D-044 + 停止轮真的是 31）
解读注意：lr 在有效轮 150 → 300 从 ×0.30 降到 ×0.09（1B-2 是常数 lr）；「衰减慢」可能部分来自 lr 小。

纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from flr_verdict import HIGH, LOW, N_EDGES, SEEDS, floor_reasons   # noqa: E402
from pilot_a4 import invalid_reasons                                # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
STOP_ROUND = 31                    # 第 1–30 轮投毒（round < stop 才投毒，轮号从 1 起）
WIN_T = (28, 30)                   # 停手前
WIN_MAIN = (51, 60)                # 主判定：与 FLR 的 floor 窗口同一批有效轮
WIN_END = (61, 70)                 # 只报告
VICTIMS = (1, 2, 3)
EDGE_KEY = "client_benign"


def edge_by_round(m: dict, edge_id: int, key: str = EDGE_KEY) -> dict:
    out = {}
    for rnd, rows in (m.get("per_edge_rounds") or {}).items():
        row = next((e for e in rows or [] if int(e.get("edge_id", -1)) == edge_id), None)
        if row is not None and row.get(key) is not None:
            out[int(rnd)] = row[key]
    return out


def window_mean(m: dict, edge_id: int, win) -> float | None:
    s = edge_by_round(m, edge_id)
    vals = [s[r] for r in range(win[0], win[1] + 1) if r in s]
    return sum(vals) / len(vals) if vals else None


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def group_mean(m: dict, edges, win):
    """若干 edge 各自取窗口均值再平均；任一 edge 缺 → None（不拿残缺的组冒充全体）。"""
    per = [window_mean(m, e, win) for e in edges]
    return None if any(v is None for v in per) else sum(per) / len(per)


def ols_slope(pairs) -> float | None:
    pts = [(x, y) for x, y in pairs if y is not None]
    if len(pts) < 3:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return None if sxx == 0 else sum((x - mx) * (y - my) for x, y in pts) / sxx


def decay_reasons(m: dict) -> list:
    reasons = list(invalid_reasons(m))
    run = m.get("run") or {}
    if run.get("attack_stop_round") != STOP_ROUND:
        reasons.append(f"run.attack_stop_round={run.get('attack_stop_round')} ≠ {STOP_ROUND}"
                       f"（停止轮没生效 = 陷阱 #7 同类）")
    if run.get("n_edges") not in (None, N_EDGES):
        reasons.append(f"run.n_edges={run.get('n_edges')} ≠ {N_EDGES}")
    if not m.get("per_edge_rounds"):
        reasons.append("没有 per_edge_rounds")
    elif max(int(r) for r in m["per_edge_rounds"]) < WIN_MAIN[1]:
        reasons.append(f"只跑到第 {max(int(r) for r in m['per_edge_rounds'])} 轮，"
                       f"不到主判定窗口末（{WIN_MAIN[1]}）")
    return reasons


def _r(v):
    return None if v is None else round(v, 4)


def judge_seed(g8, flr, g6a, seed: int) -> dict:
    if g8 is None or flr is None:
        return {"seed": seed, "verdict": "missing",
                "missing": [n for n, x in (("G8", g8), ("FLR", flr)) if x is None]}
    bad = decay_reasons(g8) + [f"FLR：{r}" for r in floor_reasons(flr)]
    if bad:
        return {"seed": seed, "verdict": "invalid", "reasons": bad}
    same_mal = sorted((g8.get("run") or {}).get("malicious_ids") or []) == \
        sorted((flr.get("run") or {}).get("malicious_ids") or [])
    out = {"seed": seed, "verdict": "ok", "same_malicious_ids": same_mal}
    for name, edges in (("victims", VICTIMS), ("e0", (0,))):
        a_t = group_mean(g8, edges, WIN_T)
        a_main = group_mean(g8, edges, WIN_MAIN)
        a_end = group_mean(g8, edges, WIN_END)
        floor = group_mean(flr, edges, WIN_MAIN)
        exc = None if a_main is None or floor is None else a_main - floor
        exc_end = None if a_end is None or floor is None else a_end - floor
        ret = (None if exc is None or a_t is None or (a_t - floor) <= 0
               else exc / (a_t - floor))
        out[name] = {"A_T": _r(a_t), "A_main": _r(a_main), "A_end": _r(a_end),
                     "floor": _r(floor), "excess": _r(exc), "excess_end": _r(exc_end),
                     "retention": _r(ret),
                     "no_stop_ref": _r(group_mean(g6a, edges, WIN_MAIN)) if g6a else None}
    rounds = g8.get("rounds") or []
    out["margin_p50_slope_31_70"] = _r(ols_slope(
        (r["round"], r.get("margin_p50")) for r in rounds if STOP_ROUND <= r["round"] <= WIN_END[1]))
    if out["victims"]["excess"] is None:
        out["verdict"] = "invalid"
        out["reasons"] = ["受害 edge 的窗口里没有值"]
    return out


def judge(triples: dict, low: float = LOW, high: float = HIGH) -> dict:
    """triples：{seed: (g8, flr, g6a)}。"""
    per = [judge_seed(*triples.get(s, (None, None, None)), s) for s in SEEDS]
    ok = [p for p in per if p["verdict"] == "ok"]
    vs = {p["verdict"] for p in per}
    if "invalid" in vs:
        overall = "invalid"
    elif len(ok) < len(SEEDS):
        overall = "insufficient" if ok else "missing"
    else:
        exc = [p["victims"]["excess"] for p in ok]
        overall = ("persists" if min(exc) >= high else
                   "decays_to_floor" if max(exc) <= low else "user_decides")
    text = {
        "persists": f"攻击者走后受害 edge 仍高于 floor ≥ {high}（3 seed 全部）→ 后门停在平台上："
                    f"「踢出攻击者」不够，需要主动清除；G1 的 3-C 缩为最小确认（D-074）",
        "decays_to_floor": f"攻击者走后受害 edge 回到 floor 之内 ≤ {low}（3 seed 全部）→ 自然恢复："
                           f"检测 + 踢出可能就够；G1 保留 3-C、主量用 margin（D-074）",
        "user_decides": f"excess 介于 {low} 与 {high} 之间或 seed 间不一致 → 用户定",
        "insufficient": "有效 seed 不足 3 个（不报方向）",
        "missing": "G8 或 FLR 的结果还没回来",
        "invalid": "有 run 无效（D-044 的闸 / 停止轮没生效 / FLR 不是 ρ=0）",
    }[overall]
    return {"overall": overall, "summary": text,
            "thresholds": {"low": low, "high": high},
            "windows": {"A_T": WIN_T, "main": WIN_MAIN, "end": WIN_END},
            "per_seed": per, "preregistered": True,
            "caveat": "lr 按 0.992^有效轮 衰减（1B-2 为常数 lr）；衰减慢可能部分来自 lr 小"}


def load(results_dir: Path = RESULTS) -> dict:
    def _l(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    return {s: (_l(results_dir / "G8" / f"G8__a__s{s}.metrics.json"),
                _l(results_dir / "FLR" / f"FLR__g6a__s{s}.metrics.json"),
                _l(results_dir / "G6" / f"G6__a__s{s}.metrics.json")) for s in SEEDS}


def main(argv=None):
    ap = argparse.ArgumentParser(description="G8（攻击停止后的衰减）的预注册判定，D-075")
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
