"""
harness/g5ab_verdict.py  —  G5AB（生成器语义 A / B 的缩小规模对比，D-079）的**预注册**判定

    python3 harness/g5ab_verdict.py [--json out.json]

问题：G5（3.3 收敛门控）的投毒窗口之外，Bad-PFL 的生成器怎么处理？
  A = generator_schedule: window —— 生成器只在窗口内训练、窗口后冻结（= G8 的语义）；
  B = generator_schedule: always —— 窗口只管投毒，生成器全程在训（窗口外 = ρ=0 影子攻击者）。
A 下每个 t0 的攻击者投入相同（只有窗口内 20 个有效轮）；B 下生成器在 t0 时已训了 t0 轮，
成熟度随 t0 增长 → 可能造出「峰值随 t0 上升」的假正相关。另一面，用户在另一个库（Bad-PFL 作者代码）
的实验提示「攻击者跟着训练」可能很关键（窗口后的持久性）。先用 8 个 run 看两种语义差多少。

格子（G5 的配置：4 edge、[10,0,0,0]、R5、equal_random、ρ=0.2、固定长度）：
  t20  = 投毒云轮 5–8（attack_start_round 5 / attack_stop_round 9），跑 19 轮
  t140 = 投毒云轮 29–32（29 / 33），跑 43 轮
  × {A, B} × seed {42, 43}

量（主列 = 良性端池化的 fresh-PM ASR `rounds[].local_benign_asr`，与 G5 的判定口径相同，D-078）：
  peak      窗口内 4 个评估点的最大值
  dilution  t0 + 65 / 70 / 75 个有效轮三点的均值（t20：第 17–19 云轮；t140：第 41–43 云轮）
  Δ = B − A（同格、同 seed 配对）

判定（THRESH = 0.10 ≈ 单点噪声 σ≈0.09，**没有证据**）：
  insensitive   全部 8 个 |Δ|（2 格 × 2 量 × 2 seed）< THRESH → G5 用 A
  sensitive     某个（格, 量）两个 seed 的 Δ 同号且 |Δ| 都 ≥ THRESH → 用户定（两种都跑 / 只跑 B 并注明混杂）
  user_decides  其余
  insufficient / missing / invalid 同其他判定（有效性闸 D-044 + 起止轮与语义真的生效了）
另报：峰值差在 t20 与 t140 的平均（B 的成熟度混杂会让 t140 的差更大）。

纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_a4 import invalid_reasons   # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43)
THRESH = 0.10                        # ⚠ 没有证据（D-079）
ARMS = {"A": "window", "B": "always"}
CELLS = {
    # t0（有效轮）: 投毒 cloud 轮 [start, stop)、run 长度、稀释点（t0+65/70/75 有效轮 ÷ R5）
    "t20":  {"start": 5,  "stop": 9,  "n_rounds": 19, "dilution": (17, 18, 19)},
    "t140": {"start": 29, "stop": 33, "n_rounds": 43, "dilution": (41, 42, 43)},
}
KEY = "local_benign_asr"
QUANTITIES = ("peak", "dilution")


def _series(m: dict) -> dict:
    return {int(r["round"]): r.get(KEY) for r in m.get("rounds") or [] if r.get(KEY) is not None}


def _r(v):
    return None if v is None else round(v, 4)


def run_reasons(m: dict, cell: str, arm: str) -> list:
    c = CELLS[cell]
    reasons = list(invalid_reasons(m))
    run = m.get("run") or {}
    for k, want in (("attack_start_round", c["start"]), ("attack_stop_round", c["stop"]),
                    ("generator_schedule", ARMS[arm])):
        if run.get(k) != want:
            reasons.append(f"run.{k}={run.get(k)!r} ≠ {want!r}（开关没生效 = 陷阱 #7 同类）")
    s = _series(m)
    if not s or max(s) < c["n_rounds"]:
        reasons.append(f"只跑到第 {max(s) if s else 0} 轮，不到 {c['n_rounds']}")
    return reasons


def quantities(m: dict, cell: str) -> dict:
    c = CELLS[cell]
    s = _series(m)
    win = [s[r] for r in range(c["start"], c["stop"]) if r in s]
    dil = [s[r] for r in c["dilution"] if r in s]
    return {"peak": _r(max(win)) if win else None,
            "n_peak_points": len(win),
            "dilution": _r(sum(dil) / len(dil)) if dil else None,
            "n_dilution_points": len(dil)}


def judge(runs: dict, thresh: float = THRESH) -> dict:
    """runs：{(cell, arm, seed): metrics | None}。"""
    per, invalid, missing = [], [], []
    vals = {}
    for cell in CELLS:
        for arm in ARMS:
            for seed in SEEDS:
                m = runs.get((cell, arm, seed))
                if m is None:
                    missing.append(f"{cell}-{arm} s{seed}")
                    continue
                bad = run_reasons(m, cell, arm)
                if bad:
                    invalid.append({"run": f"{cell}-{arm} s{seed}", "reasons": bad})
                    continue
                vals[(cell, arm, seed)] = quantities(m, cell)
    deltas = {}
    for cell in CELLS:
        for seed in SEEDS:
            a, b = vals.get((cell, "A", seed)), vals.get((cell, "B", seed))
            if a is None or b is None:
                continue
            for q in QUANTITIES:
                if a[q] is not None and b[q] is not None:
                    deltas[(cell, q, seed)] = round(b[q] - a[q], 4)
    for (cell, arm, seed), q in sorted(vals.items()):
        per.append({"cell": cell, "arm": arm, "seed": seed, **q})

    n_expected = len(CELLS) * len(QUANTITIES) * len(SEEDS)
    if invalid:
        overall = "invalid"
    elif missing or len(deltas) < n_expected:
        overall = "insufficient" if deltas else "missing"
    else:
        sensitive = [
            (cell, q) for cell in CELLS for q in QUANTITIES
            if all(abs(deltas[(cell, q, s)]) >= thresh for s in SEEDS)
            and len({deltas[(cell, q, s)] > 0 for s in SEEDS}) == 1]
        if sensitive:
            overall = "sensitive"
        elif all(abs(d) < thresh for d in deltas.values()):
            overall = "insensitive"
        else:
            overall = "user_decides"

    def _mean(cell, q):
        v = [deltas[(cell, q, s)] for s in SEEDS if (cell, q, s) in deltas]
        return _r(sum(v) / len(v)) if v else None

    text = {
        "insensitive": f"两种生成器语义的峰值与稀释值差都 < {thresh}（2 格 × 2 seed）→ G5 用 A"
                       f"（攻击者投入与 t0 无关，3.3 的秩相关不被生成器成熟度混杂）",
        "sensitive": f"某个量在两个 seed 上都差 ≥ {thresh} 且同号 → 生成器语义本身有影响：用户定"
                     f"（G5 两种都跑，或只跑 B 并注明成熟度混杂）",
        "user_decides": f"差值有的 ≥ {thresh} 但两个 seed 不一致 → 用户定",
        "insufficient": "8 个 run 不齐（不报方向）",
        "missing": "G5AB 的结果还没回来",
        "invalid": "有 run 无效（D-044 的闸 / 起止轮或生成器语义没生效 / 没跑满）",
    }[overall]
    return {"overall": overall, "summary": text,
            "threshold": thresh, "threshold_evidence": "none（D-079）",
            "deltas_B_minus_A": {f"{c}/{q}/s{s}": d for (c, q, s), d in sorted(deltas.items())},
            "mean_peak_delta": {cell: _mean(cell, "peak") for cell in CELLS},
            "mean_dilution_delta": {cell: _mean(cell, "dilution") for cell in CELLS},
            "per_run": per, "invalid": invalid, "missing": missing,
            "preregistered": True,
            "note": "2 个 seed 只够看大差别；这不是 3.3 的判定，只决定 G5 用哪种语义"}


def load(results_dir: Path = RESULTS) -> dict:
    out = {}
    for cell in CELLS:
        for arm in ARMS:
            for seed in SEEDS:
                p = results_dir / "G5AB" / f"G5AB__{cell}-{arm}__s{seed}.metrics.json"
                out[(cell, arm, seed)] = (json.loads(p.read_text(encoding="utf-8"))
                                          if p.is_file() else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="G5AB（生成器语义 A / B 对比）的预注册判定，D-079")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = judge(load())
    for p in res["per_run"]:
        print(f"  {p['cell']}-{p['arm']} s{p['seed']}: peak={p['peak']} dilution={p['dilution']}")
    for k, v in res["deltas_B_minus_A"].items():
        print(f"  Δ(B−A) {k}: {v}")
    for inv in res["invalid"]:
        print(f"  invalid {inv['run']}: {inv['reasons']}")
    print(res["summary"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
