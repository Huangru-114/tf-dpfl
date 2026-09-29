"""
harness/g5_verdict.py  —  G5（3.3 收敛门控）的**预注册**判定（D-081；G5 回传之前写定）

    python3 harness/g5_verdict.py [--json out.json]

问题（PLAN §3 的 3.3 行）：投毒窗口开在训练的不同阶段（t0 = 20 / 60 / 100 / 140 / 180 有效轮），
窗口内植入的峰值是否随 t0 单调上升 —— 若是，植入受「模型收敛程度」门控。

配置（D-078 / D-080）：G0-random 配置（4 edge、[10,0,0,0]、R5、equal_random、ρ=0.2）+ 生成器语义 A（window）；
每格投毒 cloud 轮 [t0/5 + 1, t0/5 + 5)，固定跑到 t0 + 75 有效轮。floor = 同 seed 的 G0-random（ρ=0，60 轮）。

量（主列 = 良性端池化 fresh-PM ASR `rounds[].local_benign_asr`；floor 取**同一批云轮**，同有效轮同 lr）：
  peak_excess      窗口内 4 个评估点的 ASR 最大值 − 同 4 轮 floor 的均值（floor 取均值以压低单点噪声）
  dilution_excess  t0 + 65 / 70 / 75 有效轮三点的 ASR 均值 − 同 3 轮 floor 的均值（**只报告**：F-072 显示稀释点已贴 floor）

统计（PLAN §3 通用规则：按 seed 配对、bootstrap 95% CI、10000 次、固定种子）：
  每个 seed 内对 5 个 t0 算 Spearman ρ(t0, peak_excess)（并列取平均秩）→ 对 seed 做 bootstrap 求均值 ρ 的 CI
  （复用 `harness/verdicts.py:bootstrap_mean_ci`）。
  ⚠ 3 个 seed 时 27 种等概率重采样里「三次都抽到最小者」的概率 1/27 ≈ 3.7% > 2.5%，
    所以 CI 下界 = 最小的 seed ρ：**「CI > 0」⇔ 三个 seed 的 ρ 都 > 0**。

判定：
  gated          CI 下界 > 0 → 植入受收敛门控（PLAN §3 的预注册结论）
  anti_gated     CI 上界 < 0 → 峰值随 t0 单调**下降**（预注册只写了第一支；这一支单独报，不当成「门控」）
  not_gated      CI 含 0
  insufficient / missing / invalid 同其他判定（有效性闸 D-044 + 起止轮与生成器语义真的生效 + 跑满 + floor 有效）

附带（只报告，不作闸）：G5 的 t20 / t140 × s42 / s43 与 G5AB 的 A 臂配置只差 meta（F-072）→ 逐轮 checksum 应全等
（跨作业的 GPU 复现检查）。

纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_a4 import invalid_reasons            # noqa: E402
from verdicts import bootstrap_mean_ci         # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
R_EDGE = 5                                      # 每 cloud 轮的 edge 轮数（有效轮 = 云轮 × 5）
T0S = (20, 60, 100, 140, 180)
KEY = "local_benign_asr"
SCHEDULE = "window"                             # D-080
G0_MIN_ROUNDS = 51                              # 最晚的稀释点：t180 的第 51 云轮


def cell_of(t0: int) -> dict:
    """t0（有效轮）→ 格子常数；与 registry.yaml 的 G5 cells 交叉核对（test_g5_verdict.py）。"""
    start = t0 // R_EDGE + 1
    return {"label": f"t{t0}", "start": start, "stop": start + 4,
            "n_rounds": (t0 + 75) // R_EDGE,
            "window": tuple(range(start, start + 4)),
            "dilution": tuple((t0 + d) // R_EDGE for d in (65, 70, 75))}


CELLS = {t0: cell_of(t0) for t0 in T0S}


def _series(m: dict) -> dict:
    return {int(r["round"]): r.get(KEY) for r in m.get("rounds") or [] if r.get(KEY) is not None}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _r(v):
    return None if v is None else round(v, 4)


def ranks(values) -> list:
    """平均秩（1 起），并列取平均。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x, y) -> float | None:
    """Spearman ρ = 秩的 Pearson 相关；任一边全并列（方差 0）→ None。"""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = ranks(list(x)), ranks(list(y))
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def run_reasons(m: dict, t0: int) -> list:
    c = CELLS[t0]
    reasons = list(invalid_reasons(m))
    run = m.get("run") or {}
    for k, want in (("attack_start_round", c["start"]), ("attack_stop_round", c["stop"]),
                    ("generator_schedule", SCHEDULE)):
        if run.get(k) != want:
            reasons.append(f"run.{k}={run.get(k)!r} ≠ {want!r}（开关没生效 = 陷阱 #7 同类）")
    s = _series(m)
    if not s or max(s) < c["n_rounds"]:
        reasons.append(f"只跑到第 {max(s) if s else 0} 轮，不到 {c['n_rounds']}")
    return reasons


def floor_reasons(g0: dict) -> list:
    reasons = list(invalid_reasons(g0))
    run = g0.get("run") or {}
    if run.get("poison_ratio") not in (0, 0.0):
        reasons.append(f"G0 的 poison_ratio={run.get('poison_ratio')!r} ≠ 0（不是 floor）")
    s = _series(g0)
    if not s or max(s) < G0_MIN_ROUNDS:
        reasons.append(f"G0 只跑到第 {max(s) if s else 0} 轮，不到 {G0_MIN_ROUNDS}")
    return reasons


def quantities(m: dict, g0: dict, t0: int) -> dict:
    c = CELLS[t0]
    s, f = _series(m), _series(g0)
    win = [s[r] for r in c["window"] if r in s]
    peak = max(win) if win else None
    floor_win = _mean(f.get(r) for r in c["window"])
    dil = _mean(s.get(r) for r in c["dilution"])
    floor_dil = _mean(f.get(r) for r in c["dilution"])
    return {"peak": _r(peak), "floor_window": _r(floor_win),
            "peak_excess": _r(None if peak is None or floor_win is None else peak - floor_win),
            "dilution": _r(dil), "floor_dilution": _r(floor_dil),
            "dilution_excess": _r(None if dil is None or floor_dil is None else dil - floor_dil),
            "n_window_points": len(win)}


def _checksums(m: dict) -> list:
    return [c.get("global") for c in m.get("checksums") or []]


def reproducibility(g5_runs: dict, g5ab: dict) -> list:
    """G5 与 G5AB-A 同 t0、同 seed 的逐轮 checksum 是否全等（只报告）。"""
    out = []
    for (t0, seed), m in sorted(g5_runs.items()):
        ref = g5ab.get((t0, seed))
        if m is None or ref is None:
            continue
        a, b = _checksums(m), _checksums(ref)
        n = min(len(a), len(b))
        first = next((i + 1 for i in range(n) if a[i] != b[i]), None)
        out.append({"cell": f"t{t0}", "seed": seed, "rounds_compared": n,
                    "identical": bool(n) and first is None, "first_diff_round": first})
    return out


def judge(g5_runs: dict, g0_runs: dict, g5ab_runs: dict | None = None) -> dict:
    """g5_runs：{(t0, seed): metrics}；g0_runs：{seed: G0-random metrics}；g5ab_runs：{(t0, seed): G5AB-A}。"""
    invalid, missing, per = [], [], []
    per_seed = {}
    for seed in SEEDS:
        g0 = g0_runs.get(seed)
        if g0 is None:
            missing.append(f"G0-random s{seed}")
            continue
        bad = floor_reasons(g0)
        if bad:
            invalid.append({"run": f"G0-random s{seed}", "reasons": bad})
            continue
        rows = []
        for t0 in T0S:
            m = g5_runs.get((t0, seed))
            if m is None:
                missing.append(f"t{t0} s{seed}")
                continue
            bad = run_reasons(m, t0)
            if bad:
                invalid.append({"run": f"t{t0} s{seed}", "reasons": bad})
                continue
            q = quantities(m, g0, t0)
            per.append({"t0": t0, "seed": seed, **q})
            rows.append((t0, q))
        if len(rows) == len(T0S) and all(q["peak_excess"] is not None for _, q in rows):
            per_seed[seed] = {
                "rho_peak": _r(spearman([t for t, _ in rows], [q["peak_excess"] for _, q in rows])),
                "rho_dilution": _r(spearman([t for t, _ in rows],
                                            [q["dilution_excess"] for _, q in rows])
                                   if all(q["dilution_excess"] is not None for _, q in rows) else None)}

    rhos = [v["rho_peak"] for v in per_seed.values() if v["rho_peak"] is not None]
    mean, lo, hi = bootstrap_mean_ci(rhos) if rhos else (None, None, None)
    if invalid:
        overall = "invalid"
    elif len(rhos) < len(SEEDS):
        overall = "insufficient" if rhos else "missing"
    elif lo is not None and lo > 0:
        overall = "gated"
    elif hi is not None and hi < 0:
        overall = "anti_gated"
    else:
        overall = "not_gated"
    dil = [v["rho_dilution"] for v in per_seed.values() if v["rho_dilution"] is not None]
    text = {
        "gated": "植入峰值随 t0 单调上升（三个 seed 的秩相关都 > 0）→ 植入受收敛门控（PLAN §3）",
        "anti_gated": "植入峰值随 t0 单调下降（三个 seed 的秩相关都 < 0）→ 越晚越难植入（预注册没有这一支，单独报）",
        "not_gated": "秩相关的 CI 含 0 → 没有证据表明植入受收敛门控",
        "insufficient": "有效 seed 不足 3 个（不报方向）",
        "missing": "G5 或 G0-random 的结果还没回来",
        "invalid": "有 run 无效（D-044 的闸 / 起止轮或生成器语义没生效 / 没跑满 / floor 无效）",
    }[overall]
    return {"overall": overall, "summary": text,
            "rho_peak_mean": _r(mean), "rho_peak_ci": [_r(lo), _r(hi)],
            "per_seed": {f"s{s}": v for s, v in sorted(per_seed.items())},
            "rho_dilution_mean": _r(sum(dil) / len(dil)) if dil else None,
            "per_run": per, "invalid": invalid, "missing": missing,
            "reproducibility_vs_G5AB_A": reproducibility(g5_runs, g5ab_runs or {}),
            "preregistered": True,
            "note": "3 个 seed 时 CI 下界 = 最小的 seed ρ（「CI > 0」⇔ 三个 seed 都 > 0）；稀释只报告（F-072）"}


def load(results_dir: Path = RESULTS):
    def _l(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    g5 = {(t0, s): _l(results_dir / "G5" / f"G5__t{t0}__s{s}.metrics.json") for t0 in T0S for s in SEEDS}
    g0 = {s: _l(results_dir / "G0" / f"G0__random__s{s}.metrics.json") for s in SEEDS}
    g5ab = {(t0, s): _l(results_dir / "G5AB" / f"G5AB__t{t0}-A__s{s}.metrics.json")
            for t0 in (20, 140) for s in (42, 43)}
    return {k: v for k, v in g5.items() if v is not None}, g0, {k: v for k, v in g5ab.items() if v}


def main(argv=None):
    ap = argparse.ArgumentParser(description="G5（3.3 收敛门控）的预注册判定，D-081")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    g5, g0, g5ab = load()
    res = judge(g5, g0, g5ab)
    for p in res["per_run"]:
        print(f"  t{p['t0']:<3} s{p['seed']}: peak={p['peak']} floor={p['floor_window']} "
              f"excess={p['peak_excess']} | dilution_excess={p['dilution_excess']}")
    for s, v in res["per_seed"].items():
        print(f"  {s}: ρ(t0, peak_excess)={v['rho_peak']}  ρ(t0, dilution_excess)={v['rho_dilution']}")
    for r in res["reproducibility_vs_G5AB_A"]:
        print(f"  复现 {r['cell']} s{r['seed']} vs G5AB-A：{'全等' if r['identical'] else '不同'}"
              f"（{r['rounds_compared']} 轮，首个不同 = {r['first_diff_round']}）")
    for inv in res["invalid"]:
        print(f"  invalid {inv['run']}: {inv['reasons']}")
    print(f"ρ 均值 {res['rho_peak_mean']}，CI {res['rho_peak_ci']} → {res['summary']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
