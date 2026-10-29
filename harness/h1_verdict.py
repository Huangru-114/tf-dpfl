#!/usr/bin/env python3
"""
阶段三 H1（集中布点主检验）的判定 —— 规则 = PLAN §4 H1 + D-101；本批 = A-pre 臂（D-111）。
**先 commit、再交 s43 / s44**（D-088 惯例）。

输入：逐 seed 的 (防御臂, 对照) metrics.json 配对，`--pair ARM CTRL` 重复给（同 seed，按 seed 配对）。
  本批：s42 = P1__col-pre__s42 / SNAP__collocated__s42；s43、s44 = H1__col-pre__s4x / SNAP__collocated__s4x。

每个配对先过闸（`p1_verdict.gate`：通用闸 + 加固开关生效 + 第 1 云轮 [HardenAgg] unhardened = 对照 [Checksum] Round 1
+ [Harden] 行数 + 两边因素相同）；任一配对不过 → `invalid`。量与 P1 同一套（`p1_verdict.quantities` → `d0_power.run_quantities`）。

标签（3 seed；Δ = 对照 − 臂）：
  受害端（V）：`d0_power.classify_h1(ΔV, "d101")`：
      protects  = 均值 ≥ 0.15 且最小 seed ≥ 0.10
      no_effect = |均值| < 0.05 且最大 |Δ| < 0.10（D-101）
      partial   = 其余
  精度：ΔMTA 均值 ≤ 0.02 且每个 seed ≤ 0.025。
  组合：protects + 精度过 → `protects_victims`；protects + 精度不过 → `costly`；否则 = no_effect / partial。
  `treats_E0`：同一条 classify_h1 作用于 ΔB0（只报告标签，不与精度组合）。
  天花板规则：对照与臂的 V 在**全部** seed 都 ≥ 0.95 → PLAN 说改读 Δmargin，但 margin 的 protects / no_effect 门槛 PLAN 没有写全
      （只有草稿「2 logit」）→ 本脚本**不替它定**：标签记 `ceiling_user_decides`，报告 Δmargin_v 逐 seed 值。
  5 seed（给了 5 个配对时另报，D-099 / PLAN §3.5）：均值 + 配对 percentile bootstrap 95% CI（固定 RNG，10000 次）；
      protects = 均值 ≥ 0.15 且 CI 下界 ≥ 0.10；no_effect = |均值| < 0.05 且 CI ⊂ (−0.10, 0.10)；其余 partial。
      5 seed 只能确认或降级 3 seed（s42–44）的标签、不能升级 → `final` 取两者中较弱的一个。
  机制（Δexcess ≥ 0.10）需要 ρ=0 的 floor run（H1 floor，条件组，还没跑）→ 这里报 `excess: null`、只能说「压低 ASR」。
  不足 3 个配对 → `insufficient`。

另报（不进判定）：每 seed 的 V / B0 / MTA / margin_v / jump 比值、加固墙钟。
纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d0_power import (MTA_EACH, MTA_MEAN, NO_EFFECT, NO_EFFECT_MAX,       # noqa: E402
                      PROTECT_MEAN, PROTECT_MIN, classify_h1)
from p1_verdict import ARM_CFG, CEIL, gate, quantities                    # noqa: E402

BOOT_DRAWS = 10000
BOOT_SEED = 20261029
ORDER = {"no_effect": 0, "partial": 1, "protects": 2}          # 「较弱」= 序号小（no_effect 与 partial 都弱于 protects）


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(xs):
    return math.fsum(xs) / len(xs)


def bootstrap_ci(d: list, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> tuple:
    """配对差的均值的 percentile bootstrap 95% CI（固定 RNG）。"""
    rng = random.Random(seed)
    n = len(d)
    means = sorted(_mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    lo, hi = means[int(0.025 * draws)], means[int(0.975 * draws) - 1]
    return lo, hi


def classify_5(d: list) -> dict:
    m = _mean(d)
    lo, hi = bootstrap_ci(d)
    if m >= PROTECT_MEAN and lo >= PROTECT_MIN:
        lab = "protects"
    elif abs(m) < NO_EFFECT and lo > -NO_EFFECT_MAX and hi < NO_EFFECT_MAX:
        lab = "no_effect"
    else:
        lab = "partial"
    return {"label": lab, "mean": _r(m), "ci95": [_r(lo), _r(hi)]}


def _weaker(a: str, b: str) -> str:
    """5 seed 只能确认或降级。no_effect 与 partial 之间互相「降级」没有定义 → 两个都弱于 protects 时保留 3 seed 的。"""
    if a == "protects" and b != "protects":
        return b
    return a


def combine(asr_label: str, mta_ok: bool) -> str:
    if asr_label == "protects":
        return "protects_victims" if mta_ok else "costly"
    return asr_label


def judge(pairs: list, cfg: str = ARM_CFG) -> dict:
    """pairs：[(arm_metrics, ctrl_metrics), …]，每个配对同 seed。"""
    out = {"preregistered": "PLAN §4 H1 + D-101；本批 D-111", "arm_cfg": cfg, "n_pairs": len(pairs)}
    per, bad = [], []
    for arm, ctrl in pairs:
        sa, sc = (arm.get("run") or {}).get("seed"), (ctrl.get("run") or {}).get("seed")
        reasons = gate(arm, ctrl, cfg)
        if list((arm.get("run") or {}).get("malicious_per_edge") or []) != [10, 0, 0, 0]:
            reasons.append("H1 只判集中布点 [10,0,0,0]")
        if reasons:
            bad += [f"s{sa}: {r}" for r in reasons]
            continue
        qa, qc = quantities(arm), quantities(ctrl)
        d = {k: (None if qc[k] is None or qa[k] is None else qc[k] - qa[k])
             for k in ("V", "B0", "MTA", "margin_v")}
        jr = (None if qa["jump_half"] is None or qc["jump_half"] in (None, 0)
              else qa["jump_half"] / qc["jump_half"])
        h = arm.get("harden") or []
        per.append({"seed": sa, "arm": {k: _r(v) for k, v in qa.items()},
                    "control": {k: _r(v) for k, v in qc.items()},
                    "dV": d["V"], "dB0": d["B0"], "dMTA": d["MTA"], "dmargin_v": d["margin_v"],
                    "jump_ratio": jr, "harden_seconds": _r(math.fsum(float(x.get("seconds") or 0) for x in h), 1),
                    "_ceil": qa["V"] is not None and qc["V"] is not None and qa["V"] >= CEIL and qc["V"] >= CEIL})
    seeds = [p["seed"] for p in per]
    if len(set(seeds)) != len(seeds):
        bad.append(f"seed 重复：{seeds}")
    out["per_seed"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in p.items() if k != "_ceil"}
                       for p in sorted(per, key=lambda p: p["seed"])]
    out["excess"] = None                                  # 要 H1 floor（ρ=0）run，还没跑
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    if len(per) < 3:
        return {**out, "verdict": "insufficient", "reasons": [f"只有 {len(per)} 个有效配对（要 ≥ 3）"]}
    missing = [f"s{p['seed']}:{k}" for p in per for k in ("dV", "dB0", "dMTA") if p[k] is None]
    if missing:
        return {**out, "verdict": "insufficient", "reasons": [f"量缺失：{missing}"]}

    per.sort(key=lambda p: p["seed"])
    first3 = per[:3]
    dv3, db3, dm3 = ([p[k] for p in first3] for k in ("dV", "dB0", "dMTA"))
    mta_ok3 = _mean(dm3) <= MTA_MEAN and max(dm3) <= MTA_EACH
    ceiling = all(p["_ceil"] for p in per)
    out["ceiling_rule"] = ceiling
    out["three_seed"] = {"seeds": [p["seed"] for p in first3],
                         "dV": [_r(x) for x in dv3], "dV_mean": _r(_mean(dv3)), "dV_min": _r(min(dv3)),
                         "dB0": [_r(x) for x in db3], "dMTA": [_r(x) for x in dm3], "dMTA_mean": _r(_mean(dm3)),
                         "mta_ok": mta_ok3,
                         "asr_label": classify_h1(dv3, "d101"), "treats_E0": classify_h1(db3, "d101")}
    if ceiling:
        out["three_seed"]["dmargin_v"] = [_r(p["dmargin_v"]) for p in first3]
        return {**out, "verdict": "ceiling_user_decides",
                "reasons": ["对照与臂的 V 在全部 seed 都 ≥ 0.95：PLAN 只给了 margin 门槛的草稿（2 logit），不替它定"]}
    verdict3 = combine(out["three_seed"]["asr_label"], mta_ok3)
    out["three_seed"]["verdict"] = verdict3
    final = verdict3
    if len(per) >= 5:
        five = per[:5]
        dv5, db5, dm5 = ([p[k] for p in five] for k in ("dV", "dB0", "dMTA"))
        mta_ok5 = _mean(dm5) <= MTA_MEAN and max(dm5) <= MTA_EACH
        c5, e5 = classify_5(dv5), classify_5(db5)
        out["five_seed"] = {"seeds": [p["seed"] for p in five], "dV": c5, "treats_E0": e5,
                            "dMTA_mean": _r(_mean(dm5)), "mta_ok": mta_ok5,
                            "verdict": combine(c5["label"], mta_ok5)}
        lab = _weaker(out["three_seed"]["asr_label"], c5["label"])
        final = combine(lab, mta_ok3 and mta_ok5)
    out["verdict"] = final
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 H1：集中布点主检验的判定（PLAN §4 H1 + D-101）")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("ARM", "CTRL"), required=True)
    ap.add_argument("--cfg", default=ARM_CFG)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pairs = [tuple(json.loads(Path(p).read_text(encoding="utf-8")) for p in pr) for pr in a.pair]
    res = judge(pairs, a.cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "three_seed", "five_seed", "reasons") if k in res},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
