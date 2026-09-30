"""
harness/g6_verdict.py  —  3-E（三层个性化）：G6 的预注册判定（S7；2026-09-30）

    python3 harness/g6_verdict.py [--json out.json]

**规则是预注册的，脚本不是**：规则在 PLAN §3 的 3-E 行（D-059 / D-071，数据回来之前定）；
本脚本在 G6 回传**之后**才写（同 `g3_did.py` 的情形），实现同一条规则，复现 REPORT §5.4 / FINDINGS F-061 / F-065③ 的快速读数。

量（末 10 个评估点；G6 每云轮都评估）：
  victim_asr   受害 edge E1–E3 的 fresh-PM 良性 ASR（`per_edge_rounds[r][e].client_benign`）：每个 edge 取末 10 点均值，再对三个 edge 平均
  pm_acc       fresh pm_acc（判定用，D-071）
  pm_acc_stale 陈旧 pm_acc（只报告；隔点列，按 `runs_table.window_mean` 以 pm_acc 认评估点）
  按 seed 配对：ΔASR = a − 臂、ΔMTA = a − 臂（臂 ∈ {b, c}；正 = 臂更低）

判定（每臂分开）：
  blocks     ΔASR 的 bootstrap CI 下界 > 0（`verdicts.bootstrap_mean_ci`；3 seed 时 ⇔ 三个 seed 都 > 0）
             且 ΔMTA ≤ 0.02（D-071）
  costly     ASR 降了（CI 下界 > 0）但 ΔMTA > 0.02
  no_block   ΔASR 的 CI 含 0 或 < 0
  0.02 是「均值」还是「每个 seed」规则没写死 → 两种读法都算：`mta_ok_mean` / `mta_ok_every_seed`；
  判定按两者都成立才算 `blocks`，只有一种成立时记 `mta_reading_matters`（数据回来时两种读法结论相同）。
注意（D-059）：判定用**原始** ASR；「三臂的 floor 相同」没有证据（划分不变，但 edge 段改变了受害模型）。
另报 ΔASR–ΔMTA 的权衡（D-071）。G6D（[3,3,2,2]、s42、止步，F-069）只作参照，不参与判定。

纯标准库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decay_verdict import edge_by_round          # noqa: E402
from pilot_a4 import invalid_reasons             # noqa: E402
from runs_table import last_k_mean, window_mean  # noqa: E402
from verdicts import bootstrap_mean_ci           # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
ARMS = {"a": 0, "b": 1, "c": 2}                  # 臂 → federation.edge_shared_blocks（D-057）
VICTIMS = (1, 2, 3)
PLACEMENT = [10, 0, 0, 0]
MTA_MAX = 0.02                                   # D-071（fresh 列）


def _r(v, nd=4):
    return None if v is None else round(v, nd)


def run_quantities(m: dict) -> dict:
    per_edge = {e: last_k_mean(list(edge_by_round(m, e).values()))[0] for e in (0,) + VICTIMS}
    vict = [per_edge[e] for e in VICTIMS]
    acc = m.get("acc_rounds") or []
    return {"victim_asr": None if any(v is None for v in vict) else sum(vict) / len(vict),
            "e0_benign_asr": per_edge[0],
            "per_edge_asr": per_edge,
            "pm_acc": last_k_mean([r.get("pm_acc") for r in acc])[0],
            "pm_acc_stale": window_mean(acc, "pm_acc_stale", anchor="pm_acc")[0]}


def run_reasons(m: dict | None, arm: str) -> list:
    if m is None:
        return ["缺"]
    reasons = list(invalid_reasons(m))
    run = m.get("run") or {}
    k = run.get("edge_shared_blocks") or 0             # S8 之前的文件没有 [设定6] → 0
    if k != ARMS[arm]:
        reasons.append(f"run.edge_shared_blocks={k} ≠ {ARMS[arm]}（开关没生效 = 陷阱 #7 同类）")
    if run.get("malicious_per_edge") != PLACEMENT:
        reasons.append(f"run.malicious_per_edge={run.get('malicious_per_edge')} ≠ {PLACEMENT}")
    if not m.get("per_edge_rounds"):
        reasons.append("没有 per_edge_rounds")
    return reasons


def judge(runs: dict, mta_max: float = MTA_MAX) -> dict:
    """runs：{(arm, seed): metrics | None}。"""
    per, invalid, missing = {}, [], []
    for arm in ARMS:
        for s in SEEDS:
            m = runs.get((arm, s))
            if m is None:
                missing.append(f"{arm}/s{s}")
                continue
            bad = run_reasons(m, arm)
            if bad:
                invalid.append({"run": f"{arm}/s{s}", "reasons": bad})
                continue
            per[(arm, s)] = run_quantities(m)

    arms = {}
    for arm in ("b", "c"):
        pairs = [s for s in SEEDS if (arm, s) in per and ("a", s) in per]
        d_asr = {s: per[("a", s)]["victim_asr"] - per[(arm, s)]["victim_asr"] for s in pairs}
        d_mta = {s: per[("a", s)]["pm_acc"] - per[(arm, s)]["pm_acc"] for s in pairs}
        d_stale = {s: (None if per[("a", s)]["pm_acc_stale"] is None or per[(arm, s)]["pm_acc_stale"] is None
                       else per[("a", s)]["pm_acc_stale"] - per[(arm, s)]["pm_acc_stale"]) for s in pairs}
        mean, lo, hi = bootstrap_mean_ci(list(d_asr.values())) if len(pairs) == len(SEEDS) else (None, None, None)
        mta_mean = sum(d_mta.values()) / len(d_mta) if d_mta else None
        ok_mean = mta_mean is not None and mta_mean <= mta_max
        ok_every = bool(d_mta) and max(d_mta.values()) <= mta_max
        if len(pairs) < len(SEEDS):
            v = "insufficient"
        elif lo is None or lo <= 0:
            v = "no_block"
        elif ok_mean and ok_every:
            v = "blocks"
        elif ok_mean or ok_every:
            v = "mta_reading_matters"
        else:
            v = "costly"
        arms[arm] = {"verdict": v,
                     "d_asr": {f"s{s}": _r(x) for s, x in d_asr.items()},
                     "d_asr_mean": _r(mean), "d_asr_ci": [_r(lo), _r(hi)],
                     "d_mta_fresh": {f"s{s}": _r(x, 5) for s, x in d_mta.items()},
                     "d_mta_fresh_mean": _r(mta_mean, 5),
                     "mta_ok_mean": ok_mean, "mta_ok_every_seed": ok_every,
                     "d_mta_stale": {f"s{s}": _r(x, 5) for s, x in d_stale.items()},
                     "asr_drop_per_mta_point": (None if not mta_mean else _r(mean / (mta_mean * 100), 3))}
    if invalid:
        overall = "invalid"
    elif missing and not per:
        overall = "missing"
    else:
        overall = {a: v["verdict"] for a, v in arms.items()}
    return {"overall": overall, "arms": arms, "mta_max": mta_max,
            "per_run": [{"arm": a, "seed": s, **{k: (_r(v) if not isinstance(v, dict) else
                                                    {str(e): _r(x) for e, x in v.items()})
                                                for k, v in q.items()}}
                        for (a, s), q in sorted(per.items())],
            "invalid": invalid, "missing": missing,
            "preregistered_rule": True, "script_written_after_data": True,
            "note": "原始 ASR（D-059：三臂 floor 相同没有证据）；MTA 用 fresh 列（D-071），陈旧列只报告；"
                    "asr_drop_per_mta_point = 平均 ΔASR / (平均 ΔMTA × 100)，即每损失 1 个百分点精度换来的 ASR 下降"}


def load(results_dir: Path = RESULTS) -> dict:
    def _l(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    return {(a, s): _l(results_dir / "G6" / f"G6__{a}__s{s}.metrics.json") for a in ARMS for s in SEEDS}


def load_g6d(results_dir: Path = RESULTS) -> dict:
    """G6D（[3,3,2,2]、s42）：只作参照 —— 没有受害 edge，量取池化良性 ASR。"""
    out = {}
    for a in ARMS:
        p = results_dir / "G6D" / f"G6D__{a}__s42.metrics.json"
        if p.is_file():
            m = json.loads(p.read_text(encoding="utf-8"))
            out[a] = {"benign_asr": last_k_mean([r.get("local_benign_asr") for r in m.get("rounds") or []])[0],
                      "pm_acc": last_k_mean([r.get("pm_acc") for r in m.get("acc_rounds") or []])[0]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-E（G6）的预注册判定，D-059 / D-071（脚本写于数据之后）")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = judge(load())
    res["g6d_reference"] = load_g6d()
    for p in res["per_run"]:
        print(f"  {p['arm']}/s{p['seed']}: 受害 ASR {p['victim_asr']}  E0 良性 {p['e0_benign_asr']}  "
              f"pm_acc {p['pm_acc']}  陈旧 {p['pm_acc_stale']}")
    for arm, v in res["arms"].items():
        print(f"  臂 {arm}: ΔASR {v['d_asr']} 均值 {v['d_asr_mean']} CI {v['d_asr_ci']} | "
              f"ΔMTA(fresh) {v['d_mta_fresh']} 均值 {v['d_mta_fresh_mean']} "
              f"（≤{MTA_MAX}：均值 {v['mta_ok_mean']}，逐 seed {v['mta_ok_every_seed']}） → {v['verdict']}")
    for inv in res["invalid"]:
        print(f"  invalid {inv['run']}: {inv['reasons']}")
    print(f"判定：{res['overall']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
