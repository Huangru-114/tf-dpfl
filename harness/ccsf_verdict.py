"""
harness/ccsf_verdict.py  —  阶段三 CCSF（CCS 的 flat 复现闸）的判定（FINDINGS N-008 CCSF 段，D-101；实现取舍 D-109）
**先 commit，再交 CCSF 的作业**（同 D-088）。

    python3 harness/ccsf_verdict.py \\
        --off experiments/defense/edge-native/results/P2/CCSF/CCSF__off__s42.metrics.json \\
        --ccs experiments/defense/edge-native/results/P2/CCSF/CCSF__ccs-full__s42.metrics.json \\
        --out experiments/defense/edge-native/analysis/ccsf_verdict.json

量（N-008 共用定义；末 10 个评估点，`d0_power.run_quantities`）：
  P   全部良性端池化 fresh-PM ASR（`rounds[].local_benign_asr`，只数非目标类 —— 本仓库主列）；
  MTA fresh pm_acc（`acc_rounds[].pm_acc`）；ΔX = X(off) − X(ccs-full)。
闸（任一不满足 → invalid）：
  通用闸（两个 run）：exit_code 0、client_failures 为空、攻击者参与过（n_malicious_participations > 0）；
  开关真的生效：ccs-full 的 run.ccs = true 且 ccs_at / ccs_cluster 都是 true；off 的 run.ccs 为 false / 缺失；
  两个 run 的 n_rounds / edge_rounds / n_edges / seed / partition / malicious_per_edge 相同；
  ccs-full 每个云轮恰好一行 [CCS]（flat：1 edge × R=1 → n_rounds 行）。
  **off 臂必须植入**：P(off) ≥ 0.5，否则 `invalid_no_attack`（攻击没进去时复现无从谈起）。
标签：
  ccs_reproduces      ΔP ≥ 0.5 且 ΔMTA ≤ 0.02
  ccs_not_reproduced  ΔP < 0.2   → 按语义对照表（D-109）排查实现，不读 CCSP，用户定
  partial_repro       其余（照做 CCSP，结论里注明）
另报（不进判定）：不过滤目标类的良性 ASR（`local_benign_asr_unfiltered`，CCS 原文的口径）、剔除的 TPR / FPR
  （按 [CCS] 行逐次累加：TPR = Σ|剔除 ∩ 恶意| / Σ|本轮 ∩ 恶意|，FPR 同理对良性）、HDBSCAN 状态计数、恶意端数据占比。
纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d0_power import run_quantities                   # noqa: E402
from pilot_a4 import invalid_reasons                   # noqa: E402
from runs_table import last_k_mean                     # noqa: E402

PLANT_MIN, REPRO_DP, NOT_DP, MTA_MAX = 0.5, 0.5, 0.2, 0.02
SAME_KEYS = ("n_rounds", "edge_rounds", "n_edges", "seed", "partition", "malicious_per_edge")


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def gate(off: dict, ccs: dict) -> list:
    reasons = []
    for name, m in (("off", off), ("ccs", ccs)):
        reasons += [f"{name}: {r}" for r in invalid_reasons(m)]
        if m.get("exit_code") is None:
            reasons.append(f"{name}: exit_code 缺失")
        if not (m.get("n_malicious_participations") or 0) > 0:
            reasons.append(f"{name}: 攻击者一次都没参与")
    ro, rc = off.get("run") or {}, ccs.get("run") or {}
    if not (rc.get("ccs") is True and rc.get("ccs_at") is True and rc.get("ccs_cluster") is True):
        reasons.append(f"ccs: run.ccs / ccs_at / ccs_cluster = {rc.get('ccs')!r} / {rc.get('ccs_at')!r} / "
                       f"{rc.get('ccs_cluster')!r}，应全为 true（开关没生效 = 陷阱 #7 同类）")
    if ro.get("ccs") not in (None, False):
        reasons.append(f"off: run.ccs = {ro.get('ccs')!r}（对照不该开 CCS）")
    for k in SAME_KEYS:
        if ro.get(k) != rc.get(k):
            reasons.append(f"run.{k} 不一致：off {ro.get(k)!r} vs ccs {rc.get(k)!r}")
    n = int(rc.get("n_rounds") or 0) * int(rc.get("n_edges") or 1) * int(rc.get("edge_rounds") or 1)
    got = len(ccs.get("ccs_decisions") or [])
    if got != n:
        reasons.append(f"[CCS] 行数 {got} ≠ n_rounds × n_edges × edge_rounds = {n}")
    return reasons


def rejection_rates(m: dict) -> dict:
    mal = set(int(i) for i in ((m.get("run") or {}).get("malicious_ids") or []))
    tp = pos = fp = neg = 0
    status = Counter()
    for d in m.get("ccs_decisions") or []:
        cur = set(int(i) for i in d.get("cur") or [])
        rej = set(int(i) for i in d.get("rejected") or [])
        tp += len(rej & cur & mal)
        pos += len(cur & mal)
        fp += len((rej & cur) - mal)
        neg += len(cur - mal)
        status[str(d.get("status"))] += 1
    return {"tpr": (tp / pos) if pos else None, "fpr": (fp / neg) if neg else None,
            "n_malicious_uploads": pos, "n_benign_uploads": neg, "status": dict(sorted(status.items()))}


def quantities(m: dict) -> dict:
    q = run_quantities(m)
    q["P_unfiltered"] = last_k_mean([r.get("local_benign_asr_unfiltered") for r in m.get("rounds") or []])[0]
    return {k: q.get(k) for k in ("P", "P_unfiltered", "MTA", "margin_p50")}


def judge(off: dict, ccs: dict) -> dict:
    out = {"preregistered": "FINDINGS N-008 CCSF（D-101）；实现 D-109"}
    qo, qc = quantities(off), quantities(ccs)
    out["off"], out["ccs"] = {k: _r(v) for k, v in qo.items()}, {k: _r(v) for k, v in qc.items()}
    out["rejection"] = {k: (_r(v) if isinstance(v, float) else v) for k, v in rejection_rates(ccs).items()}
    out["malicious_data_share"] = {"off": (off.get("run") or {}).get("malicious_data_share"),
                                   "ccs": (ccs.get("run") or {}).get("malicious_data_share")}
    bad = gate(off, ccs)
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    if qo["P"] is None or qc["P"] is None or qo["MTA"] is None or qc["MTA"] is None:
        return {**out, "verdict": "insufficient", "reasons": ["P / MTA 缺失（末 10 个评估点不够）"]}
    if qo["P"] < PLANT_MIN:
        return {**out, "verdict": "invalid_no_attack",
                "reasons": [f"P(off) = {qo['P']:.4f} < {PLANT_MIN}：攻击没植入，复现无从谈起"]}
    dP, dMTA = qo["P"] - qc["P"], qo["MTA"] - qc["MTA"]
    out["delta"] = {"dP": _r(dP), "dMTA": _r(dMTA),
                    "dP_unfiltered": _r(None if qo["P_unfiltered"] is None or qc["P_unfiltered"] is None
                                        else qo["P_unfiltered"] - qc["P_unfiltered"])}
    out["mta_ok"] = dMTA <= MTA_MAX
    if dP >= REPRO_DP and dMTA <= MTA_MAX:
        verdict = "ccs_reproduces"
    elif dP < NOT_DP:
        verdict = "ccs_not_reproduced"
    else:
        verdict = "partial_repro"
    return {**out, "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 CCSF：CCS 的 flat 复现闸（N-008）")
    ap.add_argument("--off", required=True)
    ap.add_argument("--ccs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    off = json.loads(Path(a.off).read_text(encoding="utf-8"))
    ccs = json.loads(Path(a.ccs).read_text(encoding="utf-8"))
    res = judge(off, ccs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "delta", "reasons") if k in res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
