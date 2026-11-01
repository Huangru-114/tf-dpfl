"""
harness/ccsd_verdict.py  —  阶段三 CCSD（CCS 拆分：只开对抗训练 / 只开聚类剔除）的判定（FINDINGS N-010，D-121）
N-010 已经用户确认（D-122）；本脚本先 commit，再交 CCSD 的作业（同 D-088）。

    python3 harness/ccsd_verdict.py \\
        --off          experiments/defense/edge-native/results/P2/CCSBOFF/CCSBOFF__fedrep-bnshared-off__s42.metrics.json \\
        --both         experiments/defense/edge-native/results/P2/CCSB/CCSB__fedrep-bnshared-ccs__s42.metrics.json \\
        --at-only      experiments/defense/edge-native/results/P2/CCSD/CCSD__fedrep-bnshared-at-only__s42.metrics.json \\
        --cluster-only experiments/defense/edge-native/results/P2/CCSD/CCSD__fedrep-bnshared-cluster-only__s42.metrics.json \\
        --out experiments/defense/edge-native/analysis/ccsd_verdict.json

量（与 CCSF 判定同一个定义，`ccsf_verdict.quantities`；末 10 个评估点）：
  P = 良性端池化的新鲜个性化模型攻击成功率（只数非目标类）；MTA = 新鲜个性化模型的主任务精度。
  对每个开防御的臂 x ∈ {at, cluster, both}：ΔP_x = P(off) − P(x)，ΔMTA_x = MTA(off) − MTA(x)。
闸（任一不满足 → invalid）：
  每个 run：exit_code 0（且不缺）、client_failures 为空、攻击者参与过；
  开关真的生效：off 不开 CCS；both 的 ccs_at / ccs_cluster 都是 true；at-only = true / false；cluster-only = false / true；
  四个 run 的 n_rounds / edge_rounds / n_edges / seed / partition / malicious_per_edge / method 相同，且统计量都是 shared；
  开聚类的两臂（both、cluster-only）每个云轮恰好一行 [CCS]；
  off 臂必须植入：P(off) ≥ 0.5，否则 invalid_no_attack。
标签（两条归因各自独立；FRAC = 2/3 由用户 2026-11-01 确认（D-122），没有数据依据）：
  精度代价归因 cost：ΔMTA_both ≤ 0.02 → no_cost（没有要归因的代价）；否则
    a = ΔMTA_at ≥ FRAC·ΔMTA_both，c = ΔMTA_cluster ≥ FRAC·ΔMTA_both →
    a 且 c：both_components；只有 a：adversarial_training；只有 c：clustering；都不是：only_together
  防御效果归因 defense：ΔP_both < 0.2 → no_effect；否则同样的规则，用 ΔP。
  另：每个臂单独按 CCS 复现闸的三个条件给标签（ΔP ≥ 0.5 且 ΔMTA ≤ 0.02 → meets_all；ΔP < 0.2 → no_defense；其余 partial）。
另报（不进判定）：开聚类两臂的剔除命中率 / 误伤率、各臂末 10 点主任务精度的标准差。
纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ccsf_verdict import quantities, rejection_rates   # noqa: E402
from pilot_a4 import invalid_reasons                   # noqa: E402

PLANT_MIN, MTA_MAX, NOT_DP, REPRO_DP, FRAC = 0.5, 0.02, 0.2, 0.5, 2 / 3
SAME_KEYS = ("n_rounds", "edge_rounds", "n_edges", "seed", "partition", "malicious_per_edge", "method")
# 臂 → (ccs_at, ccs_cluster)；off = 不开 CCS
SWITCHES = {"both": (True, True), "at": (True, False), "cluster": (False, True)}


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mta_sd(m: dict):
    vals = [a.get("pm_acc") for a in m.get("acc_rounds") or [] if a.get("pm_acc") is not None][-10:]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def gate(runs: dict) -> list:
    reasons, ref = [], None
    for name, m in runs.items():
        run = m.get("run") or {}
        reasons += [f"{name}: {r}" for r in invalid_reasons(m)]
        if m.get("exit_code") is None:
            reasons.append(f"{name}: exit_code 缺失")
        if not (m.get("n_malicious_participations") or 0) > 0:
            reasons.append(f"{name}: 攻击者一次都没参与")
        if (run.get("alignment") or {}).get("fedrep_bn_stats") != "shared":
            reasons.append(f"{name}: run.alignment.fedrep_bn_stats = "
                           f"{(run.get('alignment') or {}).get('fedrep_bn_stats')!r}，应为 'shared'")
        if name == "off":
            if run.get("ccs") not in (None, False):
                reasons.append(f"off: run.ccs = {run.get('ccs')!r}（对照不该开 CCS）")
        else:
            want = SWITCHES[name]
            got = (run.get("ccs_at"), run.get("ccs_cluster"))
            if run.get("ccs") is not True or got != want:
                reasons.append(f"{name}: run.ccs / ccs_at / ccs_cluster = {run.get('ccs')!r} / {got[0]!r} / {got[1]!r}，"
                               f"应为 True / {want[0]} / {want[1]}（开关没生效）")
            if want[1]:
                n = int(run.get("n_rounds") or 0) * int(run.get("n_edges") or 1) * int(run.get("edge_rounds") or 1)
                k = len(m.get("ccs_decisions") or [])
                if k != n:
                    reasons.append(f"{name}: [CCS] 行数 {k} ≠ n_rounds × n_edges × edge_rounds = {n}")
        key = {k: run.get(k) for k in SAME_KEYS}
        if ref is None:
            ref = (name, key)
        elif key != ref[1]:
            diff = {k: (ref[1][k], key[k]) for k in SAME_KEYS if ref[1][k] != key[k]}
            reasons.append(f"{name} 与 {ref[0]} 设置不同：{diff}")
    return reasons


def attribute(d_at: float, d_cl: float, d_both: float, nothing: bool, none_label: str) -> str:
    """nothing = 两个组件都开时也没有要归因的量（精度：ΔMTA ≤ 0.02；防御：ΔP < 0.2）。"""
    if nothing:
        return none_label
    a, c = d_at >= FRAC * d_both, d_cl >= FRAC * d_both
    if a and c:
        return "both_components"
    if a:
        return "adversarial_training"
    if c:
        return "clustering"
    return "only_together"


def standalone(dP: float, dMTA: float) -> str:
    if dP >= REPRO_DP and dMTA <= MTA_MAX:
        return "meets_all"
    if dP < NOT_DP:
        return "no_defense"
    return "partial"


def judge(off: dict, both: dict, at: dict, cluster: dict) -> dict:
    out = {"preregistered": "FINDINGS N-010（D-121）", "frac": _r(FRAC)}
    runs = {"off": off, "both": both, "at": at, "cluster": cluster}
    q = {k: quantities(m) for k, m in runs.items()}
    out["runs"] = {k: {"P": _r(v["P"]), "MTA": _r(v["MTA"]), "MTA_sd_last10": _r(_mta_sd(runs[k]))}
                   for k, v in q.items()}
    out["rejection"] = {k: {kk: (_r(vv) if isinstance(vv, float) else vv)
                            for kk, vv in rejection_rates(runs[k]).items()} for k in ("both", "cluster")}
    bad = gate(runs)
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    if any(q[k][x] is None for k in runs for x in ("P", "MTA")):
        return {**out, "verdict": "insufficient", "reasons": ["P / MTA 缺失（末 10 个评估点不够）"]}
    if q["off"]["P"] < PLANT_MIN:
        return {**out, "verdict": "invalid_no_attack",
                "reasons": [f"P(off) = {q['off']['P']:.4f} < {PLANT_MIN}：攻击没植入，归因无从谈起"]}
    dP = {k: q["off"]["P"] - q[k]["P"] for k in SWITCHES}
    dM = {k: q["off"]["MTA"] - q[k]["MTA"] for k in SWITCHES}
    out["delta_P"] = {k: _r(v) for k, v in dP.items()}
    out["delta_MTA"] = {k: _r(v) for k, v in dM.items()}
    out["standalone"] = {k: standalone(dP[k], dM[k]) for k in SWITCHES}
    out["cost"] = attribute(dM["at"], dM["cluster"], dM["both"], dM["both"] <= MTA_MAX, "no_cost")
    out["defense"] = attribute(dP["at"], dP["cluster"], dP["both"], dP["both"] < NOT_DP, "no_effect")
    return {**out, "verdict": f"cost={out['cost']};defense={out['defense']}"}


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 CCSD：CCS 拆分的归因（N-010）")
    ap.add_argument("--off", required=True)
    ap.add_argument("--both", required=True)
    ap.add_argument("--at-only", required=True)
    ap.add_argument("--cluster-only", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = judge(_load(a.off), _load(a.both), _load(a.at_only), _load(a.cluster_only))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "delta_P", "delta_MTA", "standalone", "reasons") if k in res},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
