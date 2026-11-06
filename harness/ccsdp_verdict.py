"""
harness/ccsdp_verdict.py  —  阶段三：缓存池能不能修好「只开聚类」失守的种子（FINDINGS N-012，D-126）
**N-012 经用户确认、先 commit，再交 CCSDP 的作业**（同 D-088）。

    python3 harness/ccsdp_verdict.py \\
        --off     <不加防御，种子 43 / 46 的 metrics.json …> \\
        --cluster <只开聚类（pool = round），同种子 …> \\
        --cache   <只开聚类 + 缓存池（pool = edge_cache），同种子 …> \\
        --out experiments/defense/edge-native/analysis/ccsdp_verdict.json

所有 run 按 run.seed 配对。量同 N-010：ΔP = P(off) − P(臂)，ΔMTA = MTA(off) − MTA(臂)（末 10 个评估点，`ccsf_verdict.quantities`）。
闸（任一不满足 → invalid）：每个 run exit_code 0（且不缺）、client_failures 为空、攻击者参与过、统计量 shared；
  off 不开 CCS；cluster 与 cache 都是 ccs_at = false、ccs_cluster = true，pool 分别为 round / edge_cache；
  所有 run 的 n_rounds / edge_rounds / n_edges / partition / malicious_per_edge / method 相同；开聚类的 run 每轮一行 [CCS]；
  三臂的种子集合相同、至少 2 个种子。不加防御的 P < 0.5 → invalid_no_attack。
标签（只看缓存池那一臂）：每个种子都满足 ΔP ≥ 0.5 且 ΔMTA ≤ 0.02 → fixes_failures；都不满足 → no_fix；其余 partial。
另报（不进判定）：相对同种子只开聚类的变化、剔除命中率 / 误伤率、放进来的攻击者上传数、聚类状态计数、共享模型精度的稳定性。
纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ccs_leak_timeline import leak, stability          # noqa: E402
from ccsf_verdict import quantities, rejection_rates   # noqa: E402
from pilot_a4 import invalid_reasons                   # noqa: E402

PLANT_MIN, REPRO_DP, MTA_MAX, MIN_SEEDS = 0.5, 0.5, 0.02, 2
SAME_KEYS = ("n_rounds", "edge_rounds", "n_edges", "partition", "malicious_per_edge", "method")
POOL = {"cluster": "round", "cache": "edge_cache"}


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _seed(m):
    return (m.get("run") or {}).get("seed")


def gate(by: dict) -> list:
    reasons, ref = [], None
    seeds = {arm: set(d) for arm, d in by.items()}
    if len({frozenset(v) for v in seeds.values()}) != 1:
        reasons.append(f"三臂种子集合不同：{ {a: sorted(v, key=str) for a, v in seeds.items()} }")
    if len(seeds["cache"]) < MIN_SEEDS:
        reasons.append(f"缓存池只有 {len(seeds['cache'])} 个种子，至少要 {MIN_SEEDS} 个")
    for arm, d in by.items():
        for s, m in d.items():
            run, tag = m.get("run") or {}, f"{arm}/s{s}"
            reasons += [f"{tag}: {r}" for r in invalid_reasons(m)]
            if m.get("exit_code") is None:
                reasons.append(f"{tag}: exit_code 缺失")
            if not (m.get("n_malicious_participations") or 0) > 0:
                reasons.append(f"{tag}: 攻击者一次都没参与")
            if (run.get("alignment") or {}).get("fedrep_bn_stats") != "shared":
                reasons.append(f"{tag}: 统计量不是 shared")
            if arm == "off":
                if run.get("ccs") not in (None, False):
                    reasons.append(f"{tag}: 对照不该开 CCS")
            else:
                got = (run.get("ccs"), run.get("ccs_at"), run.get("ccs_cluster"), run.get("ccs_pool"))
                want = (True, False, True, POOL[arm])
                if got != want:
                    reasons.append(f"{tag}: run.ccs / ccs_at / ccs_cluster / ccs_pool = {got}，应为 {want}（开关没生效）")
                n = int(run.get("n_rounds") or 0) * int(run.get("n_edges") or 1) * int(run.get("edge_rounds") or 1)
                if len(m.get("ccs_decisions") or []) != n:
                    reasons.append(f"{tag}: [CCS] 行数 {len(m.get('ccs_decisions') or [])} ≠ {n}")
            key = {k: run.get(k) for k in SAME_KEYS}
            if ref is None:
                ref = (tag, key)
            elif key != ref[1]:
                reasons.append(f"{tag} 与 {ref[0]} 设置不同：{ {k: (ref[1][k], key[k]) for k in SAME_KEYS if ref[1][k] != key[k]} }")
    return reasons


def judge(off: list, cluster: list, cache: list) -> dict:
    out = {"preregistered": "FINDINGS N-012（D-126）"}
    by = {"off": {}, "cluster": {}, "cache": {}}
    dup = []
    for arm, ms in (("off", off), ("cluster", cluster), ("cache", cache)):
        for m in ms:
            if _seed(m) in by[arm]:
                dup.append(f"{arm}: 种子 {_seed(m)} 重复")
            by[arm][_seed(m)] = m
    q = {arm: {s: quantities(m) for s, m in d.items()} for arm, d in by.items()}
    out["runs"] = {arm: {str(s): {"P": _r(v["P"]), "MTA": _r(v["MTA"]), "stability": stability(by[arm][s])}
                         for s, v in sorted(d.items(), key=lambda kv: str(kv[0]))} for arm, d in q.items()}
    for arm in ("cluster", "cache"):
        for s, m in by[arm].items():
            lk = leak(m) or {}
            rr = rejection_rates(m)
            out["runs"][arm][str(s)].update({
                "rejection": {"tpr": _r(rr["tpr"]), "fpr": _r(rr["fpr"]), "status": rr["status"]},
                "attacker_admitted": lk.get("admitted"), "attacker_selected": lk.get("selected")})
    bad = dup + gate(by)
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    if any(v["P"] is None or v["MTA"] is None for d in q.values() for v in d.values()):
        return {**out, "verdict": "insufficient", "reasons": ["P / MTA 缺失（末 10 个评估点不够）"]}
    no_attack = [s for s, v in q["off"].items() if v["P"] < PLANT_MIN]
    if no_attack:
        return {**out, "verdict": "invalid_no_attack", "reasons": [f"不加防御的 P < {PLANT_MIN}：种子 {no_attack}"]}
    seeds = sorted(by["cache"], key=str)
    d = {arm: {s: (q["off"][s]["P"] - q[arm][s]["P"], q["off"][s]["MTA"] - q[arm][s]["MTA"]) for s in seeds}
         for arm in ("cluster", "cache")}
    out["delta"] = {arm: {str(s): {"dP": _r(v[0]), "dMTA": _r(v[1])} for s, v in dd.items()} for arm, dd in d.items()}
    meets = {s: (d["cache"][s][0] >= REPRO_DP and d["cache"][s][1] <= MTA_MAX) for s in seeds}
    out["cache_meets"] = {str(s): v for s, v in meets.items()}
    n = sum(meets.values())
    out["verdict"] = "fixes_failures" if n == len(seeds) else ("no_fix" if n == 0 else "partial")
    return out


def _load(paths):
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三：缓存池能不能修好只开聚类失守的种子（N-012）")
    ap.add_argument("--off", nargs="+", required=True)
    ap.add_argument("--cluster", nargs="+", required=True)
    ap.add_argument("--cache", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = judge(_load(a.off), _load(a.cluster), _load(a.cache))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "delta", "reasons") if k in res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
