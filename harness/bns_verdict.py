"""
harness/bns_verdict.py  —  阶段三 BNS：BatchNorm 统计量共享本身是不是削弱了后门（FINDINGS N-009，D-120）
**先 commit，再交 BNS 的作业**（同 D-088）。

    python3 harness/bns_verdict.py \\
        --private experiments/defense/edge-native/legacy/CCSF-fedrep/CCSF__off__s42.metrics.json \\
                  experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnprivate-off__s43.metrics.json \\
                  experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnprivate-off__s44.metrics.json \\
        --shared  experiments/defense/edge-native/results/P2/CCSBOFF/CCSBOFF__fedrep-bnshared-off__s42.metrics.json \\
                  experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnshared-off__s43.metrics.json \\
                  experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnshared-off__s44.metrics.json \\
        --out experiments/defense/edge-native/analysis/bns_verdict.json

量：P = 全部良性端池化的新鲜个性化模型攻击成功率（`rounds[].local_benign_asr`，只数非目标类；末 10 个评估点，
    `d0_power.run_quantities`，与 CCSF 判定同一个定义）。两臂按 run.seed 配对：d_s = P(私有, s) − P(共享, s)。
闸（任一不满足 → invalid）：
  每个 run：exit_code 0（且不缺）、client_failures 为空、攻击者参与过；不开防御（run.defense 为 none / 缺、run.ccs 不为 true）；
  run.alignment.fedrep_bn_stats 与所在臂一致（private / shared —— 开关真的生效，陷阱 #7 同类）；
  六个 run 的 n_rounds / edge_rounds / n_edges / partition / malicious_per_edge / method 相同；
  两臂的种子集合相同、每臂内种子不重复。
标签（种子数 = 3 时才判，少于 3 → insufficient）：
  real_effect       每个种子 d_s > 0，且 d_s 的均值 ≥ 0.10
  not_established   其余
另报（不进判定）：各 run 的主任务精度（新鲜个性化模型，末 10 点）及其配对差、第 n_rounds 轮逐良性端的
  「< 0.1」与「> 0.9」端数、逐端攻击成功率与 yt_clean 的等级相关。
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
from d0_power import run_quantities                   # noqa: E402
from pilot_a4 import invalid_reasons                   # noqa: E402

N_SEEDS, MEAN_MIN = 3, 0.10
SAME_KEYS = ("n_rounds", "edge_rounds", "n_edges", "partition", "malicious_per_edge", "method")
ARMS = ("private", "shared")


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return statistics.correlation(_rank(a), _rank(b))


def per_client(m: dict) -> dict:
    """第 n_rounds 轮（client_final）逐个良性端：免疫 / 攻破端数、与 yt_clean 的等级相关。只报告。"""
    cf = m.get("client_final") or {}
    ids = cf.get("client_id") or []
    idx = [i for i in range(len(ids)) if not cf["malicious"][i] and cf["asr"][i] is not None]
    if not idx:
        return {"n_benign": 0}
    asr = [cf["asr"][i] for i in idx]
    yt = [cf["yt_clean"][i] for i in idx]
    pairs = [(a, y) for a, y in zip(asr, yt) if y is not None]
    return {"n_benign": len(idx), "lt_0.1": sum(a < 0.1 for a in asr), "gt_0.9": sum(a > 0.9 for a in asr),
            "median": _r(statistics.median(asr)),
            "spearman_yt_clean": _r(spearman([p[0] for p in pairs], [p[1] for p in pairs]), 2)}


def gate(runs: dict) -> list:
    """runs：{arm: [metrics, …]}。"""
    reasons, ref = [], None
    for arm, ms in runs.items():
        seeds = [(m.get("run") or {}).get("seed") for m in ms]
        if len(set(seeds)) != len(seeds):
            reasons.append(f"{arm}: 种子重复 {seeds}")
        for m in ms:
            run = m.get("run") or {}
            tag = f"{arm}/s{run.get('seed')}"
            reasons += [f"{tag}: {r}" for r in invalid_reasons(m)]
            if m.get("exit_code") is None:
                reasons.append(f"{tag}: exit_code 缺失")
            if not (m.get("n_malicious_participations") or 0) > 0:
                reasons.append(f"{tag}: 攻击者一次都没参与")
            if run.get("defense") not in (None, "none") or run.get("ccs") is True:
                reasons.append(f"{tag}: 开了防御（defense={run.get('defense')!r}, ccs={run.get('ccs')!r}）")
            bn = (run.get("alignment") or {}).get("fedrep_bn_stats")
            if bn != arm:
                reasons.append(f"{tag}: run.alignment.fedrep_bn_stats = {bn!r}，应为 {arm!r}（开关没生效）")
            key = {k: run.get(k) for k in SAME_KEYS}
            if ref is None:
                ref = (tag, key)
            elif key != ref[1]:
                diff = {k: (ref[1][k], key[k]) for k in SAME_KEYS if ref[1][k] != key[k]}
                reasons.append(f"{tag} 与 {ref[0]} 设置不同：{diff}")
    sp = {(m.get("run") or {}).get("seed") for m in runs["private"]}
    ss = {(m.get("run") or {}).get("seed") for m in runs["shared"]}
    if sp != ss:
        reasons.append(f"两臂种子集合不同：private {sorted(sp, key=str)} vs shared {sorted(ss, key=str)}")
    return reasons


def judge(private: list, shared: list) -> dict:
    out = {"preregistered": "FINDINGS N-009（D-120）"}
    runs = {"private": private, "shared": shared}
    by = {arm: {(m.get("run") or {}).get("seed"): m for m in ms} for arm, ms in runs.items()}
    q = {arm: {s: run_quantities(m) for s, m in d.items()} for arm, d in by.items()}
    out["runs"] = {arm: {str(s): {"P": _r(v["P"]), "MTA": _r(v["MTA"]), "clients": per_client(by[arm][s])}
                         for s, v in sorted(d.items(), key=lambda kv: str(kv[0]))} for arm, d in q.items()}
    bad = gate(runs)
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    seeds = sorted(by["private"])
    missing = [f"{arm}/s{s}" for arm in ARMS for s in seeds if q[arm][s]["P"] is None]
    if missing or len(seeds) < N_SEEDS:
        why = [f"P 缺失（末 10 个评估点不够）：{missing}"] if missing else []
        why += [f"只有 {len(seeds)} 个种子，要 {N_SEEDS} 个"] if len(seeds) < N_SEEDS else []
        return {**out, "verdict": "insufficient", "reasons": why}
    d = {s: q["private"][s]["P"] - q["shared"][s]["P"] for s in seeds}
    dm = {s: (None if q["private"][s]["MTA"] is None or q["shared"][s]["MTA"] is None
              else q["private"][s]["MTA"] - q["shared"][s]["MTA"]) for s in seeds}
    mean = statistics.mean(d.values())
    out["delta_P"] = {str(s): _r(v) for s, v in d.items()}
    out["delta_P_mean"] = _r(mean)
    out["delta_MTA"] = {str(s): _r(v) for s, v in dm.items()}
    ok = all(v > 0 for v in d.values()) and mean >= MEAN_MIN
    return {**out, "verdict": "real_effect" if ok else "not_established"}


def _load(paths):
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 BNS：统计量共享本身是不是削弱了后门（N-009）")
    ap.add_argument("--private", nargs="+", required=True)
    ap.add_argument("--shared", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = judge(_load(a.private), _load(a.shared))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "delta_P", "delta_P_mean", "reasons") if k in res},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
