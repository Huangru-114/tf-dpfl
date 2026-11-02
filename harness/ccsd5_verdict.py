"""
harness/ccsd5_verdict.py  —  阶段三 CCS 拆分实验的多种子判定（FINDINGS N-011，D-123）
**N-011 经用户确认、先 commit，再交补种子的作业**（同 D-088）。单种子判定（N-010）见 `ccsd_verdict.py`，本脚本逐种子复用它。

    python3 harness/ccsd5_verdict.py \\
        --off      <不加防御，种子 42–46 的 metrics.json …> \\
        --both     <两个都开，种子 42–44 …> \\
        --at-only  <只开对抗训练，种子 42–44 …> \\
        --cluster-only <只开聚类剔除，种子 42–46 …> \\
        --out experiments/defense/edge-native/analysis/ccsd5_verdict.json

所有 run 按 run.seed 配对（参数顺序无关）。量同 N-010：ΔP = P(off) − P(臂)，ΔMTA = MTA(off) − MTA(臂)（末 10 个评估点，同种子相减）。

闸：每个种子的那几个 run 过 `ccsd_verdict.gate`（崩溃 / 异常被吞 / 攻击者没参与 / 开关与臂不符 / 统计量不是 shared /
    设置不一致 / 开聚类的臂每轮一行聚类记录）；任何一个种子不过 → invalid。不加防御的 P < 0.5 的种子 → invalid_no_attack。

一、头条：只开聚类是否单独就够（要 5 个种子都有 off 与 cluster-only，否则 insufficient）
    均值 ΔP ≥ 0.5 且 均值 ΔMTA ≤ 0.02，且 ≥ 4 个种子各自满足这两条 → cluster_alone_sufficient；
    均值不满足（任一条）→ not_sufficient；均值满足但各自满足的种子 < 4 → mixed。
    另报 ΔP、ΔMTA 均值的 95% 区间（t 分布，自由度 = 种子数 − 1）。
二、归因（要 3 个种子四个臂都有，否则 insufficient）
    三个种子的均值套 N-010 的同一规则（`ccsd_verdict.attribute`，阈值 2/3）。
    对抗训练臂：某种子它的共享模型精度（末 10 点 em_acc）< 0.2（接近 10 类随机猜）→ 这个种子上它的攻击成功率降幅记 confounded_by_collapse，
    不进防御归因的均值；三个种子全被混淆 → 对抗训练臂在防御归因里视为「没达到阈值」，并标 at_defense_confounded。
    逐种子按 N-010 各打一个标签；三个种子的 (cost, defense) 标签都相同 → consistent，否则 seed_dependent。
纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ccsd_verdict as V                                # noqa: E402

N_HEAD, N_ATTR, MIN_EACH, COLLAPSE_EM = 5, 3, 4, 0.2
# t 分布 0.975 分位数（自由度 1–10）
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
ARMS = ("off", "both", "at", "cluster")


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _seed(m):
    return (m.get("run") or {}).get("seed")


def em_last10(m: dict):
    vals = [a.get("em_acc") for a in m.get("acc_rounds") or [] if a.get("em_acc") is not None][-10:]
    return statistics.mean(vals) if vals else None


def ci95(xs):
    if len(xs) < 2:
        return None
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    h = T975[len(xs) - 1] * sd / math.sqrt(len(xs))
    return [_r(m - h), _r(m + h)]


def _index(runs: dict) -> tuple[dict, list]:
    """{arm: {seed: metrics}}；同一臂种子重复 → 原因。"""
    by, bad = {}, []
    for arm, ms in runs.items():
        d = {}
        for m in ms:
            s = _seed(m)
            if s in d:
                bad.append(f"{arm}: 种子 {s} 重复")
            d[s] = m
        by[arm] = d
    return by, bad


def judge(off: list, both: list, at: list, cluster: list) -> dict:
    out = {"preregistered": "FINDINGS N-011（D-123）；单种子规则 N-010（ccsd_verdict.py）"}
    by, bad = _index({"off": off, "both": both, "at": at, "cluster": cluster})
    seeds = sorted({s for d in by.values() for s in d}, key=str)
    # 闸：每个种子把它有的臂一起过一遍（种子 45 / 46 只有 off 与 cluster）
    for s in seeds:
        sub = {arm: by[arm][s] for arm in ARMS if s in by[arm]}
        if "off" not in sub:
            bad.append(f"种子 {s}：没有不加防御的对照")
            continue
        bad += [f"种子 {s}：{r}" for r in V.gate(sub)]
    q = {arm: {s: V.quantities(m) for s, m in d.items()} for arm, d in by.items()}
    out["runs"] = {arm: {str(s): {"P": _r(v["P"]), "MTA": _r(v["MTA"]), "EM": _r(em_last10(by[arm][s]))}
                         for s, v in sorted(d.items(), key=lambda kv: str(kv[0]))} for arm, d in q.items()}
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    if any(v["P"] is None or v["MTA"] is None for d in q.values() for v in d.values()):
        return {**out, "verdict": "insufficient", "reasons": ["P / MTA 缺失（末 10 个评估点不够）"]}
    no_attack = [s for s, v in q["off"].items() if v["P"] < V.PLANT_MIN]
    if no_attack:
        return {**out, "verdict": "invalid_no_attack", "reasons": [f"不加防御的 P < {V.PLANT_MIN}：种子 {no_attack}"]}

    def deltas(arm, ss):
        return ({s: q["off"][s]["P"] - q[arm][s]["P"] for s in ss}, {s: q["off"][s]["MTA"] - q[arm][s]["MTA"] for s in ss})

    # ── 一、头条 ──
    hs = sorted(set(by["off"]) & set(by["cluster"]), key=str)
    dP, dM = deltas("cluster", hs)
    each = {str(s): (dP[s] >= V.REPRO_DP and dM[s] <= V.MTA_MAX) for s in hs}
    head = {"seeds": [str(s) for s in hs], "delta_P": {str(s): _r(v) for s, v in dP.items()},
            "delta_MTA": {str(s): _r(v) for s, v in dM.items()}, "each_meets": each}
    if len(hs) < N_HEAD:
        head["label"] = "insufficient"
    else:
        mP, mM = statistics.mean(dP.values()), statistics.mean(dM.values())
        head.update({"mean_delta_P": _r(mP), "mean_delta_MTA": _r(mM),
                     "ci95_delta_P": ci95(list(dP.values())), "ci95_delta_MTA": ci95(list(dM.values()))})
        if not (mP >= V.REPRO_DP and mM <= V.MTA_MAX):
            head["label"] = "not_sufficient"
        elif sum(each.values()) >= MIN_EACH:
            head["label"] = "cluster_alone_sufficient"
        else:
            head["label"] = "mixed"
    out["headline"] = head

    # ── 二、归因 ──
    ats = sorted(set.intersection(*(set(by[a]) for a in ARMS)), key=str)
    attr = {"seeds": [str(s) for s in ats]}
    if len(ats) < N_ATTR:
        attr["label"] = "insufficient"
    else:
        per_seed = {str(s): V.judge(by["off"][s], by["both"][s], by["at"][s], by["cluster"][s]) for s in ats}
        attr["per_seed"] = {s: {"cost": r.get("cost"), "defense": r.get("defense")} for s, r in per_seed.items()}
        d = {arm: deltas(arm, ats) for arm in ("both", "at", "cluster")}
        collapsed = [s for s in ats if (em_last10(by["at"][s]) or 0.0) < COLLAPSE_EM]
        attr["at_collapsed_seeds"] = [str(s) for s in collapsed]
        mean = {arm: (statistics.mean(d[arm][0].values()), statistics.mean(d[arm][1].values())) for arm in d}
        at_ok = [s for s in ats if s not in collapsed]
        at_dP = statistics.mean(d["at"][0][s] for s in at_ok) if at_ok else None
        attr["mean_delta_P"] = {"both": _r(mean["both"][0]), "at": _r(at_dP), "cluster": _r(mean["cluster"][0])}
        attr["mean_delta_MTA"] = {arm: _r(mean[arm][1]) for arm in mean}
        attr["at_defense_confounded"] = not at_ok
        attr["cost"] = V.attribute(mean["at"][1], mean["cluster"][1], mean["both"][1],
                                   mean["both"][1] <= V.MTA_MAX, "no_cost")
        attr["defense"] = V.attribute(at_dP if at_dP is not None else -math.inf, mean["cluster"][0], mean["both"][0],
                                      mean["both"][0] < V.NOT_DP, "no_effect")
        labels = {(v["cost"], v["defense"]) for v in attr["per_seed"].values()}
        attr["consistency"] = "consistent" if len(labels) == 1 else "seed_dependent"
        attr["label"] = f"cost={attr['cost']};defense={attr['defense']}"
    out["attribution"] = attr
    out["verdict"] = f"headline={head['label']};attribution={attr['label']}" + \
        (f";{attr['consistency']}" if "consistency" in attr else "")
    return out


def _load(paths):
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 CCS 拆分实验的多种子判定（N-011）")
    ap.add_argument("--off", nargs="+", required=True)
    ap.add_argument("--both", nargs="+", required=True)
    ap.add_argument("--at-only", nargs="+", required=True)
    ap.add_argument("--cluster-only", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = judge(_load(a.off), _load(a.both), _load(a.at_only), _load(a.cluster_only))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "reasons") if k in res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
