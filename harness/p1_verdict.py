"""
harness/p1_verdict.py  —  阶段三 P1（在线上传前加固探路）的判定（FINDINGS N-008 P1 段，D-101；SA1 的闸写法 D-108）
**先 commit，再打开任何 P1 数据**（同 D-088）。

    python3 harness/p1_verdict.py \\
        --arm experiments/defense/edge-native/results/P2/P1/P1__col-pre__s42.metrics.json \\
        --control experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json \\
        --out experiments/defense/edge-native/analysis/p1_verdict.json

量（N-008 共用定义；末 10 个评估点；Δ = 对照 − 防御臂，正 = 防御臂更低）：
  V 受害 edge E1–E3 良性端 fresh ASR 三 edge 均值；P 全部良性端池化 ASR；margin_v 受害 edge 的 margin 中位数三均值；
  margin 池化；MTA fresh pm_acc；jump(g) = 第 g 云轮云聚合后评估点 − 第 g−1 云轮末全量点（受害三均值）；
  jump_half = g = 2 … n/2 的均值（`g1_verdict.sawtooth`）。量的实现复用 `d0_power.run_quantities`（SNAP 的同一套）。
闸（任一不满足 → invalid）：
  通用闸：exit_code 0、client_failures 为空、攻击者参与过（n_malicious_participations > 0）—— 两个 run 都要；
  防御臂：run.edge_harden = 登记的配置、对照的 run.edge_harden 为 "none" 或缺失；
  `[Harden]` 行数 = 4 × n_rounds（A-pre）；
  **D-108 的「第一次加固之前」闸**：防御臂第 1 云轮 `[HardenAgg] unhardened` = 对照 `[Checksum] Round 1`
  （= 第一次加固之前的训练与对照逐位相同；第 1 云轮之后的 [Checksum] 本来就因加固而不同）。
标签（集中布点）：
  go_main_col  （ΔV ≥ 0.15 或 jump_half(臂) ≤ 0.5 × jump_half(对照)）且 ΔMTA ≤ 0.02；
               天花板规则：对照与臂的 V 都 ≥ 0.95 时，ΔV 那一项换成 Δmargin_v ≥ 2.0
  stop_col     ΔV < 0.05 且 jump_half 减少 < 20% 且 Δmargin_v < 1.0
  marginal     其余（只报告）
分散布点（`--placement distributed`，本次没有 run）：go_main_dist =（ΔP ≥ 0.10 或 Δmargin ≥ 2.0）且 ΔMTA ≤ 0.02；
  stop_dist = ΔP < 0.05 且 Δmargin < 1.0。
另报（不进判定）：B0、P、加固发散次数、加固墙钟、两条 run 逐云轮的 V 与 jump。
纯标准库（+ 已有 harness 模块），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from d0_power import run_quantities                  # noqa: E402
from g1_verdict import sawtooth, victim_full          # noqa: E402
from pilot_a4 import invalid_reasons                  # noqa: E402

ARM_CFG = "sau-e8-train-high"
DV_GO, JUMP_GO, MTA_MAX, CEIL, DMARGIN_GO = 0.15, 0.5, 0.02, 0.95, 2.0
DV_STOP, JUMP_STOP, DMARGIN_STOP = 0.05, 0.2, 1.0
DP_GO, DP_STOP = 0.10, 0.05


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _sub(a, b):
    return None if a is None or b is None else a - b


def gate(arm: dict, ctrl: dict, cfg: str = ARM_CFG, n_edges: int = 4) -> list:
    reasons = []
    for name, m in (("arm", arm), ("control", ctrl)):
        reasons += [f"{name}: {r}" for r in invalid_reasons(m)]
        if m.get("exit_code") is None:
            reasons.append(f"{name}: exit_code 缺失")
        if not (m.get("n_malicious_participations") or 0) > 0:
            reasons.append(f"{name}: 攻击者一次都没参与")
    ra, rc = arm.get("run") or {}, ctrl.get("run") or {}
    if ra.get("edge_harden") != cfg:
        reasons.append(f"arm: run.edge_harden={ra.get('edge_harden')!r} ≠ {cfg!r}（开关没生效 = 陷阱 #7 同类）")
    if rc.get("edge_harden") not in (None, "none"):
        reasons.append(f"control: run.edge_harden={rc.get('edge_harden')!r}（对照不该加固）")
    for k in ("malicious_per_edge", "n_rounds", "edge_rounds", "seed", "partition"):
        if ra.get(k) != rc.get(k):
            reasons.append(f"run.{k} 不一致：arm {ra.get(k)!r} vs control {rc.get(k)!r}")
    n = int(ra.get("n_rounds") or 0)
    h = arm.get("harden") or []
    if len(h) != n_edges * n:
        reasons.append(f"[Harden] 行数 {len(h)} ≠ {n_edges} × {n}")
    agg = {int(d["round"]): d["unhardened"] for d in arm.get("harden_agg") or []}
    ck = {int(d["round"]): d["global"] for d in ctrl.get("checksums") or []}
    if 1 not in agg or 1 not in ck:
        reasons.append("缺第 1 云轮的 [HardenAgg]（arm）或 [Checksum]（control）")
    elif agg[1] != ck[1]:
        reasons.append(f"第一次加固之前的训练与对照不同：arm unhardened {agg[1]} ≠ control Round 1 {ck[1]}（D-108）")
    return reasons


def quantities(m: dict) -> dict:
    q = run_quantities(m)
    st = sawtooth(m) if (m.get("run") or {}).get("malicious_per_edge") == [10, 0, 0, 0] else None
    q["jump_half"] = None if st is None else st["d_jump"]
    q["n_jump"] = None if st is None else st["n_g_used"]
    return q


def judge(arm: dict, ctrl: dict, placement: str = "collocated", cfg: str = ARM_CFG) -> dict:
    out = {"preregistered": "FINDINGS N-008 P1（D-101）；闸 D-108", "placement": placement, "arm_cfg": cfg}
    bad = gate(arm, ctrl, cfg)
    qa, qc = quantities(arm), quantities(ctrl)
    out["arm"], out["control"] = ({k: _r(v) for k, v in qa.items()}, {k: _r(v) for k, v in qc.items()})
    h = arm.get("harden") or []
    out["harden"] = {"n": len(h), "n_diverged": sum(1 for d in h if d.get("diverged_at") is not None),
                     "seconds_total": _r(math.fsum(float(d.get("seconds") or 0) for d in h), 1)}
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    d = {"dV": _sub(qc["V"], qa["V"]), "dP": _sub(qc["P"], qa["P"]), "dMTA": _sub(qc["MTA"], qa["MTA"]),
         "dmargin_v": _sub(qc["margin_v"], qa["margin_v"]), "dmargin": _sub(qc["margin_p50"], qa["margin_p50"]),
         "dB0": _sub(qc["B0"], qa["B0"])}
    jr = (None if qa["jump_half"] is None or qc["jump_half"] in (None, 0)
          else qa["jump_half"] / qc["jump_half"])
    d["jump_ratio"] = jr
    out["delta"] = {k: _r(v) for k, v in d.items()}
    need = ["dMTA"] + (["dV", "dmargin_v", "jump_ratio"] if placement == "collocated" else ["dP", "dmargin"])
    missing = [k for k in need if d.get(k) is None]
    if missing:
        return {**out, "verdict": "insufficient", "reasons": [f"量缺失：{missing}"]}
    mta_ok = d["dMTA"] <= MTA_MAX
    if placement == "collocated":
        ceiling = qa["V"] >= CEIL and qc["V"] >= CEIL
        asr_go = (d["dmargin_v"] >= DMARGIN_GO) if ceiling else (d["dV"] >= DV_GO)
        go = (asr_go or jr <= JUMP_GO) and mta_ok
        stop = d["dV"] < DV_STOP and (1.0 - jr) < JUMP_STOP and d["dmargin_v"] < DMARGIN_STOP
        out["ceiling_rule"] = ceiling
        verdict = "go_main_col" if go else ("stop_col" if stop else "marginal")
    else:
        go = (d["dP"] >= DP_GO or d["dmargin"] >= DMARGIN_GO) and mta_ok
        stop = d["dP"] < DP_STOP and d["dmargin"] < DMARGIN_STOP
        verdict = "go_main_dist" if go else ("stop_dist" if stop else "marginal")
    out["mta_ok"] = mta_ok
    out["verdict"] = verdict
    # 只报告：逐云轮的受害均值（全量点），看累积趋势
    fa, fc = victim_full(arm), victim_full(ctrl)
    out["victim_by_round"] = {str(g): [_r(fc.get(g)), _r(fa.get(g))] for g in sorted(set(fa) | set(fc))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 P1：在线上传前加固的判定（N-008）")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--placement", choices=("collocated", "distributed"), default="collocated")
    ap.add_argument("--cfg", default=ARM_CFG)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    arm = json.loads(Path(a.arm).read_text(encoding="utf-8"))
    ctrl = json.loads(Path(a.control).read_text(encoding="utf-8"))
    res = judge(arm, ctrl, a.placement, a.cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "delta", "reasons") if k in res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
