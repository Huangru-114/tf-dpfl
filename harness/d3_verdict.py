"""
harness/d3_verdict.py  —  阶段三 D3 离线探针的判定（PREREG-v2 §五 / 预注册 N-015；实现取舍 D-138）

    python3 harness/d3_verdict.py                 # 读 experiments/defense/edge-native/analysis/d3/*.json → analysis/d3_verdict.json

**时序**：规则（`experiments/thesis/PREREG-v2.md` @ 25163a5）→ 实现取舍 D-138 → 本脚本（写于任何 D3 数据之前，用合成数据测通）→ 交作业 → 数据回来只运行。

输入：`fedavg/analysis/d3_probes.py` 写的 JSON（每个 run 一个，schema 1）；P-c 的后门强度 S 与 P-a 的底线直接读已入库的 metrics.json。

P-a 重训最后一块（SNAP 集中 s42–44 × 第 15 / 60 轮，攻击者所在边缘 E0）
  A = E0 良性端 ASR 的等权均值（主列语义：ξ 在当前攻击者模型上重求；各臂同一批探针图、同一个随机起点）；
  f = (A_none − A_stage4) / (A_none − 底线_E0)；none = 不重训、只精确重估统计量；底线_E0 = G0-C1 同种子 E0 良性端第 51–60 轮均值。
  闸（任一不过 → invalid）：重现闸；none / stage4 都没发散；E0 良性端干净精度 none − stage4 ≤ 0.05。
  三个种子 × 两个快照轮共 6 个 f（D-138：预注册写「三个种子都」、对象写两个轮 → 6 个值都满足才贴标签）：
    都 ≥ 0.5 → mainly_last_block（后门主要在最后一块）；都 < 0.25 → mainly_earlier_layers（主要在前面的层，6.1 要改写）；其余 inconclusive。
  另报（不进判定）：逐轮标签、对照臂（只训 stem–stage3 / 全部主干）的 f、固定 ξ 的 f、margin。

P-b 受害模型上白盒求扰动（TBS k1-snap s42–44 第 60 轮；SNAP 集中作参照）
  V_列 = 受害边缘（E1–E3）各自良性端 ASR 均值、再对边缘平均；白盒列 = ξ 在受害端自己的 fresh 个性化模型上求，主列 = 在攻击者模型上求（配对：同 u、同 δ）。
  闸：重现闸。
    三个种子都 V_白盒 − V_主列 ≥ 0.30 且 V_白盒 ≥ 0.5 → trigger_not_transferable（触发器迁移不过去，后门功能在受害模型里存在）；
    三个种子都 |V_白盒 − V_主列| < 0.10 → not_transmitted（后门没传过来）；其余 inconclusive。
  另报：只 ξ（受害者模型 / 攻击者模型上求）、只 δ、无触发器；SNAP 参照的同一组数；run 记录的主列。

P-c 统计量偏移 vs 后门强度（9 个 run × 第 6 / 15 / 60 轮 × 4 边缘 = 108 个边缘模型）
  S = metrics.json 的 per_edge_detail_rounds[t][e].margin_p50；Δ = JSON 的 kl_all（KL(加触发器 ‖ 干净)，各 BN 层均值）。
  闸：27 个 (run, 轮) 的重现闸全过、108 个 Δ 都是有限数。
    合并 Spearman ≥ 0.5、三个种子内各自 > 0、置换 p < 0.01 → feedback_moderate（反馈循环从「弱」升为「中」）；
    合并 Spearman ≤ 0.2 → not_supported；其余 inconclusive。
  置换（D-138 ③）：108 个单位自由置换 S，单侧（ρ_置换 ≥ ρ_观测），p = (1 + 计数) / (1 + n)。
  另报（不进判定）：每轮内部的 Spearman、按轮分层置换的 p、各阶段 KL 的 Spearman、去掉目标类的 Δ 的 Spearman。

缺 JSON / 缺轮 → insufficient（不能当成「没通过」）；有数据但不合格（闸不过）→ invalid，优先于 insufficient。阈值比较带 1e-9 容差（同 D-133）。
纯标准库（+ numpy 的 `fedavg/analysis/d3_spec.py` 读清单与 metrics，同 p0_verdict 的先例），不 import TF。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "fedavg"))
from analysis import d3_spec as S            # noqa: E402
from decay_verdict import window_mean        # noqa: E402

EN = ROOT / "experiments/defense/edge-native"
D3_DIR = EN / "analysis/d3"
CONFIG = EN / "d3_configs.yaml"
RESULTS = EN / "results/P2"
G0_RESULTS = ROOT / "experiments/attack/hfl-mechanism/results/P2/G0"
OUT = EN / "analysis/d3_verdict.json"
PREREG = "N-015（experiments/thesis/PREREG-v2.md @ 25163a5）§五；实现取舍 D-138；判定脚本写于规则之后、数据之前"

SEEDS = (42, 43, 44)
TOL = 1e-9
ATTACKER_EDGE = 0
VICTIMS = (1, 2, 3)
FLOOR_WIN = (51, 60)

# ── 预注册阈值（PREREG-v2 §五；用户 11-22 全部照草案，D-135 ③）──
PA_MAIN = 0.5               # 都 f ≥ 0.5 → 主要在最后一块
PA_EARLY = 0.25             # 都 f < 0.25 → 主要在前面的层
PA_ACC_GATE = 0.05          # E0 良性端干净精度降幅 ≤ 0.05
PB_GAP = 0.30               # 都 白盒 − 主列 ≥ 0.30
PB_WB = 0.5                 # 且 白盒 ≥ 0.5
PB_SAME = 0.10              # 都 |白盒 − 主列| < 0.10
PC_RHO = 0.5                # 合并 Spearman ≥ 0.5
PC_P = 0.01                 # 置换 p < 0.01
PC_NONE = 0.2               # 合并 Spearman ≤ 0.2 → 不支持


def _ge(a, b):
    return a >= b - TOL


def _lt(a, b):
    return a < b - TOL


def _le(a, b):
    return a <= b + TOL


def _gt(a, b):
    return a > b + TOL


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _mean(xs):
    v = [float(x) for x in xs if x is not None]
    return sum(v) / len(v) if v else None


# ══════════════════════════════════════════════════════════════════════════
# 读文件
# ══════════════════════════════════════════════════════════════════════════

def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_d3(d3_dir: Path, run_id: str):
    return _load_json(Path(d3_dir) / f"{run_id}.json")


def load_metrics(results: Path, run_id: str):
    return _load_json(Path(results) / S.group_of(run_id) / f"{run_id}.metrics.json")


def floor_e0(g0_results: Path, seed: int):
    """底线_E0 = G0-C1（不投毒、同配置）同种子 E0 良性端第 51–60 轮均值（0.0974 / 0.0633 / 0.0731）。"""
    m = _load_json(Path(g0_results) / f"G0__C1__s{int(seed)}.metrics.json")
    return None if m is None else window_mean(m, ATTACKER_EDGE, FLOOR_WIN)


def _round(j, t):
    return ((j or {}).get("rounds") or {}).get(str(int(t)))


# ══════════════════════════════════════════════════════════════════════════
# 统计：Spearman（平均秩）+ 置换
# ══════════════════════════════════════════════════════════════════════════

def ranks(xs) -> list:
    """平均秩（并列取平均），1 起。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _centered(v):
    m = sum(v) / len(v)
    c = [x - m for x in v]
    return c, math.sqrt(sum(x * x for x in c))


def pearson(a, b):
    if len(a) != len(b) or len(a) < 3:
        return None
    ca, na = _centered(a)
    cb, nb = _centered(b)
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(ca, cb)) / (na * nb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def perm_p(s, d, *, n: int, seed: int, groups=None):
    """
    单侧置换检验：打乱 S 的秩（groups 给出时只在同组内打乱 = 按轮分层），ρ_置换 ≥ ρ_观测 的比例；p = (1 + 计数) / (1 + n)。
    返回 (ρ_观测, p)；ρ 无定义 → (None, None)。
    """
    rs, rd = ranks(s), ranks(d)
    obs = pearson(rs, rd)
    if obs is None:
        return None, None
    cs, ns = _centered(rs)
    cd, nd = _centered(rd)
    rng = random.Random(int(seed))
    idx_groups = ([list(range(len(s)))] if groups is None else
                  [[i for i, g in enumerate(groups) if g == key] for key in dict.fromkeys(groups)])
    cnt = 0
    perm = list(cs)
    for _ in range(int(n)):
        for idx in idx_groups:
            vals = [cs[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                perm[i] = v
        rho = sum(x * y for x, y in zip(perm, cd)) / (ns * nd)
        if rho >= obs - 1e-12:
            cnt += 1
    return obs, (1 + cnt) / (1 + int(n))


# ══════════════════════════════════════════════════════════════════════════
# 公共：JSON 的有效性
# ══════════════════════════════════════════════════════════════════════════

def json_problems(j, want_sha: str, run_id: str) -> list:
    if j is None:
        return []
    errs = []
    if j.get("schema") != S.SCHEMA:
        errs.append(f"{run_id}: schema = {j.get('schema')!r}")
    if j.get("run_id") != run_id:
        errs.append(f"{run_id}: JSON 里的 run_id = {j.get('run_id')!r}")
    if j.get("config_sha") != want_sha:
        errs.append(f"{run_id}: 参数清单 sha {j.get('config_sha')!r} ≠ 当前 {want_sha!r}（清单在数据之后被改过，或 JSON 是旧的）")
    return errs


# ══════════════════════════════════════════════════════════════════════════
# P-a
# ══════════════════════════════════════════════════════════════════════════

def _f(a_before, a_after, floor):
    if None in (a_before, a_after, floor):
        return None
    den = float(a_before) - float(floor)
    if den <= TOL:
        return None                         # ASR_前 已在底线上：没有可清除的后门，f 无定义
    return (float(a_before) - float(a_after)) / den


def _pa_label(fs):
    if any(v is None for v in fs) or not fs:
        return "insufficient"
    if all(_ge(v, PA_MAIN) for v in fs):
        return "mainly_last_block"
    if all(_lt(v, PA_EARLY) for v in fs):
        return "mainly_earlier_layers"
    return "inconclusive"


def judge_pa(cfg, d3s: dict, floors: dict) -> dict:
    sec = cfg["pa"]
    label_arm = sec["label_arm"]
    controls = [a for a in sec["arms"] if a != label_arm]
    rounds = [int(t) for t in sec["rounds"]]
    rows, missing, invalid = [], [], []
    for run in sec["runs"]:
        seed = S.seed_of(run)
        j = d3s.get(run)
        for t in rounds:
            R = _round(j, t)
            pa = (R or {}).get("pa")
            if pa is None:
                missing.append(f"{run} 第 {t} 轮")
                continue
            arms = pa.get("arms") or {}
            none, lab = arms.get("none") or {}, arms.get(label_arm) or {}
            reasons = []
            if not ((R.get("reproduce") or {}).get("pass")):
                reasons.append(f"{run} 第 {t} 轮重现闸不过：{(R.get('reproduce') or {}).get('reasons')}")
            for name, arm in (("none", none), (label_arm, lab)):
                if arm.get("diverged"):
                    reasons.append(f"{run} 第 {t} 轮 {name} 臂发散")
            a0, a1 = (none.get("main") or {}).get("asr"), (lab.get("main") or {}).get("asr")
            c0, c1 = (none.get("main") or {}).get("acc"), (lab.get("main") or {}).get("acc")
            drop = None if None in (c0, c1) else float(c0) - float(c1)
            if drop is not None and not _le(drop, PA_ACC_GATE):
                reasons.append(f"{run} 第 {t} 轮 E0 良性端干净精度降 {drop:.4f} > {PA_ACC_GATE}（可能是破坏而不是清除）")
            invalid.extend(reasons)
            fl = floors.get(seed)
            row = {"run": run, "seed": seed, "round": t, "floor_e0": _r(fl),
                   "asr_none": _r(a0), f"asr_{label_arm}": _r(a1), "acc_drop": _r(drop),
                   "f": _r(_f(a0, a1, fl)),
                   "f_fixed_xi": _r(_f((none.get("fixed") or {}).get("asr"), (lab.get("fixed") or {}).get("asr"), fl)),
                   "margin_none": _r((none.get("main") or {}).get("margin_p50")),
                   f"margin_{label_arm}": _r((lab.get("main") or {}).get("margin_p50")),
                   "controls": {}, "gate_ok": not reasons}
            for cname in controls:
                arm = arms.get(cname) or {}
                row["controls"][cname] = {
                    "diverged": bool(arm.get("diverged")),
                    "asr": _r((arm.get("main") or {}).get("asr")),
                    "f": (None if arm.get("diverged") else _r(_f(a0, (arm.get("main") or {}).get("asr"), fl))),
                    "acc_drop": _r(None if None in (c0, (arm.get("main") or {}).get("acc"))
                                   else float(c0) - float((arm.get("main") or {}).get("acc"))),
                    "margin": _r((arm.get("main") or {}).get("margin_p50"))}
            rows.append(row)
    out = {"rule": f"6 个 f（3 种子 × 第 {rounds} 轮）都 ≥ {PA_MAIN} → mainly_last_block；都 < {PA_EARLY} → mainly_earlier_layers；"
                   f"闸：重现、未发散、干净精度降 ≤ {PA_ACC_GATE}",
           "label_arm": label_arm, "rows": rows,
           "per_round": {str(t): _pa_label([r["f"] for r in rows if r["round"] == t]) for t in rounds}}
    if invalid:                                   # 有数据但不合格 → invalid 优先于缺数据（坏数据必须暴露出来）
        out.update(label="invalid", reasons=invalid)
    elif missing or len(rows) < len(sec["runs"]) * len(rounds):
        out.update(label="insufficient", reasons=missing or ["行数不足"])
    else:
        lab = _pa_label([r["f"] for r in rows])
        out.update(label=lab, reasons=[] if lab != "insufficient" else ["有 f 无定义（ASR_前 已在底线上）"])
    return out


# ══════════════════════════════════════════════════════════════════════════
# P-b
# ══════════════════════════════════════════════════════════════════════════

PB_COLUMNS = ("main", "whitebox", "xi_only_victim", "xi_only_attacker", "delta_only", "clean")


def _pb_row(run, j, t):
    R = _round(j, t)
    pb = (R or {}).get("pb")
    if pb is None:
        return None, [f"{run} 第 {t} 轮"], []
    V = pb.get("V") or {}
    wb, mn = V.get("whitebox"), V.get("main")
    reasons = []
    if not ((R.get("reproduce") or {}).get("pass")):
        reasons.append(f"{run} 第 {t} 轮重现闸不过：{(R.get('reproduce') or {}).get('reasons')}")
    row = {"run": run, "seed": S.seed_of(run), "round": t,
           "V": {c: _r(V.get(c)) for c in PB_COLUMNS},
           "gap": _r(None if None in (wb, mn) else float(wb) - float(mn)),
           "recorded_V": _r(pb.get("recorded_V")), "gate_ok": not reasons}
    return row, [], reasons


def judge_pb(cfg, d3s: dict) -> dict:
    sec = cfg["pb"]
    t = int(sec["rounds"][0])
    rows, missing, invalid = [], [], []
    for run in sec["runs"]:
        row, miss, bad = _pb_row(run, d3s.get(run), t)
        missing += miss
        invalid += bad
        if row is not None:
            rows.append(row)
    ref = []
    for run in sec.get("reference_runs") or []:
        row, _, _ = _pb_row(run, d3s.get(run), t)
        if row is not None:
            ref.append(row)
    out = {"rule": f"3 个种子都 V_白盒 − V_主列 ≥ {PB_GAP} 且 V_白盒 ≥ {PB_WB} → trigger_not_transferable；"
                   f"都 |V_白盒 − V_主列| < {PB_SAME} → not_transmitted；闸：重现",
           "round": t, "rows": rows, "reference": ref}
    if invalid:
        out.update(label="invalid", reasons=invalid)
        return out
    if missing or len(rows) < len(sec["runs"]):
        out.update(label="insufficient", reasons=missing or ["行数不足"])
        return out
    gaps = [r["gap"] for r in rows]
    wbs = [r["V"]["whitebox"] for r in rows]
    if any(v is None for v in gaps + wbs):
        out.update(label="insufficient", reasons=["有 V 无定义"])
    elif all(_ge(g, PB_GAP) and _ge(w, PB_WB) for g, w in zip(gaps, wbs)):
        out.update(label="trigger_not_transferable", reasons=[])
    elif all(_lt(abs(g), PB_SAME) for g in gaps):
        out.update(label="not_transmitted", reasons=[])
    else:
        out.update(label="inconclusive", reasons=[])
    return out


# ══════════════════════════════════════════════════════════════════════════
# P-c
# ══════════════════════════════════════════════════════════════════════════

def pc_units(cfg, d3s: dict, metrics: dict):
    """→ (单位列表, 缺失, 无效)。单位 = {run, seed, round, edge, S, delta, delta_nontarget, kl_stage…}。"""
    sec = cfg["pc"]
    units, missing, invalid = [], [], []
    for run in sec["runs"]:
        j, m = d3s.get(run), metrics.get(run)
        if m is None:
            missing.append(f"{run} 的 metrics.json")
            continue
        for t in sec["rounds"]:
            R = _round(j, t)
            pc = (R or {}).get("pc")
            if pc is None:
                missing.append(f"{run} 第 {t} 轮")
                continue
            if not ((R.get("reproduce") or {}).get("pass")):
                invalid.append(f"{run} 第 {t} 轮重现闸不过：{(R.get('reproduce') or {}).get('reasons')}")
            rec = S.recorded_edges(m, t)
            for e in range(4):
                ed = (pc.get("edges") or {}).get(str(e))
                s_val = (rec.get(e) or {}).get(S.MARGIN_COL)
                if ed is None or s_val is None:
                    missing.append(f"{run} 第 {t} 轮 edge {e}")
                    continue
                d = ed.get("kl_all")
                if d is None or not math.isfinite(float(d)):
                    invalid.append(f"{run} 第 {t} 轮 edge {e} 的 Δ 不是有限数：{d!r}")
                    continue
                u = {"run": run, "seed": S.seed_of(run), "round": int(t), "edge": e,
                     "S": float(s_val), "delta": float(d), "delta_nontarget": ed.get("kl_all_nontarget")}
                for k, v in ed.items():
                    if k.startswith("kl_stage") or k == "kl_stem":
                        u[k] = v
                units.append(u)
    return units, missing, invalid


def judge_pc(cfg, d3s: dict, metrics: dict) -> dict:
    sec = cfg["pc"]
    perm = sec["permutation"]
    n_want = len(sec["runs"]) * len(sec["rounds"]) * 4
    units, missing, invalid = pc_units(cfg, d3s, metrics)
    out = {"rule": f"合并 Spearman ≥ {PC_RHO}、三个种子内各自 > 0、自由置换单侧 p < {PC_P} → feedback_moderate；"
                   f"合并 ≤ {PC_NONE} → not_supported；闸：重现、Δ 有限",
           "n_units": len(units), "n_expected": n_want,
           "permutation": {"kind": perm["kind"], "n": int(perm["n"]), "seed": int(perm["seed"]), "sided": perm["sided"]},
           "units": [{k: (_r(v, 6) if isinstance(v, float) else v) for k, v in u.items()} for u in units]}
    if invalid:
        out.update(label="invalid", reasons=invalid)
        return out
    if missing or len(units) < n_want:
        out.update(label="insufficient", reasons=missing or ["单位数不足"])
        return out
    s = [u["S"] for u in units]
    d = [u["delta"] for u in units]
    rho, p = perm_p(s, d, n=perm["n"], seed=perm["seed"])
    per_seed = {}
    for seed in SEEDS:
        sub = [u for u in units if u["seed"] == seed]
        per_seed[str(seed)] = _r(spearman([u["S"] for u in sub], [u["delta"] for u in sub]))
    out.update(pooled_spearman=_r(rho), perm_p=_r(p, 6), per_seed_spearman=per_seed)
    # ── 描述性（不进判定）──
    rounds = [u["round"] for u in units]
    rho_s, p_s = perm_p(s, d, n=perm["n"], seed=perm["seed"], groups=rounds)
    desc = {"within_round_spearman": {str(t): _r(spearman([u["S"] for u in units if u["round"] == t],
                                                           [u["delta"] for u in units if u["round"] == t]))
                                      for t in sec["rounds"]},
            "round_stratified_perm_p": _r(p_s, 6),
            "stage_spearman": {}, "nontarget_spearman": None}
    for k in ("kl_stem", "kl_stage1", "kl_stage2", "kl_stage3", "kl_stage4"):
        vals = [u.get(k) for u in units]
        if all(v is not None for v in vals):
            desc["stage_spearman"][k] = _r(spearman(s, vals))
    nt = [u.get("delta_nontarget") for u in units]
    if all(v is not None for v in nt):
        desc["nontarget_spearman"] = _r(spearman(s, nt))
    out["descriptive"] = desc
    if rho is None or any(v is None for v in per_seed.values()):
        out.update(label="insufficient", reasons=["Spearman 无定义（S 或 Δ 全部相同）"])
    elif _ge(rho, PC_RHO) and all(_gt(v, 0.0) for v in per_seed.values()) and _lt(p, PC_P):
        out.update(label="feedback_moderate", reasons=[])
    elif _le(rho, PC_NONE):
        out.update(label="not_supported", reasons=[])
    else:
        out.update(label="inconclusive", reasons=[])
    return out


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════

def run_all(*, d3_dir=D3_DIR, config=CONFIG, results=RESULTS, g0_results=G0_RESULTS) -> dict:
    cfg = S.load_config(config)
    sha = S.content_sha(cfg)
    runs = S.all_runs(cfg)
    d3s = {r: load_d3(d3_dir, r) for r in runs}
    problems = [p for r in runs for p in json_problems(d3s[r], sha, r)]
    metrics = {r: load_metrics(results, r) for r in cfg["pc"]["runs"]}
    floors = {s: floor_e0(g0_results, s) for s in SEEDS}
    res = {"preregistered": PREREG, "config_sha": sha,
           "thresholds": {"pa": {"main": PA_MAIN, "early": PA_EARLY, "acc_gate": PA_ACC_GATE},
                          "pb": {"gap": PB_GAP, "whitebox": PB_WB, "same": PB_SAME},
                          "pc": {"rho": PC_RHO, "p": PC_P, "none": PC_NONE}, "tol": TOL},
           "floors_e0": {str(k): _r(v) for k, v in floors.items()},
           "files": {r: d3s[r] is not None for r in runs}}
    if problems:
        res.update(problems=problems,
                   pa={"label": "invalid", "reasons": problems}, pb={"label": "invalid", "reasons": problems},
                   pc={"label": "invalid", "reasons": problems})
        return res
    res["pa"] = judge_pa(cfg, d3s, floors)
    res["pb"] = judge_pb(cfg, d3s)
    res["pc"] = judge_pc(cfg, d3s, metrics)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--d3", default=str(D3_DIR))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--g0", default=str(G0_RESULTS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    res = run_all(d3_dir=Path(a.d3), config=Path(a.config), results=Path(a.results), g0_results=Path(a.g0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k in ("pa", "pb", "pc"):
        print(f"[d3] {k}: {res[k]['label']}" + (f"  {res[k].get('reasons')[:2]}" if res[k].get("reasons") else ""))
    print(f"[d3] → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
