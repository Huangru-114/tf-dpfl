"""
harness/d3pa2_verdict.py  —  P-a 追加探针（加大训练量）的选择与判定（预注册 N-016，决定 D-140；**写于标定与正式数据之前**）

    # ① 标定回传后：按规则选训练量 → 冻结清单（提交之后才交正式作业）
    python3 harness/d3pa2_verdict.py select experiments/defense/edge-native/analysis/d3pa2/SNAP__collocated__s42.calibrate.json
    # ② 正式回传后：判定
    python3 harness/d3pa2_verdict.py judge

输入：`fedavg/analysis/d3_pa2.py` 写的 JSON；清单 `experiments/defense/edge-native/d3pa2_configs.yaml`。
量：A = E0 良性端主列攻击成功率（ξ 在当前攻击者模型上重求）；f = (A_基线 − A_臂) / (A_基线 − 底线_E0)，
基线 = 不重训、只精确重估统计量；底线_E0 = G0-C1 同种子 E0 良性端第 51–60 轮均值（同 D3 P-a，`d3_verdict.floor_e0`）；
精度降幅 = 基线的 E0 良性端干净精度 − 该臂的。

选择（只看标定 = SNAP 集中 s42 × 第 15 / 60 轮 × 「训全部主干」，12 档）：
  合格 = 两轮都 f ≥ 0.5、两轮精度降幅都 ≤ 0.05、都没发散；合格者里取网格顺序第一个
  （网格已按 学习率 × 步数 升序、再学习率升序、再 probe 先于 reinit 排好，`d3_spec.pa2_grid`）→ selected；
  没有合格的 → not_removable（在学习率 ≤ 0.1、≤ 300 步内，训全部主干的干净微调都清不掉这个后门；P-a 到此为止）。
  标定的重现闸不过 → invalid；缺轮 / 缺档 → insufficient。

判定（冻结的那一档 × 三臂 × s42–44 × 第 15 / 60 轮）：
  闸（任一不过 → invalid）：JSON 的清单 sha / 冻结清单 sha / 选定档与本地一致；重现闸；
    「只训最后一块」与「训全部主干」两臂没发散、精度降幅 ≤ 0.05。
  阳性对照：「训全部主干」6 个 f 都 ≥ 0.5，否则 uninformative。
  标签（「只训最后一块」6 个 f）：都 ≥ 0.5 → last_block_suffices；都 < 0.25 → last_block_insufficient；其余 inconclusive。
  缺文件 / 缺轮 → insufficient；顺序：invalid → insufficient → uninformative → 标签（同 D-139 ⑨）。
  另报（不进判定）：「只训前面几层」的 f、固定 ξ 的 f、margin、权重改变量、逐轮标签。
阈值比较带 1e-9 容差。纯标准库（+ `fedavg/analysis/d3_spec.py` 读清单、`d3_verdict.floor_e0` 取底线），不 import TF。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "fedavg"))
from analysis import d3_spec as S            # noqa: E402
from d3_verdict import G0_RESULTS, floor_e0  # noqa: E402

EN = ROOT / "experiments/defense/edge-native"
CONFIG = EN / "d3pa2_configs.yaml"
D3PA2_DIR = EN / "analysis/d3pa2"
FROZEN = EN / "analysis/d3pa2_frozen.json"
OUT = EN / "analysis/d3pa2_verdict.json"
PREREG = "N-016（experiments/defense/edge-native/FINDINGS.md）；D-140；判定脚本写于标定与正式数据之前"

TOL = 1e-9
REMOVE = 0.5          # f ≥ 0.5 = 清掉了（阳性对照、选择、last_block_suffices）
EARLY = 0.25          # 都 f < 0.25 → last_block_insufficient
ACC_GATE = 0.05       # 精度降幅 ≤ 0.05


def _ge(a, b):
    return a >= b - TOL


def _lt(a, b):
    return a < b - TOL


def _le(a, b):
    return a <= b + TOL


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def sha12(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def check_thresholds(cfg):
    want = {"remove": REMOVE, "early": EARLY, "acc_gate": ACC_GATE}
    if cfg.get("thresholds") != want:
        raise ValueError(f"清单的 thresholds {cfg.get('thresholds')} 与判定脚本的常数 {want} 不一致")


def f_value(a_base, a_arm, floor):
    if None in (a_base, a_arm, floor):
        return None
    den = float(a_base) - float(floor)
    if den <= TOL:
        return None
    return (float(a_base) - float(a_arm)) / den


def arm_reading(R: dict, budget_id: str, arm: str, floor) -> dict | None:
    """一个 (轮, 档, 臂) 的读数；没有这一格 → None。"""
    row = (((R.get("budgets") or {}).get(budget_id) or {}).get("arms") or {}).get(arm)
    if row is None:
        return None
    base = (R.get("baseline") or {})
    h = row.get("harden") or {}
    if row.get("diverged"):
        return {"diverged": True, "f": None, "acc_drop": None, "harden": h}
    b_main, a_main = base.get("main") or {}, row.get("main") or {}
    b_fix, a_fix = base.get("fixed") or {}, row.get("fixed") or {}
    drop = (None if None in (b_main.get("acc"), a_main.get("acc"))
            else float(b_main["acc"]) - float(a_main["acc"]))
    # f 与精度降幅**不舍入**：判定直接拿它们和阈值比（先舍入会让 0.49996 变成 0.5 而过线）
    return {"diverged": False, "asr_base": _r(b_main.get("asr")), "asr": _r(a_main.get("asr")),
            "f": f_value(b_main.get("asr"), a_main.get("asr"), floor),
            "f_fixed_xi": f_value(b_fix.get("asr"), a_fix.get("asr"), floor),
            "acc_base": _r(b_main.get("acc")), "acc": _r(a_main.get("acc")), "acc_drop": drop,
            "margin_base": _r(b_main.get("margin_p50")), "margin": _r(a_main.get("margin_p50")),
            "steps": h.get("steps"), "lr": h.get("lr"), "dw_body_absmax": _r(h.get("dw_body_absmax"), 6)}


def json_problems(j, cfg, *, stage, run_id, frozen=None, frozen_sha=None) -> list:
    errs = []
    if j.get("schema") != S.SCHEMA or j.get("tool") != "d3_pa2":
        errs.append(f"{run_id}: 不是 d3_pa2 的 schema {S.SCHEMA} 输出")
    if j.get("stage") != stage:
        errs.append(f"{run_id}: stage = {j.get('stage')!r}，应为 {stage!r}")
    if j.get("run_id") != run_id:
        errs.append(f"{run_id}: JSON 里的 run_id = {j.get('run_id')!r}")
    if j.get("config_sha") != S.content_sha(cfg):
        errs.append(f"{run_id}: 参数清单 sha {j.get('config_sha')!r} ≠ 当前 {S.content_sha(cfg)!r}")
    if frozen is not None:
        if j.get("frozen_selected") != frozen.get("selected"):
            errs.append(f"{run_id}: 跑的档 {j.get('frozen_selected')!r} ≠ 冻结清单选的 {frozen.get('selected')!r}")
        if frozen_sha is not None and j.get("frozen_sha") != frozen_sha:
            errs.append(f"{run_id}: 冻结清单 sha {j.get('frozen_sha')!r} ≠ 本地 {frozen_sha!r}（集群上用的不是这份冻结清单）")
    return errs


# ══════════════════════════════════════════════════════════════════════════
# ① 选择
# ══════════════════════════════════════════════════════════════════════════

def select(calib: dict | None, cfg: dict, *, calib_path=None, g0_results=G0_RESULTS) -> dict:
    check_thresholds(cfg)
    sec = cfg["calibrate"]
    run, arm = sec["run"], sec["arm"]
    rounds = [int(t) for t in sec["rounds"]]
    grid = S.pa2_grid(cfg)
    out = {"schema": S.SCHEMA, "preregistered": PREREG, "config_sha": S.content_sha(cfg),
           "rule": f"标定 {run} × 第 {rounds} 轮 × {arm}：两轮都 f ≥ {REMOVE}、精度降幅 ≤ {ACC_GATE}、没发散 → 合格；"
                   f"取网格顺序（学习率 × 步数 → 学习率 → probe 先于 reinit）第一个合格的",
           "calibration_file": None if calib_path is None else str(Path(calib_path).name),
           "calibration_sha": None if calib_path is None else sha12(calib_path),
           "selected": None, "table": []}
    if calib is None:
        out.update(label="insufficient", reasons=["没有标定 JSON"])
        return out
    probs = json_problems(calib, cfg, stage="calibrate", run_id=run)
    if probs:
        out.update(label="invalid", reasons=probs)
        return out
    floor = floor_e0(g0_results, S.seed_of(run))
    out["floor_e0"] = _r(floor)
    missing, invalid = [], []
    for t in rounds:
        R = (calib.get("rounds") or {}).get(str(t))
        if R is None:
            missing.append(f"第 {t} 轮")
        elif not (R.get("reproduce") or {}).get("pass"):
            invalid.append(f"第 {t} 轮重现闸不过：{(R.get('reproduce') or {}).get('reasons')}")
    if invalid:
        out.update(label="invalid", reasons=invalid)
        return out
    for b in grid:
        row = {"id": b["id"], "index": b["index"], "rounds": {}, "eligible": True, "why_not": []}
        for t in rounds:
            R = (calib.get("rounds") or {}).get(str(t))
            rd = None if R is None else arm_reading(R, b["id"], arm, floor)
            if rd is None:
                missing.append(f"第 {t} 轮 {b['id']}")
                row["eligible"] = False
                continue
            row["rounds"][str(t)] = rd
            if rd["diverged"]:
                row["eligible"] = False
                row["why_not"].append(f"第 {t} 轮发散")
                continue
            if rd["f"] is None or not _ge(rd["f"], REMOVE):
                row["eligible"] = False
                row["why_not"].append(f"第 {t} 轮 f = {rd['f']}")
            if rd["acc_drop"] is None or not _le(rd["acc_drop"], ACC_GATE):
                row["eligible"] = False
                row["why_not"].append(f"第 {t} 轮精度降 {rd['acc_drop']}")
        out["table"].append(row)
    if missing:
        out.update(label="insufficient", reasons=missing)
        return out
    pick = next((r["id"] for r in out["table"] if r["eligible"]), None)
    if pick is None:
        out.update(label="not_removable", reasons=["12 档里没有一档在两轮都能让训全部主干 f ≥ 0.5 且精度降幅 ≤ 0.05"])
    else:
        out.update(label="selected", selected=pick, reasons=[])
    return out


# ══════════════════════════════════════════════════════════════════════════
# ② 判定
# ══════════════════════════════════════════════════════════════════════════

def _label(fs):
    if not fs or any(v is None for v in fs):
        return "insufficient"
    if all(_ge(v, REMOVE) for v in fs):
        return "last_block_suffices"
    if all(_lt(v, EARLY) for v in fs):
        return "last_block_insufficient"
    return "inconclusive"


def judge(cfg: dict, frozen: dict | None, mains: dict, *, frozen_sha=None, g0_results=G0_RESULTS) -> dict:
    """mains = {run_id: 正式 JSON 或 None}。"""
    check_thresholds(cfg)
    sec = cfg["main"]
    lab_arm, ctl_arm = sec["label_arm"], sec["control_arm"]
    others = [a for a in sec["arms"] if a not in (lab_arm, ctl_arm)]
    rounds = [int(t) for t in sec["rounds"]]
    out = {"preregistered": PREREG, "config_sha": S.content_sha(cfg),
           "thresholds": {"remove": REMOVE, "early": EARLY, "acc_gate": ACC_GATE, "tol": TOL},
           "rule": f"阳性对照：{ctl_arm} 6 个 f 都 ≥ {REMOVE}（否则 uninformative）；{lab_arm} 6 个 f 都 ≥ {REMOVE} → last_block_suffices，"
                   f"都 < {EARLY} → last_block_insufficient；闸：清单 / 冻结 sha、重现、两臂未发散且精度降 ≤ {ACC_GATE}",
           "frozen": None if frozen is None else {k: frozen.get(k) for k in ("label", "selected", "calibration_sha")},
           "rows": []}
    if frozen is None:
        out.update(label="insufficient", reasons=["没有冻结清单（标定还没选）"])
        return out
    if frozen.get("config_sha") != S.content_sha(cfg):
        out.update(label="invalid", reasons=["冻结清单的参数清单 sha 与当前清单不一致"])
        return out
    if frozen.get("label") == "not_removable":
        out.update(label="not_removable", reasons=["标定判 not_removable：P-a 到此为止，不跑正式阶段"])
        return out
    if frozen.get("label") != "selected" or not frozen.get("selected"):
        out.update(label="insufficient", reasons=[f"冻结清单没有选出训练量（label = {frozen.get('label')!r}）"])
        return out
    bid = frozen["selected"]
    invalid, missing = [], []
    for run in sec["runs"]:
        j = mains.get(run)
        if j is None:
            missing.append(f"{run} 的正式 JSON")
            continue
        invalid += json_problems(j, cfg, stage="main", run_id=run, frozen=frozen, frozen_sha=frozen_sha)
        seed = S.seed_of(run)
        floor = floor_e0(g0_results, seed)
        for t in rounds:
            R = (j.get("rounds") or {}).get(str(t))
            if R is None:
                missing.append(f"{run} 第 {t} 轮")
                continue
            if not (R.get("reproduce") or {}).get("pass"):
                invalid.append(f"{run} 第 {t} 轮重现闸不过：{(R.get('reproduce') or {}).get('reasons')}")
            row = {"run": run, "seed": seed, "round": t, "floor_e0": _r(floor), "arms": {}}
            for arm in [lab_arm, ctl_arm] + others:
                rd = arm_reading(R, bid, arm, floor)
                if rd is None:
                    missing.append(f"{run} 第 {t} 轮 {arm} 臂")
                    continue
                row["arms"][arm] = rd
                if arm in (lab_arm, ctl_arm):
                    if rd["diverged"]:
                        invalid.append(f"{run} 第 {t} 轮 {arm} 臂发散")
                    elif rd["acc_drop"] is not None and not _le(rd["acc_drop"], ACC_GATE):
                        invalid.append(f"{run} 第 {t} 轮 {arm} 臂精度降 {rd['acc_drop']} > {ACC_GATE}（可能是破坏而不是清除）")
            out["rows"].append(row)
    if invalid:
        out.update(label="invalid", reasons=invalid)
        return out
    if missing:
        out.update(label="insufficient", reasons=missing)
        return out
    ctl = [r["arms"][ctl_arm]["f"] for r in out["rows"]]
    lab = [r["arms"][lab_arm]["f"] for r in out["rows"]]
    out["control_f"] = ctl
    out["label_f"] = lab
    out["per_round"] = {str(t): _label([r["arms"][lab_arm]["f"] for r in out["rows"] if r["round"] == t]) for t in rounds}
    if any(v is None for v in ctl) or not all(_ge(v, REMOVE) for v in ctl):
        out.update(label="uninformative",
                   reasons=[f"阳性对照（{ctl_arm}）不是 6 个都 f ≥ {REMOVE}：{ctl}"])
        return out
    lb = _label(lab)
    out.update(label=lb, reasons=[] if lb != "insufficient" else ["有 f 无定义（基线已在底线上）"])
    return out


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════

def _load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("calibration")
    s.add_argument("--config", default=str(CONFIG))
    s.add_argument("--out", default=str(FROZEN))
    j = sub.add_parser("judge")
    j.add_argument("--config", default=str(CONFIG))
    j.add_argument("--d3pa2", default=str(D3PA2_DIR))
    j.add_argument("--frozen", default=str(FROZEN))
    j.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    cfg = S.load_pa2_config(a.config)
    if a.cmd == "select":
        res = select(_load(a.calibration), cfg, calib_path=a.calibration if Path(a.calibration).exists() else None)
        print(f"[d3pa2] select: {res['label']}" + (f" → {res['selected']}" if res.get("selected") else "")
              + (f"  {res['reasons'][:2]}" if res.get("reasons") else ""))
    else:
        fz = _load(a.frozen)
        mains = {r: _load(Path(a.d3pa2) / f"{r}.main.json") for r in cfg["main"]["runs"]}
        res = judge(cfg, fz, mains, frozen_sha=sha12(a.frozen) if fz is not None else None)
        print(f"[d3pa2] judge: {res['label']}" + (f"  {res['reasons'][:2]}" if res.get("reasons") else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[d3pa2] → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
