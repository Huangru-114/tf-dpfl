"""
analysis/d3_spec.py  —  阶段三 D3 离线探针的纯算术部分（**不 import TF**）

规则：PREREG-v2 §五（预注册 N-015，`experiments/thesis/PREREG-v2.md` @ 25163a5）；实现取舍：DECISIONS D-138。
TF 部分在 `analysis/d3_probes.py`（只读集群上的快照，零训练），判定在 `harness/d3_verdict.py`（纯标准库）。

三个探针（都只读已有快照）：
  P-a  重训最后一块：SNAP 集中第 15 / 60 轮，攻击者所在边缘 E0 只训 stage4（对照：只训 stem–stage3、训全部主干），
       训后 E0 各端在自己训练分片上**精确重估** BN 运行统计量（基线同样重估），量 E0 良性端 ASR / margin / 干净精度；
  P-b  受害模型上白盒求扰动：TBS k1-snap（与 SNAP 集中作参照）第 60 轮，受害边缘良性端在自己的 fresh 个性化模型上求 ξ，
       与主列（ξ 在攻击者模型上求）配对比较；另报只 ξ、只 δ；
  P-c  统计量偏移：9 个 run × 第 6 / 15 / 60 轮 × 4 边缘，边缘干净集加触发器前后各 BN 层输入矩的高斯 KL（层均值）。

这里放的是探针两侧都要、又能在本地秒级测的东西：参数清单的读取与 sha、每个 run 要跑哪些探针、
精确重估的动量日程、run 记录的逐 edge 数、重现闸、逐客户端记录 → 逐 edge 汇总。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 1
TAG = 0xD3                          # 本工具全部随机数的键（与训练 / 评估 / P0 的键都不重叠）
PROBE_TAGS = {"pa": 0xA, "pb": 0xB, "pc": 0xC, "train": 0x7}
PROBES = ("pa", "pb", "pc")
# metrics.json 里 per_edge_detail_rounds 的列名（harness/collect_metrics.EDGE_DETAIL_COLUMNS 的同一份；
# 文件自带 per_edge_detail_columns，读的时候按名字取，不依赖这个顺序）
MARGIN_COL = "margin_p50"


# ══════════════════════════════════════════════════════════════════════════
# 参数清单（experiments/defense/edge-native/d3_configs.yaml）
# ══════════════════════════════════════════════════════════════════════════

def load_config(path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict):
    errs = []
    if int(cfg.get("schema", 0)) != SCHEMA:
        errs.append(f"schema = {cfg.get('schema')!r}，本工具只认 {SCHEMA}")
    for p in PROBES:
        sec = cfg.get(p) or {}
        if not sec.get("runs") or not sec.get("rounds"):
            errs.append(f"{p}: 缺 runs / rounds")
    pa = cfg.get("pa") or {}
    arms = pa.get("arms") or {}
    if pa.get("label_arm") not in arms:
        errs.append(f"pa.label_arm = {pa.get('label_arm')!r} 不在 arms {sorted(arms)} 里")
    if "none" in arms:
        errs.append("pa.arms 不能叫 none（none 是「不重训、只重估」的基线，由工具自己加）")
    re_ = pa.get("reestimate") or {}
    if re_.get("method") != "precise" or bool(re_.get("augment")):
        errs.append("pa.reestimate 只实现了 method=precise、augment=false（D-138 ①）")
    if (cfg.get("pc") or {}).get("kl_direction") != "triggered_given_clean":
        errs.append("pc.kl_direction 只实现了 triggered_given_clean = KL(加触发器 ‖ 干净)")
    perm = (cfg.get("pc") or {}).get("permutation") or {}
    if perm.get("kind") != "free" or perm.get("sided") != "greater" or int(perm.get("n", 0)) < 1000:
        errs.append("pc.permutation：判定用自由置换（D-138 ③）、单侧（greater）、n ≥ 1000")
    if errs:
        raise ValueError("d3_configs.yaml 不合法：\n  " + "\n  ".join(errs))


def content_sha(cfg: dict) -> str:
    """参数清单的内容 sha（与 YAML 的排版、注释无关）：排序后的 JSON 的 sha256 前 16 位。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def plan_for(cfg: dict, run_id: str) -> dict:
    """这个 run 要跑哪些探针、各在哪几轮 → {probe: [轮, …]}（只列出有份的探针）。P-b 的参照 run 也算 P-b。"""
    out = {}
    for p in PROBES:
        sec = cfg.get(p) or {}
        runs = list(sec.get("runs") or []) + list(sec.get("reference_runs") or [])
        if run_id in runs:
            out[p] = sorted(int(r) for r in sec["rounds"])
    return out


def rounds_needed(plan: dict) -> list:
    return sorted({int(r) for rs in plan.values() for r in rs})


def all_runs(cfg: dict) -> list:
    """全部要交作业的 run（去重、保持首次出现的顺序）。"""
    seen = {}
    for p in PROBES:
        sec = cfg.get(p) or {}
        for r in list(sec.get("runs") or []) + list(sec.get("reference_runs") or []):
            seen.setdefault(r, None)
    return list(seen)


def group_of(run_id: str) -> str:
    """run_id = <组>__<格>__s<种子> → 组名（metrics.json 在 results/P2/<组>/ 下）。"""
    return str(run_id).split("__", 1)[0]


def seed_of(run_id: str) -> int:
    tail = str(run_id).rsplit("__", 1)[-1]
    if not tail.startswith("s") or not tail[1:].isdigit():
        raise ValueError(f"run_id {run_id!r} 末段不是 s<种子>")
    return int(tail[1:])


def arm_prefixes(cfg: dict) -> dict:
    """{臂名: 层名前缀元组或 None（= 全部主干）}，顺序 = 清单顺序。"""
    return {k: (None if v is None else tuple(str(p) for p in v))
            for k, v in ((cfg.get("pa") or {}).get("arms") or {}).items()}


def select_by_prefix(owner_names, prefixes) -> list:
    """owner_names[i] = 第 i 个变量所属层的名字；prefixes=None → 全部下标。返回名字以任一前缀开头的下标（升序）。"""
    if prefixes is None:
        return list(range(len(owner_names)))
    pre = tuple(prefixes)
    return [i for i, n in enumerate(owner_names) if n is not None and str(n).startswith(pre)]


# ══════════════════════════════════════════════════════════════════════════
# 精确重估 BN 运行统计量（D-138 ①）
# ══════════════════════════════════════════════════════════════════════════

def seq_batches(n: int, batch: int) -> list:
    """不打乱、不丢尾：[(start, stop), …]。"""
    if n <= 0 or batch <= 0:
        return []
    return [(s, min(s + batch, n)) for s in range(0, n, batch)]


def precise_momenta(sizes) -> list:
    """
    第 k 批前把每个 BN 层的 momentum 设成 n_前 / (n_前 + n_k)（n_前 = 之前各批的样本数之和）。
    Keras 的更新是 moving ← m·moving + (1 − m)·batch_stat → 第 1 批 m = 0（旧统计量完全不留），
    之后各批按样本数加权平均：结束时 moving = Σ_k n_k·stat_k / Σ_k n_k（各批统计量的样本数加权均值）。
    方差的口径（有偏 / 无偏）就是 Keras 自己更新 moving_variance 用的那个 —— 由 TF 测试钉住。
    """
    out, seen = [], 0
    for n in sizes:
        n = int(n)
        if n <= 0:
            raise ValueError(f"批大小必须 > 0：{n}")
        out.append(float(seen) / float(seen + n))
        seen += n
    return out


def apply_momenta(stats, momenta, init=None):
    """precise_momenta 的参照实现（numpy，测试用）：按 Keras 的更新式依次混合 → 应等于样本数加权均值。"""
    cur = None if init is None else np.asarray(init, np.float64)
    for s, m in zip(stats, momenta):
        s = np.asarray(s, np.float64)
        cur = s if cur is None and m == 0.0 else m * cur + (1.0 - m) * s
    return cur


# ══════════════════════════════════════════════════════════════════════════
# run 记录的数 + 重现闸
# ══════════════════════════════════════════════════════════════════════════

def recorded_edges(metrics: dict, t: int) -> dict:
    """第 t 轮全量点的逐 edge 记录 → {edge: {"client_benign": …, "margin_p50": …}}（缺的记 None）。"""
    out = {}
    for d in (metrics.get("per_edge_rounds") or {}).get(str(int(t))) or []:
        out[int(d["edge_id"])] = {"client_benign": d.get("client_benign"), MARGIN_COL: None}
    cols = list(metrics.get("per_edge_detail_columns") or [])
    if cols and "edge_id" in cols and MARGIN_COL in cols:
        ei, mi = cols.index("edge_id"), cols.index(MARGIN_COL)
        for row in (metrics.get("per_edge_detail_rounds") or {}).get(str(int(t))) or []:
            if row and row[ei] is not None:
                out.setdefault(int(row[ei]), {"client_benign": None, MARGIN_COL: None})[MARGIN_COL] = row[mi]
    return out


def reproduce_check(recomputed: dict, recorded: dict, tol: float, edges=None) -> dict:
    """
    离线重算的第 t 轮全量点主列（`p0_snapshot.eval_full_main(..., detail=True)`）与 run 记录逐 edge 比：
    良性 ASR 与 margin_p50 的差都 ≤ tol 才算过（快照复原对了的证据；P0 的 V0 ③ 只看受害 edge，这里四台都看，含 E0）。
    任何一方缺值 → 不过（不能把「没法比」当成「比过了」）。
    """
    edges = sorted(set(recorded) | set(recomputed)) if edges is None else sorted(int(e) for e in edges)
    per, reasons, worst = {}, [], 0.0
    for e in edges:
        a, b = recomputed.get(e) or {}, recorded.get(e) or {}
        row = {}
        for k in ("client_benign", MARGIN_COL):
            x, y = a.get(k), b.get(k)
            if x is None or y is None:
                row[k] = None
                reasons.append(f"edge {e} 的 {k} 缺值（重算 {x!r}，记录 {y!r}）")
                continue
            d = abs(float(x) - float(y))
            row[k] = round(d, 6)
            worst = max(worst, d)
            if d > float(tol) + 1e-12:
                reasons.append(f"edge {e} 的 {k} 差 {d:.4g} > {tol}")
        per[str(e)] = row
    return {"pass": not reasons and bool(edges), "tol": float(tol), "max_abs_diff": round(worst, 6),
            "per_edge": per, "reasons": reasons or ([] if edges else ["没有可比的 edge"])}


# ══════════════════════════════════════════════════════════════════════════
# 逐客户端记录 → 逐 edge 汇总
# ══════════════════════════════════════════════════════════════════════════

def _mean(vals):
    v = [float(x) for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def edge_summary(records, edges=None, malicious=False) -> dict:
    """
    records：[{client_id, edge_id, malicious, asr, margins (np 数组), acc}, …]（一列触发器、一个模型状态下的逐端记录）。
    → {edge: {"asr": 良性端（malicious=False）ASR 等权均值, "margin_p50": 良性端触发样本 margin 池化中位数,
              "acc": 良性端干净精度均值, "n": 端数}}。
    口径与 run 的 `per_edge[].client_benign`（逐端等权平均）和 `[EvalDetailEdge] margin_p50`（池化分位数）相同。
    """
    by = {}
    for r in records:
        if bool(r.get("malicious")) != bool(malicious):
            continue
        by.setdefault(int(r["edge_id"]), []).append(r)
    keys = sorted(by) if edges is None else sorted(int(e) for e in edges)
    out = {}
    for e in keys:
        rs = by.get(e) or []
        m = [np.asarray(r.get("margins") if r.get("margins") is not None else [], np.float64) for r in rs]
        pooled = np.concatenate(m) if m else np.zeros(0)
        out[str(e)] = {"asr": _mean(r.get("asr") for r in rs),
                       MARGIN_COL: (float(np.percentile(pooled, 50)) if pooled.size else None),
                       "acc": _mean(r.get("acc") for r in rs),
                       "n": len(rs)}
    return out


def group_mean(summary: dict, edges, key="asr"):
    """若干 edge 的某个量再取等权均值（V 的口径）；任一 edge 缺 → None。"""
    vals = [(summary.get(str(int(e))) or {}).get(key) for e in edges]
    return None if not vals or any(v is None for v in vals) else float(np.mean(vals))


def client_rows(records) -> list:
    """逐端记录 → JSON 友好的紧凑行（不含逐样本 margin）。"""
    return [{"cid": int(r["client_id"]), "edge": int(r["edge_id"]), "mal": bool(r.get("malicious")),
             "asr": _r(r.get("asr")), "margin_med": _r(r.get("margin_med")), "acc": _r(r.get("acc")),
             "n": int(r.get("n", 0))} for r in records]


def trim_info(info: dict) -> dict:
    """EdgeHardener.harden 的 info → 回传用的精简版（不带逐步轨迹）。"""
    keep = ("id", "n_clean", "probe_acc", "lr", "augment", "steps", "loss_first", "loss_last",
            "diverged_at", "dw_body_absmax", "seconds")
    return {k: info.get(k) for k in keep if k in info}


def _r(v, nd=6):
    return None if v is None else round(float(v), nd)


def jsonable(x):
    """numpy 标量 / 数组 → Python 原生类型（写 JSON 用）。"""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x
