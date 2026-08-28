"""
probe/writer.py  —  探针产物落盘（**纯 numpy/stdlib，不 import tf**）。

回程协议（CLAUDE.md）：只回 **layer 级** CSV 与 summary JSON —— 每个 checkpoint
几百行、KB 量级。**逐参数数组一律留在集群**：resnet10 一个 Δθ 就是 ~20MB，
回传三个 twin × 若干客户端 × 若干 checkpoint 直接撞「单文件 > 10MB」红线。
occupation 关系也只回**分箱曲线**，不回散点。
"""
import csv
import json
import math
from pathlib import Path

from .analyze import ROW_COLUMNS


def _jsonable(o):
    """把 numpy 标量/数组与 nan/inf 变成 JSON 能表达的东西（nan → null）。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item") and getattr(o, "ndim", None) == 0:
        return _jsonable(o.item())
    if hasattr(o, "tolist"):
        return _jsonable(o.tolist())
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


def write_rows(path, rows) -> Path:
    """Long-format CSV，列名固定（对齐用户 §28 的 Client-level schema）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and math.isnan(r[k])
                            else r.get(k, "")) for k in ROW_COLUMNS})
    return path


def write_summary(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def verdicts(summaries, *, ratio_threshold: float = 3.0) -> dict:
    """
    Stage 1 的三条判据，直接算成数字，免得靠人眼在 CSV 里翻。

    判据 1（仪表）：良性端的 ‖Δθ_BD‖ —— 良性端两个 twin 走的是**逐字节相同**的代码
                    路径，所以这个值只反映批序冻结与状态还原是否真的生效。
                    它不为 ~0 就说明仪表坏了，下面所有数字都不可读。
    判据 2（信号）：恶意端 ‖Δθ_BD‖ / ‖Δθ_stochastic‖ 是否显著 > 1。
                    不显著 → Q1 的答案是「参数空间不可定位」，**这是结论不是 bug**。
    判据 3（量级）：恶意端与良性端的 stochastic 噪声地板量级是否可比。
    """
    mal = [s for s in summaries if s.get("role") == "malicious"]
    ben = [s for s in summaries if s.get("role") == "benign"]

    def _f(xs):
        xs = [x for x in xs if isinstance(x, float) and not math.isnan(x)]
        return {"n": len(xs), "min": min(xs) if xs else None,
                "max": max(xs) if xs else None,
                "mean": (sum(xs) / len(xs)) if xs else None}

    benign_bd = _f([s["bd_norm"] for s in ben])
    mal_ratio = _f([s["bd_over_stochastic"] for s in mal])
    return {
        "instrument_ok": {
            "benign_bd_norm": benign_bd,
            "note": "良性端两个 twin 代码路径逐字节相同 → 这个值就是批序冻结/状态还原的残差。",
            "pass": bool(benign_bd["max"] is not None and benign_bd["max"] < 1e-6),
        },
        "signal": {
            "malicious_bd_over_stochastic": mal_ratio,
            "threshold": ratio_threshold,
            "pass": bool(mal_ratio["min"] is not None and mal_ratio["min"] > ratio_threshold),
        },
        "noise_floor": {
            "benign_stochastic_norm": _f([s["stochastic_norm"] for s in ben]),
            "malicious_stochastic_norm": _f([s["stochastic_norm"] for s in mal]),
        },
    }
