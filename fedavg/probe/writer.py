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


def _stats(xs):
    xs = [x for x in xs if isinstance(x, float) and not math.isnan(x)]
    return {"n": len(xs), "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
            "mean": (sum(xs) / len(xs)) if xs else None}


def hardware_floor(summaries) -> dict:
    """
    良性端的 ‖Δθ_BD‖ —— 纯硬件/实现噪声。

    良性端没有攻击 mixin，翻 `is_malicious` 是无操作 → 两条 twin 走**逐字节相同**的
    代码路径。所以这个量里不含任何算法内容，只有「同一段计算跑两遍是否给出同一个数」。
    `determinism=cpu` 下它必须精确为 0；GPU 上（cuDNN 的非确定性 kernel）它不为 0，
    那时它就是**必须被超过的地板**，而不是"仪表坏了"。
    """
    return _stats([s["bd_norm"] for s in summaries if s.get("role") == "benign"])


def verdicts(summaries, *, ratio_threshold: float = 3.0,
             determinism: str = "cpu", tol: float = 1e-6) -> dict:
    """
    Stage 1 的判据，直接算成数字，免得靠人眼在 CSV 里翻。

    三个量语义各不相同，必须分开报（混起来会把硬件问题读成科学结论）：

        hardware_floor  良性端 ‖Δθ_BD‖        同一段计算跑两遍的残差（见上）
        sgd_floor       ‖Δθ_stochastic‖       换 order_seed 引起的**真实** SGD 噪声
        signal          恶意端 ‖Δθ_BD‖        后门位移 + 上面两者

    判据 1（仪表）：`determinism=cpu` 时 hardware_floor 必须 < tol。
                    不为 0 → 批序冻结或状态还原漏了东西，下面所有数字都不可读。
                    `gpu-measure` 时不做这个断言 —— 那个模式的地板本来就非零，
                    它被并进判据 2 的分母里。
    判据 2（信号）：signal / max(hardware_floor, sgd_floor) 是否显著 > 1。
                    **除以 max 而不是只除以 sgd_floor**：GPU 模式下硬件地板可能是
                    两者中更大的那个，只跟 SGD 噪声比会高估信号。
                    不显著 → Q1 的答案是「参数空间不可定位」，**这是结论不是 bug**。
    """
    mal = [s for s in summaries if s.get("role") == "malicious"]
    ben = [s for s in summaries if s.get("role") == "benign"]

    hw = hardware_floor(summaries)
    hw_max = hw["max"] or 0.0

    ratios = []
    for s in mal:
        floor = max(hw_max, s.get("stochastic_norm") or 0.0)
        if floor > 1e-12 and not math.isnan(s["bd_norm"]):
            ratios.append(s["bd_norm"] / floor)
    sig = _stats(ratios)

    instrument_pass = (determinism != "cpu"
                       or (hw["max"] is not None and hw["max"] < tol))
    return {
        "determinism": determinism,
        "instrument_ok": {
            "hardware_floor": hw,
            "tol": tol,
            "note": ("良性端两条 twin 代码路径逐字节相同 → 这个值只反映批序冻结/状态还原"
                     "（cpu 模式）或硬件非确定性（gpu-measure 模式）。"),
            "pass": bool(instrument_pass),
            "asserted": determinism == "cpu",
        },
        "signal": {
            "signal_over_floor": sig,
            "floor_used": "max(hardware_floor, sgd_floor)",
            "threshold": ratio_threshold,
            # 取**最差**的恶意端：用均值会被另一个高比值的端掩盖过去
            "pass": bool(sig["min"] is not None and sig["min"] > ratio_threshold),
        },
        "noise_floor": {
            "benign_stochastic_norm": _stats([s["stochastic_norm"] for s in ben]),
            "malicious_stochastic_norm": _stats([s["stochastic_norm"] for s in mal]),
        },
    }
