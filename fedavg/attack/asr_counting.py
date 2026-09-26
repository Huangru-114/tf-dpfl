"""
attack/asr_counting.py  —  ASR 的四种计数口径（AUDIT A06 / D-019）

  过滤    只数真实标签 ≠ target 的触发样本（本仓库主口径）
  不过滤  全部触发样本都数，真实标签就是 target 的也算命中（官方 utils.py:29-48）
× 仅良性 / 全体（含恶意端），均为逐客户端等权平均。主指标 = 过滤 × 仅良性。

为什么四列都要：3-B 刻意改变各 edge 的 y_t 占比（C2 约 30%、C3 < 1%），不过滤的口径会被
占比机械抬高；「不过滤 × 全体」是与论文可比的那一列。

纯 numpy，不 import TF（attack/backdoor_eval.py 从这里 import）。
"""

from __future__ import annotations

import numpy as np


def asr_counts(pred, y, target_label) -> dict:
    """一批触发样本的预测 → {hits_filtered, n_filtered, hits_all, n_all}。"""
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    hit = pred == int(target_label)
    elig = y != int(target_label)
    return {"hits_filtered": int(np.sum(hit & elig)), "n_filtered": int(np.sum(elig)),
            "hits_all": int(np.sum(hit)), "n_all": int(len(y))}


def rates_from_counts(c: dict):
    """计数 → (过滤 ASR 或 None, 不过滤 ASR 或 None)。分母为 0 = 无定义，不是 0（陷阱 #13）。"""
    f = c["hits_filtered"] / c["n_filtered"] if c["n_filtered"] else None
    u = c["hits_all"] / c["n_all"] if c["n_all"] else None
    return (None if f is None else float(f)), (None if u is None else float(u))


def mean_defined(xs):
    """逐客户端等权平均；None（该端无定义）先剔除，全是 None → None。"""
    vals = [v for v in xs if v is not None]
    return float(np.mean(vals)) if vals else None
