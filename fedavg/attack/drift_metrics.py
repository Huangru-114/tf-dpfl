"""
attack/drift_metrics.py  —  参数漂移 / 表示漂移的**纯 numpy** 核（Experiment 3C）。

刻意不 import tensorflow：这样这两个数学核可以本地秒级 L1，不必进容器。
特征抽取（需要 TF forward）在 attack/backdoor_eval.py:evaluate_drift 里，调用本模块。

约定：漂移都相对「本轮起点的全局模型」anchor 度量（edge 在本轮 edge_rounds 里偏离共识多远）。
"""
import numpy as np


def _flat_at(weights, idx):
    """把 weights 里 idx 指定的那些张量展平拼接成一维（idx 为空 → 空数组）。"""
    if not idx:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([np.asarray(weights[i], dtype=np.float64).ravel() for i in idx])


def param_drift(edge_weights, anchor_weights, base_idx):
    """
    参数漂移（只在 backbone 索引 base_idx 上）：
      abs = ‖θ_edge − θ_anchor‖₂            （绝对漂移量，用户要求单独记录）
      rel = abs / ‖θ_anchor‖₂               （相对漂移，跨 R_edge/尺度可比）
    Returns: (abs, rel)
    """
    d = _flat_at(edge_weights, base_idx) - _flat_at(anchor_weights, base_idx)
    a = _flat_at(anchor_weights, base_idx)
    absd = float(np.linalg.norm(d))
    rel  = absd / (float(np.linalg.norm(a)) + 1e-12)
    return absd, rel


def repr_shift(E_edge, E_anchor):
    """
    表示漂移：探针每个样本的 (1 − cos(F_edge(x), F_anchor(x)))，对样本做 **mean 与 median**。
    median 版对少数极端样本更稳健（用户要求同时给出）。
    E_edge, E_anchor: (N, D) 同形。
    Returns: (mean, median)
    """
    ee = np.asarray(E_edge, dtype=np.float64)
    ea = np.asarray(E_anchor, dtype=np.float64)
    if ee.shape != ea.shape or ee.size == 0:
        return float("nan"), float("nan")
    num = np.sum(ee * ea, axis=1)
    den = np.linalg.norm(ee, axis=1) * np.linalg.norm(ea, axis=1) + 1e-12
    d = 1.0 - num / den
    return float(np.mean(d)), float(np.median(d))
