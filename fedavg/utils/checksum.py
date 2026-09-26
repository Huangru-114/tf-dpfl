"""
utils/checksum.py  —  权重的短哈希（AUDIT A15 / D-028 的验收：`[Checksum]` 行）

每个云轮聚合后打一行 `[Checksum] Round N | global=<sha256 前 12 位>`。
同 seed 两次 run 的前 5 轮逐轮相同，才算 `enable_op_determinism()` 真的生效
（F-002：P1 同 seed 重跑第 1 轮即分叉，来源没有证据）。

纯 numpy + hashlib，不 import TF。
"""

import hashlib

import numpy as np


def weights_checksum(weights, n: int = 12) -> str:
    """权重列表（numpy 数组）→ sha256 十六进制前 n 位。形状与 dtype 也进哈希。"""
    h = hashlib.sha256()
    for w in weights:
        a = np.ascontiguousarray(np.asarray(w))
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:n]
