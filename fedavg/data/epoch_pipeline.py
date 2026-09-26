"""
data/epoch_pipeline.py  —  每 epoch 的取数（AUDIT A25 / D-026，开关 data.batch_pipeline）

旧行为（legacy，默认）按方法各不相同（FINDINGS F-025 / F-026）：
  · FedRep（client/hier_fedrep.py）构造时 `list(self.dataset)` 缓存一次 → 增强冻结、
    **batch 组成冻结**，每 epoch 只打乱 batch 的顺序，并永远丢**同一个**尾批 →
    seed42 下 3.2% 的训练样本从未参与训练（单端最高 19.5%）。
  · FedAvg 每 epoch 重读 tf.data（重洗、重增强、训练尾批）。
  · 生成器另读一遍 tf.data，循环取批。
per_epoch（官方语义，Bad-PFL main.py:78-80 的 SubsetRandomSampler + drop_last）：
  每 epoch 重洗 + 重增强、`drop_last` 且尾批每 epoch 轮换；三处走同一个取数函数
  （`FLClientBase.epoch_batches`，它调这里的两个纯函数）。

增强与 partition._augment 同分布：左右翻转（p=0.5）+ 四周补 4 个 0 再随机裁回原尺寸
（偏移在 {0..8} 上均匀）。补的 0 在**模型输入空间**里（与旧 tf.image.pad_to_bounding_box
一致；标准化时它是「均值灰」而不是黑）。

所有随机都来自调用方给的 seeded Generator（客户端的 data_rng = [seed, client_id, 1]），
不碰全局 np.random，也不碰 Python random。纯 numpy，不 import TF。
"""

from __future__ import annotations

import numpy as np

PAD = 4


def epoch_index_batches(n: int, batch_size: int, rng: np.random.Generator,
                        drop_last: bool = True) -> list:
    """一个 epoch 的批索引（相对本客户端训练集的位置 0..n-1）。每次调用重洗一次。"""
    n, bs = int(n), int(batch_size)
    perm = rng.permutation(n)
    nb = n // bs if drop_last else -(-n // bs)
    return [perm[i * bs:(i + 1) * bs] for i in range(nb)]


def augment_batch(x: np.ndarray, rng: np.random.Generator, pad: int = PAD) -> np.ndarray:
    """(N,H,W,C) → 同形状：逐样本随机左右翻转 + 补 pad 个 0 后随机裁剪。"""
    x = np.asarray(x, dtype=np.float32)
    n, h, w, _ = x.shape
    flip = rng.random(n) < 0.5
    x = np.where(flip[:, None, None, None], x[:, :, ::-1, :], x)
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oy = rng.integers(0, 2 * pad + 1, size=n)
    ox = rng.integers(0, 2 * pad + 1, size=n)
    out = np.empty_like(x)
    for i in range(n):
        out[i] = padded[i, oy[i]:oy[i] + h, ox[i]:ox[i] + w, :]
    return out
