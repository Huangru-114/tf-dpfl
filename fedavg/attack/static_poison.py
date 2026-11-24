"""
attack/static_poison.py  —  静态触发器的「逐批投毒」（攻击基线第一档：BadNets / Blended / DBA，D-141）

静态触发器（不依赖模型的 BadNets 白块 / Blended 混合图 / DBA 局部像素条）原来走 `vanilla` 策略：
开跑前一次性把恶意端 20% 的训练图换成「带触发器 + 目标标签」的副本（attack/backdoor.py:build_poisoned_dataset）。
P2 的每轮重洗取数路径（data.batch_pipeline = per_epoch）从原始数组按索引取数，不读那个替换过的数据集
→ 恶意端会静默变成良性端（config_validate 因此拒绝 vanilla + per_epoch）。

`static` 策略改成在 `on_batch` 钩子里**每批**投毒（client/client_static.py），照 Bad-PFL 的写法：
  · 哪几张被投毒 = attack/poison_mask.poison_mask(self.rng, n, ρ, backdoor.poison_sampling)
    —— 与 Bad-PFL 对照读同一个开关，「投毒率 ρ」的含义完全相同（用户 11-24 拍板）；
  · 钩子在 epoch_batches 做完数据增强**之后**才被调用 → 先增强、再加触发器（与 Bad-PFL 相同）。

本模块只放纯算术（**不 import TF**），本地 L1 秒级可测：被选中的行换成 trigger(x) + 目标标签，
其余行逐位不变。
"""

from __future__ import annotations

import numpy as np


def apply_static_poison(x, y, mask, trigger_fn, target: int):
    """
    x : (n, H, W, C) 模型输入空间的一批图（numpy 或可转成 numpy 的张量）；不会被就地修改。
    y : (n,) 整数标签；不会被就地修改。
    mask : (n,) bool，True = 这一张投毒。
    trigger_fn : numpy (k, H, W, C) → (k, H, W, C)，只作用于被选中的行（静态触发器逐样本独立，
                 只喂选中的行与整批喂进去再挑出来逐位相同）。
    返回 (x_out, y_out, n_poisoned)。x_out 是 float32 副本，y_out 保持 y 的 dtype。
    """
    x_out = np.array(x, dtype=np.float32, copy=True)
    y_out = np.array(y, copy=True)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(x_out),):
        raise ValueError(f"mask 形状 {mask.shape} 与批大小 {len(x_out)} 不符")
    k = int(mask.sum())
    if k == 0:
        return x_out, y_out, 0
    x_out[mask] = np.asarray(trigger_fn(x_out[mask]), dtype=np.float32)
    y_out[mask] = int(target)
    return x_out, y_out, k
