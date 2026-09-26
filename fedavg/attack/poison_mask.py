"""
attack/poison_mask.py  —  动态投毒「这一批里哪几个样本被投毒」（AUDIT A03 / D-016）

两种口径（开关 `backdoor.poison_sampling`，alignment.py）：

  exact_k   （旧，默认）每批恰好 k = round(n·ρ) 个，位置由 rng.shuffle 决定。
            batch 32 下有取整偏差：ρ=0.2 → 实际 0.1875（−6.3%），ρ=0.05 → +25%，
            ρ=0.02 → +56%（FINDINGS F-020）。**rng 的调用序列与改动前逐字节相同**
            （只有一次 rng.shuffle），所以冻结的 P1 配置重跑不受影响。
  bernoulli （官方 fba.py:48）每个样本独立以概率 ρ 被选中（`rand() <= ρ`）。
            这里写成 `rng.random(n) < ρ`：两者只在 rand 恰好等于 ρ 时不同（测度零），
            而 `<` 让 ρ=0 精确地一个都不选、ρ=1 精确地全选（rand ∈ [0,1)）。

一个都没选中时，调用方应**原样返回**这一批、不去算 ξ/δ（官方 fba.py:49-50）。

纯 numpy，不 import TF。
"""

from __future__ import annotations

import numpy as np

MODES = ("exact_k", "bernoulli")


def poison_mask(rng: np.random.Generator, n: int, rho: float,
                mode: str = "exact_k") -> np.ndarray:
    """长度 n 的 bool 掩码。rng 必须是客户端自己的 seeded 流（不碰全局 np.random）。"""
    n = int(n)
    if mode == "exact_k":
        k = int(round(n * float(rho)))
        mask = np.zeros(n, dtype=bool)
        if k <= 0:
            return mask                    # 旧代码在这里直接 return，不消耗 rng
        mask[:k] = True
        rng.shuffle(mask)
        return mask
    if mode == "bernoulli":
        rho = float(rho)
        if rho <= 0.0:
            return np.zeros(n, dtype=bool)
        return rng.random(n) < rho
    raise ValueError(f"poison_sampling={mode!r}，合法取值 {MODES}")
