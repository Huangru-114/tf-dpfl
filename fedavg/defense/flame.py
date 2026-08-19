"""
defense/flame.py  –  FLAME 鲁棒聚合（Nguyen et al., USENIX Security'22）

三个组件（对更新 update = client_weights − ref_weights 运算）：
  1. 余弦聚类过滤：基于更新方向的两两余弦相似度，保留与多数方向一致的「良性簇」，
     剔除方向异常的更新。原文用 HDBSCAN；本实现用无依赖的 majority-cosine 近似
     （每个 client 数其余弦相似度 ≥ 阈值的邻居数，保留邻居数 ≥ m/2 的多数簇），
     在注释中标注为近似实现。
  2. 范数裁剪（自适应）：clip = median(更新 L2 范数)，把每个被保留更新缩放到范数 ≤ clip，
     限制单个 client 的影响（防放大攻击）。
  3. 加噪：对裁剪后的加权平均更新加 N(0, (noise_scale·clip)²) 高斯噪声（差分隐私式平滑，
     抹掉残留后门）。

返回 ref_weights + (裁剪加权均值更新 + 噪声)。
"""

import numpy as np

from .base_defense import (BaseDefense, flatten_weights, unflatten_weights,
                           weighted_mean)


class FlameDefense(BaseDefense):

    name = "flame"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = self.config or {}
        self.noise_scale = float(cfg.get("noise_scale", 0.001))
        # 最近一次 aggregate 接纳的 client 索引。L1 测试断言它、L2 smoke 用它填
        # metrics.json 的 admitted_count。纯记录，不参与算法。
        self.last_admitted = None

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        m = len(client_updates)
        ref_flat = flatten_weights(ref_weights)
        updates = np.stack(
            [flatten_weights(upd[0]) - ref_flat for upd in client_updates], axis=0)  # (m,d)
        samples = np.array([upd[1] for upd in client_updates], dtype=np.float64)

        # ── 1. 余弦聚类过滤（majority-cosine 近似 HDBSCAN）─────────────────────
        norms = np.linalg.norm(updates, axis=1)
        safe = np.where(norms > 0, norms, 1.0)
        unit = updates / safe[:, None]
        cos_sim = unit @ unit.T                       # (m, m) ∈ [-1, 1]
        np.fill_diagonal(cos_sim, -np.inf)
        # 阈值取 off-diagonal 余弦相似度的中位数：良性更新彼此更相似（高于中位数）。
        finite = cos_sim[np.isfinite(cos_sim)]
        thresh = float(np.median(finite)) if finite.size else 0.0
        neighbor_cnt = np.sum(cos_sim >= thresh, axis=1)
        admitted = np.where(neighbor_cnt >= (m / 2.0))[0]
        if admitted.size == 0:                        # 退化保护
            admitted = np.arange(m)
        self.last_admitted = [int(i) for i in admitted]
        print(f"    [Defense:flame] admitted {admitted.size}/{m} updates "
              f"(cos_thresh={thresh:.3f})")

        # ── 2. 自适应范数裁剪 ────────────────────────────────────────────────
        clip = float(np.median(norms[admitted]))
        if clip <= 0:
            clip = float(np.median(norms)) if np.median(norms) > 0 else 1.0
        clipped = []
        weights_sel = []
        for i in admitted:
            scale = min(1.0, clip / norms[i]) if norms[i] > 0 else 1.0
            clipped.append(updates[i] * scale)
            weights_sel.append(samples[i])
        clipped = np.stack(clipped, axis=0)
        weights_sel = np.array(weights_sel, dtype=np.float64)
        agg_update = np.average(clipped, axis=0, weights=weights_sel)

        # ── 3. 加噪 ─────────────────────────────────────────────────────────
        sigma = self.noise_scale * clip
        if sigma > 0:
            agg_update = agg_update + np.random.normal(0.0, sigma, size=agg_update.shape)

        return unflatten_weights(ref_flat + agg_update, ref_weights)
