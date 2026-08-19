"""
defense/dnc.py  –  DnC (Divide-and-Conquer) 鲁棒聚合
                   （Shejwalkar & Houmansadr, NDSS'21 "Manipulating the Byzantine"）

思路：恶意更新在主奇异方向上偏离良性群。多次随机投影 + 子空间奇异向量打分，累计离群分，
剔除分数最高的若干 client，剩余做样本加权平均。

流程（重复 num_projections 次，累计分数）：
  1. 随机抽 subspace_dim 个坐标（降维提速；维度过大时）。
  2. 取该子空间的更新，减去均值做中心化。
  3. 求中心化矩阵的最大右奇异向量 v。
  4. 离群分 = (centered_i · v)²。
最终剔除累计分最高的 num_remove 个 client（num_remove = filter_mult · num_malicious），
其余按样本数加权平均其完整权重。
"""

import numpy as np

from .base_defense import BaseDefense, flatten_weights, weighted_mean


class DnCDefense(BaseDefense):

    name = "dnc"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = self.config or {}
        self.num_malicious   = int(cfg.get("num_malicious", 1))
        self.num_projections = int(cfg.get("num_projections", 5))
        self.subspace_dim    = int(cfg.get("subspace_dim", 10000))
        self.filter_mult     = float(cfg.get("filter_mult", 1.0))

    def _ensure_rng(self):
        """随机投影必须可复现：同一 seed 下每轮抽到的子空间序列固定。"""
        if getattr(self, "rng", None) is None:
            self.rng = np.random.default_rng(
                int((self.config or {}).get("seed", 42)))

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        self._ensure_rng()
        m = len(client_updates)
        num_remove = int(np.ceil(self.filter_mult * self.num_malicious))
        num_remove = min(num_remove, m - 1)
        if num_remove < 1:
            self.last_admitted = list(range(m))      # 退化：全部接纳
            return weighted_mean(client_updates)

        flats = np.stack([flatten_weights(upd[0]) for upd in client_updates], axis=0)  # (m,d)
        d = flats.shape[1]
        sub_dim = min(self.subspace_dim, d)

        total_scores = np.zeros(m, dtype=np.float64)
        for _ in range(self.num_projections):
            idx = (self.rng.choice(d, sub_dim, replace=False)
                   if sub_dim < d else np.arange(d))
            sub = flats[:, idx]
            centered = sub - sub.mean(axis=0, keepdims=True)
            # 最大右奇异向量（对应最大奇异值方向）
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                v = vt[0]
            except np.linalg.LinAlgError:
                v = centered[np.argmax(np.linalg.norm(centered, axis=1))]
                v = v / (np.linalg.norm(v) + 1e-12)
            proj = centered @ v
            total_scores += proj ** 2

        # 剔除离群分最高的 num_remove 个，保留其余
        keep = np.argsort(total_scores)[: m - num_remove].tolist()
        self.last_admitted = sorted(int(i) for i in keep)
        print(f"    [Defense:dnc] m={m} remove {num_remove} → keep {len(keep)} clients "
              f"(kept ids by order: {sorted(keep)})")
        return weighted_mean(client_updates, indices=keep)
