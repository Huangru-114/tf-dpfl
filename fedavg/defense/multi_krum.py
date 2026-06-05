"""
defense/multi_krum.py  –  multi-Krum 鲁棒聚合（Blanchard et al., NeurIPS'17）

参照 PFedBA serverbase.py：展平各 client 权重 → 两两欧氏距离 → 每个 client 取与最近
(m − f − 2) 个邻居的距离和作为 Krum 分数 → 选分数最低的 (m − f) 个 client → 对其原始权重
按样本数加权平均。

f = 估计的恶意客户端数（per-edge，由 config 给出，默认按 n_malicious/n_edges 估算）。
本实现不偷看真实标签——f 是防御方的估计值。若 m ≤ 2f+2（Krum 可行性不满足），自动把 f
收缩到可行上界并告警；若仍不可行则回退普通加权平均。
"""

import numpy as np

from .base_defense import BaseDefense, flatten_weights, weighted_mean


class MultiKrumDefense(BaseDefense):

    name = "multi_krum"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.num_malicious = int((self.config or {}).get("num_malicious", 1))

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        m = len(client_updates)
        f = self.num_malicious

        # Krum 可行性：需要 m - f - 2 >= 1 且 m - f >= 1。收缩 f 到可行上界。
        if m - f - 2 < 1:
            f_feasible = max(0, (m - 3) // 2)
            if f_feasible != f:
                print(f"    [Defense:multi_krum] f={f} infeasible for m={m}, "
                      f"shrink to f={f_feasible}")
            f = f_feasible
        if m - f - 2 < 1:
            print(f"    [Defense:multi_krum] m={m} too small, fallback to weighted mean")
            return weighted_mean(client_updates)

        flats = np.stack([flatten_weights(upd[0]) for upd in client_updates], axis=0)  # (m, d)

        # 两两平方欧氏距离矩阵
        dist = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                d = np.sum((flats[i] - flats[j]) ** 2)
                dist[i, j] = dist[j, i] = d

        # Krum 分数：每个 client 到最近 (m - f - 2) 个邻居的距离和
        n_neighbors = m - f - 2
        scores = np.empty(m, dtype=np.float64)
        for i in range(m):
            nearest = np.sort(dist[i])[1:n_neighbors + 1]  # 排除自身(0)
            scores[i] = nearest.sum()

        # 选分数最低的 (m - f) 个 client 做加权平均
        n_select = m - f
        selected = np.argsort(scores)[:n_select].tolist()
        print(f"    [Defense:multi_krum] m={m} f={f} → selected {len(selected)} clients "
              f"(ids by order: {selected})")
        return weighted_mean(client_updates, indices=selected)
