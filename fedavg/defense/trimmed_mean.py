"""
defense/trimmed_mean.py  –  坐标级（coordinate-wise）鲁棒聚合

TrimmedMeanDefense：逐坐标排序后去掉两端各 trim_ratio 比例的极值再平均
                    （Yin et al., ICML'18 "Byzantine-Robust Distributed Learning"）。
MedianDefense：     逐坐标取中位数（= trim_ratio→0.5 的极限）。

两者均按层逐坐标运算，不需要展平，天然兼容任意层形状。样本数不参与（坐标级防御
对每个坐标独立排序，加权语义不明确，按等权处理——与文献一致）。
"""

import numpy as np

from .base_defense import BaseDefense, weighted_mean


class TrimmedMeanDefense(BaseDefense):

    name = "trimmed_mean"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.trim_ratio = float((self.config or {}).get("trim_ratio", 0.1))

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        m = len(client_updates)
        k = int(np.floor(self.trim_ratio * m))
        # 去极值后至少要留 1 个；若过度修剪则回退普通加权平均。
        if m - 2 * k < 1:
            return weighted_mean(client_updates)

        agg = []
        n_layers = len(client_updates[0][0])
        for j in range(n_layers):
            stacked = np.stack([upd[0][j] for upd in client_updates], axis=0)  # (m, *shape)
            ordered = np.sort(stacked, axis=0)
            trimmed = ordered[k:m - k] if k > 0 else ordered
            agg.append(np.mean(trimmed, axis=0).astype(client_updates[0][0][j].dtype))
        return agg


class MedianDefense(BaseDefense):

    name = "median"

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        agg = []
        n_layers = len(client_updates[0][0])
        for j in range(n_layers):
            stacked = np.stack([upd[0][j] for upd in client_updates], axis=0)
            agg.append(np.median(stacked, axis=0).astype(client_updates[0][0][j].dtype))
        return agg
