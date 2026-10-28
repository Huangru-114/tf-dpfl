"""
defense/ccs.py  —  CCS（Cui et al., ICASSP 2026）的服务端：按**上传物里**的 BN moving 统计量 HDBSCAN 剔除，剩余端样本加权平均

客户端那一半（良性端对抗训练）在 `client/ccs_mixin.py`，经 `client_mixin` 组合进所有客户端（compose.py）。
两个组件各有一个开关（`defense.ccs.at` / `cluster`）→ CCSP 的三臂 = 两个布尔（D-109）。

服务端照官方 `hdbscan_detect.py` 的意图（F-101：发布的版本取到的统计量是空的）：
  · 向量 = 上传物的全部 BN moving_mean / moving_variance（`models.cnn.get_bn_stat_indices` 的索引，升序拼接）；
    本仓库 FedRep（A27 / D-032）下上传物带的是 **body 阶段末**的统计量 —— 攻击端含投毒批次；
  · 聚类与剔除 = `analysis/bn_cluster.detect`（余弦距离、min_cluster_size = n // 2 + 2、min_samples = 1、
    allow_single_cluster、留最大簇；全噪声 / 退化 → 不剔除）；
  · 剩余端**按样本加权**（官方代码 `server/fedavg.py:727-736`；原文写的是无权平均 → 记为与原文的偏差，D-109）。
    走 `aggregation.fedavg.aggregate`（与 off 臂同一个函数、同一个顺序）→ 不剔除时与 off 臂的聚合逐位相同。
  · pool：`round` = 只聚本轮上传（官方）；`edge_cache` = 本轮上传 ∪ 本 edge 其余端最近一次上传（D-109 Q4，分层用）；
    每个 edge 一个防御实例（`RobustAggregationMixin._init_defense`），缓存天然按 edge 分开。

纯 numpy + scikit-learn，不 import TF。统计量索引由 main.py 建好 edge server 之后注入（`set_stat_indices`）。
"""

from __future__ import annotations

import numpy as np

from aggregation.fedavg import aggregate as fedavg_aggregate
from utils.kvline import fmt_list, format_kv
from analysis.bn_cluster import detect, stat_vector

from .base_defense import BaseDefense
from .ccs_spec import ccs_params


class _LazyClientMixin:
    """`client_mixin` 用到时才 import（mixin 依赖 TF；本模块要能在没有 TF 的 L1 里 import）。"""

    def __get__(self, obj, owner):
        from client.ccs_mixin import CCSClientMixin
        return CCSClientMixin


class CCSDefense(BaseDefense):
    name = "ccs"
    layers = frozenset({"client", "edge"})
    client_mixin = _LazyClientMixin()
    client_mixin_ref = "client.ccs_mixin.CCSClientMixin"   # config_validate 只核对「有没有」，不 import

    def __init__(self, ccs: dict | None = None):
        """ccs = `defense.ccs` 块（工厂经 from_config 传进来；不是 defense.params）。"""
        super().__init__(ccs)
        self.p = ccs_params({"defense": {"ccs": dict(ccs or {})}})
        self.stat_idx: list | None = None
        self._cache: dict = {}          # edge_cache：client_id → 最近一次上传的统计量向量
        self.last_status = None         # ok / all_noise / degenerate / empty / off
        self.label = None               # edge id（[CCS] 行的 edgeK 段）

    @classmethod
    def from_config(cls, config: dict):
        return cls(((config or {}).get("defense") or {}).get("ccs") or {})

    def set_stat_indices(self, stat_idx, label=None):
        self.stat_idx = [int(i) for i in stat_idx]
        self.label = None if label is None else int(label)

    def _vectors(self, client_updates):
        if self.stat_idx is None:
            raise RuntimeError("CCSDefense：没有注入 BN 统计量索引（main.py 应在建 edge server 后调 set_stat_indices）")
        return [stat_vector([u[0][i] for i in self.stat_idx]) for u in client_updates]

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        n = len(client_updates)
        if not self.p["cluster"]:
            self.last_admitted = list(range(n))
            self.last_status = "off"
            return fedavg_aggregate(client_updates)

        ids = [int(getattr(u, "client_id", j)) for j, u in enumerate(client_updates)]
        cur = dict(zip(ids, self._vectors(client_updates)))
        if self.p["pool"] == "edge_cache":
            pool = {k: v for k, v in self._cache.items() if k not in cur}
            pool.update(cur)
            self._cache.update(cur)
        else:
            pool = cur
        if self.p["pool"] == "edge_cache" and len(pool) < self.p["min_pool"]:
            res = {"admitted": sorted(pool), "rejected": [], "status": "degenerate"}
        else:
            res = detect(pool)
        rejected = set(res["rejected"]) & set(cur)
        self.last_status = res["status"]
        self.last_admitted = [j for j, cid in enumerate(ids) if cid not in rejected]
        # 剔除 TPR / FPR 的原料（[Decision] 行只有被剔除的 id，没有本轮有谁）
        print("    " + format_kv("[CCS]", {"cur": fmt_list(sorted(cur)), "n_pool": len(pool),
                                         "status": res["status"], "rejected": fmt_list(sorted(rejected))},
                                 edge_id=self.label))
        kept = [client_updates[j] for j in self.last_admitted]
        if not kept:
            # 只可能出在 edge_cache：最大簇全是缓存里的端 → 本轮上传全部剔除 → 本层模型不变
            return [np.array(w, copy=True) for w in ref_weights]
        return fedavg_aggregate(kept)
