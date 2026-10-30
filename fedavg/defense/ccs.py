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

纯 numpy + scikit-learn，不 import TF。统计量索引由 `EdgeServerBase.__init__` 建好防御后注入（`set_stat_indices`，F-107）。
"""

from __future__ import annotations

import numpy as np

from aggregation.fedavg import aggregate as fedavg_aggregate
from utils.kvline import fmt_list, format_kv
from analysis.bn_cluster import detect, stat_vector

from .base_defense import BaseDefense
from .ccs_spec import ccs_params, finite_upload


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
            raise RuntimeError("CCSDefense：没有注入 BN 统计量索引（应由 EdgeServerBase.__init__ 调 set_stat_indices，F-107）")
        return [stat_vector([u[0][i] for i in self.stat_idx]) for u in client_updates]

    def _finite_gate(self, client_updates, ids):
        """D-116：上传物含 NaN / inf 的端先剔除（不进聚类、不进聚合、不进缓存）；有效占比过低 → 停止训练。

        返回 (有效端的下标列表, 非有限端的 id 列表)。停止用 RuntimeError（edge 聚合在主线程，不会被
        `_collect_updates_*` 吞掉 → run 以非 0 退出，日志里有 [CCSStop] 行与 traceback）。
        """
        n = len(client_updates)
        ok = [j for j, u in enumerate(client_updates) if finite_upload(u[0])]
        bad = [ids[j] for j in range(n) if j not in set(ok)]
        frac = (len(ok) / n) if n else 1.0
        if n and frac < self.p["min_valid_frac"]:
            print("    " + format_kv("[CCSStop]", {"n_valid": len(ok), "n": n, "frac": round(frac, 4),
                                                 "min_valid_frac": self.p["min_valid_frac"],
                                                 "nonfinite": fmt_list(sorted(bad))}, edge_id=self.label))
            raise RuntimeError(f"CCS：本次 edge 聚合只有 {len(ok)}/{n} 个上传物是有限的（< {self.p['min_valid_frac']}），"
                               f"非有限的端 {sorted(bad)} —— 停止训练（D-116）")
        return ok, bad

    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        n = len(client_updates)
        ids_all = [int(getattr(u, "client_id", j)) for j, u in enumerate(client_updates)]
        ok, bad = self._finite_gate(client_updates, ids_all)
        if not self.p["cluster"]:
            self.last_admitted = list(ok)
            self.last_status = "off"
            if bad:
                print("    " + format_kv("[CCS]", {"cur": fmt_list(sorted(ids_all)), "n_pool": len(ok),
                                                 "status": "off", "rejected": fmt_list([]),
                                                 "n_valid": len(ok), "nonfinite": fmt_list(sorted(bad))},
                                         edge_id=self.label))
            return fedavg_aggregate([client_updates[j] for j in ok])

        ids = [ids_all[j] for j in ok]
        cur = dict(zip(ids, self._vectors([client_updates[j] for j in ok])))
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
        # 下标是相对**全部**上传的（record_decision 按它翻译成 client_id）；非有限的端不在其中 = 被剔除
        self.last_admitted = [j for j in ok if ids_all[j] not in rejected]
        # 剔除 TPR / FPR 的原料（[Decision] 行只有被剔除的 id，没有本轮有谁）。cur = 本轮全部上传（含非有限的端），
        # n_valid / nonfinite = D-116 的有限性检查（老日志没有这两列 → 解析成 None / []）
        print("    " + format_kv("[CCS]", {"cur": fmt_list(sorted(ids_all)), "n_pool": len(pool),
                                         "status": res["status"], "rejected": fmt_list(sorted(rejected)),
                                         "n_valid": len(ok), "nonfinite": fmt_list(sorted(bad))},
                                 edge_id=self.label))
        kept = [client_updates[j] for j in self.last_admitted]
        if not kept:
            # 只可能出在 edge_cache：最大簇全是缓存里的端 → 本轮上传全部剔除 → 本层模型不变
            return [np.array(w, copy=True) for w in ref_weights]
        return fedavg_aggregate(kept)
