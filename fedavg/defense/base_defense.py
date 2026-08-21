"""
defense/base_defense.py  –  鲁棒聚合防御基类（参照 xtLyu/PFedBA 把防御写在 server 基类）

设计要点（与本仓库实际数据格式对齐）：
  - 客户端更新格式与 aggregation/fedavg.aggregate 完全一致：
        client_updates = [(weights, n_samples, loss, train_time), ...]
        weights = List[np.ndarray]   # 模型每层权重，get_weights() 顺序
  - 防御对 List[np.ndarray] 直接运算（不是 dict[str, tf.Tensor]），纯 numpy，无 TF/GPU 依赖。
  - aggregate() 统一返回 List[np.ndarray]（聚合后的完整权重列表），与 aggregate 同形，
    便于各 edge server 无差别替换其「加权平均步」（FedAvg 整体、pFedMe/Ditto 的 mean、
    FedRep 取 backbone 索引）。

参考：
  - Trimmed Mean / multi-Krum：PFedBA serverbase.py（展平 client 参数 + 排序去极值 / 两两欧氏距离）。
  - FLAME（Nguyen et al., USENIX'22）：余弦聚类过滤 + 范数裁剪 + 加噪。
  - DnC（Shejwalkar & Houmansadr, NDSS'21）：随机投影子空间 + 奇异向量离群打分。
"""

from abc import ABC, abstractmethod

import numpy as np


def flatten_weights(weights: list) -> np.ndarray:
    """把 List[np.ndarray]（每层权重）展平为单个一维向量。"""
    return np.concatenate([w.reshape(-1) for w in weights]).astype(np.float64)


def unflatten_weights(flat: np.ndarray, ref_weights: list) -> list:
    """把一维向量按 ref_weights 的层形状还原为 List[np.ndarray]（dtype 跟随 ref）。"""
    out, offset = [], 0
    for w in ref_weights:
        size = w.size
        out.append(flat[offset:offset + size].reshape(w.shape).astype(w.dtype))
        offset += size
    return out


def weighted_mean(client_updates: list, indices=None) -> list:
    """对 client_updates（可选子集 indices）做样本数加权平均，返回 List[np.ndarray]。"""
    subset = client_updates if indices is None else [client_updates[i] for i in indices]
    total = sum(n for _, n, *_ in subset)
    agg = [np.zeros_like(w) for w in subset[0][0]]
    for weights, n, *_ in subset:
        coeff = n / total
        for i, lw in enumerate(weights):
            agg[i] += coeff * lw
    return agg


class BaseDefense(ABC):
    """鲁棒聚合防御基类。"""

    name = "base"

    # ── 本防御**能**作用在哪几层（类属性，单一事实来源）────────────────────
    # ⊆ {"client", "edge", "cloud", "post_hoc"}
    #   client   需要客户端配合（主动防御）：必须同时给出 client_mixin
    #   edge     edge 层鲁棒聚合（现有 5 种都是这一类）
    #   cloud    cloud 层鲁棒聚合
    #   post_hoc 训练后处理（Simple-Tuning）
    #
    # 用户在 config 里用 `defense.layers` 选**实际启用**哪几层；
    # config_validate 校验 `config.layers ⊆ cls.layers`，声明与接线对不上就拒绝启动
    # —— 与 METHOD_SUPPORTS_DEFENSE 同一套「启动前挡住静默跑错」的思路。
    #
    # 默认 {"edge", "cloud"}：本基类下的 5 种防御都只依赖
    # 「一批 (weights, n_samples, …) + 一个参考权重」这个接口，与这批更新来自
    # client 还是来自 edge 无关，所以两层都能跑。默认**启用**的层仍由 config 决定
    # （defense.layers 缺省 ["edge"]），本属性只说明「能不能」，不说明「要不要」。
    layers = frozenset({"edge", "cloud"})

    # 声明了 "client" 的防御必须给一个客户端行为 mixin（见 client/compose.py）。
    # 它会被组合到**所有**客户端（防御方分不出谁是恶意的），且在攻击 mixin 内层。
    client_mixin = None

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # ── 判决记账 ──────────────────────────────────────────────────────
        # last_admitted      : 本轮接纳的**位置索引**（子类在 aggregate 里填）
        # last_admitted_ids  : 对应的 client_id（record_decision 填）
        # last_rejected_ids  : 被剔除的 client_id
        # 筛选类防御（flame / multi_krum / dnc）必须填 last_admitted；
        # 坐标类防御（trimmed_mean / median）不做客户端级筛选，保持 None。
        self.last_admitted     = None
        self.last_admitted_ids = None
        self.last_rejected_ids = None

    def record_decision(self, client_updates: list):
        """
        把 last_admitted（位置索引）翻译成 client_id。

        **为什么必须有这一步**：位置索引只在本次调用内有意义，无法回答
        「防御到底剔掉了谁」——而防御轴的核心指标（检出率 TPR / 误伤率 FPR）
        只能按 client_id 算。由 EdgeServerBase.robust_mean 在聚合后统一调用。
        """
        ids = [getattr(u, "client_id", None) for u in client_updates]
        if self.last_admitted is None:          # 坐标类防御：无客户端级判决
            self.last_admitted_ids = None
            self.last_rejected_ids = None
            return
        admitted = set(int(i) for i in self.last_admitted)
        self.last_admitted_ids = [ids[i] for i in sorted(admitted)
                                  if 0 <= i < len(ids)]
        self.last_rejected_ids = [cid for j, cid in enumerate(ids)
                                  if j not in admitted]

    def make_control(self, server, round_idx: int) -> dict:
        """
        本轮下发给客户端的额外载荷（主动防御用：裁剪上界、挑战样本、参考统计量…）。

        由 `EdgeServerBase.broadcast_to_clients` 每次广播时调用，最终进到
        `FLClientBase.set_control`。默认空 dict —— 现有 5 种聚合层防御都不需要
        客户端配合，返回 {} 时下行路径与改动前逐元素相同。
        """
        return {}

    @abstractmethod
    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        """
        Args:
            client_updates: [(weights, n_samples, loss, train_time), ...]
            ref_weights:    List[np.ndarray]，聚合前 edge 模型权重（广播点），
                            FLAME/DnC 用作更新增量参考；坐标类防御忽略。
        Returns:
            List[np.ndarray]：鲁棒聚合后的完整权重列表。
        """
        raise NotImplementedError

    def __repr__(self):
        return f"<Defense:{self.name}>"
