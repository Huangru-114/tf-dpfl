"""
server/robust_aggregation.py  –  鲁棒聚合 mixin（edge 与 cloud 共用）

**为什么抽出来**：原来 `robust_mean` 只长在 `EdgeServerBase` 上，于是防御轴天然
只能作用在 edge 层，cloud 层完全不设防 —— 恶意更新只要过了 edge 那一关，到 cloud
就是无条件加权平均。抽成 mixin 后 `CloudServer` 也能继承，防御按层配置：

    defense:
      name:   flame
      layers: [edge]        # 默认，行为与改动前逐元素相同；未来可 [edge, cloud]

本会话只建通道，**不改默认行为**：`layers` 缺省为 `["edge"]`，cloud 层拿到的
defense 是 None → `robust_mean` 回退 `aggregation.fedavg.aggregate`，
与改动前的 `_aggregate_global` 逐元素相等（守卫见
tests/test_cloud_aggregate_default.py）。
"""

from aggregation.fedavg import aggregate as fedavg_aggregate
from defense import create_defense


class RobustAggregationMixin:
    """
    提供 `robust_mean`。使用者需在 __init__ 里调用 `_init_defense(config, layer)`，
    并实现 `_default_ref_weights()`（聚合前的本层模型权重，作为更新增量的参考点）。
    """

    def _init_defense(self, config: dict, layer: str):
        """按层构建防御实例；该层未启用时为 None（robust_mean 回退普通加权平均）。"""
        self.defense_layer = layer
        self.defense = create_defense(config, layer=layer)

    def _default_ref_weights(self) -> list:
        """本层聚合前的模型权重（广播点）。子类必须实现。"""
        raise NotImplementedError

    def _defense_label(self) -> str:
        """本层在日志里的标识（edge0 / cloud）。子类覆盖。"""
        return getattr(self, "defense_layer", "?")

    def _log_decision(self, updates: list):
        """
        发一条**格式统一**的防御判决行。

        **为什么统一在这里发**：各防御原本各打各的
        （flame「admitted 7/10」、multi_krum「selected 7 clients」、
        dnc「keep 7 clients」、trimmed_mean/median 干脆不打），
        于是回程解析器要么写 5 个脆弱正则、要么只认得 flame ——
        实际是后者，`admitted_count` 这个 CLAUDE.md 要求的字段
        **5 个防御里只有 1 个读得出来**，defense 轴 4/5 是瞎的。

        robust_mean 是所有 PFL 方法、两个层都已经收口经过的唯一入口，
        在这里发一次，新增防御自动被覆盖，解析器只需要一条正则。
        """
        name = getattr(self.defense, "name", "?")
        total = len(updates)
        ids = getattr(self.defense, "last_admitted_ids", None)
        if ids is None:
            # 坐标类防御（trimmed_mean / median）不做客户端级筛选
            print(f"    [Decision] {self._defense_label()} | {name} | "
                  f"coordinate-wise | n={total}")
            return
        rejected = getattr(self.defense, "last_rejected_ids", None) or []
        print(f"    [Decision] {self._defense_label()} | {name} | "
              f"admitted {len(ids)}/{total} | rejected={list(rejected)}")

    def robust_mean(self, updates: list, ref_weights: list = None) -> list:
        """
        统一的「加权平均步」替身：有防御时走鲁棒聚合，否则回退普通样本加权 FedAvg。

        Args:
            updates:     [(weights, n_samples, loss, train_time), ...]
            ref_weights: 聚合前本层模型权重（广播点）；None 时取 _default_ref_weights()。
                         FLAME/DnC 用作更新增量参考；坐标类防御忽略。
        Returns:
            List[np.ndarray]：聚合后的完整权重列表（与 aggregate 同形）。
        """
        if getattr(self, "defense", None) is None:
            return fedavg_aggregate(updates)
        if ref_weights is None:
            ref_weights = self._default_ref_weights()
        agg = self.defense.aggregate(updates, ref_weights)
        # 防御接纳/剔除的判决按 client_id 记账（不是位置索引），供 TPR/FPR 统计
        self.defense.record_decision(updates)
        self._log_decision(updates)
        return agg
