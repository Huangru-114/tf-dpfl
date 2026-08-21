"""
defense/  –  后门防御模块（鲁棒聚合，参照 xtLyu/PFedBA 写在 server 基类）

5 种聚合层防御，对本仓库实际的 client 更新格式 List[np.ndarray] 直接运算：
  - trimmed_mean / median  坐标级去极值 / 中位数
  - multi_krum             两两欧氏距离选近邻
  - flame                  余弦聚类过滤 + 范数裁剪 + 加噪
  - dnc                    随机投影子空间 + 奇异向量离群打分

经 RobustAggregationMixin.robust_mean 统一接入所有 PFL 方法的聚合步
（EdgeServerBase 与 CloudServer 共用同一份实现）。

防御按层启用：`defense.layers` 缺省 ["edge"]，与改动前行为一致。
每个防御类用 `layers` 类属性声明自己**能**作用在哪几层；
需要客户端配合的主动防御另给 `client_mixin`（组合进所有客户端，见 client/compose.py）。
"""

from .base_defense import BaseDefense
from .factory import (create_defense, create_post_hoc_defense,
                      configured_layers, defense_class, DEFAULT_LAYERS)

__all__ = ["BaseDefense", "create_defense", "create_post_hoc_defense",
           "configured_layers", "defense_class", "DEFAULT_LAYERS"]
