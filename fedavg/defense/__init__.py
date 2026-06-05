"""
defense/  –  后门防御模块（鲁棒聚合，参照 xtLyu/PFedBA 写在 server 基类）

5 种聚合层防御，对本仓库实际的 client 更新格式 List[np.ndarray] 直接运算：
  - trimmed_mean / median  坐标级去极值 / 中位数
  - multi_krum             两两欧氏距离选近邻
  - flame                  余弦聚类过滤 + 范数裁剪 + 加噪
  - dnc                    随机投影子空间 + 奇异向量离群打分

经 EdgeServerBase.robust_mean 统一接入所有 PFL 方法的 edge 聚合步。
"""

from .base_defense import BaseDefense
from .factory import create_defense

__all__ = ["BaseDefense", "create_defense"]
