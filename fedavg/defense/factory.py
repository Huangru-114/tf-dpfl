"""
defense/factory.py  –  防御工厂

按 config["defense"] 构建防御实例。name="none"（或缺省）返回 None，表示不设防——
EdgeServerBase.robust_mean 会回退到原始 aggregation.fedavg.aggregate，完全向后兼容。

config 结构（见 config.yaml）：
  defense:
    name:   "none" | "trimmed_mean" | "median" | "multi_krum" | "flame" | "dnc"
    params: {trim_ratio, num_malicious, noise_scale, num_projections, subspace_dim, ...}

num_malicious 缺省时按 ceil(backdoor.n_malicious / n_edges) 估算（防御方估计值，
不偷看真实标签）。
"""

import math

from .trimmed_mean import TrimmedMeanDefense, MedianDefense
from .multi_krum import MultiKrumDefense
from .flame import FlameDefense
from .dnc import DnCDefense

_REGISTRY = {
    "trimmed_mean": TrimmedMeanDefense,
    "median":       MedianDefense,
    "multi_krum":   MultiKrumDefense,
    "flame":        FlameDefense,
    "dnc":          DnCDefense,
}

# 训练后、客户端侧的后处理防御（不在聚合层；见 simple_tuning.py）。
# 这些名字在 create_defense 中返回 None（聚合不变），由 create_post_hoc_defense 单独构建。
_POST_HOC = {"simple_tuning"}


def _estimate_num_malicious(config: dict) -> int:
    """按 backdoor.n_malicious / n_edges 估算 per-edge 恶意数（至少 1）。"""
    bd = config.get("backdoor", {}) or {}
    fed = config.get("federation", {}) or {}
    n_mal = int(bd.get("n_malicious", 0) or 0)
    n_edges = max(1, int(fed.get("n_edges", 1) or 1))
    if n_mal <= 0:
        return 1
    return max(1, math.ceil(n_mal / n_edges))


def create_defense(config: dict):
    """根据 config 构建防御实例；name=none 返回 None。"""
    dcfg = (config.get("defense") or {}) if isinstance(config, dict) else {}
    name = str(dcfg.get("name", "none")).lower()
    if name in ("none", "", "off", "disabled"):
        return None
    if name in _POST_HOC:
        # 后处理防御不在聚合层生效 → 聚合回退普通加权平均（见 create_post_hoc_defense）。
        return None
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown defense: {name!r}. "
            f"Available: {['none'] + list(_REGISTRY) + list(_POST_HOC)}")

    params = dict(dcfg.get("params", {}) or {})
    # num_malicious 缺省时自动估算（multi_krum / dnc 需要）
    if "num_malicious" not in params:
        params["num_malicious"] = _estimate_num_malicious(config)

    defense = _REGISTRY[name](params)
    print(f"[Defense] enabled: {name} | params={params}")
    return defense


def create_post_hoc_defense(config: dict, num_classes: int, lr_default: float):
    """
    构建训练后、客户端侧的后处理防御（目前仅 Simple-Tuning）。

    返回带 .tune(model, clean_dataset) 方法的对象；name 非后处理类时返回 None。
    在 BackdoorCloudServer 的分层评估处接入：对每个良性 client 的本地模型先 tune 再测 ASR。
    """
    from .simple_tuning import SimpleTuningDefense

    dcfg = (config.get("defense") or {}) if isinstance(config, dict) else {}
    name = str(dcfg.get("name", "none")).lower()
    if name not in _POST_HOC:
        return None
    return SimpleTuningDefense(config, num_classes=num_classes, lr_default=lr_default)
