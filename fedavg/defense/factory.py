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
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown defense: {name!r}. Available: {['none'] + list(_REGISTRY)}")

    params = dict(dcfg.get("params", {}) or {})
    # num_malicious 缺省时自动估算（multi_krum / dnc 需要）
    if "num_malicious" not in params:
        params["num_malicious"] = _estimate_num_malicious(config)

    defense = _REGISTRY[name](params)
    print(f"[Defense] enabled: {name} | params={params}")
    return defense
