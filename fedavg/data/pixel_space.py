"""
data/pixel_space.py  —  模型输入空间的唯一定义（标准化常数、合法像素范围、ε 换算）

为什么单独成模块（AUDIT A10 / D-025，FINDINGS F-027）：
  本仓库喂给模型的是**逐通道标准化**的图，官方 Bad-PFL 是 [0,1]。凡是「像素空间里
  定义的量」—— Bad-PFL 的 ε/σ（4/255）、PGD 的 clamp 范围、BadNet 的白块 —— 都要
  换算到模型输入空间。以前这件事散在三处（dataset.py 标准化、client_badpfl.py 除 STD、
  triggers.py 自带一份常数），G7「官方预处理」一关标准化，三处不跟着变，ξ 预算就放大
  1/std ≈ 3.8–4.1 倍。现在三处都问这里。

`data.normalize`（默认 true，G7 设 false）经 alignment.get_switch 读取：
  true  → x_in = (x/255 − μ)/s，合法范围 [(0−μ)/s, (1−μ)/s]，ε_in = ε/s
  false → x_in = x/255，       合法范围 [0, 1]，              ε_in = ε

纯 numpy，**不 import TF**。
"""

from __future__ import annotations

import numpy as np

from alignment import get_switch

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

_STATS = {
    "cifar10":  (CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
}


def dataset_name(config: dict) -> str:
    return str(((config or {}).get("data") or {}).get("dataset", "cifar10")).lower()


def normalizes(config: dict) -> bool:
    return bool(get_switch(config or {}, "data.normalize"))


def dataset_stats(name: str):
    """数据集自己的 (μ, s)。未登记的数据集回退 CIFAR-10（与历史行为一致）。"""
    mean, std = _STATS.get(str(name).lower(), _STATS["cifar10"])
    return mean.copy(), std.copy()


def pixel_stats(config: dict):
    """模型输入空间的 (μ, s)，逐通道 float32。关标准化时是 (0, 1)。"""
    mean, std = dataset_stats(dataset_name(config))
    if not normalizes(config):
        return np.zeros_like(mean), np.ones_like(std)
    return mean, std


def valid_range(config: dict):
    """[0,1] 像素在模型输入空间里的上下界 (lo, hi)，逐通道。"""
    mean, std = pixel_stats(config)
    return ((0.0 - mean) / std).astype(np.float32), ((1.0 - mean) / std).astype(np.float32)


def to_input_space(eps: float, config: dict) -> np.ndarray:
    """像素空间的幅度（如 4/255）→ 模型输入空间，逐通道。"""
    _, std = pixel_stats(config)
    return (np.float32(eps) / std).astype(np.float32)


def normalize_images(x_uint8, config: dict) -> np.ndarray:
    """uint8 图 → 模型输入。与 dataset.load_cifar* 用的是同一个函数。"""
    mean, std = pixel_stats(config)
    return ((np.asarray(x_uint8).astype("float32") / 255.0 - mean) / std).astype(np.float32)


def to_unit(x_in, config: dict):
    """模型输入 → [0,1] 像素（官方生成器吃 [0,1] 图，A14 / D-020）。支持 numpy 与 tf 张量。"""
    mean, std = pixel_stats(config)
    return x_in * std + mean
