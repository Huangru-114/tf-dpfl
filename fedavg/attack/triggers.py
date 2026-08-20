"""
attack/triggers.py  –  后门触发器（在标准化后的图像空间上操作）

数据约定（见 data/dataset.py）：
  CIFAR-10 图像逐通道标准化 normalized = (raw/255 - CIFAR10_MEAN) / CIFAR10_STD。
  因此「像素值=1.0（白）」在标准化空间对应 (1.0 - mean) / std（逐通道）。
  Blended 在标准化空间做线性混合，与在 [0,1] 像素空间混合再标准化等价：
    ((1-α)c + α t - m)/s = (1-α)(c-m)/s + α(t-m)/s

所有 trigger 函数输入/输出均为「已标准化的 numpy 数组」(N,H,W,C) float32。
"""

import os

import numpy as np
import tensorflow as tf

# 与 data/dataset.py 中的 CIFAR-10 逐通道标准化常数保持一致（此处本地定义，
# 避免 import data.dataset 连带引入 tensorflow_datasets 等重依赖）。
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def make_badnet_trigger(mean=CIFAR10_MEAN, std=CIFAR10_STD, size=3, value=1.0):
    """
    BadNet：在右下角放 size×size 的白色方块（像素值=value，默认 1.0）。
    返回一个作用于标准化 numpy batch 的函数。
    """
    val_std = (np.float32(value) - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)  # (C,)

    def apply(x):
        x = np.array(x, dtype=np.float32, copy=True)
        x[:, -size:, -size:, :] = val_std  # 广播到 (size,size,C)
        return x

    return apply


def make_blended_trigger(image_path, mean=CIFAR10_MEAN, std=CIFAR10_STD,
                         alpha=0.2, img_size=32):
    """
    Blended：把 trigger 图案以 α 混合叠加：poisoned = (1-α)·clean + α·trigger。
    trigger 图案从 image_path 读取，resize 到 img_size 并按 CIFAR 同样方式标准化。
    """
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size]).numpy().astype("float32") / 255.0
    trig_std = (img - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)  # (H,W,C)
    a = float(alpha)

    def apply(x):
        x = np.array(x, dtype=np.float32, copy=True)
        return (1.0 - a) * x + a * trig_std

    return apply


def make_dba_local_trigger(poison_pattern, mean=CIFAR10_MEAN, std=CIFAR10_STD,
                           value=1.0):
    """
    DBA 局部触发器（AI-secure/DBA, image_helper.add_pixel_pattern）：
    把 poison_pattern 列出的像素坐标 [[r,c], ...] 置为 value（标准化空间）。
    每个恶意客户端分到一个局部 pattern，本地投毒只用自己的 pattern。

    Args:
        poison_pattern : 坐标列表，如 [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
    返回作用于标准化 numpy batch 的函数 apply(x)。
    """
    val_std = (np.float32(value) - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)  # (C,)
    coords  = np.asarray(poison_pattern, dtype=np.int64)
    rows, cols = coords[:, 0], coords[:, 1]

    def apply(x):
        x = np.array(x, dtype=np.float32, copy=True)
        x[:, rows, cols, :] = val_std   # (N,P,C) ← (C,) 广播
        return x

    return apply


def make_dba_global_trigger(all_patterns, mean=CIFAR10_MEAN, std=CIFAR10_STD,
                            value=1.0):
    """
    DBA 全局触发器：所有局部 pattern 的坐标合并（ASR 评估用，与官方
    centralized_test_trigger=True 一致）。
    Args:
        all_patterns : List[局部 pattern]，每个是坐标列表
    """
    merged = [pt for pattern in all_patterns for pt in pattern]
    return make_dba_local_trigger(merged, mean=mean, std=std, value=value)


def build_trigger(bd_cfg, img_size=32):
    """
    根据 config['backdoor'] 构建**投毒/评估侧静态 trigger 函数** apply(x)。

    DBA：返回**全局触发器**（合并所有局部 pattern），用于 ASR 评估；
    各恶意客户端的局部投毒触发器由 attack/backdoor.build_dba_poisoned_datasets 单独构建。
    """
    kind = bd_cfg.get("trigger", "badnet")
    if kind == "badnet":
        return make_badnet_trigger(
            size=int(bd_cfg.get("badnet_size", 3)),
            value=float(bd_cfg.get("badnet_value", 1.0)),
        )
    if kind == "blended":
        path = bd_cfg.get("blended_image", "attack/triggers/hello_kitty.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Blended trigger image not found: {path!r}. "
                f"请把 hello-kitty 图片放到该路径，或设置 backdoor.blended_image。"
            )
        return make_blended_trigger(
            path,
            alpha=float(bd_cfg.get("blended_alpha", 0.2)),
            img_size=int(img_size),
        )
    if kind == "dba":
        patterns = bd_cfg.get("dba_patterns")
        if not patterns:
            raise ValueError("trigger='dba' 需要 backdoor.dba_patterns（局部触发器坐标列表）。")
        return make_dba_global_trigger(
            patterns, value=float(bd_cfg.get("badnet_value", 1.0)))
    raise ValueError(f"Unknown backdoor trigger: {kind!r} (choose 'badnet'/'blended'/'dba')")
