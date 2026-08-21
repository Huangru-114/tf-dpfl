"""
models/autoencoder.py  –  Bad-PFL 触发器生成器（TF 版，对照官方 generator.py）

官方 Autoencoder（PyTorch）：
  Encoder 4×Conv2d(k=4,s=2,p=1)  通道 3→16→32→64→128，每层 +BN+ReLU
  Decoder 4×ConvTranspose2d(k=4,s=2,p=1) 通道 128→64→32→16→3，前 3 层 +BN+ReLU，末层 Tanh
输入 32×32 → Encoder 逐层减半 16→8→4→2 → Decoder 逐层加倍 4→8→16→32 → 输出 32×32×3 ∈ [-1,1]。

使用时输出再乘 epsilon（4/255 像素空间，标准化空间下逐通道换算）得到触发器扰动 δ。
本生成器**只在恶意客户端本地训练、不参与联邦聚合**。
"""

import tensorflow as tf


def build_autoencoder(img_size: int = 32, channels: int = 3) -> tf.keras.Model:
    """构建与官方等价的 Encoder-Decoder 触发器生成器（输出 ∈ [-1,1]，与输入同形状）。

    官方是固定 4 层、bottleneck 2×2 的结构，只在 32×32 上定义。4 次 stride-2
    半减后要能整数还原，要求 `img_size` 为 16 的倍数（8×8 在官方 torch 版同样
    会崩：bottleneck 退化为 0）。这里显式拦截，避免小尺寸下 decoder 输出形状
    与输入不符（如 img_size=8 → 输出 16×16），错误信息比广播 shape mismatch 清楚。
    """
    assert img_size % 16 == 0, (
        f"4-layer autoencoder 需 img_size 为 16 的倍数（bottleneck=img_size/16），"
        f"收到 img_size={img_size}。CIFAR 用 32；L1 测试请用 16。")
    L = tf.keras.layers
    inp = tf.keras.Input(shape=(img_size, img_size, channels))

    x = inp
    # ── Encoder：4×(Conv stride2 + BN + ReLU)，filters 16/32/64/128 ──────────
    for f in (16, 32, 64, 128):
        x = L.Conv2D(f, 4, strides=2, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)

    # ── Decoder：4×(ConvTranspose stride2)，filters 64/32/16/3 ───────────────
    for f in (64, 32, 16):
        x = L.Conv2DTranspose(f, 4, strides=2, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    x = L.Conv2DTranspose(channels, 4, strides=2, padding="same")(x)
    out = L.Activation("tanh")(x)

    return tf.keras.Model(inp, out, name="badpfl_generator")
