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


# ══════════════════════════════════════════════════════════════════════════
# A14 / D-020：结构、初始化、BN 对齐官方 generator.py（开关 backdoor.badpfl_generator）
# ══════════════════════════════════════════════════════════════════════════
#
# 官方 `generator.py:6-40`（2026-09-26 按 raw 重读）：
#   encoder 4×[Conv2d(k4,s2,p1) + BN + ReLU]，3→16→32→64→128
#   decoder 3×[ConvTranspose2d(k4,s2,p1) + BN + ReLU]，128→64→32→16，
#           末层 ConvTranspose2d(16→3) + Tanh（**无 BN**；论文表 5 写了 BN，不采纳）
#   全部带 bias（torch 默认）；BN 是 torch 默认 eps 1e-5、momentum 0.1。
#
# 与旧版（build_autoencoder）的差别，逐项：
#   · bias：旧版前 7 层 use_bias=False → 加上
#   · 初始化：旧版 glorot_uniform、bias 0 → torch 默认 kaiming_uniform(a=√5)，
#     方差 1/(3·fan_in) = VarianceScaling(1/3, "fan_in", "uniform")；bias ~ U(±1/√fan_in)。
#     ConvTranspose 的 fan_in：torch 权重形状 (in, out, k, k) 取 size(1)·k² = out·k²；
#     Keras ConvT kernel 形状 (k, k, out, in)，VarianceScaling 取 k²·shape[-2] = out·k²，
#     两边同一个数 —— bias 的界也按它算。
#   · BN：eps 1e-5；momentum 0.9（= torch 0.1）。**统计量在 official 模式下不会被用到**
#     （A05：投毒 / 评估时生成器一律用 batch 统计），写上只为与官方一致。
#   · padding 保留 "same"：偶数输入、k4 s2 时 SAME 的 pad_total = 2 → 前后各 1，
#     与 torch p1 逐元素相等（L1：tests/test_badpfl_official.py）。
#   · 输入：官方吃 [0,1] 图。调用方负责把标准化图反变换回去（client_badpfl._atk_gen_input）。

def _torch_default_init(fan_in: int):
    """(kernel_initializer, bias_initializer)，复刻 torch Conv/ConvT 的默认初始化。"""
    import math
    k = tf.keras.initializers.VarianceScaling(scale=1.0 / 3.0, mode="fan_in",
                                              distribution="uniform")
    bound = 1.0 / math.sqrt(fan_in)
    b = tf.keras.initializers.RandomUniform(-bound, bound)
    return k, b


def build_autoencoder_official(img_size: int = 32, channels: int = 3) -> tf.keras.Model:
    """官方 Autoencoder 的 TF 复刻。输入 [0,1] 图，输出 ∈ [-1,1]、与输入同形状。"""
    assert img_size % 16 == 0, (
        f"4-layer autoencoder 需 img_size 为 16 的倍数，收到 img_size={img_size}。")
    L = tf.keras.layers
    inp = tf.keras.Input(shape=(img_size, img_size, channels))

    def bn():
        return L.BatchNormalization(momentum=0.9, epsilon=1e-5)

    x, c_in = inp, channels
    for f in (16, 32, 64, 128):
        ki, bi = _torch_default_init(c_in * 16)          # Conv2d：fan_in = in·k²
        x = L.Conv2D(f, 4, strides=2, padding="same", use_bias=True,
                     kernel_initializer=ki, bias_initializer=bi)(x)
        x = bn()(x)
        x = L.ReLU()(x)
        c_in = f
    for f in (64, 32, 16):
        ki, bi = _torch_default_init(f * 16)             # ConvT：fan_in = out·k²
        x = L.Conv2DTranspose(f, 4, strides=2, padding="same", use_bias=True,
                              kernel_initializer=ki, bias_initializer=bi)(x)
        x = bn()(x)
        x = L.ReLU()(x)
    ki, bi = _torch_default_init(channels * 16)
    x = L.Conv2DTranspose(channels, 4, strides=2, padding="same", use_bias=True,
                          kernel_initializer=ki, bias_initializer=bi)(x)
    out = L.Activation("tanh")(x)
    return tf.keras.Model(inp, out, name="badpfl_generator_official")


def build_generator(config: dict, img_size: int = None, channels: int = 3) -> tf.keras.Model:
    """按开关 backdoor.badpfl_generator 选生成器（legacy = 旧版，默认）。"""
    from alignment import get_switch
    img = int(img_size if img_size is not None else config["data"]["img_size"])
    if get_switch(config, "backdoor.badpfl_generator") == "official":
        return build_autoencoder_official(img_size=img, channels=channels)
    return build_autoencoder(img_size=img, channels=channels)
