"""
models/gtsrb_model.py
────────────────────────────────────────────────────────────────────────────
TensorFlow/Keras models for German Traffic Sign Recognition Benchmark (GTSRB).

GTSRB properties:
  • 43 classes
  • Variable-size images, typically resized to 32×32 or 48×48
  • Highly imbalanced class distribution

Two architectures:

1. GTSRBConvNet  – compact 4-layer CNN, lightweight enough for IoT / edge
2. GTSRBMobileNet – MobileNetV2-inspired depthwise-separable network,
                   good accuracy/parameter trade-off

Both are DP-compatible (GroupNorm, no BatchNorm).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Compact CNN for GTSRB ────────────────────────────────────────────────

def GTSRBConvNet(
    input_shape=(32, 32, 3),
    num_classes: int = 43,
) -> keras.Model:
    """
    4-layer convolutional network for GTSRB.

    Design choices:
      • Spatial Transformer-like preprocessing (simple affine augmentation)
      • Dropout for regularisation
      • GroupNorm for DP compatibility

    Based on: Sermanet & LeCun (2011) "Traffic Sign Recognition with
    Multi-Scale Convolutional Networks".
    """
    inputs = keras.Input(shape=input_shape, name="traffic_sign")
    x = inputs

    # Branch 1 (multi-scale: pool early feature map for final FC)
    x1 = layers.Conv2D(32, 5, padding="same", use_bias=False, name="c1")(x)
    x1 = layers.GroupNormalization(groups=16, name="gn1")(x1)
    x1 = layers.Activation("relu")(x1)
    x1 = layers.MaxPooling2D(2)(x1)          # → 16×16×32
    x1 = layers.Dropout(0.2)(x1)

    x2 = layers.Conv2D(64, 5, padding="same", use_bias=False, name="c2")(x1)
    x2 = layers.GroupNormalization(groups=32, name="gn2")(x2)
    x2 = layers.Activation("relu")(x2)
    x2 = layers.MaxPooling2D(2)(x2)          # → 8×8×64
    x2 = layers.Dropout(0.2)(x2)

    x3 = layers.Conv2D(128, 3, padding="same", use_bias=False, name="c3")(x2)
    x3 = layers.GroupNormalization(groups=32, name="gn3")(x3)
    x3 = layers.Activation("relu")(x3)
    x3 = layers.MaxPooling2D(2)(x3)          # → 4×4×128
    x3 = layers.Dropout(0.2)(x3)

    # Flatten + head
    f1 = layers.Flatten()(x1)               # 16×16×32 = 8192
    f2 = layers.Flatten()(x2)               # 8×8×64   = 4096
    f3 = layers.Flatten()(x3)               # 4×4×128  = 2048
    f = layers.Concatenate()([f1, f2, f3])   # multi-scale fusion

    f = layers.Dense(256, activation="relu", name="fc1")(f)
    f = layers.Dropout(0.3)(f)
    outputs = layers.Dense(num_classes, name="logits")(f)

    return keras.Model(inputs, outputs, name="GTSRBConvNet")


# ── Depthwise-separable MobileNet-style for GTSRB ────────────────────────

def _find_groups(channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def _dw_block(x, filters: int, stride: int = 1, expand: int = 6, name: str = "dw"):
    """Inverted residual block (MobileNetV2-style, no BN → GN)."""
    in_ch = x.shape[-1]
    expand_ch = in_ch * expand

    residual = x
    # Expand
    if expand != 1:
        x = layers.Conv2D(expand_ch, 1, use_bias=False, name=f"{name}_exp")(x)
        x = layers.GroupNormalization(groups=_find_groups(expand_ch), name=f"{name}_gn0")(x)
        x = layers.Activation("relu6")(x)
    # Depthwise
    dw_ch = expand_ch if expand != 1 else in_ch
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                                use_bias=False, name=f"{name}_dw")(x)
    x = layers.GroupNormalization(groups=_find_groups(dw_ch), name=f"{name}_gn1")(x)
    x = layers.Activation("relu6")(x)
    # Project
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_proj")(x)
    x = layers.GroupNormalization(groups=_find_groups(filters), name=f"{name}_gn2")(x)

    if stride == 1 and in_ch == filters:
        x = layers.Add()([x, residual])
    return x


def GTSRBMobileNet(
    input_shape=(32, 32, 3),
    num_classes: int = 43,
    width_multiplier: float = 1.0,
) -> keras.Model:
    """
    MobileNetV2-inspired network for GTSRB.

    Lightweight (~1.2M params at wm=1.0), suitable for edge/IoT inference.
    Replaces BatchNorm with GroupNorm for DP-SGD compatibility.

    Parameters
    ----------
    width_multiplier : float in (0, 2]
        Scale the number of channels. 0.5 for ultra-lightweight IoT models.
    """

    def ch(c):
        return max(1, int(c * width_multiplier))

    inputs = keras.Input(shape=input_shape, name="traffic_sign")
    x = layers.Conv2D(ch(32), 3, strides=1, padding="same",
                      use_bias=False, name="stem")(inputs)
    x = layers.GroupNormalization(groups=min(32, ch(32)), name="stem_gn")(x)
    x = layers.Activation("relu6")(x)

    # MobileNetV2 inverted residual blocks (adapted for 32×32)
    configs = [
        # (out_channels, stride, expand, n_blocks)
        (ch(16),  1, 1, 1),
        (ch(24),  1, 6, 2),
        (ch(32),  2, 6, 3),
        (ch(64),  2, 6, 4),
        (ch(96),  1, 6, 3),
        (ch(160), 2, 6, 3),
        (ch(320), 1, 6, 1),
    ]
    for block_idx, (out_ch, stride, expand, n) in enumerate(configs):
        for i in range(n):
            s = stride if i == 0 else 1
            x = _dw_block(x, out_ch, stride=s, expand=expand,
                          name=f"b{block_idx}_{i}")

    x = layers.Conv2D(ch(1280), 1, use_bias=False, name="head_conv")(x)
    x = layers.GroupNormalization(groups=min(32, ch(1280)), name="head_gn")(x)
    x = layers.Activation("relu6")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, outputs, name="GTSRBMobileNet")
