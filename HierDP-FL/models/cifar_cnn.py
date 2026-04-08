"""
models/cifar_cnn.py
────────────────────────────────────────────────────────────────────────────
TensorFlow/Keras CNN models for CIFAR-10 / CIFAR-100.

Architectures
─────────────
• CifarCNN3Conv  – lightweight 3-conv CNN (mirrors HierFL's cifar_cnn_3conv)
• CifarResNet18  – ResNet-18 adapted for 32×32 inputs

Note on DP compatibility
─────────────────────────
Batch Normalization is NOT used here because DP-SGD requires per-sample
gradient computation, which is incompatible with BatchNorm's cross-sample
statistics.  GroupNorm / LayerNorm are used instead.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Helper: GroupNorm ────────────────────────────────────────────────────

def group_norm(x, groups: int = 32, name: str = "gn"):
    """
    Group Normalization (compatible with DP per-sample gradients).
    Falls back to LayerNorm when channels < groups.
    """
    channels = x.shape[-1]
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return layers.GroupNormalization(groups=g, name=name)(x)


# ── CifarCNN3Conv ────────────────────────────────────────────────────────
def CifarCNN3Conv(
    input_shape: tuple = (32, 32, 3),
    num_classes: int = 10,
    use_group_norm: bool = True,
) -> keras.Model:
    """
    2-layer convolutional CNN for CIFAR-10 with SELU activations.

    Architecture mirrors CNNCifar_SELU_full:
      Conv(64,5) → [GN] → SELU → MaxPool(3,2)
      Conv(64,5) → [GN] → SELU → MaxPool(3,2)
      Flatten → Dense(384) → SELU → Dense(192) → SELU → Dense(num_classes)

    DP notes: GroupNorm instead of BatchNorm; SELU for self-normalizing properties.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs

    for filters in [64, 64]:
        x = layers.Conv2D(filters, 5, padding="valid", use_bias=not use_group_norm)(x)
        if use_group_norm:
            x = layers.GroupNormalization(groups=min(32, filters))(x)
        x = layers.Activation("selu")(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(384, activation="selu")(x)
    x = layers.Dense(192, activation="selu")(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, outputs, name="CifarCNN3Conv")

# def CifarCNN3Conv(
#     input_shape=(32, 32, 3),
#     num_classes: int = 10,
#     use_group_norm: bool = True,
# ) -> keras.Model:
#     """
#     3-layer convolutional CNN for CIFAR-10.

#     Architecture mirrors HierFL's `cifar_cnn_3conv_layer`:
#       Conv(64) → GN → ReLU → MaxPool
#       Conv(128) → GN → ReLU → MaxPool
#       Conv(256) → GN → ReLU → MaxPool
#       Flatten → Dense(512) → Dense(num_classes)

#     DP notes: GroupNorm instead of BatchNorm.
#     """
#     inputs = keras.Input(shape=input_shape, name="image")
#     x = inputs

#     for filters, pool in [(64, True), (128, True), (256, True)]:
#         x = layers.Conv2D(filters, 3, padding="same", use_bias=not use_group_norm)(x)
#         if use_group_norm:
#             x = layers.GroupNormalization(groups=min(32, filters))(x)
#         x = layers.Activation("relu")(x)
#         if pool:
#             x = layers.MaxPooling2D(2)(x)

#     x = layers.Flatten()(x)
#     x = layers.Dense(512, activation="relu")(x)
#     outputs = layers.Dense(num_classes, name="logits")(x)

#     return keras.Model(inputs, outputs, name="CifarCNN3Conv")


# ── ResNet-18 for 32×32 inputs ───────────────────────────────────────────

def _resnet_block(x, filters: int, stride: int = 1, name: str = "rb"):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False, name=f"{name}_c1")(x)
    x = layers.GroupNormalization(groups=min(32, filters), name=f"{name}_gn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same",
                      use_bias=False, name=f"{name}_c2")(x)
    x = layers.GroupNormalization(groups=min(32, filters), name=f"{name}_gn2")(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 use_bias=False, name=f"{name}_sc")(shortcut)
        shortcut = layers.GroupNormalization(groups=min(32, filters),
                                             name=f"{name}_scgn")(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def CifarResNet18(
    input_shape=(32, 32, 3),
    num_classes: int = 10,
) -> keras.Model:
    """
    ResNet-18 adapted for 32×32 CIFAR inputs.
    Uses GroupNorm instead of BatchNorm for DP compatibility.
    """
    inputs = keras.Input(shape=input_shape, name="image")

    # Initial conv (smaller than ImageNet version: 3×3 instead of 7×7, no MaxPool)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.GroupNormalization(groups=32, name="stem_gn")(x)
    x = layers.Activation("relu")(x)

    # ResNet stages: [64×2, 128×2, 256×2, 512×2]
    for i, (filters, stride) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)]):
        for j in range(2):
            s = stride if j == 0 else 1
            x = _resnet_block(x, filters, stride=s, name=f"s{i}b{j}")

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    return keras.Model(inputs, outputs, name="CifarResNet18")
