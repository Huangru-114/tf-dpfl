import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ─────────────────────────────────────────────────────────
# 工具函数：GroupNorm（FL友好，替代BN）
# ─────────────────────────────────────────────────────────
def gn(x, groups=8):
    return layers.GroupNormalization(groups=groups)(x)


# ─────────────────────────────────────────────────────────
# 模型 1：ResNet-56  (~0.85M, ~93-94%)
# 最轻量，不一定够95%，但可作为对照
# ─────────────────────────────────────────────────────────
def _resnet_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = gn(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = gn(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = gn(shortcut)
    return layers.ReLU()(x + shortcut)

def build_resnet56(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    x = gn(x)
    x = layers.ReLU()(x)
    for _ in range(9):   # 9 blocks × 3 stages = ResNet-56
        x = _resnet_block(x, 16)
    x = _resnet_block(x, 32, stride=2)
    for _ in range(8):
        x = _resnet_block(x, 32)
    x = _resnet_block(x, 64, stride=2)
    for _ in range(8):
        x = _resnet_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="resnet56")


# ─────────────────────────────────────────────────────────
# 模型 2：WideResNet-28-4  (~5.9M, ~95.5%)  ← 推荐主力基线
# width_factor=4 是 95% 的甜蜜点，参数量仍远低于 10M
#现在实际设定为16-4，训练时间减半，仅降低1.6%准确率，FL通信成本大幅降低
# ─────────────────────────────────────────────────────────
def _wide_block(x, filters, stride=1, dropout_rate=0.3):
    shortcut = x
    x = gn(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = gn(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
    return x + shortcut

def build_wideresnet28_4(input_shape=(32, 32, 3), num_classes=10):
    k            = 4          # width factor
    n_blocks     = 2          # (28 - 4) / 6 = 4
    base_filters = [16, 16*k, 32*k, 64*k]   # [16, 64, 128, 256]

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(base_filters[0], 3, padding="same", use_bias=False)(inputs)

    for stage, filters in enumerate(base_filters[1:]):
        stride = 1 if stage == 0 else 2
        x = _wide_block(x, filters, stride=stride)
        for _ in range(n_blocks - 1):
            x = _wide_block(x, filters)

    x = gn(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="wideresnet28_4")


# ─────────────────────────────────────────────────────────
# 模型 3：DenseNet-BC-L100-k12  (~0.8M, ~95.3%)
# 参数最少却能到95%，FL通信成本最低；代价是训练显存略高
# ─────────────────────────────────────────────────────────
def _dense_layer(x, growth_rate):
    out = gn(x, groups=min(8, x.shape[-1]))
    out = layers.ReLU()(out)
    out = layers.Conv2D(4 * growth_rate, 1, use_bias=False)(out)   # bottleneck
    out = gn(out, groups=min(8, out.shape[-1]))
    out = layers.ReLU()(out)
    out = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)(out)
    return layers.Concatenate()([x, out])

def _dense_block(x, n_layers, growth_rate):
    for _ in range(n_layers):
        x = _dense_layer(x, growth_rate)
    return x

def _transition(x, compression=0.5):
    filters = max(1, int(x.shape[-1] * compression))
    x = gn(x, groups=min(8, x.shape[-1]))
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, use_bias=False)(x)
    return layers.AveragePooling2D(2)(x)

def build_densenet_bc(input_shape=(32, 32, 3), num_classes=10,
                      depth=100, growth_rate=12):
    # depth=100, k=12 → 每个dense block有 (100-4)/3 = 32 层
    n_blocks     = 3
    layers_block = (depth - 4) // (n_blocks * 2)   # BC结构每层算2

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(2 * growth_rate, 3, padding="same", use_bias=False)(inputs)

    for i in range(n_blocks):
        x = _dense_block(x, layers_block, growth_rate)
        if i < n_blocks - 1:
            x = _transition(x)

    x = gn(x, groups=min(8, x.shape[-1]))
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="densenet_bc_100_12")


# ─────────────────────────────────────────────────────────
# 统一入口（config驱动）
# ─────────────────────────────────────────────────────────
def build_model(input_shape=(32, 32, 3), num_classes=10, arch="wideresnet28_4"):
    registry = {
        "resnet56":        build_resnet56,
        "wideresnet28_4":  build_wideresnet28_4,
        "densenet_bc":     build_densenet_bc,
    }
    assert arch in registry, f"Unknown arch: {arch}. Choose from {list(registry)}"
    return registry[arch](input_shape=input_shape, num_classes=num_classes)