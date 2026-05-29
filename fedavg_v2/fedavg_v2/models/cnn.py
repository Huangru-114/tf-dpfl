import numpy as np
import tensorflow as tf


# def build_model(input_shape=(32, 32, 3), num_classes=10):
#     """
#     适合 CIFAR-10 的小型 CNN。

#     架构：
#         Conv(32) → Conv(64) → Pool →
#         Conv(128) → Pool →
#         Flatten → Dense(256) → Dense(num_classes)

#     参数量约 2.19M，对 FedAvg baseline 实验合适。
#     后续做 DP-FedAvg 时可以缩减 Dense 层降低参数量。

#     Args:
#         input_shape: (H, W, C)，默认 (32, 32, 3)
#         num_classes: 分类数，CIFAR-10 为 10

#     Returns:
#         tf.keras.Model
#     """
#     inputs = tf.keras.Input(shape=input_shape)

#     # Block 1
#     x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2)(x)
#     x = tf.keras.layers.Dropout(0.25)(x)

#     # Block 2
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2)(x)
#     x = tf.keras.layers.Dropout(0.25)(x)

#     # Classifier
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
#     return model


def build_model(input_shape=(32, 32, 3), num_classes=10, dim=1024):
    """
    FedAvgCNN 的 TensorFlow 复现：
        Conv2D(32, kernel=5, stride=1, valid) + ReLU + MaxPool2D(2, strides=2)
        Conv2D(64, kernel=5, stride=1, valid) + ReLU + MaxPool2D(2, strides=2)
        Flatten
        Dense(512) + ReLU
        Dense(num_classes, activation='softmax')

    参数：
        input_shape: 输入图像形状 (H, W, C)，默认 (28, 28, 1) 对应 MNIST
        num_classes: 分类数量，默认 10
        dim: Flatten 后的维度，默认 1024（对应 MNIST 28×28×1 输入）

    返回：
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 第一卷积块
    x = tf.keras.layers.Conv2D(32, 5, strides=1, padding='valid', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 第二卷积块
    x = tf.keras.layers.Conv2D(64, 5, strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 分类器
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fedavg_cnn")
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
    dummy = np.random.rand(4, 32, 32, 3).astype("float32")
    out   = model(dummy, training=False)
    print("output shape:", out.shape)  # (4, 10)
