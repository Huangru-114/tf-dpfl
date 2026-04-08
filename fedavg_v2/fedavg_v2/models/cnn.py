import numpy as np
import tensorflow as tf


def build_model(input_shape=(32, 32, 3), num_classes=10):
    """
    适合 CIFAR-10 的小型 CNN。

    架构：
        Conv(32) → Conv(64) → Pool →
        Conv(128) → Pool →
        Flatten → Dense(256) → Dense(num_classes)

    参数量约 2.19M，对 FedAvg baseline 实验合适。
    后续做 DP-FedAvg 时可以缩减 Dense 层降低参数量。

    Args:
        input_shape: (H, W, C)，默认 (32, 32, 3)
        num_classes: 分类数，CIFAR-10 为 10

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classifier
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    dummy = np.random.rand(4, 32, 32, 3).astype("float32")
    out   = model(dummy, training=False)
    print("output shape:", out.shape)  # (4, 10)
