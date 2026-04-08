import tensorflow as tf


def build_model_old(input_shape=(32, 32, 3), num_classes=43):
    """
    适合 GTSRB 的小型 CNN。
    使用 GroupNorm 代替 BatchNorm，方便后续扩展 DP-FedAvg。
    （BatchNorm 在小 batch 或 DP 噪声下表现不稳定）

    架构：
        Conv(32) → Conv(64) → Pool → Conv(128) → Pool → FC(256) → Output

    Args:
        input_shape: (H, W, C)，默认 (32, 32, 3)
        num_classes: 分类数，GTSRB 为 43

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

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gtsrb_cnn")
    return model

def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)  # 补上 BN
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)  # 补上 BN
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
    x = tf.keras.layers.SpatialDropout2D(0.05)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)  # 补上 BN
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

    # Classifier
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x) 

    return tf.keras.Model(inputs, outputs, name="cifar_cnn_3conv")

if __name__ == "__main__":
    model = build_model()
    model.summary()
    # 验证前向传播
    import numpy as np
    dummy = np.random.rand(4, 32, 32, 3).astype("float32")
    out   = model(dummy, training=False)
    print("output shape:", out.shape)   # (4, 43)
