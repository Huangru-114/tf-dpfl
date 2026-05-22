import tensorflow as tf


def build_gtsrb_cnn(input_shape=(32, 32, 3), num_classes=43):
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

def build_cifar_cnn_3conv(input_shape=(32, 32, 3), num_classes=10):
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

def build_net_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    将 PyTorch 中的 Net 模型改写为 TensorFlow 实现。
    原模型描述：一个四层 CNN（两个卷积层 + 两个全连接层），用于 MNIST 图像分类。

    架构（与原始 Net 完全一致）：
        Conv2D(16, kernel=2, stride=1) + ReLU
        MaxPool2D(pool_size=2, strides=1)
        Dropout(0.25)
        Conv2D(32, kernel=2, stride=1) + ReLU
        MaxPool2D(pool_size=2, strides=1)
        Dropout(0.5)
        Flatten
        Dense(128) + ReLU
        Dense(num_classes, activation='softmax')   # 原始使用 log_softmax，此处用 softmax 配合分类交叉熵

    参数：
        input_shape: 输入图像形状 (H, W, C)，默认 (28, 28, 1) 对应 MNIST
        num_classes: 分类数量，默认 10

    返回：
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 第一卷积块
    x = tf.keras.layers.Conv2D(16, 2, strides=1, padding='valid', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # 第二卷积块
    x = tf.keras.layers.Conv2D(32, 2, strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # 分类器
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="net_cnn")
    return model

def build_fedavg_cnn(input_shape=(32, 32, 3), num_classes=10, dim=1024):
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


def build_model(input_shape=(32, 32, 3), num_classes=10, arch="cifar_cnn_3conv"):
    registry = {
        "cifar_cnn_3conv":  build_cifar_cnn_3conv,
        "net_cnn":  build_net_cnn,
        "gtsrb_cnn":  build_gtsrb_cnn,
        "fedavg_cnn":  build_fedavg_cnn,
    }
    assert arch in registry, f"Unknown arch: {arch}. Choose from {list(registry)}"
    return registry[arch](input_shape=input_shape, num_classes=num_classes)
