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

def build_fedavg_cnn(input_shape=(32, 32, 3), num_classes=10, rep_dim=64):
    """
    FedAvgCNN 的 TensorFlow 复现 + FedRep 式低维瓶颈表示层。

    架构：
        Conv2D(32,5) + ReLU + MaxPool(2)
        Conv2D(64,5) + ReLU + MaxPool(2)
        Flatten
        Dense(512) + ReLU
        Dense(rep_dim) + ReLU         ← 低维共享表示（瓶颈）；rep_dim<=0 时跳过
        Dense(num_classes) + softmax  ← 分类头（FedRep/Rep 类方法的个性化部分）

    为什么要瓶颈层（rep_dim=64）：
        rep 类方法（FedRep/Ditto-Rep/pFedMe-Rep）的个性化 head 是线性分类器，
        参数量≈rep_dim×num_classes。若 head 输入维 d ≥ 每客户端样本数 n，线性系统欠定，
        head 可记住整个本地训练集 → 过拟合（PM loss 升、acc 降），且客户端越多 n 越小越严重。
        CIFAR-10、100 client 时 n≈450，而 d=512 → d>n 会过拟合；压到 rep_dim=64 后 d<n，
        head（64×10+10=650 参数）回到泛化区。rep_dim=0 可关闭瓶颈（消融对照）。

    参数：
        input_shape: 输入图像形状 (H, W, C)
        num_classes: 分类数量
        rep_dim:     瓶颈表示维度（head 输入维）；<=0 时不加瓶颈层（head 直接接 512）。
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
    if rep_dim and rep_dim > 0:
        # 低维共享表示（瓶颈）—— head 的输入维降到 rep_dim
        x = tf.keras.layers.Dense(rep_dim, activation='relu', name='representation')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fedavg_cnn")
    return model


def build_model(input_shape=(32, 32, 3), num_classes=10, arch="cifar_cnn_3conv",
                rep_dim=64):
    registry = {
        "cifar_cnn_3conv":  build_cifar_cnn_3conv,
        "net_cnn":  build_net_cnn,
        "gtsrb_cnn":  build_gtsrb_cnn,
        "fedavg_cnn":  build_fedavg_cnn,
    }
    assert arch in registry, f"Unknown arch: {arch}. Choose from {list(registry)}"
    if arch == "fedavg_cnn":
        # 仅 fedavg_cnn 支持低维瓶颈（rep_dim）；其余 arch 保持原样
        return build_fedavg_cnn(input_shape=input_shape, num_classes=num_classes,
                                rep_dim=rep_dim)
    return registry[arch](input_shape=input_shape, num_classes=num_classes)


def get_base_head_indices(model, num_classes):
    """
    FedRep 式 backbone / head 切分（索引方案，不实体化两个子模型）。

    head = 模型最后一个 Dense(num_classes) 的 kernel + bias，
           即 get_weights() 列表末尾的两个元素。
    backbone = 其余所有权重（卷积、BN 的 trainable gamma/beta 与
               non-trainable moving_mean/variance、head 之前的 Dense）。

    判定方式：从 get_weights() 末尾向前扫描，找
        - shape == (num_classes,) 的 1-D bias
        - shape[-1] == num_classes 的 2-D kernel
    这两个即 head。兼容 cifar_cnn_3conv（末尾 Dense(1024)→Dense(512)→
    Dense(num_classes)），只取最后这个 Dense。

    Keras 约定（已验证）：
        model.weights 与 model.get_weights() 索引一一对齐；
        model.trainable_variables 是 model.weights 的有序子集
        （BN moving_mean/variance 不在其中）。
        因此可用 id() 把 get_weights 索引映射到 trainable 索引。

    Returns:
        {
          "head_weight_indices":    [...],  # get_weights() 中 head 的 2 个索引
          "base_weight_indices":    [...],  # 其余索引
          "head_trainable_indices": [...],  # trainable_variables 中 head 索引
          "base_trainable_indices": [...],  # trainable_variables 中 backbone 索引
        }
    """
    weights = model.get_weights()
    n = len(weights)

    # ── 从末尾扫描定位 head 的 kernel 与 bias ─────────────────────────────
    head_bias_idx = head_kernel_idx = None
    for i in range(n - 1, -1, -1):
        w = weights[i]
        if head_bias_idx is None and w.ndim == 1 and w.shape[0] == num_classes:
            head_bias_idx = i
            continue
        if head_bias_idx is not None and w.ndim == 2 and w.shape[-1] == num_classes:
            head_kernel_idx = i
            break

    if head_bias_idx is None or head_kernel_idx is None:
        raise ValueError(
            f"Cannot locate head (Dense(num_classes={num_classes})) in get_weights(); "
            f"found bias_idx={head_bias_idx}, kernel_idx={head_kernel_idx}."
        )

    head_weight_indices = sorted([head_kernel_idx, head_bias_idx])
    head_weight_set     = set(head_weight_indices)
    base_weight_indices = [i for i in range(n) if i not in head_weight_set]

    # ── get_weights 索引 → trainable_variables 索引（按变量身份映射） ──────
    tv_map = {id(v): j for j, v in enumerate(model.trainable_variables)}
    model_weights = model.weights  # 与 get_weights() 同序
    head_trainable_indices = [
        tv_map[id(model_weights[i])]
        for i in head_weight_indices
        if id(model_weights[i]) in tv_map
    ]
    head_tv_set = set(head_trainable_indices)
    base_trainable_indices = [
        j for j in range(len(model.trainable_variables)) if j not in head_tv_set
    ]

    return {
        "head_weight_indices":    head_weight_indices,
        "base_weight_indices":    base_weight_indices,
        "head_trainable_indices": head_trainable_indices,
        "base_trainable_indices": base_trainable_indices,
    }
