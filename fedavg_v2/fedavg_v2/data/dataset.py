import numpy as np
import tensorflow as tf


def load_cifar10(config: dict):
    """
    从 tf.keras.datasets 加载 CIFAR-10。
    第一次运行自动下载到 ~/.keras/datasets/，之后直接读缓存。

    Returns:
        train_ds:  tf.data.Dataset，供 server 全局评估用（未分区）
        test_ds:   tf.data.Dataset，全局测试集
        x_train:   np.ndarray (50000, 32, 32, 3)，供 partition.py 直接使用
        y_train:   np.ndarray (50000,)
        x_test:    np.ndarray (10000, 32, 32, 3)，供 per-client 测试集生成使用
        y_test:    np.ndarray (10000,)
    """
    batch_size  = config["data"]["batch_size"]
    shuffle_buf = config["data"]["shuffle_buffer"]

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 归一化到 [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # y shape 是 (N, 1)，压成 (N,)
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(shuffle_buf)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"[Dataset] CIFAR-10 loaded | "
          f"train={x_train.shape[0]} | test={x_test.shape[0]}")

    # 返回 numpy 数组供 partition.py 直接使用，省去 extract_numpy 步骤
    return train_ds, test_ds, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    config = {"data": {"batch_size": 64, "shuffle_buffer": 10000}}
    train_ds, test_ds, x_train, y_train, x_test, y_test = load_cifar10(config)
    for images, labels in train_ds.take(1):
        print("images shape:", images.shape)   # (64, 32, 32, 3)
        print("labels shape:", labels.shape)   # (64,)
        print("pixel range:", images.numpy().min(), "~", images.numpy().max())
    print("x_test shape:", x_test.shape)       # (10000, 32, 32, 3)
    print("y_test shape:", y_test.shape)        # (10000,)