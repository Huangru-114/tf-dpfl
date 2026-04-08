import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def preprocess(sample, img_size: int = 32):
    """
    把 tfds 返回的原始 dict 转成 (image, label) 元组。
    - cast + /255：uint8 → float32，归一化到 [0, 1]
    - resize：统一图片尺寸
    """
    image = tf.cast(sample["image"], tf.float32) / 255.0
    image = tf.image.resize(image, [img_size, img_size])
    label = sample["label"]
    return image, label


def augment(image, label):
    """
    训练集专用数据增强，测试集不用。
    随机左右翻转 + 随机亮度扰动。
    交通标志不做上下翻转，因为倒置的标志没有语义意义。
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label



# def load_cifar10(config: dict):
#     """
#     从 tf.keras.datasets 加载 CIFAR-10。
#     第一次运行自动下载到 ~/.keras/datasets/，之后直接读缓存。
#     返回已经构建好 pipeline 的 train_ds 和 test_ds。
#     """
#     batch_size  = config["data"]["batch_size"]
#     shuffle_buf = config["data"]["shuffle_buffer"]

#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#     # 归一化到 [0, 1]
#     x_train = x_train.astype("float32") / 255.0
#     x_test  = x_test.astype("float32")  / 255.0

#     # y shape 是 (N, 1)，压成 (N,)
#     y_train = y_train.squeeze()
#     y_test  = y_test.squeeze()

#     train_ds = (
#         tf.data.Dataset.from_tensor_slices((x_train, y_train))
#         .shuffle(shuffle_buf)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     test_ds = (
#         tf.data.Dataset.from_tensor_slices((x_test, y_test))
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     return train_ds, test_ds, x_train, y_train
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def load_cifar10(config: dict):
    batch_size  = config["data"]["batch_size"]
    shuffle_buf = config["data"]["shuffle_buffer"]

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 改为逐通道标准化，比单纯 /255 效果更好
    x_train = (x_train.astype("float32") / 255.0 - CIFAR10_MEAN) / CIFAR10_STD
    x_test  = (x_test.astype("float32")  / 255.0 - CIFAR10_MEAN) / CIFAR10_STD

    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    def augment(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(
            tf.pad(img, [[4,4],[4,4],[0,0]], mode="REFLECT"), [32, 32, 3]
        )
        return img, lbl

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .cache()                                          # 缓存到内存，加速后续 epoch
        .shuffle(shuffle_buf, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds, x_train, y_train


def load_gtsrb(config: dict):
    """
    加载 GTSRB 数据集，返回 train 和 test 的 tf.data.Dataset。

    Args:
        config: 从 config.yaml 读入的字典

    Returns:
        train_ds, test_ds
    """
    img_size    = config["data"]["img_size"]
    batch_size  = config["data"]["batch_size"]
    shuffle_buf = config["data"]["shuffle_buffer"]
    data_dir    = config["data"]["data_dir"]

    raw = tfds.load("gtsrb", as_supervised=False, data_dir=data_dir)
    train_raw = raw["train"]
    test_raw  = raw["test"]

    train_ds = (
        train_raw
        .map(lambda s: preprocess(s, img_size),
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(shuffle_buf)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_raw
        .map(lambda s: preprocess(s, img_size),
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds


def get_dataset_info():
    """返回 GTSRB 的基本统计信息。"""
    _, info = tfds.load("gtsrb", with_info=True)
    return {
        "num_classes": info.features["label"].num_classes,  # 43
        "num_train":   info.splits["train"].num_examples,   # 39270
        "num_test":    info.splits["test"].num_examples,    # 12630
    }


if __name__ == "__main__":
    config = {
        "data": {
            "img_size": 32,
            "batch_size": 64,
            "shuffle_buffer": 10000
        }
    }
    train_ds, test_ds = load_gtsrb(config)
    for images, labels in train_ds.take(1):
        print("images shape:", images.shape)   # (64, 32, 32, 3)
        print("labels shape:", labels.shape)   # (64,)
        print("pixel range:", images.numpy().min(), "~", images.numpy().max())
