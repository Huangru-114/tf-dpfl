import tensorflow as tf
import numpy as np

# **不要在模块顶层 import tensorflow_datasets**：它只被 GTSRB 的两个 loader 用到，
# CIFAR 路径完全不需要；而它的导入链
#   tfds → tensorflow_metadata → google.protobuf.runtime_version
# 在 protobuf < 5.27 的容器里会抛
#   ImportError: cannot import name 'runtime_version' from 'google.protobuf'
# 这条被 absl 记成 ERROR 后流程仍会继续（非致命），但它会出现在每一次 run 的日志
# 最前面，极易被当成故障主因去排查。改成用到时再导入，CIFAR 的 run 就完全看不到它。
def _tfds():
    import tensorflow_datasets as tfds
    return tfds

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

# Mimer 上 CIFAR 数据集的共享路径（CIFAR-10 和 CIFAR-100 均在此）
# **注意这是 Chalmers Mimer 的路径**，别的集群上不存在，会走下面的 fallback。
CIFAR_MIMER_PATH = "/mimer/NOBACKUP/Datasets/CIFAR"


def resolve_keras_home(tag: str = "CIFAR") -> str | None:
    """
    决定 keras 数据缓存根目录（`KERAS_HOME`），并在**容器里看不见目标**时给出人话报错。

    优先级：
      1. `TFDPFL_KERAS_HOME` —— 显式覆盖，集群上由 cluster_env.sh 自动导出
      2. 已经设好的 `KERAS_HOME`
      3. `CIFAR_MIMER_PATH`（Chalmers Mimer 专用，别处不存在）
      4. 都没有 → 返回 None，走 keras 默认的 `~/.keras`

    **为什么需要第 4 步的额外检查（踩过的坑）**：
    `~/.keras/datasets` 常常是指向共享盘的**符号链接**。在 apptainer 容器里，
    若链接目标所在的盘没有被 `--bind` 进来，这个链接就是**悬空**的 ——
    `os.path.isdir()` 为 False，于是 keras 的 `os.makedirs(datadir, exist_ok=True)`
    去 mkdir，撞上已存在的链接本身，抛出

        FileExistsError: [Errno 17] File exists: '/home/<user>/.keras/datasets'

    这句话完全看不出真实原因（既不是「已存在」的问题，也不是权限问题），
    所以这里提前拦截并把链接指向哪里、该 bind 什么都打出来。
    """
    import os

    for src, path in (("TFDPFL_KERAS_HOME", os.environ.get("TFDPFL_KERAS_HOME")),
                      ("KERAS_HOME",        os.environ.get("KERAS_HOME")),
                      ("CIFAR_MIMER_PATH",  CIFAR_MIMER_PATH)):
        if path and os.path.isdir(path):
            os.environ["KERAS_HOME"] = path
            print(f"[Data] {tag}: KERAS_HOME = {path}  (来源: {src})")
            return path

    # 走 keras 默认的 ~/.keras —— 检查 datasets 是不是悬空符号链接
    default_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
    if os.path.islink(default_dir) and not os.path.isdir(default_dir):
        target = os.readlink(default_dir)

        # 先尝试**自愈**：软链本身就表达了「数据缓存放这儿」的意图，
        # 常见情形只是那个目录还没被建出来（不是不可见）。能建就建。
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as e:
            create_err = f"{type(e).__name__}: {e}"
        else:
            create_err = None
            if os.path.isdir(default_dir):        # 软链现在活了
                print(f"[Data] {tag}: 软链目标不存在，已创建 {target}")
                return None

        raise RuntimeError(
            f"\n[Data] {tag}: `{default_dir}` 是**悬空的符号链接**，且无法自动修复。\n"
            f"  它指向：{target}\n"
            + (f"  尝试创建该目录失败：{create_err}\n" if create_err else
               f"  已创建该目录，但软链仍然不可用 —— 说明目标在当前环境里**看不见**。\n")
            + f"  在 apptainer 容器里，这通常是因为该盘没有被 --bind 进来\n"
            f"  （容器默认只挂 $PWD 和 $HOME）。\n"
            f"  不处理的话 keras 会抛一句与真实原因无关的\n"
            f"  `FileExistsError: [Errno 17] File exists: {default_dir}`。\n"
            f"  三种修法（任选其一）：\n"
            f"    1) 手工建目录：mkdir -p {target}\n"
            f"    2) 让容器能看到目标盘：TFDPFL_BIND=<覆盖 {target} 的路径>\n"
            f"    3) 绕开 $HOME 的软链，直接指定缓存根：\n"
            f"       TFDPFL_KERAS_HOME={os.path.dirname(target)}\n"
            f"       （keras 会在其下找 datasets/，正好是 {target}）\n")

    print(f"[Data] {tag}: 用 keras 默认缓存 {default_dir}")
    return None


def load_cifar10(config: dict):
    import os
    batch_size  = config["data"]["batch_size"]
    shuffle_buf = config["data"]["shuffle_buffer"]

    resolve_keras_home("CIFAR-10")

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

    return train_ds, test_ds, x_train, y_train, x_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# CIFAR-100
# 和 CIFAR-10 使用同一个 Mimer 路径：/mimer/NOBACKUP/Datasets/CIFAR/
# ══════════════════════════════════════════════════════════════════════════════

CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)


def load_cifar100(config: dict):
    """
    加载 CIFAR-100（100 类，每类 600 张，32×32）。
    与 CIFAR-10 共用同一个 Mimer 路径，KERAS_HOME 已在 load_cifar10 中设置，
    若直接调用此函数则在此处再设置一次。
    返回：train_ds, test_ds, x_train, y_train, x_test, y_test
    """
    import os
    batch_size  = config["data"]["batch_size"]
    shuffle_buf = config["data"]["shuffle_buffer"]

    resolve_keras_home("CIFAR-100")

    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.cifar100.load_data(label_mode="fine")

    x_train = (x_train.astype("float32") / 255.0 - CIFAR100_MEAN) / CIFAR100_STD
    x_test  = (x_test.astype("float32")  / 255.0 - CIFAR100_MEAN) / CIFAR100_STD
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    def augment_c100(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(
            tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"), [32, 32, 3]
        )
        return img, lbl

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .cache()
        .shuffle(shuffle_buf, reshuffle_each_iteration=True)
        .map(augment_c100, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"[Data] CIFAR-100 | train={len(x_train)} test={len(x_test)} classes=100")
    return train_ds, test_ds, x_train, y_train, x_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# ImageNet (ILSVRC2012, face-blurred)
# Mimer 路径：/mimer/NOBACKUP/Datasets/ImageNet/Face-blurred_ILSVRC2012-2017
#   train_blurred.zip  ← 训练集（按 WordNet synset 目录组织的 JPEG）
#   val_blurred.zip    ← 验证集
#   map_clsloc.txt     ← synset → class_index (1-based) 映射
# 访问前提：在 SUPR 加入 imagenet-license-agreement 组后重新登录。
# C3SE 建议直接读取 zip，不解压，避免大量小文件写入 $TMPDIR。
# ══════════════════════════════════════════════════════════════════════════════

IMAGENET_MIMER_PATH = (
    "/mimer/NOBACKUP/Datasets/ImageNet/Face-blurred_ILSVRC2012-2017"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _parse_imagenet_map(dataroot: str) -> dict:
    """解析 map_clsloc.txt → {synset: 0-based class index}。"""
    mapping = {}
    with open(f"{dataroot}/map_clsloc.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = int(parts[1]) - 1   # 1-based → 0-based
    return mapping


def _imagenet_generator(zip_path: str, img_paths: list, labels: list,
                         img_size: int):
    """逐张从 zip 流式读取、解码、归一化，不需要全量载入内存。"""
    from zipfile import ZipFile
    zf = ZipFile(zip_path)
    for path, label in zip(img_paths, labels):
        raw = zf.read(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        yield img, label


def _scan_imagenet_zip(zip_path: str, class_map: dict) -> tuple:
    """扫描 zip 内所有 .jpg/.JPEG，返回 (img_paths, labels)。"""
    from zipfile import ZipFile
    print(f"[Data] Scanning {zip_path} ...")
    with ZipFile(zip_path) as zf:
        all_paths = zf.namelist()
    img_paths, labels = [], []
    for p in all_paths:
        if not p.lower().endswith((".jpeg", ".jpg")):
            continue
        synset = p.split("/")[-2]
        label  = class_map.get(synset, -1)
        if label >= 0:
            img_paths.append(p)
            labels.append(label)
    return img_paths, labels


def load_imagenet(config: dict):
    """
    加载 ImageNet from Mimer，直接从 zip 流式读取（不解压）。
    ImageNet 训练集约 120 万张，不做全量 numpy 加载，
    x_train / x_test 返回 None，仅返回标签数组用于 partition。
    返回：train_ds, test_ds, None, y_train_np, None, y_val_np
    """
    import os
    dataroot    = config["data"].get("imagenet_path", IMAGENET_MIMER_PATH)
    batch_size  = config["data"]["batch_size"]
    img_size    = config["data"].get("img_size", 224)
    shuffle_buf = config["data"]["shuffle_buffer"]

    if not os.path.isdir(dataroot):
        raise FileNotFoundError(
            f"ImageNet not found at '{dataroot}'.\n"
            "Please join the 'imagenet-license-agreement' group on SUPR "
            "and re-login to Alvis."
        )

    class_map = _parse_imagenet_map(dataroot)
    print(f"[Data] ImageNet class map: {len(class_map)} synsets")

    train_zip = os.path.join(dataroot, "train_blurred.zip")
    val_zip   = os.path.join(dataroot, "val_blurred.zip")

    train_paths, train_labels = _scan_imagenet_zip(train_zip, class_map)
    val_paths,   val_labels   = _scan_imagenet_zip(val_zip,   class_map)
    print(f"[Data] ImageNet | train={len(train_paths)} val={len(val_paths)} "
          f"classes={len(class_map)}")

    sig = (
        tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def augment_imagenet(img, lbl):
        pad = int(img_size * 0.1)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(
            tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT"),
            [img_size, img_size, 3]
        )
        return img, lbl

    train_ds = (
        tf.data.Dataset.from_generator(
            lambda: _imagenet_generator(train_zip, train_paths,
                                        train_labels, img_size),
            output_signature=sig
        )
        .shuffle(shuffle_buf, reshuffle_each_iteration=True)
        .map(augment_imagenet, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_generator(
            lambda: _imagenet_generator(val_zip, val_paths,
                                        val_labels, img_size),
            output_signature=sig
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return (train_ds, test_ds,
            None, np.array(train_labels, dtype=np.int32),
            None, np.array(val_labels,   dtype=np.int32))
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

    raw = _tfds().load("gtsrb", as_supervised=False, data_dir=data_dir)
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
    _, info = _tfds().load("gtsrb", with_info=True)
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