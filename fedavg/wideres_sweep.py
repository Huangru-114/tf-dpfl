"""
Wide-ResNet on CIFAR-10 — W&B Sweep
目标: ~5M 参数, 95%+ 准确率

运行:
  wandb sweep this_file.py --program this_file.py   # 或手动创建 sweep
  python train.py                                    # 单次运行（默认配置）
  wandb agent <SWEEP_ID>                             # sweep agent
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import wandb
from wandb.integration.keras import WandbMetricsLogger

# ─────────────────────────────────────────────────────────────
# Sweep 配置
# ─────────────────────────────────────────────────────────────

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        # WRN-22-4 ≈ 3.9M | WRN-28-4 ≈ 5.9M
        "depth":          {"value": 16},
        "width":          {"value": 4},
        "dropout":        {"values": [0.0, 0.1, 0.3]},
        "lr":             {"distribution": "log_uniform_values", "min": 0.05, "max": 0.3},
        "weight_decay":   {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
        "batch_size":     {"value": 256},
        "lr_schedule":    {"value": "cosine"},
        "use_cutout":     {"value": True},
        "epochs":         {"value": 100},
        "warmup_epochs":  {"value": 5},
    },
}

DEFAULT_CONFIG = {
    "depth": 28, "width": 4, "dropout": 0.3,
    "lr": 0.1, "weight_decay": 5e-4, "batch_size": 128,
    "lr_schedule": "cosine", "use_cutout": True,
    "epochs": 200, "warmup_epochs": 5,
}

# ─────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

def cutout(img, size=16):
    h, w = 32, 32
    cy = tf.random.uniform([], 0, h, tf.int32)
    cx = tf.random.uniform([], 0, w, tf.int32)
    y1, y2 = tf.clip_by_value(cy - size//2, 0, h), tf.clip_by_value(cy + size//2, 0, h)
    x1, x2 = tf.clip_by_value(cx - size//2, 0, w), tf.clip_by_value(cx + size//2, 0, w)
    mask = tf.pad(tf.ones([y2-y1, x2-x1, 3], dtype=img.dtype), [[y1, h-y2],[x1, w-x2],[0,0]])
    return img * (1.0 - mask)

def build_datasets(batch_size, use_cutout):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    y_tr, y_te = y_tr.flatten().astype(np.int32), y_te.flatten().astype(np.int32)

    def norm(img, lbl):
        return (tf.cast(img, tf.float32) / 255.0 - MEAN) / STD, lbl

    def augment(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(tf.pad(img, [[4,4],[4,4],[0,0]], "REFLECT"), [32,32,3])
        if use_cutout:
            img = cutout(img)
        return img, lbl

    train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
                .shuffle(50000).map(norm).map(augment)
                .batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    test_ds  = (tf.data.Dataset.from_tensor_slices((x_te, y_te))
                .map(norm).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return train_ds, test_ds

# ─────────────────────────────────────────────────────────────
# Wide-ResNet
# ─────────────────────────────────────────────────────────────

def wrn_block(x, out_ch, stride, dropout, wd, name):
    in_ch = x.shape[-1]
    shortcut = x

    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn1")(x)
    x = layers.Activation("relu")(x)

    if stride != 1 or in_ch != out_ch:
        shortcut = layers.Conv2D(out_ch, 1, strides=stride, padding="same",
                                 use_bias=False, kernel_regularizer=regularizers.l2(wd),
                                 name=f"{name}_sc")(x)

    x = layers.Conv2D(out_ch, 3, strides=stride, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(wd),
                      name=f"{name}_c1")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn2")(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(out_ch, 3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(wd), name=f"{name}_c2")(x)
    return layers.Add()([x, shortcut])

def build_wrn(depth, width, dropout, weight_decay):
    assert (depth - 4) % 6 == 0
    n, k, wd = (depth - 4) // 6, width, weight_decay
    ch = [16, 16*k, 32*k, 64*k]

    inp = layers.Input([32, 32, 3])
    x = layers.Conv2D(ch[0], 3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(wd))(inp)

    for i, (out_ch, stride) in enumerate(zip(ch[1:], [1, 2, 2])):
        x = wrn_block(x, out_ch, stride, dropout, wd, f"g{i+1}_b0")
        for j in range(1, n):
            x = wrn_block(x, out_ch, 1, dropout, wd, f"g{i+1}_b{j}")

    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(10, kernel_regularizer=regularizers.l2(wd))(x)
    return tf.keras.Model(inp, out)

# ─────────────────────────────────────────────────────────────
# 学习率调度
# ─────────────────────────────────────────────────────────────

class LRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, total_epochs, warmup_epochs, base_lr):
        super().__init__()
        self.schedule, self.total, self.warmup, self.base = schedule, total_epochs, warmup_epochs, base_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            lr = self.base * (epoch + 1) / self.warmup
        elif self.schedule == "cosine":
            p = (epoch - self.warmup) / (self.total - self.warmup)
            lr = 1e-6 + 0.5 * (self.base - 1e-6) * (1 + np.cos(np.pi * p))
        else:  # multistep × 0.2 at 50%, 75%
            lr = self.base
            for ms in [0.5, 0.75]:
                if epoch >= self.total * ms: lr *= 0.2
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        wandb.log({"lr": lr}, commit=False)

# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────

def train():
    with wandb.init() as run:
        c = wandb.config

        train_ds, test_ds = build_datasets(c.batch_size, c.use_cutout)

        model = build_wrn(c.depth, c.width, c.dropout, c.weight_decay)
        n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
        print(f"WRN-{c.depth}-{c.width}  参数: {n_params/1e6:.2f}M")
        wandb.log({"params_M": n_params / 1e6})

        model.compile(
            optimizer=tf.keras.optimizers.SGD(c.lr, momentum=0.9, nesterov=True),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            epochs=c.epochs,
            validation_data=test_ds,
            callbacks=[
                LRScheduler(c.lr_schedule, c.epochs, c.warmup_epochs, c.lr),
                WandbMetricsLogger(log_freq="epoch"),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=20, restore_best_weights=True
                ),
            ],
            verbose=2,
        )

        _, final_acc = model.evaluate(test_ds, verbose=0)
        wandb.log({"final_test_accuracy": final_acc})
        model.save_weights(f"wrn{c.depth}_{c.width}_{run.name}.h5")

# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="创建并启动 sweep")
    parser.add_argument("--agent", type=str, default=None, help="指定 SWEEP_ID 直接启动 agent")
    parser.add_argument("--project", type=str, default="wrn-cifar10")
    args = parser.parse_args()

    wandb.login()

    if args.agent:
        wandb.agent(args.agent, function=train, project=args.project)
    elif args.sweep:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
        print(f"\nSweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=train, count=30)   # count 按需调整
    else:
        # 单次运行，使用默认配置
        wandb.init(project=args.project, config=DEFAULT_CONFIG)
        train()