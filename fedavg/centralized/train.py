import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import time
import numpy as np
import tensorflow as tf

from data.dataset    import load_cifar10
from models.resnet      import build_model
from utils.logger    import CentralizedLogger


def run_centralized(config_path: str = "config/config.yaml"):
    """
    集中式训练 baseline。
    使用和 FL 实验完全相同的：
      - 模型结构
      - 数据集（全量训练集）
      - 总训练 epoch 数（= FL 的 n_rounds × local_epochs）
      - batch size
      - 优化器和学习率

    这样得到的精度曲线是 FL 能达到的理论上界。
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── 随机种子 ──────────────────────────────────────────
    seed = config.get("seed", 42)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ── 数据 ──────────────────────────────────────────────
    print("[Centralized] Loading CIFAR-10...")
    train_ds, test_ds, x_train, y_train = load_cifar10(config)

    # ── 模型 ──────────────────────────────────────────────
    print("[Centralized] Building model...")
    model = build_model(
        input_shape=(config["data"]["img_size"],
                     config["data"]["img_size"], 3),
        num_classes=config["data"]["num_classes"],
        arch=config["model"]["arch"]
    )
    model.summary()

    lr        = config["training"]["learning_rate"]
    decay_rate   = config["training"]["lr_decay"]
    batch_size   = config["data"]["batch_size"]
    n_train      = x_train.shape[0]                        
    steps_per_epoch = max(1, n_train // batch_size)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = lr,
        decay_steps           = steps_per_epoch,
        decay_rate            = decay_rate,
        staircase             = True
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()

    # ── 计算总 epoch 数，和 FL 实验对齐 ──────────────────
    # FL 的总训练量：n_rounds × local_epochs
    # 集中式直接跑同样多的 epoch
    n_rounds      = config["federation"]["n_rounds"]
    edge_rounds   = config["federation"]["edge_rounds"]
    local_epochs  = config["training"]["local_epochs"]
    total_epochs  = n_rounds * local_epochs * edge_rounds

    print(f"[Centralized] Total epochs: {total_epochs} "
          f"(= {n_rounds} rounds × {local_epochs} local epochs)")

    # ── Logger ────────────────────────────────────────────
    logger = None
    if config.get("wandb", {}).get("enabled", False):
        # 修改 run_name 区分集中式和 FL
        centralized_config = dict(config)
        centralized_config["wandb"] = dict(config["wandb"])
        centralized_config["wandb"]["run_name"] = (
            "centralized-" + config["wandb"]["run_name"]
        )
        centralized_config["mode"] = "centralized"
        logger = CentralizedLogger(centralized_config)

    # ── 训练循环 ──────────────────────────────────────────
    history = {
        "epoch":       [],
        "train_loss":  [],
        "test_loss":   [],
        "test_acc":    [],
        "epoch_time":  [],
        # 和 FL 对齐的 x 轴：累计 total_epochs
        "total_epochs": [],
    }

    print("\n" + "=" * 52)
    print(" Centralized Training Start")
    print("=" * 52)

    for epoch in range(1, total_epochs + 1):
        t_start     = time.time()
        batch_losses = []

        for images, labels in train_ds:
            loss = _train_step(model, optimizer, loss_fn, images, labels)
            batch_losses.append(loss.numpy())

        train_loss = float(np.mean(batch_losses))
        test_loss, test_acc = _evaluate(model, loss_fn, test_ds)
        epoch_time = time.time() - t_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)
        history["total_epochs"].append(epoch)

        print(f"  [Epoch {epoch:>3}/{total_epochs}] "
              f"train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | "
              f"test_acc={test_acc:.4f} | "
              f"time={epoch_time:.1f}s")

        if logger is not None:
            logger.log_epoch(epoch, {
                "train_loss":   train_loss,
                "test_loss":    test_loss,
                "test_acc":     test_acc,
                "epoch_time":   epoch_time,
                "total_epochs": epoch,
            })

    if logger is not None:
        logger.finish()

    _print_summary(history)
    return history


@tf.function
def _train_step(model, optimizer, loss_fn, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss        = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def _evaluate(model, loss_fn, test_ds):
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in test_ds:
        predictions    = model(images, training=False)
        loss           = loss_fn(labels, predictions)
        total_loss    += loss.numpy() * images.shape[0]
        total_correct += np.sum(
            np.argmax(predictions.numpy(), axis=1) == labels.numpy()
        )
        total_samples += images.shape[0]
    return total_loss / total_samples, total_correct / total_samples


def _print_summary(history: dict):
    best_idx = int(np.argmax(history["test_acc"]))
    print("\n" + "=" * 52)
    print(" Centralized Training Summary")
    print("=" * 52)
    print(f"  Best  test_acc : {history['test_acc'][best_idx]:.4f} "
          f"(epoch {history['epoch'][best_idx]})")
    print(f"  Final test_acc : {history['test_acc'][-1]:.4f}")
    print(f"  Total epochs   : {len(history['epoch'])}")
    print("=" * 52)


if __name__ == "__main__":
    run_centralized()