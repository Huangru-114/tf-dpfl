import numpy as np
import tensorflow as tf
import time

class FLClient:
    def __init__(self, client_id: int, dataset: tf.data.Dataset,
                 model: tf.keras.Model, config: dict):
        """
        Args:
            client_id: 客户端编号，用于日志
            dataset:   本地数据集（partition.py 分配好的）
            model:     全局模型的独立副本（clone_model 生成）
            config:    超参数字典
        """
        self.client_id = client_id
        self.dataset   = dataset
        self.model     = model
        self.config    = config

        # 统计本地样本数，用于 FedAvg 加权聚合
        self.n_samples = sum(images.shape[0] for images, _ in dataset)

        lr = config["training"]["learning_rate"]
        decay_rate   = config["training"]["lr_decay"]

        steps_per_epoch = max(1, self.n_samples // config["data"]["batch_size"])
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=steps_per_epoch * config["training"]["local_epochs"],
            decay_rate=decay_rate,
            staircase=True
        )         
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.9)
        self.loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()

    def set_weights(self, global_weights: list):
        """
        接收服务端广播的全局权重，覆盖本地模型。
        每轮训练开始前由 server.broadcast() 调用。
        """
        self.model.set_weights(global_weights)

    def local_train(self, round_idx: int):
        """
        执行 E 轮本地 SGD 训练。

        Args:
            round_idx: 当前通信轮次，用于日志

        Returns:
            weights:   训练后的本地模型权重 (list of np.ndarray)
            n_samples: 本地数据集样本数，供服务端加权聚合
            avg_loss:  本轮平均训练 loss
        """
        local_epochs = self.config["training"]["local_epochs"]
        epoch_losses = []

        t_start = time.time()                        # ← 开始计时

        for epoch in range(local_epochs):
            batch_losses = []
            for images, labels in self.dataset:
                loss = self._train_step(images, labels)
                batch_losses.append(loss.numpy())
            epoch_losses.append(np.mean(batch_losses))

        train_time = time.time() - t_start           # ← 训练耗时

        avg_loss = float(np.mean(epoch_losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
            f"loss={avg_loss:.4f} | time={train_time:.1f}s")

        return self.model.get_weights(), self.n_samples, avg_loss, train_time

    @tf.function
    def _train_step(self, images, labels):
        """
        单个 batch 的前向 + 反向传播。
        @tf.function 将其编译为静态图，比 eager 模式快 2~3 倍。
        """
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss        = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss

    def evaluate(self):
        """
        在本地数据上评估当前模型（用于调试）。
        FL 的正式评估应在服务端用全局测试集完成。

        Returns:
            loss (float), accuracy (float)
        """
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in self.dataset:
            predictions   = self.model(images, training=False)
            loss          = self.loss_fn(labels, predictions)
            total_loss   += loss.numpy() * images.shape[0]
            total_correct += np.sum(
                np.argmax(predictions.numpy(), axis=1) == labels.numpy()
            )
            total_samples += images.shape[0]

        return total_loss / total_samples, total_correct / total_samples
