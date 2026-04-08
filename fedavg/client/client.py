import numpy as np
import tensorflow as tf
import time

class FLClient:
    def __init__(self, client_id: int, dataset: tf.data.Dataset,
                 model: tf.keras.Model, config: dict, n_samples: int = None):
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
        if n_samples is not None:
            self.n_samples = n_samples          # 直接用传入的值
        else:
            # fallback：遍历统计（慢，尽量避免）
            self.n_samples = sum(
                images.shape[0] for images, _ in dataset
            )

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
        # 保存两层参考权重
        self.global_weights = None   # Cloud 全局模型
        self.edge_weights   = None   # 所属 Edge 模型

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """
        接收服务端广播的全局权重，覆盖本地模型。
        global_weights: Cloud 全局权重（每轮开始时由 Cloud 下发）
        edge_weights:   Edge 模型权重（每个 edge round 开始时由 Edge 下发）

        在 HierFAVG 里，Edge 广播给客户端的是 edge 模型，
        同时客户端需要保存全局模型作为远端参考。
        每轮训练开始前由 server.broadcast() 调用。
        """
        self.model.set_weights(edge_weights)
        self.global_weights = [w.copy() for w in global_weights]
        if edge_weights is not None:
            self.edge_weights = [w.copy() for w in edge_weights]

    def local_train(self, round_idx: int):
        method = self.config["training"].get(
            "drift_correction", "fedavg"
        )
        ref    = self.config["training"].get(
            "proximal_ref", "global"   # "global" / "edge" / "both"
        )

        # 决定近端项的参考权重
        if ref == "edge" and self.edge_weights is not None:
            ref_weights = self.edge_weights
        elif ref == "both":
            ref_weights = None   # 下面单独处理
        else:
            ref_weights = self.global_weights

        if method == "fedavg":
            return self._local_train(round_idx)
        elif method == "fedprox":
            return self._local_train_fedprox(round_idx, ref_weights)
        elif method == "feddyn":
            return self._local_train_feddyn(round_idx, ref_weights)
        elif method == "scaffold":
            return self._local_train_scaffold(round_idx)
        elif method == "pfedme":
            return self._local_train_pfedme(round_idx)
        elif method == "perfedavg":
            return self._local_train_perfedavg(round_idx)
        else:
            raise ValueError(f"Unknown method: {method}")

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

    def _local_train_fedprox(self, round_idx, ref_weights):
        mu           = self.config["training"].get("mu", 0.01)
        local_epochs = self.config["training"]["local_epochs"]
        epoch_losses = []
        t_start      = time.time()

        ref_tf = [tf.constant(w) for w in ref_weights]

        for epoch in range(local_epochs):
            batch_losses = []
            for images, labels in self.dataset:
                loss = self._train_step_fedprox(
                    images, labels, ref_tf, mu
                )
                batch_losses.append(loss.numpy())
            epoch_losses.append(np.mean(batch_losses))

        train_time = time.time() - t_start
        avg_loss   = float(np.mean(epoch_losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"FedProx(ref={self.config['training'].get('proximal_ref','global')}) | "
              f"loss={avg_loss:.4f}")
        return self.model.get_weights(), self.n_samples, avg_loss, train_time

    @tf.function
    def _train_step_fedprox(self, images, labels, ref_tf, mu):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss        = self.loss_fn(labels, predictions)
            prox        = tf.add_n([
                tf.reduce_sum(tf.square(w - r))
                for w, r in zip(
                    self.model.trainable_variables, ref_tf
                )
            ])
            loss = loss + (mu / 2.0) * prox
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return loss

    def _local_train_fedprox_both(self, round_idx):
        """
        双近端项版本：同时参考 edge 模型和全局模型。
        loss = F_k(w) + (μ₁/2)||w-w_edge||² + (μ₂/2)||w-w_cloud||²
        """
        mu1          = self.config["training"].get("mu_edge",   0.1)
        mu2          = self.config["training"].get("mu_global",  0.01)
        local_epochs = self.config["training"]["local_epochs"]
        epoch_losses = []
        t_start      = time.time()

        edge_tf   = [tf.constant(w) for w in self.edge_weights]
        global_tf = [tf.constant(w) for w in self.global_weights]

        for epoch in range(local_epochs):
            batch_losses = []
            for images, labels in self.dataset:
                loss = self._train_step_fedprox_both(
                    images, labels, edge_tf, global_tf, mu1, mu2
                )
                batch_losses.append(loss.numpy())
            epoch_losses.append(np.mean(batch_losses))

        train_time = time.time() - t_start
        avg_loss   = float(np.mean(epoch_losses))
        return (self.model.get_weights(), self.n_samples,
                avg_loss, train_time)

    @tf.function
    def _train_step_fedprox_both(self, images, labels,
                                   edge_tf, global_tf, mu1, mu2):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss        = self.loss_fn(labels, predictions)

            prox_edge = tf.add_n([
                tf.reduce_sum(tf.square(w - r))
                for w, r in zip(
                    self.model.trainable_variables, edge_tf
                )
            ])
            prox_global = tf.add_n([
                tf.reduce_sum(tf.square(w - r))
                for w, r in zip(
                    self.model.trainable_variables, global_tf
                )
            ])
            loss = (loss
                    + (mu1 / 2.0) * prox_edge
                    + (mu2 / 2.0) * prox_global)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
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
