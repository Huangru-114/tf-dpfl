"""
client/hier_perfedavg.py  –  Hier-PerFedAvg 客户端（Fallah et al., NeurIPS 2020, FO-MAML）

三层 HFL 下的 Per-FedAvg：
  - edge 下发元参数 θ_E，client 在本地数据上算一阶 MAML 元梯度上传给 edge；
  - edge 聚合元梯度做一步元更新（见 server/hier_perfedavg.py）。
Per-FedAvg 没有持久个性化模型，PM 评估时对下发模型做少步微调（personalize）。

参考 client/client.py 中 compute_meta_gradient / _local_train_perfedavg 实现。
"""

import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase


class HierPerFedAvgClient(FLClientBase):

    def __init__(
        self,
        client_id: int,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        config: dict,
        n_samples: int = None,
    ):
        super().__init__(client_id, dataset, model, config, n_samples)

        # get_weights() 中 trainable 权重的位置（把 trainable 梯度写回完整列表用）
        tv_set = {id(v) for v in model.trainable_variables}
        self._trainable_indices = [
            i for i, w in enumerate(model.weights) if id(w) in tv_set
        ]

    # ══════════════════════════════════════════════════════════════════════
    # 元梯度（FO-MAML）：edge 通过 mode="meta_grad" 调用
    # ══════════════════════════════════════════════════════════════════════

    def compute_meta_gradient(self, edge_weights: list, round_idx: int):
        """
        一阶 MAML 元梯度：g = ∇_{θ_E} F(θ_E − α·∇F(θ_E, D1), D2)。

        每个 epoch 把数据二分为 support(D1) / query(D2)，累积多 epoch 元梯度。

        Returns:
            (accumulated_grads, n_samples, query_loss)
            accumulated_grads：完整 get_weights() 长度，non-trainable 位置为 0
        """
        alpha  = self.config["training"].get("alpha_inner", 0.01)
        epochs = self.config["training"]["local_epochs"]

        all_batches = list(self.dataset)
        mid         = max(1, len(all_batches) // 2)
        support     = all_batches[:mid]
        query       = all_batches[mid:] if len(all_batches) > mid else all_batches

        accumulated_grads = [np.zeros_like(w) for w in edge_weights]
        l_support = l_query = None

        for _ in range(epochs):
            # 内层：θ' = θ_E − α·∇F(θ_E, D1)
            self.model.set_weights(edge_weights)
            with tf.GradientTape() as tape:
                l_support = tf.reduce_mean(
                    [self.loss_fn(y, self.model(x, training=True)) for x, y in support])
            inner_grads = tape.gradient(l_support, self.model.trainable_variables)

            adapted = [w.copy() for w in edge_weights]
            for idx, g in zip(self._trainable_indices, inner_grads):
                adapted[idx] = edge_weights[idx] - alpha * g.numpy()

            # 外层元梯度：∇F(θ', D2)
            self.model.set_weights(adapted)
            with tf.GradientTape() as tape:
                l_query = tf.reduce_mean(
                    [self.loss_fn(y, self.model(x, training=True)) for x, y in query])
            meta_grads = tape.gradient(l_query, self.model.trainable_variables)

            for idx, g in zip(self._trainable_indices, meta_grads):
                accumulated_grads[idx] += g.numpy() / epochs

        self.model.set_weights(edge_weights)
        query_loss = float(l_query.numpy())
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-PerFedAvg | support={float(l_support):.4f} query={query_loss:.4f}")
        return accumulated_grads, self.n_samples, query_loss

    # ══════════════════════════════════════════════════════════════════════
    # 本地训练（抽象方法实现）：逐 batch FO-MAML 元参数更新，作为 fallback
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """
        逐 batch Per-FedAvg 元参数更新（与 client/client.py 的实现一致）。

        注意：Hier-PerFedAvg 的 edge 走 mode="meta_grad" 聚合元梯度，正常不会
        调用本方法；此处实现仅为满足基类抽象接口并提供可用的 fallback。
        """
        alpha  = self.config["training"].get("alpha_inner", 0.01)
        epochs = self.config["training"]["local_epochs"]
        ref_weights = self.edge_weights if self.edge_weights is not None \
            else self.global_weights
        meta_w = [w.copy() for w in ref_weights]
        losses, t0 = [], time.time()

        for _ in range(epochs):
            bl = []
            for images, labels in self.dataset:
                half = max(1, images.shape[0] // 2)
                D1x, D1y = images[:half], labels[:half]
                D2x, D2y = images[half:], labels[half:]
                if D2x.shape[0] == 0:
                    D2x, D2y = D1x, D1y

                self.model.set_weights(meta_w)
                with tf.GradientTape() as it:
                    l1 = self.loss_fn(D1y, self.model(D1x, training=True))
                ig = it.gradient(l1, self.model.trainable_variables)
                adapted = [w.copy() for w in meta_w]
                for idx, g in zip(self._trainable_indices, ig):
                    adapted[idx] = meta_w[idx] - alpha * g.numpy()

                self.model.set_weights(adapted)
                with tf.GradientTape() as ot:
                    l2 = self.loss_fn(D2y, self.model(D2x, training=True))
                mg = ot.gradient(l2, self.model.trainable_variables)

                self.model.set_weights(meta_w)
                self.optimizer.apply_gradients(zip(mg, self.model.trainable_variables))
                meta_w = self.model.get_weights()
                bl.append(float(l2.numpy()))
            losses.append(np.mean(bl))

        self.model.set_weights(meta_w)
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-PerFedAvg(local, α={alpha}) | loss={avg:.4f}")
        return meta_w, self.n_samples, avg, time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # 个性化评估：下发模型 + 少步微调，再在测试集上评估
    # ══════════════════════════════════════════════════════════════════════

    def personalize_and_evaluate(self, edge_weights: list, steps: int = 1,
                                 fallback_dataset: tf.data.Dataset = None):
        """
        Per-FedAvg 无持久个性化模型，PM 评估时对下发的 edge 元参数做 steps 步
        微调（在本地训练数据上），微调后的模型留在 self.model 供后续评估。

        Returns:
            (loss, accuracy) on test set（evaluate_on）
        """
        self.model.set_weights(edge_weights)
        for _ in range(int(steps)):
            for x, y in self.dataset:
                with tf.GradientTape() as tape:
                    loss = self.loss_fn(y, self.model(x, training=True))
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return self.evaluate_on(fallback_dataset=fallback_dataset)
