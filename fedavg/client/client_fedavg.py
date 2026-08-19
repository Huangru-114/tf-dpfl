"""
client/client_fedavg.py  –  Hier-FedAvg 客户端（三层 FedAvg 的 client 层）

最朴素的三层 FedAvg：client 在本地数据上做普通 SGD，上传模型权重，
edge / cloud 各做样本加权 FedAvg。无任何个性化机制。

可选「最后微调」（final fine-tuning）：
  FedAvg 本身不产出个性化模型，PM 评估时可对下发的 edge 模型做少步本地微调，
  得到「FedAvg + 微调」这一经典个性化 baseline。由 main.py 在最终 PM 评估处
  通过 config["evaluation"]["final_finetune"] 触发，调用 personalize_and_evaluate。
"""

import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase


class FedAvgClient(FLClientBase):

    # ══════════════════════════════════════════════════════════════════════
    # 训练
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """
        普通本地 SGD（local_epochs 个 epoch），上传训练后的模型权重。

        self.model 已由 set_weights 对齐到 edge 权重；self.dataset 在
        make_client_dataset(shuffle=True) 时已 shuffle。

        Returns:
            (upload_weights, n_samples, avg_loss, elapsed)
        """
        epochs     = int(self.config["training"]["local_epochs"])
        losses, t0 = [], time.time()

        self.on_round_start(round_idx)

        for _ in range(epochs):
            bl = []
            for x, y in self.dataset:
                x, y = self.on_batch(x, y)
                bl.append(float(self._train_step(x, y).numpy()))
            if bl:
                losses.append(np.mean(bl))

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-FedAvg | loss={avg:.4f}")
        upload = self.on_upload(self.model.get_weights(), round_idx)
        return upload, self.n_samples, avg, time.time() - t0

    @tf.function
    def _train_step(self, images, labels):
        """产出 upload 的训练步 → 叠加 on_extra_loss（基类返回 0.0，数值恒等）。"""
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            loss = tf.cast(loss, tf.float32)
            total = loss + self.on_extra_loss()
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # ══════════════════════════════════════════════════════════════════════
    # 个性化评估（最后微调选项）：下发模型 + 少步微调，再评估
    # ══════════════════════════════════════════════════════════════════════

    def personalize_and_evaluate(self, edge_weights: list, steps: int = 1,
                                 fallback_dataset: tf.data.Dataset = None):
        """
        「FedAvg + 微调」：把下发的 edge 模型在本地训练数据上微调 steps 个 epoch，
        微调后的模型留在 self.model 供后续评估。

        Returns:
            (loss, accuracy) on test set（evaluate_on）
        """
        self.model.set_weights(edge_weights)
        for _ in range(int(steps)):
            for x, y in self.dataset:
                self._train_step(x, y)
        return self.evaluate_on(fallback_dataset=fallback_dataset)
