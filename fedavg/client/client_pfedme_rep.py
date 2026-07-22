"""
clients/hier_pfedme_rep.py – PFedMeRep 客户端
在 PFedMeClient 基础上增加 head 私有化与 backbone 聚合。
使用 legacy optimizer 避免变量切换时的状态问题。
"""

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import SGD as LegacySGD

from .client_base import FLClientBase


class HierPFedMeRepClient(FLClientBase):

    def __init__(
        self,
        client_id: int,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        config: dict,
        n_samples: int = None,
    ):
        super().__init__(client_id, dataset, model, config, n_samples)

        # ── 替换基类的 optimizer 为 legacy SGD，避免变量切换问题 ──
        lr    = config["training"]["learning_rate"]
        decay = config["training"]["lr_decay"]
        steps = max(1, self.n_samples // config["data"]["batch_size"])
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=steps * config["training"]["local_epochs"],
            decay_rate=decay,
            staircase=True,
        )
        self.optimizer = LegacySGD(learning_rate=self.lr_schedule)
        # 其他属性（loss_fn 等）已在 super().__init__ 中定义，无需重复

        # 拆分 backbone 和 head
        self._backbone_layers = self.model.layers[:-1]   # 除最后一层外均为 backbone
        self._head_layer = self.model.layers[-1]         # 最后一层 Dense

        # 私有 head 权重缓存
        self._head_weights = self._extract_head_weights()

        # 本地锚点模型（完整结构，但实际只使用 backbone 部分）
        self._local_model = tf.keras.models.clone_model(model)
        self._local_model.set_weights(model.get_weights())

        # 超参 tf.Variable
        self._lam2_var: tf.Variable | None = None
        self._eta2_var: tf.Variable | None = None

        # dataset 缓存
        self._batch_list = list(self.dataset)
        self._batch_size = config["data"]["batch_size"]

        # 训练后暂存
        self._personalized_weights: list | None = None
        self._local_weights: list | None = None

    def _extract_head_weights(self):
        """返回 head 层权重 (kernel, bias)"""
        full = self.model.get_weights()
        return [full[-2], full[-1]]

    def _split_backbone_head_indices(self):
        """返回 (backbone_indices, head_indices) 基于权重列表长度"""
        n = len(self.model.get_weights())
        return list(range(n-2)), [n-2, n-1]

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """
        接收广播权重：只更新 backbone，head 保持私有。
        """
        self.global_weights = [w.copy() for w in global_weights]
        backbone_idx, head_idx = self._split_backbone_head_indices()

        if edge_weights is not None:
            self.edge_weights = [w.copy() for w in edge_weights]
            backbone_src = [edge_weights[i] for i in backbone_idx]
        else:
            self.edge_weights = None
            backbone_src = [global_weights[i] for i in backbone_idx]

        # 构造完整权重：backbone_src + 私有 head
        total_len = len(self.model.get_weights())
        new_full = [None] * total_len
        for i, w in zip(backbone_idx, backbone_src):
            new_full[i] = w
        for i, w in zip(head_idx, self._head_weights):
            new_full[i] = w
        self.model.set_weights(new_full)

        # 同步锚点模型的 backbone（head 部分可不管）
        self._local_model.set_weights(new_full)

        print(f"  [Client {self.client_id:>2}] PFedMeRep: backbone updated, head kept private.")

    def local_train(self, round_idx: int):
        """
        本地训练：
          1. 固定 backbone，训练 head (plocal_epochs 轮)
          2. 固定 head，对 backbone 执行原 PFedMe 的 Moreau 优化
        """
        lam2 = float(self.config["training"].get("lambda2_hier", 35.0))
        eta2 = float(self.config["training"]["learning_rate"])
        plocal_epochs = int(self.config["training"].get("plocal_epochs", 5))
        epochs = int(self.config["training"]["local_epochs"])
        inner_steps = int(self.config["training"].get("inner_steps_hier", 1))

        # 初始化/更新超参变量
        if self._lam2_var is None:
            self._lam2_var = tf.Variable(lam2, trainable=False, dtype=tf.float32)
            self._eta2_var = tf.Variable(eta2, trainable=False, dtype=tf.float32)
        else:
            self._lam2_var.assign(lam2)
            self._eta2_var.assign(eta2)

        # 准备数据集
        batch_list = [(x, y) for x, y in self._batch_list if x.shape[0] == self._batch_size]

        losses = []
        t0 = time.time()

        # ========================= 阶段 1：训练 head =========================
        # 注意：不改变 layer.trainable，只手动控制变量列表
        print(f"  [Client {self.client_id:>2}] Stage 1: Training head for {plocal_epochs} epochs (backbone frozen).")
        for _ in range(plocal_epochs):
            random.shuffle(batch_list)
            for x, y in batch_list:
                loss = self._train_head_step(x, y)
                losses.append(float(loss.numpy()))

        # 更新私有 head 缓存（训练后 head 变化了）
        self._head_weights = self._extract_head_weights()

        # ========================= 阶段 2：Backbone Moreau 优化 =========================
        print(f"  [Client {self.client_id:>2}] Stage 2: PFedMe on backbone (λ2={lam2}, K={inner_steps}, epochs={epochs}).")
        for _ in range(epochs):
            random.shuffle(batch_list)
            for x, y in batch_list:
                # 内层 K 步优化（仅更新 backbone）
                for _ in range(inner_steps):
                    loss = self._inner_backbone_step(x, y)
                    losses.append(float(loss.numpy()))
                # Moreau 步：更新锚点 backbone
                self._moreau_backbone_step()

        # 保存个性化模型权重（完整模型，用于评估）
        self._personalized_weights = self.model.get_weights()

        # 上传 backbone 锚点（从 _local_model 中提取 backbone 部分）
        upload_weights = self._extract_backbone_weights(self._local_model)
        self._local_weights = self._local_model.get_weights()

        avg_loss = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - t0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | PFedMeRep | avg_loss={avg_loss:.4f}")
        return upload_weights, self.n_samples, avg_loss, elapsed

    # ---------- 辅助方法 ----------
    def _extract_backbone_weights(self, model):
        """从模型中提取 backbone 部分权重列表"""
        full = model.get_weights()
        n = len(full)
        return [full[i] for i in range(n-2)]

    def _train_head_step(self, images, labels):
        """单步训练 head（不更新 backbone）"""
        with tf.GradientTape() as tape:
            preds = self.model(images, training=True)
            loss = self.loss_fn(labels, preds)
        # 只对 head 变量求导并更新
        grads = tape.gradient(loss, self._head_layer.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._head_layer.trainable_variables))
        return loss

    @tf.function
    def _inner_backbone_step(self, images, labels):
        """
        内层更新 backbone（head 不更新），目标：
        F(θ̃_backbone; D) + (λ2/2) · ‖θ̃_backbone - Θ_backbone‖²
        """
        # 收集 backbone 可训练变量
        backbone_vars = []
        for layer in self._backbone_layers:
            backbone_vars.extend(layer.trainable_variables)
        # 对应锚点变量（从 _local_model 中取）
        anchor_vars = []
        for layer in self._backbone_layers:
            anchor_vars.extend(layer.trainable_variables)

        with tf.GradientTape() as tape:
            preds = self.model(images, training=True)
            task_loss = self.loss_fn(labels, preds)
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - a))
                for v, a in zip(backbone_vars, anchor_vars)
            ])
            total_loss = task_loss + (self._lam2_var / 2.0) * prox
        grads = tape.gradient(total_loss, backbone_vars)
        self.optimizer.apply_gradients(zip(grads, backbone_vars))
        return task_loss

    @tf.function
    def _moreau_backbone_step(self):
        """
        Moreau 包络梯度步，更新本地锚点 Θ_backbone：
        Θ_backbone ← Θ_backbone - η2·λ2·(Θ_backbone - θ̃_backbone)
        """
        for backbone_layer, anchor_layer in zip(self._backbone_layers, self._local_model.layers[:-1]):
            for b_var, a_var in zip(backbone_layer.trainable_variables, anchor_layer.trainable_variables):
                a_var.assign(a_var - self._eta2_var * self._lam2_var * (a_var - b_var))