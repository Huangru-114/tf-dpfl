"""
client/client_pfedme.py  –  Hier-pFedMe 客户端

变量角色对照：
  ref_weights        edge 下发权重  θ_n^{t,i}   ←→  pFedMe 中的 server.model (w)
  self.model         内层个性化模型  θ̃           ←→  pFedMe 中的 user.model
  self._local_model  本地锚点模型   Θ            ←→  pFedMe 中的 local_model (θ)

数据遍历方式（与 PFLlib 对齐）：
  - local_epochs 含义：完整数据集遍历次数（真正的 epoch），而非 batch 步数。
  - 每个 epoch 对 _batch_list 重新随机排列（等价于 DataLoader shuffle=True）。
  - Moreau 步在每个 batch 的 K 步内层优化之后立即执行，而非每个 epoch 结束时。

性能优化（避免热循环瓶颈）：
  1. _local_model 与 self.model 结构相同，直接 set/get_weights 对齐，无需索引。
  2. _lam_var / _plr_var / _moreau_lr_var / _mu_var 用 tf.Variable 存超参，
     @tf.function 通过引用捕获，避免每次传 tf.constant() 触发 retrace。
  3. dataset 一次性缓存为 list（_batch_list），每 epoch 原地 shuffle 后顺序遍历。
"""

import random
import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase


class PFedMeClient(FLClientBase):

    def __init__(
        self,
        client_id: int,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        config: dict,
        n_samples: int = None,
    ):
        super().__init__(client_id, dataset, model, config, n_samples)

        # ── 本地锚点 Θ（与 self.model 结构相同的模型副本） ────────────────
        self._local_model = tf.keras.models.clone_model(model)
        self._local_model.set_weights(model.get_weights())

        # ── pFedMe 双学习率（对齐 PFLlib：内层 personal_lr ≠ 外层 moreau_lr）──────
        # 之前内层与 Moreau 都用同一个 learning_rate(0.02) + λ=35 → 有效系数 lr·λ≈0.70，
        # 比 PFLlib（内层 plr·λ=0.01·15=0.15、外层 lr·λ=0.005·15=0.075）大 5~10 倍，
        # 导致本地锚点 Θ 每个 batch 就塌缩到 θ̃ 上、K>1 发散。现分离两个 lr 并对齐 PFLlib。
        tr = config["training"]
        self._plr0       = float(tr.get("personal_lr_pfedme", 0.01))   # 内层 θ̃ 个性化 lr
        self._moreau_lr0 = float(tr.get("moreau_lr_pfedme",   0.005))  # 外层 Θ Moreau lr
        self._plr_var       = tf.Variable(self._plr0,       trainable=False, dtype=tf.float32)
        self._moreau_lr_var = tf.Variable(self._moreau_lr0, trainable=False, dtype=tf.float32)
        self._lam_var = tf.Variable(float(tr.get("lambda2_hier", 15.0)),
                                    trainable=False, dtype=tf.float32)
        self._mu_var  = tf.Variable(float(tr.get("mu_pfedme", 0.001)),
                                    trainable=False, dtype=tf.float32)
        # 内层个性化优化器：用 personal_lr，独立于 base self.optimizer
        self._inner_opt = tf.keras.optimizers.SGD(learning_rate=self._plr_var)

        # ── dataset 缓存为 list，每 epoch shuffle 后遍历 ─────────────────
        # 等价于 PFLlib 的 DataLoader(shuffle=True, drop_last=True)。
        # drop_last：若最后一个 batch 样本数不足完整 batch，跳过，与 PFLlib 一致。
        self._batch_list  = list(self.dataset)
        self._batch_size  = config["data"]["batch_size"]

        # ── 训练后暂存，供外部（EdgeServer）评估用 ────────────────────────
        self._personalized_weights: list | None = None  # θ̃（内层个性化模型）
        self._local_weights:        list | None = None  # Θ（本地锚点）

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """接收广播权重并初始化本地模型（以 edge_weights 优先）。"""
        self.global_weights = [w.copy() for w in global_weights]
        if edge_weights is not None:
            self.edge_weights = [w.copy() for w in edge_weights]
            self.model.set_weights(edge_weights)
            self._local_model.set_weights(edge_weights)  # Θ 从 edge 权重重置
            print(f"  [Client {self.client_id:>2}] Received edge weights. Set local model to edge weights.")
        else:
            self.edge_weights = None
            self.model.set_weights(global_weights)
            self._local_model.set_weights(global_weights)  # Θ 从 global 权重重置
            print(f"  [Client {self.client_id:>2}] Received global weights." )

    def apply_round_lr(self, round_idx: int):
        """pFedMe：personal_lr（内层）与 moreau_lr（外层）都按 global round 衰减，对齐 PFLlib。"""
        super().apply_round_lr(round_idx)
        r = max(0, int(round_idx))
        self._plr_var.assign(self._plr0 * (self.lr_gamma ** r))
        self._moreau_lr_var.assign(self._moreau_lr0 * (self.lr_gamma ** r))

    # ══════════════════════════════════════════════════════════════════════
    # 训练
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """
        Hier-pFedMe 本地训练（Algorithm 1, Lines 6–14）。

        数据遍历方式与 PFLlib 对齐：
          for epoch in range(local_epochs):          # 真正的 epoch
              for x, y in shuffled_batches:          # 完整数据集
                  for _ in range(inner_steps):       # K 步内层优化
                      inner_step(x, y)
                  moreau_step()                      # 每 batch 一次

        Returns:
            upload_weights : 上传 Θ（本地锚点），非 θ̃（内层模型）
            n_samples      : 本地样本数
            avg_loss       : 所有 batch 的平均 task loss
            elapsed        : 训练耗时（秒）
        """
        lam         = float(self.config["training"].get("lambda2_hier", 15.0))
        epochs      = int(self.config["training"]["local_epochs"])
        inner_steps = int(self.config["training"].get("inner_steps_hier", 5))
        # personal_lr / moreau_lr / μ 已在 __init__ 设好，并由 apply_round_lr 按轮衰减；
        # λ 每轮重读（允许 config 中途调整）。
        self._lam_var.assign(lam)
        losses, t0 = [], time.time()

        print(
            f"  [Client {self.client_id:>2}] Starting Hier-pFedMe | "
            f"λ={lam}, plr={float(self._plr_var.numpy()):.4g}, "
            f"moreau_lr={float(self._moreau_lr_var.numpy()):.4g}, μ={float(self._mu_var.numpy()):.4g}, "
            f"K={inner_steps}, epochs={epochs}, batches/epoch={len(self._batch_list)}"
        )
        # self.evaluate_on(fallback_dataset=self.test_dataset)  # 训练前快照（调试用）

        # ── 主训练循环（与 PFLlib 结构一致） ──────────────────────────────
        for _ in range(epochs):

            # 每个 epoch 重新 shuffle batch 顺序（等价于 DataLoader shuffle=True）
            epoch_batches = self._batch_list.copy()
            random.shuffle(epoch_batches)

            # drop_last：丢弃样本数不足 batch_size 的最后一个 batch
            epoch_batches = [
                (x, y) for x, y in epoch_batches
                if x.shape[0] == self._batch_size
            ]

            for x, y in epoch_batches:
                # K 步内层优化（更新 θ̃）
                for _ in range(inner_steps):
                    loss = self._inner_step(x, y)

                # Moreau 步（更新 Θ）—— 每 batch 后立即执行，与 PFLlib 一致
                self._moreau_step()
                losses.append(float(loss.numpy()))

        # self.evaluate_on(fallback_dataset=self.test_dataset)  # 训练后快照（调试用）

        # ── 上传准备 ──────────────────────────────────────────────────────
        # 先保存此刻的 θ̃（个性化模型），再用 Θ 覆盖 self.model 用于上传。
        # 顺序不能颠倒：assign 之后 self.model 的 trainable_variables 变成 Θ 的值，
        # 若先 assign 再 get_weights，_personalized_weights 存的将是 Θ 而非 θ̃。
        self._personalized_weights = self.model.get_weights()   # ← 先保存 θ̃

        # Algorithm Line 14: 上传 Θ
        # BN 滑动统计来自 θ̃（self.model）的训练；
        # 将 Θ（_local_model）的 trainable 参数 assign 回 self.model，再 get_weights()。
        # for m_var, l_var in zip(self.model.trainable_variables,
        #                         self._local_model.trainable_variables):
        #     m_var.assign(l_var)
        upload_weights = self._local_model.get_weights()

        self._local_weights = self._local_model.get_weights()   # 保存 Θ

        avg = float(np.mean(losses)) if losses else 0.0
        print(
            f"  [Client {self.client_id:>2}] Round {round_idx} | "
            f"Hier-pFedMe(λ2={lam}, K={inner_steps}) | loss={avg:.4f}"
        )

        return upload_weights, self.n_samples, avg, time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # TF 图内核心步骤（用 tf.Variable 超参避免 retrace）
    # ══════════════════════════════════════════════════════════════════════

    @tf.function
    def _inner_step(self, images, labels):
        """
        θ̃ 的一步梯度更新（内层个性化优化器，用 personal_lr），以 Θ 为近端锚点。

        目标：F(θ̃; D) + (λ/2)·‖θ̃ − Θ‖² + (μ/2)·‖θ̃‖²
        梯度：∇F + λ·(θ̃ − Θ) + μ·θ̃   （对齐 PFLlib pFedMeOptimizer）
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - t))
                for v, t in zip(self.model.trainable_variables,
                                self._local_model.trainable_variables)
            ])
            l2 = tf.add_n([
                tf.reduce_sum(tf.square(v))
                for v in self.model.trainable_variables
            ])
            total_loss = (loss
                          + (self._lam_var / 2.0) * prox
                          + (self._mu_var / 2.0) * l2)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self._inner_opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss

    @tf.function
    def _moreau_step(self):
        """
        Moreau 包络梯度步，更新本地锚点 Θ（用 moreau_lr，外层小步长）。

        Θ ← Θ − moreau_lr · λ · (Θ − θ̃)   （对齐 PFLlib：localweight − lamda·lr·(··)）
        """
        for t, v in zip(self._local_model.trainable_variables,
                        self.model.trainable_variables):
            t.assign(t - self._moreau_lr_var * self._lam_var * (t - v))