"""
client/hier_pfedme_rep.py  –  Hier-pFedMe-Rep 客户端

把 FedRep 式 backbone/head 拆分引入三层 HFL，backbone 个性化机制用 pFedMe
（Moreau envelope）。结构沿用 client_pfedme.py，但内层优化与 Moreau 步都
**限定在 backbone**，head 走 FedRep 式私有训练。

变量角色：
  φ_e               edge 下发的 backbone
  θ̃_bb              内层个性化 backbone（self.model 的 backbone，本地保留供 PM 评估）
  Θ_bb              本地锚点 backbone（Moreau center，上传给 edge 聚合）
  self._head_weights 私有 head（永不上传，轮次间 warm-start）

关键顺序约束（FedRep 理论）：
  每轮先训 head（冻 backbone，多步 plocal_epochs），再训 backbone（冻 head，
  少步 local_epochs）。理由：backbone 的梯度方向依赖 head 接近最优，head 没
  收敛会让 backbone 更新有偏。

数据遍历与 PFLlib 对齐：
  for epoch in local_epochs: for batch in shuffled(drop_last):
      for _ in inner_steps: inner_step(backbone)     # K 步内层
      moreau_step(backbone)                          # 每 batch 一次
"""

import random
import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase
from models.cnn import get_base_head_indices


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

        # ── backbone / head 切分索引 ─────────────────────────────────────
        num_classes = config["data"]["num_classes"]
        split = get_base_head_indices(model, num_classes)
        self._head_w_idx  = split["head_weight_indices"]
        self._base_w_idx  = split["base_weight_indices"]
        self._head_tv_idx = split["head_trainable_indices"]
        self._base_tv_idx = split["base_trainable_indices"]

        tv = model.trainable_variables
        self._head_tvars = [tv[j] for j in self._head_tv_idx]
        self._base_tvars = [tv[j] for j in self._base_tv_idx]

        # ── 本地锚点 Θ（Moreau center），仅 backbone 参与 ─────────────────
        self._anchor_model = tf.keras.models.clone_model(model)
        self._anchor_model.set_weights(model.get_weights())
        atv = self._anchor_model.trainable_variables
        self._base_anchor_tvars = [atv[j] for j in self._base_tv_idx]

        # ── 持久状态 ──────────────────────────────────────────────────────
        self._head_weights: list | None = None  # 私有 head（warm-start）

        # ── dataset 缓存 ──────────────────────────────────────────────────
        self._batch_list = list(self.dataset)
        self._batch_size = config["data"]["batch_size"]

        # ── head / backbone 各自的优化器 ─────────────────────────────────
        # 同一个 SGD 实例只能服务于「构建时」的那组变量（Keras 3 强约束），
        # 因此 head 阶段与 backbone 内层优化必须用独立优化器。
        # 个性化 head 用**单独、更低**的学习率（head_lr_rep），抑制 head 过拟合（PM loss
        # 后期回升）；backbone 仍用基类 lr_schedule。SGD(momentum=0) 无状态耦合。
        self._head_lr0    = float(config["training"].get("head_lr_rep", 0.005))
        self._head_lr_var = tf.Variable(self._head_lr0, trainable=False,
                                        dtype=tf.float32, name=f"head_lr_c{client_id}")
        self._head_opt = tf.keras.optimizers.SGD(learning_rate=self._head_lr_var)
        self._base_opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)

        # ── 超参 tf.Variable（避免 @tf.function retrace） ─────────────────
        self._lam2_var: tf.Variable | None = None
        self._eta2_var: tf.Variable | None = None

    def apply_round_lr(self, round_idx: int):
        """除基类 lr（backbone）外，个性化 head 的 lr 也按 global round 衰减。"""
        super().apply_round_lr(round_idx)
        r = max(0, int(round_idx))
        self._head_lr_var.assign(self._head_lr0 * (self.lr_gamma ** r))

    # ══════════════════════════════════════════════════════════════════════
    # 权重管理（override：只 set backbone，保留私有 head）
    # ══════════════════════════════════════════════════════════════════════

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """
        接收 edge 下发权重，只覆盖 backbone，保留本地私有 head。

        同时把锚点 Θ_bb 从 φ_e 重置（pFedMe 每个 edge round 从 edge 权重重置，
        与 client_pfedme.py 一致）。首轮从广播权重捕获初始 head。
        """
        self.global_weights = (
            [w.copy() for w in global_weights] if global_weights is not None else None
        )
        src = edge_weights if edge_weights is not None else global_weights
        self.edge_weights = [w.copy() for w in src]

        new_w = [w.copy() for w in src]
        if self._head_weights is not None:
            for k, idx in enumerate(self._head_w_idx):
                new_w[idx] = self._head_weights[k].copy()
        else:
            self._head_weights = [src[idx].copy() for idx in self._head_w_idx]
        self.model.set_weights(new_w)
        self._anchor_model.set_weights(src)  # Θ_bb 从 φ_e 重置
        print(f"  [Client {self.client_id:>2}] Received edge backbone (private head kept).")

    # ══════════════════════════════════════════════════════════════════════
    # 训练
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """
        Hier-pFedMe-Rep 本地训练：

          阶段 1：训 head（冻 backbone），plocal_epochs，普通 CE → 保存私有 head
          阶段 2：pFedMe on backbone（冻 head），local_epochs：
                    每 batch K 步内层（CE + (λ2/2)‖θ̃_bb − Θ_bb‖²）+ 1 步 Moreau

        Returns:
            (upload_weights, n_samples, avg_loss, elapsed)
            upload_weights：完整列表，backbone=Θ_bb（锚点），head=私有 head（上游忽略）
            self.model 停在 [θ̃_bb, 私有 head] 供 PM 评估
        """
        plocal      = int(self.config["training"].get("plocal_epochs", 5))
        local       = int(self.config["training"].get("local_epochs", 1))
        lam2        = float(self.config["training"].get("lambda2_hier", 35.0))
        eta2        = float(self.config["training"]["learning_rate"])
        inner_steps = int(self.config["training"].get("inner_steps_hier", 1))

        if self._lam2_var is None:
            self._lam2_var = tf.Variable(lam2, trainable=False, dtype=tf.float32)
            self._eta2_var = tf.Variable(eta2, trainable=False, dtype=tf.float32)
        else:
            self._lam2_var.assign(lam2)
            self._eta2_var.assign(eta2)

        losses, t0 = [], time.time()
        print(f"  [Client {self.client_id:>2}] Hier-pFedMe-Rep | "
              f"plocal={plocal}, local={local}, λ2={lam2}, K={inner_steps}")

        # set_weights 已把 model 设为 [φ_e, 私有 head]，Θ_bb 重置为 φ_e

        # ── 阶段 1：head 训练（冻 backbone） ──────────────────────────────
        for _ in range(plocal):
            for x, y in self._shuffled_batches():
                self._train_head_step(x, y)
        self._head_weights = [v.numpy() for v in self._head_tvars]

        # ── 阶段 2：pFedMe on backbone（冻 head） ─────────────────────────
        for _ in range(local):
            for x, y in self._shuffled_batches():
                for _ in range(inner_steps):
                    loss = self._inner_step(x, y)
                self._moreau_step()
                losses.append(float(loss.numpy()))

        # ── 上传 Θ_bb（backbone）+ 私有 head ──────────────────────────────
        upload_weights = self._anchor_model.get_weights()  # backbone=Θ_bb
        for k, idx in enumerate(self._head_w_idx):
            upload_weights[idx] = self._head_weights[k]
        # self.model 停在 [θ̃_bb, 私有 head] —— 供 PM 评估 ✓

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-pFedMe-Rep(λ2={lam2}, K={inner_steps}) | loss={avg:.4f}")
        return upload_weights, self.n_samples, avg, time.time() - t0

    # ── 数据遍历辅助 ───────────────────────────────────────────────────────
    def _shuffled_batches(self):
        eb = self._batch_list.copy()
        random.shuffle(eb)
        return [(x, y) for x, y in eb if x.shape[0] == self._batch_size]

    # ══════════════════════════════════════════════════════════════════════
    # TF 图内步骤（限定 backbone，用 tf.Variable 超参避免 retrace）
    # ══════════════════════════════════════════════════════════════════════

    @tf.function
    def _train_head_step(self, images, labels):
        """训 head（冻 backbone）：梯度只对 head trainable 变量。"""
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
        grads = tape.gradient(loss, self._head_tvars)
        self._head_opt.apply_gradients(zip(grads, self._head_tvars))
        return loss

    @tf.function
    def _inner_step(self, images, labels):
        """
        θ̃_bb 的一步内层更新，以 Θ_bb 为近端锚点（冻 head）。

        目标：F(θ̃) + (λ2/2)·‖θ̃_bb − Θ_bb‖²，梯度只对 backbone 变量。
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - t))
                for v, t in zip(self._base_tvars, self._base_anchor_tvars)
            ])
            total = loss + (self._lam2_var / 2.0) * prox
        grads = tape.gradient(total, self._base_tvars)
        self._base_opt.apply_gradients(zip(grads, self._base_tvars))
        return loss

    @tf.function
    def _moreau_step(self):
        """Moreau 包络步，更新 backbone 锚点：Θ_bb ← Θ_bb − η2·λ2·(Θ_bb − θ̃_bb)。"""
        for t, v in zip(self._base_anchor_tvars, self._base_tvars):
            t.assign(t - self._eta2_var * self._lam2_var * (t - v))
