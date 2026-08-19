"""
client/hier_ditto.py  –  Hier-Ditto 客户端（整模型 Ditto，无 backbone/head 拆分）

参考 client_pfedme.py 的结构（clone 锚点、tf.Variable 超参避免 retrace、
_batch_list 缓存 + 每 epoch shuffle + drop_last）。区别在个性化机制：
pFedMe 用 Moreau envelope，本类用 Ditto（双模型）。

Ditto 双模型（Li et al., ICML 2021）：
  w_k   本地/共享模型：从 edge 权重 θ_e 起做普通本地 SGD，上传给 edge 聚合。
  v_k   个性化模型：以 θ_e 为近端锚点训练 min F(v) + (λ/2)‖v − θ_e‖²，
        本地保留（warm-start），用于 PM 评估。

变量角色对照：
  ref_weights / θ_e   edge 下发权重 ←→ pFedMe 中的 server.model
  w_k（上传）          ←→ pFedMe 上传的 Moreau center
  self.model（结束时）= v_k（个性化模型），供 evaluate_on 使用
"""

import random
import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase


class HierDittoClient(FLClientBase):

    def __init__(
        self,
        client_id: int,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        config: dict,
        n_samples: int = None,
    ):
        super().__init__(client_id, dataset, model, config, n_samples)

        # ── 近端锚点 clone（存 θ_e，供 Ditto 近端项参考） ─────────────────
        self._anchor_model = tf.keras.models.clone_model(model)
        self._anchor_model.set_weights(model.get_weights())

        # ── 持久状态 ──────────────────────────────────────────────────────
        self._v_weights: list | None = None             # 个性化模型 v_k（warm-start）
        self._personalized_weights: list | None = None  # 评估快照（= v_k）

        # ── dataset 缓存 ──────────────────────────────────────────────────
        self._batch_list = list(self.dataset)
        self._batch_size = config["data"]["batch_size"]

        # ── 超参 tf.Variable（避免 @tf.function retrace） ─────────────────
        self._lam_var: tf.Variable | None = None

    # ══════════════════════════════════════════════════════════════════════
    # 权重管理
    # ══════════════════════════════════════════════════════════════════════

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """接收广播权重；self.model 与锚点都对齐到 edge 权重 θ_e。"""
        self.global_weights = (
            [w.copy() for w in global_weights] if global_weights is not None else None
        )
        src = edge_weights if edge_weights is not None else global_weights
        self.edge_weights = [w.copy() for w in src]
        self.model.set_weights(src)
        self._anchor_model.set_weights(src)  # 近端锚点 = θ_e
        print(f"  [Client {self.client_id:>2}] Received edge weights (Hier-Ditto).")

    # ══════════════════════════════════════════════════════════════════════
    # 训练
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """
        Hier-Ditto 本地训练（双模型）：

          阶段 A：共享模型 w_k —— 从 θ_e 起做普通本地 SGD（local_epochs）→ 上传
          阶段 B：个性化模型 v_k —— 以 θ_e 为近端锚点训练 Ditto 近端目标
                  （ditto_pers_epochs，warm-start）→ 本地保留，self.model 停在 v_k

        Returns:
            (upload_weights=w_k, n_samples, avg_loss, elapsed)
        """
        lam         = float(self.config["training"].get(
            "lambda_ditto", self.config["training"].get("mu", 0.1)))
        epochs      = int(self.config["training"]["local_epochs"])
        pers_epochs = int(self.config["training"].get("ditto_pers_epochs", epochs))

        if self._lam_var is None:
            self._lam_var = tf.Variable(lam, trainable=False, dtype=tf.float32)
        else:
            self._lam_var.assign(lam)

        losses, t0 = [], time.time()
        print(f"  [Client {self.client_id:>2}] Hier-Ditto | "
              f"local={epochs}, pers={pers_epochs}, λ={lam}")

        self.on_round_start(round_idx)

        # ── 阶段 A：共享模型 w_k（普通本地 SGD，从 θ_e 起） ───────────────
        self.model.set_weights(self.edge_weights)
        for _ in range(epochs):
            for x, y in self._shuffled_batches():
                x, y = self.on_batch(x, y)
                loss = self._plain_step(x, y)
                losses.append(float(loss.numpy()))
        w_upload = self.model.get_weights()

        # ── 阶段 B：个性化模型 v_k（Ditto 近端，锚点 = θ_e，warm-start） ───
        self.model.set_weights(
            self._v_weights if self._v_weights is not None else self.edge_weights)
        for _ in range(pers_epochs):
            for x, y in self._shuffled_batches():
                x, y = self.on_batch(x, y)
                self._prox_step(x, y)
        self._v_weights = self.model.get_weights()
        self._personalized_weights = self._v_weights
        # self.model 现停在 v_k —— 供 PM 评估 ✓

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-Ditto(λ={lam}) | loss={avg:.4f}")
        w_upload = self.on_upload(w_upload, round_idx)
        return w_upload, self.n_samples, avg, time.time() - t0

    # ── 数据遍历辅助 ───────────────────────────────────────────────────────
    def _shuffled_batches(self):
        eb = self._batch_list.copy()
        random.shuffle(eb)
        return [(x, y) for x, y in eb if x.shape[0] == self._batch_size]

    # ══════════════════════════════════════════════════════════════════════
    # TF 图内步骤
    # ══════════════════════════════════════════════════════════════════════

    @tf.function
    def _plain_step(self, images, labels):
        """
        共享模型 w_k 的一步普通 SGD。

        这是产出 upload 的那一步 → 叠加 on_extra_loss（基类返回 0.0，数值恒等）。
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            total = loss + self.on_extra_loss()
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def _prox_step(self, images, labels):
        """
        个性化模型 v_k 的一步 Ditto 近端更新。

        目标：F(v) + (λ/2)·‖v − θ_e‖²（θ_e 存于 _anchor_model）。
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - a))
                for v, a in zip(self.model.trainable_variables,
                                self._anchor_model.trainable_variables)
            ])
            total = loss + (self._lam_var / 2.0) * prox
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
