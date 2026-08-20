"""
client/hier_ditto_rep.py  –  Hier-Ditto-Rep 客户端

把 FedRep 式 backbone/head 拆分引入三层 HFL，backbone 个性化机制用 Ditto
（双 backbone 结构）。

变量角色：
  φ_e               edge 下发的 backbone（self.edge_weights 的 backbone 切片）
  w_k               上传 backbone（普通本地训练得到，交给 edge 聚合）
  v_k               个性化 backbone（Ditto 近端项靠近 φ_e，本地保留，warm-start）
  self._head_weights 私有 head（永不上传，轮次间 warm-start）

模型拆分（索引方案，单一 tf.keras.Model）：
  head     = get_weights() 末尾两个元素（最后一个 Dense(num_classes) 的 kernel+bias）
  backbone = 其余所有权重

关键顺序约束（FedRep 理论）：
  每轮先训 head（冻 backbone，多步 plocal_epochs），再训 backbone（冻 head，
  少步 local_epochs），plocal_epochs > local_epochs。理由：backbone 的梯度方向
  依赖 head 接近最优，head 没收敛会让 backbone 更新有偏。

head 处理：
  head 完全私有，永不上传、永不被 edge/cloud 聚合覆盖。set_weights 接收 edge
  下发的完整权重后，只覆盖 backbone 索引、保留本地 head（把 head 索引位置替换
  成 self._head_weights）。上传时给出完整权重列表，但 head 位置在 edge/cloud 上
  被平均后会被 client 重新覆盖，因此实际不参与个性化模型。
"""

import random
import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase
from models.cnn import get_base_head_indices


class HierDittoRepClient(FLClientBase):

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

        # 预捕获 trainable 变量子集（set_weights 原地赋值，变量对象不变）
        tv = model.trainable_variables
        self._head_tvars = [tv[j] for j in self._head_tv_idx]
        self._base_tvars = [tv[j] for j in self._base_tv_idx]

        # ── Ditto 近端锚点（存 φ_e，仅 backbone 参与近端项） ──────────────
        self._anchor_model = tf.keras.models.clone_model(model)
        self._anchor_model.set_weights(model.get_weights())
        atv = self._anchor_model.trainable_variables
        self._base_anchor_tvars = [atv[j] for j in self._base_tv_idx]

        # ── 持久状态 ──────────────────────────────────────────────────────
        self._head_weights:  list | None = None  # 私有 head（warm-start）
        self._pers_backbone: list | None = None  # 个性化 backbone v_k（完整列表）

        # ── dataset 缓存 ──────────────────────────────────────────────────
        self._batch_list = list(self.dataset)
        self._batch_size = config["data"]["batch_size"]

        # ── head / backbone 各自的优化器 ─────────────────────────────────
        # 同一个 SGD 实例只能服务于「构建时」的那组变量（Keras 3 强约束），
        # 因此 head 阶段与 backbone 阶段必须用独立优化器。
        # 个性化 head 用**单独、更低**的学习率（head_lr_rep），抑制 head 过拟合（PM loss
        # 后期回升）；backbone 仍用基类 lr_schedule。SGD(momentum=0) 无状态耦合。
        self._head_lr0    = float(config["training"].get("head_lr_rep", 0.005))
        self._head_lr_var = tf.Variable(self._head_lr0, trainable=False,
                                        dtype=tf.float32, name=f"head_lr_c{client_id}")
        self._head_opt = tf.keras.optimizers.SGD(learning_rate=self._head_lr_var)
        self._base_opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)

        # ── 超参 tf.Variable（避免 @tf.function retrace） ─────────────────
        self._lam_var: tf.Variable | None = None

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

        首轮（self._head_weights 为 None）时从广播权重捕获初始 head。
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
        print(f"  [Client {self.client_id:>2}] Received edge backbone (private head kept).")

    # ══════════════════════════════════════════════════════════════════════
    # 训练
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """
        Hier-Ditto-Rep 本地训练（双 backbone）：

          阶段 1：训 head（冻 backbone），plocal_epochs，普通 CE → 保存私有 head
          阶段 2：训上传 backbone w_k（冻 head），local_epochs，普通 CE → 上传
          阶段 3：训个性化 backbone v_k（冻 head），local_epochs，
                  Ditto 近端 CE + (λ/2)‖v_bb − φ_e_bb‖² → 本地保留，self.model 停在
                  [v_k, 私有 head] 供 PM 评估

        Returns:
            (upload_weights, n_samples, avg_loss, elapsed)
            upload_weights：完整列表，backbone=w_k，head=私有 head（上游忽略）
        """
        plocal = int(self.config["training"].get("plocal_epochs", 5))
        local  = int(self.config["training"].get("local_epochs", 1))
        lam    = float(self.config["training"].get(
            "lambda_ditto", self.config["training"].get("mu", 0.1)))

        if self._lam_var is None:
            self._lam_var = tf.Variable(lam, trainable=False, dtype=tf.float32)
        else:
            self._lam_var.assign(lam)

        losses, t0 = [], time.time()
        print(f"  [Client {self.client_id:>2}] Hier-Ditto-Rep | "
              f"plocal={plocal}, local={local}, λ={lam}")

        # set_weights 已把 model 设为 [φ_e backbone, 私有 head]

        # ── 阶段 1：head 训练（冻 backbone） ──────────────────────────────
        for _ in range(plocal):
            for x, y in self._shuffled_batches():
                self._train_head_step(x, y)
        self._head_weights = [v.numpy() for v in self._head_tvars]

        # ── 阶段 2：上传 backbone w_k 训练（冻 head，普通 CE） ─────────────
        for _ in range(local):
            for x, y in self._shuffled_batches():
                loss = self._train_backbone_plain_step(x, y)
                losses.append(float(loss.numpy()))
        upload_weights = self.model.get_weights()  # backbone=w_k, head=私有 head

        # ── 阶段 3：个性化 backbone v_k 训练（Ditto 近端，冻 head） ────────
        # 近端锚点 = φ_e（edge backbone）
        self._anchor_model.set_weights(self.edge_weights)
        # 载入 v_k（warm-start；首轮 = φ_e）+ 私有 head
        pers_full = ([w.copy() for w in self._pers_backbone]
                     if self._pers_backbone is not None
                     else [w.copy() for w in self.edge_weights])
        for k, idx in enumerate(self._head_w_idx):
            pers_full[idx] = self._head_weights[k].copy()
        self.model.set_weights(pers_full)

        for _ in range(local):
            for x, y in self._shuffled_batches():
                self._train_backbone_ditto_step(x, y)
        self._pers_backbone = self.model.get_weights()
        # self.model 现停在 [v_k, 私有 head] —— 供 PM 评估 ✓

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-Ditto-Rep | loss={avg:.4f}")
        return upload_weights, self.n_samples, avg, time.time() - t0

    # ── 数据遍历辅助 ───────────────────────────────────────────────────────
    def _shuffled_batches(self):
        """每 epoch 重新 shuffle 并 drop_last（等价 DataLoader(shuffle=True))。"""
        eb = self._batch_list.copy()
        random.shuffle(eb)
        return [(x, y) for x, y in eb if x.shape[0] == self._batch_size]

    # ══════════════════════════════════════════════════════════════════════
    # TF 图内步骤（各阶段独立 @tf.function，固定捕获变量子集，避免 retrace）
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
    def _train_backbone_plain_step(self, images, labels):
        """训上传 backbone w_k（冻 head）：普通 CE，梯度只对 backbone 变量。"""
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
        grads = tape.gradient(loss, self._base_tvars)
        self._base_opt.apply_gradients(zip(grads, self._base_tvars))
        return loss

    @tf.function
    def _train_backbone_ditto_step(self, images, labels):
        """
        训个性化 backbone v_k（冻 head）：Ditto 近端。

        目标：F(v) + (λ/2)·‖v_bb − φ_e_bb‖²，梯度只对 backbone 变量。
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - a))
                for v, a in zip(self._base_tvars, self._base_anchor_tvars)
            ])
            total = loss + (self._lam_var / 2.0) * prox
        grads = tape.gradient(total, self._base_tvars)
        self._base_opt.apply_gradients(zip(grads, self._base_tvars))
        return loss
