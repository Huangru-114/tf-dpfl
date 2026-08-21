"""
client/hier_fedrep.py  –  Hier-FedRep 客户端

把 FedRep（Collins et al., ICML 2021）式 backbone/head 拆分引入三层 HFL：
本质就是 **Hier-FedAvg + 私有分类头**——backbone 共享聚合，head 完全私有、
永不上传，轮次间 warm-start。相比 Hier-Ditto-Rep，去掉了 Ditto 双 backbone
与近端项，更轻量。

变量角色：
  φ_e               edge 下发的 backbone（self.edge_weights 的 backbone 切片）
  w_k               上传 backbone（冻 head 后本地训练得到，交给 edge 聚合）
  self._head_weights 私有 head（永不上传，轮次间 warm-start）
  self.model（结束时）= [w_k, 私有 head] —— 个性化模型，供 PM 评估

模型拆分（索引方案，单一 tf.keras.Model）：
  head     = get_weights() 中最后一个 Dense(num_classes) 的 kernel+bias
  backbone = 其余所有权重
  （由 models.cnn.get_base_head_indices 统一计算，与 Hier-*-Rep 一致）

关键顺序约束（FedRep 理论）：
  每轮先训 head（冻 backbone，多步 plocal_epochs），再训 backbone（冻 head，
  少步 local_epochs），plocal_epochs > local_epochs。理由：backbone 的梯度方向
  依赖 head 接近最优，head 没收敛会让 backbone 更新有偏。
"""

import random
import time

import numpy as np
import tensorflow as tf

from .client_base import FLClientBase
from models.cnn import get_base_head_indices


class HierFedRepClient(FLClientBase):

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

        # ── 持久状态 ──────────────────────────────────────────────────────
        self._head_weights: list | None = None   # 私有 head（warm-start）

        # ── dataset 缓存 ──────────────────────────────────────────────────
        self._batch_list = list(self.dataset)
        self._batch_size = config["data"]["batch_size"]

        # ── head / backbone 各自的优化器 ─────────────────────────────────
        # 同一 SGD 实例只能服务于构建时那组变量（Keras 3 强约束），故 head 阶段
        # 与 backbone 阶段必须用独立优化器。
        # 个性化 head 用**单独、更低**的学习率（head_lr_rep）：head 是个性化部分，
        # 三层结构下每 global round 训 edge_rounds×plocal_epochs 个 epoch、warm-start，
        # 用 backbone 同样的 lr 会让 head 过拟合本地数据（PM loss 后期回升）。低 head_lr
        # 减缓 head 记忆、配合瓶颈一起抑制过拟合。backbone 仍用基类 lr_schedule。
        self._head_lr0    = float(config["training"].get("head_lr_rep", 0.005))
        self._head_lr_var = tf.Variable(self._head_lr0, trainable=False,
                                        dtype=tf.float32, name=f"head_lr_c{client_id}")
        self._head_opt = tf.keras.optimizers.SGD(learning_rate=self._head_lr_var)
        self._base_opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)

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
        Hier-FedRep 本地训练：

          阶段 1：训 head（冻 backbone），plocal_epochs，普通 CE → 保存私有 head
          阶段 2：训上传 backbone w_k（冻 head），local_epochs，普通 CE → 上传
          self.model 结束时停在 [w_k, 私有 head]，即个性化模型，供 PM 评估

        Returns:
            (upload_weights, n_samples, avg_loss, elapsed)
            upload_weights：完整列表，backbone=w_k，head=私有 head（上游聚合时忽略 head）
        """
        plocal = int(self.config["training"].get("plocal_epochs", 5))
        local  = int(self.config["training"].get("local_epochs", 1))

        losses, t0 = [], time.time()
        print(f"  [Client {self.client_id:>2}] Hier-FedRep | "
              f"plocal={plocal}, local={local}")

        # set_weights 已把 model 设为 [φ_e backbone, 私有 head]

        self.on_round_start(round_idx)

        # ── 阶段 1：head 训练（冻 backbone） ──────────────────────────────
        for _ in range(plocal):
            for x, y in self._shuffled_batches():
                x, y = self.on_batch(x, y)
                self._train_head_step(x, y)
        self._head_weights = [v.numpy() for v in self._head_tvars]

        # ── 阶段 2：上传 backbone w_k 训练（冻 head，普通 CE） ─────────────
        for _ in range(local):
            for x, y in self._shuffled_batches():
                x, y = self.on_batch(x, y)
                loss = self._train_backbone_step(x, y)
                losses.append(float(loss.numpy()))
        upload_weights = self.model.get_weights()  # backbone=w_k（交 edge 聚合）, head=私有

        # ── 个性化模型（供 PM 评估）= [共享表示 φ_e, 私有 head] ──────────────────
        # FedRep 定义：个性化模型 = 共享表示 + 本地 head。head 是在收到的共享 backbone
        # φ_e 上训练的（阶段1），故评估必须用 φ_e 与之配对。
        # 之前用 phase2 漂移后的 w_k 评估 → head/backbone 失配，且 w_k 过拟合本地少数类
        # → PM loss 随训练单调上升、acc 偏低（客户端越多数据越少、越严重）。
        # 注意：上传给 edge 聚合的仍是 w_k（upload_weights 已在上面捕获），此处只改 self.model
        # 的评估状态，不影响聚合。
        pm_weights = [w.copy() for w in self.edge_weights]   # φ_e backbone（+ edge head 占位）
        for k, idx in enumerate(self._head_w_idx):
            pm_weights[idx] = self._head_weights[k]
        self.model.set_weights(pm_weights)
        # self.model 现停在 [φ_e 共享表示, 私有 head] —— 一致的个性化模型，供 PM 评估 ✓

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-FedRep | loss={avg:.4f}")
        upload_weights = self.on_upload(upload_weights, round_idx)
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
    def _train_backbone_step(self, images, labels):
        """
        训上传 backbone w_k（冻 head）：普通 CE，梯度只对 backbone 变量。

        这是产出 upload 的那一步 → 叠加 on_extra_loss（基类返回 0.0，数值恒等）。
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            total = loss + self.on_extra_loss()
        grads = tape.gradient(total, self._base_tvars)
        self._base_opt.apply_gradients(zip(grads, self._base_tvars))
        return loss
