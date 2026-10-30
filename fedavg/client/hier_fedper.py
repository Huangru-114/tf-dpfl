"""
client/hier_fedper.py  –  Hier-FedPer 客户端（阶段三 CCSF，D-113 / D-114）

FedPer（Arivazhagan et al., 2019）引入三层 HFL：body 共享聚合，个性化层私有。与 FedRep 的唯一区别是**训练**：
FedPer 每一步对全部可训练变量（body + 个性化层）一起求梯度，没有「先训 head、再训 body」的两阶段。

照 CCS 官方代码（`chenjian0924/paper-code @ b4b8653`，FINDINGS F-109）：
  · 私有部分 = state_dict 里名字含 `classifier` / `project` 的参数（`src/client/fedper.py`）
    → 本仓库 = head + `proj_*`（`models.cnn.get_base_head_indices` + `PRIVATE_HEAD_PREFIXES`，与 FedRep 同一划分）；
  · `fit()` 不重写 → 沿用 `FedAvgClient`：每一步训全部参数；
  · BN running 统计量是 buffer、跟着 base 聚合 → **固定共享**：不读 FedRep 专属的 `training.fedrep_bn_stats`
    （P2 模板里它是 private，`[设定4]` 照实打印，对本方法无效；本方法自己打 `[设定13]`）；
  · 学习率 / epoch / batch 由配置给（CCSF：lr 0.1 常数 = `lr_decay: 1.0`、2 epoch、batch 64，D-113）；SGD 无 momentum。

其余都继承 HierFedRepClient：切分索引、私有部分 warm-start、`set_weights`（只覆盖 body）、`private_state`、
取数（`_epoch()`）、个性化模型的组装（φ_e + 私有部分）。`plocal_epochs` 不用（没有 head 阶段）。
CCS 的对抗训练经 `on_body_step` 整步替换联合步，梯度对 `step_vars()`（本类 = 全部可训练变量）。
"""

import time

import numpy as np
import tensorflow as tf

from .hier_fedrep import HierFedRepClient


class HierFedPerClient(HierFedRepClient):

    def __init__(self, client_id: int, dataset: tf.data.Dataset, model: tf.keras.Model,
                 config: dict, n_samples: int):
        super().__init__(client_id, dataset, model, config, n_samples)
        # BN 统计量随 body 共享（官方 FedPer：buffers 跟着 base）—— 不读 training.fedrep_bn_stats
        self._bn_private = False
        self._stat_w_idx = []
        self._stats = None
        # 联合步：全部可训练变量（body + 私有层），顺序 = model.trainable_variables；共享 body 的 lr 日程
        self._joint_tvars = list(model.trainable_variables)
        self._joint_opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)

    # ── CCS mixin 的训练步接口（HierFedRepClient 的定义 = body 变量 + body 优化器）────────
    def step_vars(self):
        return self._joint_tvars

    def step_optimizer(self):
        return self._joint_opt

    @tf.function
    def _train_joint_step(self, images, labels):
        """FedPer 的一步：普通 CE（+ on_extra_loss），梯度对全部可训练变量。"""
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            total = loss + self.on_extra_loss()
        grads = tape.gradient(total, self._joint_tvars)
        self._joint_opt.apply_gradients(zip(grads, self._joint_tvars))
        return loss

    def local_train(self, round_idx: int):
        """
        Hier-FedPer 本地训练：local_epochs 个 epoch，每批一步联合更新（投毒经 on_batch，CCS 经 on_body_step）。

        Returns:
            (upload_weights, n_samples, avg_loss, elapsed)
            upload_weights：完整列表（body 交 edge 聚合；私有层位置上游忽略）
        """
        local = int(self.config["training"].get("local_epochs", 1))
        losses, t0 = [], time.time()
        print(f"  [Client {self.client_id:>2}] Hier-FedPer | local={local}")

        self.on_round_start(round_idx)
        for _ in range(local):
            for x, y in self._epoch():
                x, y = self.on_batch(x, y)
                loss = self.on_body_step(x, y)          # 缺省 None；CCS 的对抗训练整步替换
                if loss is None:
                    loss = self._train_joint_step(x, y)
                losses.append(float(loss.numpy()))

        self._head_weights = [v.numpy() for v in self._head_tvars]
        upload_weights = self.model.get_weights()

        # 个性化模型（供 PM 评估）= [φ_e, 私有层]（与 HierFedRepClient 同一组装；统计量共享 → 随 φ_e）
        pm_weights = [w.copy() for w in self.edge_weights]
        for k, idx in enumerate(self._head_w_idx):
            pm_weights[idx] = self._head_weights[k]
        self.model.set_weights(pm_weights)

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | Hier-FedPer | loss={avg:.4f}")
        upload_weights = self.on_upload(upload_weights, round_idx)
        return upload_weights, self.n_samples, avg, time.time() - t0
