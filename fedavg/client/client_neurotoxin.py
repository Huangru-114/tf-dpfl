"""
client/client_neurotoxin.py  –  Neurotoxin（ICML 2022）攻击 mixin

官方参考：jhcknzzm/Federated-Learning-Backdoor，grad_mask_cv + apply_grad_mask。
核心思想：把后门藏到**良性训练几乎不更新**的坐标里，使其不被后续良性轮覆盖 → 持久。
做法：在**干净数据**上算 benign 梯度，屏蔽幅值 top-k% 坐标（良性最常更新、最易冲掉
后门的），只把恶意更新投影到剩余的低梯度坐标上；并**每轮持续参与**让后门逐轮累积。

本实现要点
──────────
  - **是 mixin，不是 Client 子类**（CLAUDE.md 陷阱 #1）。它不继承 FedAvgClient，
    而是与任意 PFL 方法类组合（见 client/compose.py），因此恶意客户端仍然跑
    pFedMe / Ditto / FedRep 的 local_train，只是在钩子上叠加掩码投影。
  - benign 梯度用**真实干净数据梯度**（在收到的 edge 模型上算），不用 edge 权重差代理。
  - 掩码投影挂在 `on_upload` 上，作用于**上传权重列表**，基准是 `self.edge_weights`。
    `upload − edge_weights` 才是聚合端真正看到的 delta，且对六个 PFL 方法**全都有定义**
    （FedAvg 传完整权重、pFedMe 传锚点 Θ、Ditto 传阶段 A 的 w_k、Rep 家族传
    backbone+私有 head）。旧实现取 `self.model` 训练前后之差 —— 在非 FedAvg 方法下
    那是**个性化模型**，与上传物无关。
  - `on_upload` 是**纯函数**：绝不 `self.model.set_weights(...)`。旧实现会把掩码后的
    上传物写回 self.model，即污染个性化模型，破坏 PM 评估 / 遗忘曲线 / local ASR。
  - 连续参与由 main.py 对 neurotoxin 设 Q_eff=1（每轮在场）保证累积。
  - 投毒数据为静态 BadNet（build_poisoned_dataset 注入 self.dataset）；干净数据另由
    set_clean_dataset 注入，仅用于算 benign 梯度掩码。

mask_ratio = 被屏蔽（置 0 更新）的坐标比例 = benign 幅值 top-k%。
> 这个符号方向是否与官方一致**尚未证实**（CLAUDE.md 陷阱 #4，本会话不处理）。

已知残留（决策 A）：Rep 家族上传列表里的 head 槽位会被一起掩码。那些槽位上游平均后
客户端收到时会被私有 head 覆盖，因此是**惰性的**（不影响任何结果），但不等价。
"""

import numpy as np
import tensorflow as tf


class NeurotoxinMixin:

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = (config or {}).get("backdoor", {})
        self._atk_mask_ratio   = float(bd.get("neurotoxin_mask_ratio", 0.05))
        self._atk_mask_batches = int(bd.get("neurotoxin_mask_batches", 0))  # 0 = 全部 clean batch
        self._atk_clean_dataset = None   # 未投毒本地训练集，仅用于算 benign 梯度掩码
        self._atk_mask = None            # 本轮掩码（on_round_start 算，on_upload 用）

        # trainable_variables → get_weights() 索引映射（把梯度掩码对齐到完整权重列表）。
        tv_ids = {id(v): j for j, v in enumerate(model.weights)}
        self._atk_tv_to_w_idx = [tv_ids[id(v)] for v in model.trainable_variables]

    def set_clean_dataset(self, clean_dataset):
        """注入未投毒的本地训练集（main.py 在替换投毒数据集前捕获并传入）。"""
        self._atk_clean_dataset = clean_dataset

    # ── 在当前模型上算 clean 数据梯度（不 apply），@tf.function 线程安全 ──────────
    @tf.function
    def _atk_benign_grads(self, images, labels):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=False))
        return tape.gradient(loss, self.model.trainable_variables)

    # ── Neurotoxin 掩码：屏蔽 benign 真实梯度幅值 top-k% 坐标（→0），其余→1 ────────
    def _atk_compute_mask(self):
        if self._atk_clean_dataset is None or self._atk_mask_ratio <= 0.0:
            return None

        # 在收到的 edge 模型上累计若干 batch 的 benign 梯度（Σgrad）
        acc, nb = None, 0
        for x, y in self._atk_clean_dataset:
            g = [gi.numpy() for gi in self._atk_benign_grads(x, y)]
            acc = g if acc is None else [a + gi for a, gi in zip(acc, g)]
            nb += 1
            if self._atk_mask_batches > 0 and nb >= self._atk_mask_batches:
                break
        if acc is None:
            return None

        # 全局 top-k%（按 |Σgrad|）定阈值
        # > 官方是**逐层**阈值，这里是全局阈值（陷阱 #4，本会话不处理）。
        flat = np.concatenate([np.abs(a).reshape(-1) for a in acc])
        k = int(len(flat) * self._atk_mask_ratio)
        if k <= 0:
            return None
        thr = np.partition(flat, -k)[-k]                # 第 k 大的阈值
        grad_mask = [(np.abs(a) < thr).astype(np.float32) for a in acc]

        # 对齐到完整 get_weights() 列表：trainable 位置用上面的掩码，其余=1（不屏蔽）
        full = [np.ones_like(w, dtype=np.float32) for w in self.model.get_weights()]
        for tv_i, w_i in enumerate(self._atk_tv_to_w_idx):
            full[w_i] = grad_mask[tv_i]
        return full

    # ══════════════════════════════════════════════════════════════════════
    # 钩子
    # ══════════════════════════════════════════════════════════════════════

    def on_round_start(self, round_idx: int):
        """在收到的 edge 模型上（尚未训练）用干净数据算 benign 梯度掩码。"""
        super().on_round_start(round_idx)
        self._atk_mask = self._atk_compute_mask() if self.is_malicious else None

    def on_upload(self, upload: list, round_idx: int):
        """
        Neurotoxin 投影：只保留 benign 改动最小坐标上的更新。

        纯函数 —— 只返回新列表，不碰 self.model。
        """
        upload = super().on_upload(upload, round_idx)       # 先走内层（防御 mixin）
        if not self.is_malicious or self._atk_mask is None:
            return upload
        ref = self.edge_weights
        if ref is None or len(ref) != len(upload):
            print(f"  [Client {self.client_id:>2}] Neurotoxin: edge_weights 不可用或"
                  f"长度不匹配，跳过掩码投影。")
            return upload
        projected = [r + (u - r) * m for r, u, m in zip(ref, upload, self._atk_mask)]
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Neurotoxin(mask=top-{self._atk_mask_ratio:.0%}, ref=edge_weights)")
        return projected
