"""
client/client_neurotoxin.py  –  Neurotoxin（ICML 2022）恶意客户端（忠实重写）

官方参考：jhcknzzm/Federated-Learning-Backdoor，grad_mask_cv + apply_grad_mask。
核心思想：把后门藏到**良性训练几乎不更新**的坐标里，使其不被后续良性轮覆盖 → 持久。
做法：在**干净数据**上算 benign 梯度，屏蔽幅值 top-k% 坐标（良性最常更新、最易冲掉后门的），
只把恶意更新投影到剩余的低梯度坐标上；并**每轮持续参与**让后门逐轮累积。

本实现要点：
  - benign 梯度用**真实干净数据梯度**（在收到的 edge 模型上算），不再用 edge 权重差代理
    （代理在 fix-frequency 稀疏参与下失效：首轮为 None、跨轮是陈旧聚合差、方向错）。
  - mask 一轮内固定，故「逐步 mask 梯度」≡「对整轮累计 update 做一次 mask」。复用父类
    FedAvgClient.local_train（与所有客户端一致的 @tf.function 训练路径），训练后对
    (w_after − w_before) 做一次 mask 投影。避免手写 eager 循环与良性 @tf.function 在线程池
    并发污染（曾导致 reshape 报错、恶意更新被丢弃、ASR=0）。
  - 连续参与由 main.py 对 neurotoxin 设 Q_eff=1（每轮在场）保证累积。
  - 投毒数据为静态 BadNet（build_poisoned_dataset 注入 self.dataset）；干净数据另由
    set_clean_dataset 注入，仅用于算 benign 梯度 mask。

mask_ratio = 被屏蔽（置 0 更新）的坐标比例 = benign 幅值 top-k%。
"""

import numpy as np
import tensorflow as tf

from .client_fedavg import FedAvgClient


class NeurotoxinClient(FedAvgClient):

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = config.get("backdoor", {})
        self.mask_ratio = float(bd.get("neurotoxin_mask_ratio", 0.05))
        self.mask_batches = int(bd.get("neurotoxin_mask_batches", 0))  # 0 = 全部 clean batch
        self._clean_dataset = None     # 未投毒本地训练集，仅用于算 benign 梯度 mask

        # trainable_variables → get_weights() 索引映射（用于把梯度 mask 对齐到完整权重列表）。
        # fedavg_cnn 无 BN，trainable 即全部权重；一般情况下非 trainable 位置 mask=1（占位）。
        tv_ids = {id(v): j for j, v in enumerate(model.weights)}
        self._tv_to_w_idx = [tv_ids[id(v)] for v in model.trainable_variables]

    def set_clean_dataset(self, clean_dataset):
        """注入未投毒的本地训练集（main.py 在替换投毒数据集前捕获并传入）。"""
        self._clean_dataset = clean_dataset

    # ── 在当前模型上算 clean 数据梯度（不 apply），@tf.function 线程安全 ──────────
    @tf.function
    def _benign_grads(self, images, labels):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=False))
        return tape.gradient(loss, self.model.trainable_variables)

    # ── Neurotoxin mask：屏蔽 benign 真实梯度幅值 top-k% 坐标（→0），其余→1 ────────
    def _compute_mask(self):
        if self._clean_dataset is None or self.mask_ratio <= 0.0:
            return None

        # 在收到的 edge 模型上累计若干 batch 的 benign 梯度（Σgrad）
        acc = None
        nb = 0
        for x, y in self._clean_dataset:
            g = self._benign_grads(x, y)
            g = [gi.numpy() for gi in g]
            acc = g if acc is None else [a + gi for a, gi in zip(acc, g)]
            nb += 1
            if self.mask_batches > 0 and nb >= self.mask_batches:
                break
        if acc is None:
            return None

        # 全局 top-k%（按 |Σgrad|）定阈值
        flat = np.concatenate([np.abs(a).reshape(-1) for a in acc])
        k = int(len(flat) * self.mask_ratio)
        if k <= 0:
            return None
        thr = np.partition(flat, -k)[-k]                # 第 k 大的阈值
        grad_mask = [(np.abs(a) < thr).astype(np.float32) for a in acc]

        # 对齐到完整 get_weights() 列表：trainable 位置用上面的 mask，其余=1（不屏蔽）
        full = [np.ones_like(w, dtype=np.float32) for w in self.model.get_weights()]
        for tv_i, w_i in enumerate(self._tv_to_w_idx):
            full[w_i] = grad_mask[tv_i]
        return full

    def local_train(self, round_idx: int):
        if not getattr(self, "is_malicious", False):
            return super().local_train(round_idx)

        # 1) 先在收到的 edge 模型上算 benign mask（用干净数据真实梯度，模型尚未训练）
        w_before = self.model.get_weights()
        mask = self._compute_mask()

        # 2) 正常投毒训练（复用与所有客户端一致的 @tf.function 路径，避免 eager/线程污染）
        upload, n, loss, t = super().local_train(round_idx)

        # 3) Neurotoxin 投影：仅保留 benign 改动最小坐标上的更新
        masked = "off"
        if mask is not None:
            w_after = self.model.get_weights()
            w_proj = [wb + (wa - wb) * m
                      for wb, wa, m in zip(w_before, w_after, mask)]
            self.model.set_weights(w_proj)
            upload = self.model.get_weights()
            masked = f"top-{self.mask_ratio:.0%}"

        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Neurotoxin(mask={masked}) | loss={loss:.4f}")
        return upload, n, loss, t
