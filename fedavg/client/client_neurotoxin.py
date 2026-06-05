"""
client/client_neurotoxin.py  –  Neurotoxin（ICML 2022）恶意客户端

官方参考：jhcknzzm/Federated-Learning-Backdoor，grad_mask_cv + apply_grad_mask。核心：
恶意客户端只更新**良性训练改动最小**的坐标（屏蔽 benign 梯度幅值 top-k% 的坐标），
让后门更持久（不被后续良性更新冲掉）。Neurotoxin 本质是「单行方法」——把更新投影到
benign 改动最小的坐标集合上。

实现要点（关键修复）：
  - mask 在一轮内固定，因此「逐步 mask 梯度」≡「对整轮累计 update 做一次 mask」。
    故本类**复用父类 FedAvgClient.local_train（即与所有客户端一致的 @tf.function 训练路径）**，
    训练后对 (w_after − w_before) 做一次 mask 投影。
    这样避免了「手写 eager 训练循环」与良性客户端 @tf.function 在线程池里并发执行而互相
    污染（曾导致 reshape/形状错乱报错、恶意更新被丢弃、ASR 恒为 0）。
  - benign 方向代理：用连续两次收到的 edge 权重之差（self-contained，无需改任何 edge server，
    可与 FedAvg/pFedMe/Ditto/FedRep 等任意 edge 组合）；首轮无 prev → 不屏蔽（= 普通投毒，baseline）。
  - 投毒数据为静态 BadNet（build_poisoned_dataset 注入 self.dataset）。

mask_ratio = 被屏蔽（置 0 更新）的坐标比例 = benign 幅值 top-k%。
"""

import numpy as np

from .client_fedavg import FedAvgClient


class NeurotoxinClient(FedAvgClient):

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = config.get("backdoor", {})
        self.mask_ratio = float(bd.get("neurotoxin_mask_ratio", 0.05))
        self._prev_edge_full = None    # 上一次收到的 edge 完整权重
        self._pseudo_grad = None       # benign 方向代理（完整权重列表的差）

    # ── 接收 edge 权重时顺便算 benign 方向代理 ──────────────────────────────
    def set_weights(self, global_weights, edge_weights=None):
        src = edge_weights if edge_weights is not None else global_weights
        if self._prev_edge_full is not None and src is not None:
            self._pseudo_grad = [now - old for now, old
                                 in zip(src, self._prev_edge_full)]
        else:
            self._pseudo_grad = None
        self._prev_edge_full = [w.copy() for w in src] if src is not None else None
        super().set_weights(global_weights, edge_weights)

    # ── Neurotoxin 坐标 mask（benign 幅值 top-k% → 0，其余 → 1），按 get_weights 对齐 ──
    def _compute_mask(self):
        if self._pseudo_grad is None or self.mask_ratio <= 0.0:
            return None
        flat = np.concatenate([np.abs(g).reshape(-1) for g in self._pseudo_grad])
        k = int(len(flat) * self.mask_ratio)
        if k <= 0:
            return None
        thr = np.partition(flat, -k)[-k]            # 第 k 大的阈值
        return [(np.abs(g) < thr).astype(np.float32) for g in self._pseudo_grad]

    def local_train(self, round_idx: int):
        if not getattr(self, "is_malicious", False):
            return super().local_train(round_idx)

        # 1) 正常投毒训练（复用与所有客户端一致的 @tf.function 路径，避免 eager/线程污染）
        w_before = self.model.get_weights()
        upload, n, loss, t = super().local_train(round_idx)

        # 2) Neurotoxin 投影：仅保留 benign 改动最小坐标上的更新
        mask = self._compute_mask()
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
