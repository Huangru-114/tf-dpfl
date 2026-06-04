"""
client/client_neurotoxin.py  –  Neurotoxin（ICML 2022）恶意客户端

官方参考：jhcknzzm/Federated-Learning-Backdoor，FL_Backdoor_CV/helper.grad_mask_cv
+ train_funcs.apply_grad_mask。核心：恶意客户端只更新**良性训练改动最小**的坐标
（屏蔽 benign 梯度幅值 top-k% 的坐标），让后门更持久（不被后续良性更新冲掉）。

与官方的实现差异（已适配本仓库的三层 + 并行架构）：
  - 官方用「上一轮全局模型在 clean 数据上的聚合梯度」生成 mask。
  - 本实现让恶意客户端**自给自足**地用「连续两次收到的 edge 权重之差」作为良性方向代理：
    pseudo_benign_grad = edge_weights_now − edge_weights_prev。
    这样无需修改任何 edge server（可与 FedAvg/pFedMe/Ditto/Rep 等任意 edge 组合），
    首轮无 prev 时退化为不屏蔽（= 普通投毒训练，baseline）。
  - 投毒数据沿用静态 BadNet（build_poisoned_dataset 在 client 构造时注入 self.dataset）。

仅当 self.is_malicious 为真时启用梯度屏蔽；良性时完全等同 FedAvgClient。
mask_ratio = 被屏蔽（置 0）的坐标比例 = benign 梯度幅值 top-k%。
"""

import time

import numpy as np
import tensorflow as tf

from .client_fedavg import FedAvgClient


class NeurotoxinClient(FedAvgClient):

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = config.get("backdoor", {})
        self.mask_ratio = float(bd.get("neurotoxin_mask_ratio", 0.05))

        # get_weights() 索引 ←→ trainable_variables 映射（按变量身份）
        w_id_to_idx = {id(w): i for i, w in enumerate(model.weights)}
        self._tv_w_idx = [w_id_to_idx[id(v)] for v in model.trainable_variables]

        self._prev_edge_full = None    # 上一次收到的 edge 完整权重
        self._pseudo_grad = None       # 良性方向代理（完整权重列表的差）

    # ── 接收 edge 权重时顺便算良性方向代理 ──────────────────────────────────
    def set_weights(self, global_weights, edge_weights=None):
        src = edge_weights if edge_weights is not None else global_weights
        if self._prev_edge_full is not None and src is not None:
            self._pseudo_grad = [now - old for now, old
                                 in zip(src, self._prev_edge_full)]
        else:
            self._pseudo_grad = None
        self._prev_edge_full = [w.copy() for w in src] if src is not None else None
        super().set_weights(global_weights, edge_weights)

    # ── Neurotoxin 梯度屏蔽 mask（benign 幅值 top-k% → 0，其余 → 1）─────────
    def _compute_mask(self):
        if self._pseudo_grad is None or self.mask_ratio <= 0.0:
            return None
        # 取与各 trainable 变量对应的 benign 方向分量
        grad_tv = [np.abs(self._pseudo_grad[wi]) for wi in self._tv_w_idx]
        flat = np.concatenate([g.reshape(-1) for g in grad_tv])
        k = int(len(flat) * self.mask_ratio)
        if k <= 0:
            return None
        # 第 k 大的阈值：|grad| >= thr 的坐标被屏蔽（mask=0）
        thr = np.partition(flat, -k)[-k]
        return [(np.abs(self._pseudo_grad[wi]) < thr).astype(np.float32)
                for wi in self._tv_w_idx]

    def local_train(self, round_idx: int):
        if not getattr(self, "is_malicious", False):
            return super().local_train(round_idx)

        epochs = int(self.config["training"]["local_epochs"])
        mask = self._compute_mask()       # None => 不屏蔽（首轮 / mask_ratio=0）
        tvars = self.model.trainable_variables

        losses, t0 = [], time.time()
        for _ in range(epochs):
            for x, y in self.dataset:
                with tf.GradientTape() as tape:
                    loss = self.loss_fn(y, self.model(x, training=True))
                grads = tape.gradient(loss, tvars)
                if mask is not None:
                    grads = [g * m for g, m in zip(grads, mask)]
                self.optimizer.apply_gradients(zip(grads, tvars))
                losses.append(float(loss.numpy()))

        avg = float(np.mean(losses)) if losses else 0.0
        masked = "off" if mask is None else f"top-{self.mask_ratio:.0%}"
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Neurotoxin(mask={masked}) | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0
