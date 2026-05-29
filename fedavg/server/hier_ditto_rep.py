"""
server/hier_ditto_rep.py  –  Hier-Ditto-Rep 边缘服务器

三层正则锚点（两套 Hier-Rep 共享的 edge 层设计）：
  Cloud:  全局 backbone φ*    标准 FedAvg 聚合各 edge 上传的 backbone（server.py 不改）
  Edge:   edge backbone φ_e   聚合 client backbone + Ditto 正则靠近 φ*（系数 mu_edge）
  Client: 个性化 backbone + 私有 head

edge 层只对 **backbone 索引** 做样本加权聚合，并用 Ditto 近端把 edge backbone
拉向 cloud 的全局 backbone φ*（φ* 在一个 global round 内固定，区别于 pFedMe 的
移动锚点 W_n）：

    φ_e_bb ← mean_bb − mu_edge · (φ_e_bb_prev − φ*_bb)

head 索引位置全程不参与聚合（保持广播下来的无用值），因为 client 收到后总会用
私有 head 覆盖。上传给 cloud 的是完整权重列表（长度匹配框架的 FedAvg），head
位置被平均但永不被使用。
"""

import numpy as np
import tensorflow as tf

from .edge_server_base import EdgeServerBase
from models.cnn import get_base_head_indices


class HierDittoRepEdgeServer(EdgeServerBase):

    # 日志标签；子类（Hier-pFedMe-Rep）可覆盖，edge 层聚合逻辑两套完全相同。
    _method_label = "Hier-Ditto-Rep"

    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        super().__init__(edge_id, clients, model, config)

        # ── backbone / head 切分索引 ─────────────────────────────────────
        num_classes = config["data"]["num_classes"]
        split = get_base_head_indices(model, num_classes)
        self._base_w_idx = split["base_weight_indices"]
        self._head_w_idx = split["head_weight_indices"]

    # ══════════════════════════════════════════════════════════════════════
    # 边缘轮次
    # ══════════════════════════════════════════════════════════════════════

    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """单轮 edge 内部通信：backbone-only FedAvg + Ditto 正则靠近 φ*。"""
        mu_edge = float(self.config["training"].get("mu_edge", 0.1))

        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"{self._method_label} | Selected {len(selected)}/{len(self.clients)}")

        # 步骤 1：广播 edge 完整权重 + 全局参考给 client
        edge_w = self.model.get_weights()
        for client in selected:
            client.set_weights(global_weights=self._global_weights_ref,
                               edge_weights=edge_w)

        # 步骤 2：收集 client 上传（完整列表，backbone=w_k）
        client_updates = self._collect_updates_parallel(
            selected, global_round_idx, mode="fedavg")

        # 步骤 3：仅 backbone 索引样本加权聚合 + Ditto 近端靠近 φ*
        total_n    = sum(n for _, n, *_ in client_updates)
        theta_prev = self.model.get_weights()
        gref       = self._global_weights_ref  # φ*（global round 内固定）

        new_w = [w.copy() for w in theta_prev]
        for j in self._base_w_idx:
            mean_bb  = sum(upd[0][j] * (upd[1] / total_n) for upd in client_updates)
            new_w[j] = mean_bb - mu_edge * (theta_prev[j] - gref[j])
        # head 索引位置保持不变（无意义但无害）
        self.model.set_weights(new_w)

        avg_loss = sum(l * n / total_n for _, n, l, _ in client_updates)
        avg_time = float(np.mean([t for *_, t in client_updates]))
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"loss={avg_loss:.4f}")
        return float(avg_loss), avg_time

    # ══════════════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════════════

    def run(self, global_round_idx: int):
        """执行 edge_rounds 轮聚合，上传 edge 完整权重（backbone=φ_e）给 cloud。"""
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []

        for er in range(1, edge_rounds + 1):
            loss, t = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(t)

        comm = 2 * self.model_bytes * len(self.clients) * edge_rounds
        return (self.model.get_weights(), self.n_samples,
                float(np.mean(round_losses)), float(np.mean(round_times)), comm)
