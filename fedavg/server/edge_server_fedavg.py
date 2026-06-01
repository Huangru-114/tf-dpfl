"""
server/edge_server_fedavg.py  –  Hier-FedAvg 边缘服务器（三层 FedAvg 的 edge 层）

最朴素的三层 FedAvg：
  Cloud:  全局模型      标准 FedAvg 聚合各 edge 上传的模型（server.py 不改）
  Edge:   edge 模型     样本加权 FedAvg 聚合 client 模型
  Client: 本地 SGD（client_fedavg.py）

复用 aggregation/fedavg.py 的 aggregate（与 cloud 同一套加权平均）。
"""

import numpy as np
import tensorflow as tf

from .edge_server_base import EdgeServerBase
from aggregation.fedavg import aggregate


class FedAvgEdgeServer(EdgeServerBase):

    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """单轮 edge 内部通信：广播 edge 模型 → client 本地训练 → 样本加权 FedAvg。"""
        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Hier-FedAvg | Selected {len(selected)}/{len(self.clients)}")

        # 广播 edge 模型（同时转发全局参考）
        self.broadcast_to_clients(selected, global_weights=self._global_weights_ref)

        # 收集 client 本地训练结果
        client_updates = self._collect_updates_parallel(
            selected, global_round_idx, mode="fedavg")

        # 样本加权 FedAvg 聚合
        new_w = aggregate(client_updates)
        self.model.set_weights(new_w)

        total_n  = sum(n for _, n, *_ in client_updates)
        avg_loss = sum(l * n / total_n for _, n, l, _ in client_updates)
        avg_time = float(np.mean([t for *_, t in client_updates]))
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"loss={avg_loss:.4f}")
        return float(avg_loss), avg_time

    def run(self, global_round_idx: int):
        """执行 edge_rounds 轮 FedAvg 聚合，上传 edge 模型给 cloud。"""
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []

        for er in range(1, edge_rounds + 1):
            loss, t = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(t)

        comm = 2 * self.model_bytes * len(self.clients) * edge_rounds
        return (self.model.get_weights(), self.n_samples,
                float(np.mean(round_losses)), float(np.mean(round_times)), comm)
