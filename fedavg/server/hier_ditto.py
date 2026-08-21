"""
server/hier_ditto.py  –  Hier-Ditto 边缘服务器（整模型，无 backbone/head 拆分）

参考 edge_server_pfedme.py 的结构（run_edge_round 收集 client 上传、样本加权
聚合、更新 edge 模型，run 循环 edge_rounds 后上传）。区别在 edge 更新机制：
pFedMe 用移动锚点 W_n（式 7/8），本类用 Ditto 固定锚点 φ*（cloud 下发的全局
模型，一个 global round 内固定）。

三层正则锚点：
  Cloud:  全局模型 φ*       标准 FedAvg 聚合各 edge 上传的模型（server.py 不改）
  Edge:   edge 模型 φ_e     聚合 client 共享模型 w_k + Ditto 正则靠近 φ*（mu_edge）
  Client: 个性化模型 v_k + 共享模型 w_k（上传）

edge 更新：φ_e ← mean(w_k) − mu_edge·(φ_e_prev − φ*)
"""

import numpy as np
import tensorflow as tf

from .edge_server_base import EdgeServerBase


class HierDittoEdgeServer(EdgeServerBase):

    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """单轮 edge 内部通信：FedAvg 聚合 client 共享模型 + Ditto 正则靠近 φ*。"""
        mu_edge = float(self.config["training"].get("mu_edge", 0.1))

        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Hier-Ditto | Selected {len(selected)}/{len(self.clients)}")

        # 步骤 1：广播 edge 模型 + 全局参考（走下行唯一通道 broadcast_to_clients）
        self.broadcast_to_clients(selected,
                                  global_weights=self._global_weights_ref,
                                  round_idx=global_round_idx)

        # 步骤 2：收集 client 上传的共享模型 w_k
        client_updates = self._collect_updates_parallel(
            selected, global_round_idx, mode="fedavg")

        # 步骤 3：样本加权聚合（有防御走鲁棒聚合）+ Ditto 近端靠近 φ*
        total_n    = sum(n for _, n, *_ in client_updates)
        theta_prev = self.model.get_weights()
        mean_w     = self.robust_mean(client_updates, theta_prev)
        gref       = self._global_weights_ref  # φ*（global round 内固定）
        new_w = [
            mw - mu_edge * (tp - gr)
            for mw, tp, gr in zip(mean_w, theta_prev, gref)
        ]
        self.model.set_weights(new_w)

        avg_loss = sum(l * n / total_n for _, n, l, _ in client_updates)
        avg_time = float(np.mean([t for *_, t in client_updates]))
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"loss={avg_loss:.4f}")
        return float(avg_loss), avg_time

    def run(self, global_round_idx: int):
        """执行 edge_rounds 轮聚合，上传 edge 模型 φ_e 给 cloud。"""
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []

        for er in range(1, edge_rounds + 1):
            loss, t = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(t)

        comm = 2 * self.model_bytes * len(self.clients) * edge_rounds
        return (self.model.get_weights(), self.n_samples,
                float(np.mean(round_losses)), float(np.mean(round_times)), comm)
