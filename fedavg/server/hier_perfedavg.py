"""
server/hier_perfedavg.py  –  Hier-PerFedAvg 边缘服务器

三层 HFL 下的 Per-FedAvg edge 层（参考 edge_server.py 的
run_edge_round_hier_perfedavg）：
  1. 广播当前 edge 元参数 θ_E 给选中 client
  2. 每个 client 计算一阶 MAML 元梯度 g_i
  3. NaN/inf 过滤（Dirichlet α 较小时数值易崩）
  4. 样本加权聚合元梯度
  5. edge 元参数更新：θ_E ← θ_E − β·Σ(n_i/N · g_i)
edge_rounds 轮后上传 edge 元参数给 cloud，cloud 标准 FedAvg。
"""

import numpy as np
import tensorflow as tf

from .edge_server_base import EdgeServerBase


class HierPerFedAvgEdgeServer(EdgeServerBase):

    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """单轮边缘元聚合。"""
        beta = self.config["training"].get("beta_outer", 0.01)

        selected     = self.select_clients(global_round_idx)
        edge_weights = self.model.get_weights()
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Hier-PerFedAvg | Selected {len(selected)}/{len(self.clients)}")

        # 广播 edge 元参数 + 全局参考
        self.broadcast_to_clients(selected, global_weights=self._global_weights_ref)

        # 收集元梯度（含 NaN/inf 过滤）
        meta_grads_collected, n_total, losses = [], 0, []
        for client in selected:
            grads, n, loss = client.compute_meta_gradient(edge_weights, global_round_idx)
            if any(np.isnan(g).any() or np.isinf(g).any() for g in grads):
                print(f"    [SKIP] Client {client.client_id}: NaN/inf in meta-gradient")
                continue
            meta_grads_collected.append((grads, n))
            n_total += n
            losses.append(loss)

        if not meta_grads_collected:
            print(f"  [Edge {self.edge_id}] WARNING: all clients skipped, "
                  f"keeping current edge weights")
            return float("nan"), 0.0

        # 样本加权平均元梯度
        n_params  = len(edge_weights)
        agg_grads = [
            sum((n / n_total) * grads[i] for grads, n in meta_grads_collected)
            for i in range(n_params)
        ]

        # edge 元参数更新
        new_edge_w = [w - beta * g for w, g in zip(edge_weights, agg_grads)]
        self.model.set_weights(new_edge_w)

        avg_loss = float(np.mean(losses))
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"loss={avg_loss:.4f}")
        return avg_loss, 0.0

    def run(self, global_round_idx: int):
        """执行 edge_rounds 轮元聚合，上传 edge 元参数给 cloud。"""
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []

        for er in range(1, edge_rounds + 1):
            loss, t = self.run_edge_round(global_round_idx, er)
            if not np.isnan(loss):
                round_losses.append(loss)
            round_times.append(t)

        comm     = 2 * self.model_bytes * len(self.clients) * edge_rounds
        avg_loss = float(np.mean(round_losses)) if round_losses else float("nan")
        return (self.model.get_weights(), self.n_samples,
                avg_loss, float(np.mean(round_times)), comm)
