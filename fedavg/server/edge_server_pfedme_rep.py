"""
servers/hier_pfedme_rep.py – PFedMeRep 边缘服务器（与 DittoRep 完全相同的聚合逻辑）
"""

import numpy as np
import tensorflow as tf

from .edge_server_base import EdgeServerBase


class HierPFedMeRepEdgeServer(EdgeServerBase):

    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        super().__init__(edge_id, clients, model, config)
        total_len = len(model.get_weights())
        self._backbone_indices = list(range(total_len - 2))
        self._head_indices = [total_len - 2, total_len - 1]

    def broadcast_to_clients(self, selected: list, global_weights: list = None):
        edge_full = self.model.get_weights()
        for client in selected:
            client.set_weights(global_weights=global_weights, edge_weights=edge_full)

    def run(self, global_round_idx: int):
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []
        total_comm = 0

        for er in range(1, edge_rounds + 1):
            selected = self.select_clients(global_round_idx)
            print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{er} | PFedMeRep | Selected {len(selected)}/{len(self.clients)}")

            self.broadcast_to_clients(selected, global_weights=self.model.get_weights())
            client_results = self._collect_updates_parallel(selected, global_round_idx, mode="fedavg")

            if not client_results:
                continue

            total_n = sum(n for _, n, _, _ in client_results)
            avg_loss = sum(l * n / total_n for _, n, l, _ in client_results)
            avg_time = np.mean([t for *_, t in client_results])

            # backbone 索引聚合（有防御走鲁棒聚合）；无防御时与手写样本加权均值一致。
            current_full = self.model.get_weights()
            robust_full  = self.robust_mean(client_results, current_full)
            for idx in self._backbone_indices:
                current_full[idx] = robust_full[idx]
            self.model.set_weights(current_full)

            round_losses.append(avg_loss)
            round_times.append(avg_time)
            total_comm += self.model_bytes * len(selected) * 2

        final_backbone = [self.model.get_weights()[i] for i in self._backbone_indices]
        return (final_backbone, self.n_samples,
                float(np.mean(round_losses)) if round_losses else 0.0,
                float(np.mean(round_times)) if round_times else 0.0,
                total_comm)