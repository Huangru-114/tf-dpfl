"""
edges/pfedme.py  –  Hier-pFedMe 边缘服务器

算法：Liu et al., ICME 2025（三层嵌套优化的 edge 层部分）

变量角色对照：
  self.model  θ_n^{t,i}   edge 模型，Algorithm 1 式 (7) 更新
  self.W_n    W_n^{t,i}   edge 保存的全局模型副本，Algorithm 1 式 (8) 更新

每个 global round 开始时，set_global_ref() 将 W_n 重置为 W^t。
每个 edge round 结束后 W_n 随 θ_n 更新，最终上传 W_n^{t,I} 给 Cloud。
"""

import numpy as np
import tensorflow as tf

from .edge_server_base import EdgeServerBase


class PFedMeEdgeServer(EdgeServerBase):

    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        super().__init__(edge_id, clients, model, config)

        # ── Hier-pFedMe 专属：W_n^{t,i}，Algorithm 1 式 (8) ─────────────
        # 每个 global round 开始时由 set_global_ref() 重置为 W^t。
        self.W_n: list | None = None

    # ══════════════════════════════════════════════════════════════════════
    # 权重管理（override：W_n 随 global round 重置）
    # ══════════════════════════════════════════════════════════════════════

    def set_global_ref(self, global_weights: list):
        """
        存储 Cloud 全局权重快照，并将 W_n^{t,0} 重置为 W^t。

        对应 Algorithm 1 第 4 行，每个 global round 执行一次。
        """
        super().set_global_ref(global_weights)
        self.W_n = [w.copy() for w in global_weights]
        print(f"  [Edge {self.edge_id}] W_n reset to global weights for new round.")

    # ══════════════════════════════════════════════════════════════════════
    # 边缘轮次
    # ══════════════════════════════════════════════════════════════════════

    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """
        Hier-pFedMe 单轮 edge 内部通信（Algorithm 1 第 5–18 行）：

        1. 广播 θ_n^{t,i}（edge 模型）给选中 client
        2. client 执行内层 + Moreau 优化，上传 Θ_{n,m}^{t,i,R}
        3. edge 模型更新（式 7）：
               θ_n^{t,i+1} = mean(Θ_{n,m}) − η2·λ1·(θ_n^{t,i} − W_n^{t,i})
        4. W_n 更新（式 8）：
               W_n^{t,i+1} = W_n^{t,i} − η1·λ1·(W_n^{t,i} − θ_n^{t,i+1})
        """
        lam1 = float(self.config["training"].get("lambda1_hier", 35.0))
        eta2 = float(self.config["training"]["learning_rate"])
        eta1 = float(self.config["training"].get("eta1_hier", eta2))

        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Hier-pFedMe | Selected {len(selected)}/{len(self.clients)}")

        # 步骤 1：广播 θ_n^{t,i}（edge 模型）给 client，同时传入全局参考
        edge_w = self.model.get_weights()
        for client in selected:
            client.set_weights(global_weights=self._global_weights_ref,
                               edge_weights=edge_w)
            # ── 调试：打印 client 测试集分布 ────────────────────────────
            ds = client.get_test_dataset()
            if ds is not None:
                all_labels, total = [], 0
                for x, y in ds.unbatch().batch(1):
                    all_labels.append(y.numpy()[0])
                    total += 1
                unique, counts = np.unique(all_labels, return_counts=True)
                print(f"[C{client.client_id}] Test set: "
                      f"samples={total}, "
                      f"class_dist={dict(zip(unique, counts))}")

        # 步骤 2：client 本地训练，收集 Θ_{n,m}^{t,i,R}
        client_updates = self._collect_updates_parallel(
            selected, global_round_idx, mode="fedavg"
        )

        # 步骤 3：edge 模型更新（式 7）
        # mean(Θ_{n,m}) − η2·λ1·(θ_n^{t,i} − W_n^{t,i})
        n_layers    = len(client_updates[0][0])
        # mean_Theta  = [
        #     np.mean([upd[0][j] for upd in client_updates], axis=0)
        #     for j in range(n_layers)
        # ]
        total_n = sum(n for _, n, *_ in client_updates)
        mean_Theta = [
            sum(upd[0][j] * (upd[1] / total_n) for upd in client_updates)
            for j in range(n_layers)
        ]
        
        theta_n_prev = self.model.get_weights()
        theta_n_new  = [
            mc - eta2 * lam1 * (tn - wn)
            for mc, tn, wn in zip(mean_Theta, theta_n_prev, self.W_n)
        ]
        self.model.set_weights(theta_n_new)

        # 步骤 4：W_n 更新（式 8）
        self.W_n = [
            wn - eta1 * lam1 * (wn - tn)
            for wn, tn in zip(self.W_n, theta_n_new)
        ]

        total_n  = sum(n for _, n, *_ in client_updates)
        avg_loss = sum(l * n / total_n for _, n, l, _ in client_updates)
        avg_time = float(np.mean([t for *_, t in client_updates]))

        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"loss={avg_loss:.4f}")
        return float(avg_loss), avg_time

    # ══════════════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════════════

    def run(self, global_round_idx: int):
        """
        执行 edge_rounds 轮 Hier-pFedMe 聚合，上传 W_n^{t,I} 给 Cloud。

        注意：上传的是 W_n（Algorithm 1 第 19 行），不是 θ_n（edge 模型）。
        """
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []

        for er in range(1, edge_rounds + 1):
            loss, t = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(t)

        comm     = 2 * self.model_bytes * len(self.clients) * edge_rounds
        upload_w = self.W_n 
        return (upload_w, self.n_samples,
                float(np.mean(round_losses)), float(np.mean(round_times)), comm)