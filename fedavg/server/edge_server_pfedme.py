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
        # edge 式7/8 的 Moreau 步长：用专门的小 moreau_lr（对齐 PFLlib），而非
        # learning_rate(0.02)。原来 eta·λ1 = 0.02×35 = 0.70 过激 → 边缘锚点 W_n 不稳；
        # 现 moreau_lr·λ1 = 0.005×15 = 0.075，与 client 层一致。
        lam1 = float(self.config["training"].get("lambda1_hier", 15.0))
        eta2 = float(self.config["training"].get("moreau_lr_pfedme", 0.005))
        eta1 = float(self.config["training"].get("eta1_hier", eta2))

        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Hier-pFedMe | Selected {len(selected)}/{len(self.clients)}")

        # 步骤 1：广播 θ_n^{t,i}（edge 模型）给 client，同时传入全局参考
        # 走 broadcast_to_clients（下行唯一通道），否则主动防御的下行载荷送不到。
        # 原先这里内联了一段「打印 client 测试集分布」的调试代码：它每个 edge round
        # 把每个 client 的测试集 unbatch 后逐样本遍历一遍，是纯开销，已删除。
        self.broadcast_to_clients(selected,
                                  global_weights=self._global_weights_ref,
                                  round_idx=global_round_idx)

        # 步骤 2：client 本地训练，收集 Θ_{n,m}^{t,i,R}
        client_updates = self._collect_updates_parallel(
            selected, global_round_idx, mode="fedavg"
        )

        # 步骤 3：edge 模型更新（式 7）
        # mean(Θ_{n,m}) − η2·λ1·(θ_n^{t,i} − W_n^{t,i})
        # mean(Θ_{n,m}) 改走 robust_mean：无防御=样本加权均值（行为不变），
        # 有防御=对 client 上传 Θ 做鲁棒聚合，再叠加 Moreau 锚点项。
        theta_n_prev = self.model.get_weights()
        mean_Theta   = self.robust_mean(client_updates, theta_n_prev)
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