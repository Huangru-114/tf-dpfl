import time
import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering

from aggregation.fedavg import aggregate
from models.model_utils import clone_model, get_model_bytes


class EdgeServer:
    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        """
        Args:
            edge_id: Edge Server 编号
            clients: 该 Edge 管辖的客户端列表（FLClient 实例）
            model:   Edge 本地模型，从全局模型 clone 而来
            config:  超参数字典
        """
        self.edge_id  = edge_id
        self.clients  = clients
        self.model    = model
        self.config   = config

        # 统计管辖的总样本数，供 Cloud 聚合加权用
        self.n_samples = sum(c.n_samples for c in clients)

        # ── 进阶版聚类需要的状态 ──────────────────────────
        # warm-up 结束后存储每个客户端的梯度更新，用于聚类
        self._client_gradient_cache = {}   # client_id → flat gradient
        self._cluster_assignments   = {}   # client_id → cluster_id
        self._warmup_done           = False


        self.model_bytes = get_model_bytes(model)

        print(f"[EdgeServer {edge_id}] "
              f"{len(clients)} clients | "
              f"{self.n_samples} total samples")

    def set_weights(self, global_weights: list):
        """
        接收 Cloud 广播的全局权重，覆盖本地 edge 模型。
        """
        self.model.set_weights(global_weights)

    def select_clients(self, round_idx: int):
        """
        每轮从本 Edge 管辖的客户端里按比例随机选取参与者。

        注意：客户端归属哪个 Edge 已经在 build_edge_servers() 里
        由分配策略决定了，这里只控制每轮的参与比例。
        """
        fraction = self.config["federation"]["client_fraction"]
        n_select = max(1, int(len(self.clients) * fraction))
        selected = np.random.choice(
            self.clients, n_select, replace=False
        ).tolist()
        return selected
   
    # def broadcast_to_clients(self):
    #     """
    #     把当前 edge 模型权重广播给所有管辖客户端。
    #     注意：广播的是 edge 模型，不是全局模型。
    #     在边缘内部多轮聚合时，客户端每轮都从最新的 edge 模型出发。
    #     """
    #     edge_weights = self.model.get_weights()
    #     for client in self.clients:
    #         client.set_weights(edge_weights)

    def broadcast_to_clients(self, selected_clients,
                            global_weights=None):
        """
        广播 edge 权重给选中的客户端。
        同时传入 global_weights，供 drift correction 用作远端参考。

        global_weights 由 CloudServer 在每轮开始时传入 EdgeServer，
        EdgeServer 保存一份后转发给客户端。
        """
        edge_weights = self.model.get_weights()
        for client in selected_clients:
            client.set_weights(
                global_weights=global_weights if global_weights else edge_weights,
                edge_weights=edge_weights
            )

    def set_global_ref(self, global_weights: list):
        """CloudServer 在 broadcast 阶段调用，保存全局权重供转发。"""
        self._global_weights_ref = [w.copy() for w in global_weights]

    def run_edge_round(self, global_round_idx, edge_round_idx):
        selected = self.select_clients(global_round_idx)

        print(f"  [Edge {self.edge_id}] "
              f"G-Round {global_round_idx} | "
              f"E-Round {edge_round_idx} | "
              f"Selected {len(selected)}/{len(self.clients)} clients")

        # 广播 edge 权重，同时传入全局参考权重
        global_ref = getattr(self, '_global_weights_ref', None)
        self.broadcast_to_clients(selected, global_weights=global_ref)

        client_updates = []
        for client in selected:
            weights, n, loss, t = client.local_train(global_round_idx)
            client_updates.append((weights, n, loss, t))

            # warm-up 阶段记录梯度
            warmup_rounds = self.config["federation"].get("warmup_rounds", 5)
            if (self.config["federation"].get("client_selection") == "clustered"
                    and global_round_idx <= warmup_rounds):
                self._collect_gradient_update(
                    client,
                    self.model.get_weights()
                )

        new_weights = aggregate(client_updates)
        self.model.set_weights(new_weights)

        total_n  = sum(n for _, n, _, _ in client_updates)
        avg_loss = sum(
            l * n / total_n for _, n, l, _ in client_updates
        )
        avg_time = float(np.mean([t for _, _, _, t in client_updates]))

        print(f"  [Edge {self.edge_id}] G-Round {global_round_idx} | "
            f"E-Round {edge_round_idx} | loss={avg_loss:.4f}")
        return float(avg_loss), avg_time
    # def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
    #     """
    #     执行一轮边缘内部聚合：
    #         broadcast → 客户端本地训练 → FedAvg 聚合 → 更新 edge 模型

    #     Args:
    #         global_round_idx: 当前全局通信轮次（用于日志）
    #         edge_round_idx:   当前边缘内部轮次（用于日志）

    #     Returns:
    #         avg_loss:   本轮边缘平均 loss
    #         avg_time:   客户端平均训练时间
    #     """
    #     # 广播当前 edge 模型给所有客户端
    #     self.broadcast_to_clients()

    #     # 所有管辖客户端训练（边缘层不做采样，全员参与）
    #     client_updates = []
    #     for client in self.clients:
    #         weights, n_samples, loss, train_time = client.local_train(
    #             round_idx=global_round_idx
    #         )
    #         client_updates.append((weights, n_samples, loss, train_time))

    #     # FedAvg 聚合，更新 edge 模型
    #     new_weights = aggregate(client_updates)
    #     self.model.set_weights(new_weights)

    #     total_n   = sum(n for _, n, _, _ in client_updates)
    #     avg_loss  = sum(loss * n / total_n for _, n, loss, _ in client_updates)
    #     avg_time  = float(np.mean([t for _, _, _, t in client_updates]))

    #     print(f"  [Edge {self.edge_id}] "
    #           f"G-Round {global_round_idx} | "
    #           f"E-Round {edge_round_idx} | "
    #           f"loss={avg_loss:.4f}")

    #     return float(avg_loss), avg_time

    def run(self, global_round_idx: int):
        """
        执行完整的边缘聚合流程：
        连续跑 edge_rounds 轮内部聚合，然后把结果上传给 Cloud。

        Args:
            global_round_idx: 当前全局通信轮次

        Returns:
            weights:    edge 聚合后的模型权重，上传给 Cloud
            n_samples:  管辖总样本数，供 Cloud 加权聚合
            avg_loss:   edge_rounds 轮的平均 loss
            avg_time:   客户端平均训练时间
            comm_bytes: 本次边缘通信量（client↔edge）
        """
        edge_rounds = self.config["federation"]["edge_rounds"]
        t_start     = time.time()
        round_losses = []
        round_times  = []

        for er in range(1, edge_rounds + 1):
            loss, avg_time = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(avg_time)

        # 通信量：client↔edge 的双向传输
        # edge_rounds 轮 × 每轮双向 × 每个客户端一份模型
        comm_bytes = (
            2 * self.model_bytes * len(self.clients) * edge_rounds
        )

        return (
            self.model.get_weights(),
            self.n_samples,
            float(np.mean(round_losses)),
            float(np.mean(round_times)),
            comm_bytes
        )