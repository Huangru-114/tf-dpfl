import numpy as np
import tensorflow as tf
import time
from models.model_utils import get_model_bytes

from aggregation.fedavg import aggregate



class CloudServer:
    def __init__(self, global_model: tf.keras.Model,
                 edge_servers: list,
                 test_dataset: tf.data.Dataset,
                 config: dict):
        """
        Args:
            global_model:  全局模型，整个实验只有这一个
            edge_servers:  所有 EdgeServer 实例的列表
            test_dataset:  全局测试集
            config:        超参数字典
        """
        self.global_model = global_model
        self.edge_servers = edge_servers
        self.test_dataset = test_dataset
        self.config       = config
        self.loss_fn      = tf.keras.losses.SparseCategoricalCrossentropy()

        self.model_bytes      = get_model_bytes(global_model)
        self.total_comm_bytes = 0

        print(f"[CloudServer] "
              f"{len(edge_servers)} edge servers | "
              f"model size: {self.model_bytes / 1024 / 1024:.2f} MB")

        self.history = {
            "round":              [],
            "total_epoch":        [],
            "global_loss":        [],
            "global_acc":         [],
            "avg_edge_loss":      [],
            "round_time":         [],
            "avg_client_time":    [],
            "comm_bytes_round":   [],
            "comm_bytes_total":   [],
        }

    def select_clients(self, round_idx: int):
        """
        按采样比例随机选取参与本轮的客户端。
        client_fraction=1.0 时全员参与。

        Returns:
            selected: List[FLClient]
        """
        fraction = self.config["federation"]["client_fraction"]
        n_select = max(1, int(len(self.clients) * fraction))
        selected = np.random.choice(self.clients, n_select, replace=False).tolist()

        print(f"\n[Round {round_idx:>3}] "
              f"Selected {n_select}/{len(self.clients)} clients")
        return selected

    def broadcast_to_edges(self):
        """
        把当前全局权重广播给所有 Edge Server。
        Edge Server 再负责向下广播给自己的客户端。
        """
        global_weights = self.global_model.get_weights()
        for edge in self.edge_servers:
            edge.set_weights(global_weights)

    def collect_and_aggregate(self, round_idx: int):
        """
        让所有 Edge Server 执行边缘聚合，收集结果后 Cloud 做全局聚合。

        Returns:
            avg_edge_loss:   各 Edge 的加权平均 loss
            avg_client_time: 所有客户端的平均训练时间
            comm_client_edge: client↔edge 总通信量
        """
        edge_updates     = []
        comm_client_edge = 0

        for edge in self.edge_servers:
            weights, n_samples, avg_loss, avg_time, comm = edge.run(round_idx)
            edge_updates.append((weights, n_samples, avg_loss, avg_time))
            comm_client_edge += comm

        # Cloud 层全局聚合
        new_weights = aggregate(edge_updates)
        self.global_model.set_weights(new_weights)

        total_n        = sum(n for _, n, _, _ in edge_updates)
        avg_edge_loss  = sum(
            loss * n / total_n for _, n, loss, _ in edge_updates
        )
        avg_client_time = float(
            np.mean([t for _, _, _, t in edge_updates])
        )

        return float(avg_edge_loss), avg_client_time, comm_client_edge

    def evaluate_global(self):
        """在全局测试集上评估全局模型。"""
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in self.test_dataset:
            predictions    = self.global_model(images, training=False)
            loss           = self.loss_fn(labels, predictions)
            total_loss    += loss.numpy() * images.shape[0]
            total_correct += np.sum(
                np.argmax(predictions.numpy(), axis=1) == labels.numpy()
            )
            total_samples += images.shape[0]

        return total_loss / total_samples, total_correct / total_samples

    def run_round(self, round_idx: int):
        """
        执行完整的一轮 HierFAVG 通信：

        Phase 1（边缘聚合）：
            Cloud 广播 → Edges → Clients 训练
            → Edge 内部聚合 edge_rounds 次

        Phase 2（全局聚合）：
            Cloud 收集各 Edge 模型 → 全局 FedAvg → 评估
        """
        t_start = time.time()

        print(f"\n[Round {round_idx:>3}] "
              f"Phase 1: Cloud broadcasting to "
              f"{len(self.edge_servers)} edges...")

        # Phase 1：广播全局权重到所有 Edge
        self.broadcast_to_edges()

        # Edge 内部聚合（包含客户端训练）
        avg_edge_loss, avg_client_time, comm_client_edge = \
            self.collect_and_aggregate(round_idx)

        # Phase 2：全局评估
        print(f"[Round {round_idx:>3}] Phase 2: Cloud aggregating...")
        global_loss, global_acc = self.evaluate_global()

        # 通信量：
        #   cloud↔edge：双向 × edge 数量 × 模型大小
        #   client↔edge：由 EdgeServer 统计后传回
        comm_cloud_edge   = 2 * self.model_bytes * len(self.edge_servers)
        comm_this_round   = comm_cloud_edge + comm_client_edge
        self.total_comm_bytes += comm_this_round
        round_time = time.time() - t_start

        metrics = {
            "round":            round_idx,
            "edge_rounds":     self.config["federation"]["edge_rounds"],
            "local_epochs":    self.config["training"]["local_epochs"],
            "global_loss":      global_loss,
            "global_acc":       global_acc,
            "avg_edge_loss":    avg_edge_loss,
            "round_time":       round_time,
            "avg_client_time":  avg_client_time,
            "comm_bytes_round": comm_this_round,
            "comm_bytes_total": self.total_comm_bytes,
            "comm_mb_round":    comm_this_round / 1024 / 1024,
            "comm_mb_total":    self.total_comm_bytes / 1024 / 1024,
            "comm_mb_cloud_edge":   comm_cloud_edge / 1024 / 1024,
            "comm_mb_client_edge":  comm_client_edge / 1024 / 1024,
        }

        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)

        print(
            f"  [Cloud] acc={global_acc:.4f} | "
            f"loss={global_loss:.4f} | "
            f"time={round_time:.1f}s | "
            f"comm={comm_this_round / 1024 / 1024:.1f}MB "
            f"(total={self.total_comm_bytes / 1024 / 1024:.0f}MB)"
        )

        return metrics

    def run(self, logger=None):
        """执行完整实验，循环 n_rounds 轮全局通信。"""
        n_rounds = self.config["federation"]["n_rounds"]

        for r in range(1, n_rounds + 1):
            metrics = self.run_round(r)
            if logger is not None:
                logger.log_round(r, metrics)

        print("\n[Cloud] Training complete.")
        return self.history