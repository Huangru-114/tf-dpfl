import time
import numpy as np
import tensorflow as tf

from aggregation.fedavg     import aggregate
from models.model_utils     import get_model_bytes


class FLServer:
    def __init__(self, global_model: tf.keras.Model, clients: list,
                 test_dataset: tf.data.Dataset, config: dict):
        """
        Args:
            global_model: 全局模型，整个实验只有这一个
            clients:      所有 FLClient 实例的列表
            test_dataset: 全局测试集，用于每轮评估
            config:       超参数字典
        """
        self.global_model     = global_model
        self.clients          = clients
        self.test_dataset     = test_dataset
        self.config           = config
        self.loss_fn          = tf.keras.losses.SparseCategoricalCrossentropy()

        # 通信量统计
        self.model_bytes      = get_model_bytes(global_model)
        self.total_comm_bytes = 0
        print(f"[Setup] Model size: {self.model_bytes / 1024 / 1024:.2f} MB "
              f"per transmission")

        self.history = {
            "round":            [],
            "global_loss":      [],
            "global_acc":       [],
            "avg_client_loss":  [],
            "round_time":       [],
            "avg_client_time":  [],
            "comm_bytes_round": [],
            "comm_bytes_total": [],
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

    def broadcast(self, selected_clients: list):
        """
        把当前全局权重广播给选中的客户端。
        未被选中的客户端本轮不参与。
        """
        global_weights = self.global_model.get_weights()
        for client in selected_clients:
            client.set_weights(global_weights)

    def collect_and_aggregate(self, selected_clients: list, round_idx: int):
        """
        让选中的客户端各自做本地训练，收集结果后 FedAvg 聚合。

        Returns:
            avg_client_loss: 加权平均 client loss
            avg_client_time: 平均训练耗时
        """
        client_updates = []
        for client in selected_clients:
            weights, n_samples, loss, train_time = client.local_train(round_idx)
            client_updates.append((weights, n_samples, loss, train_time))

        new_weights = aggregate(client_updates)
        self.global_model.set_weights(new_weights)

        total_n          = sum(n for _, n, _, _ in client_updates)
        avg_client_loss  = sum(
            loss * n / total_n for _, n, loss, _ in client_updates
        )
        avg_client_time  = float(
            np.mean([t for _, _, _, t in client_updates])
        )

        return float(avg_client_loss), avg_client_time

    def evaluate_global(self):
        """
        在全局测试集上评估当前全局模型。

        Returns:
            loss (float), accuracy (float)
        """
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
        执行完整的一轮联邦学习通信：
            select → broadcast → local_train → aggregate → evaluate

        Returns:
            metrics: dict，包含本轮所有指标
        """
        t_round_start = time.time()

        selected   = self.select_clients(round_idx)
        n_selected = len(selected)

        self.broadcast(selected)

        avg_client_loss, avg_client_time = self.collect_and_aggregate(
            selected, round_idx
        )

        global_loss, global_acc = self.evaluate_global()

        avg_pm_acc = avg_pm_loss = None
        pm_losses, pm_accs, pm_ns = [], [], []
        for client in self.clients:
            c_loss, c_acc = client.evaluate_on(
                fallback_dataset=self.test_dataset
            )
            pm_losses.append(c_loss)
            pm_accs.append(c_acc)
            pm_ns.append(client.n_samples)
        total_pm_n  = sum(pm_ns)
        avg_pm_acc  = float(sum(a * n / total_pm_n for a, n in zip(pm_accs, pm_ns)))
        avg_pm_loss = float(sum(l * n / total_pm_n for l, n in zip(pm_losses, pm_ns)))


        # 通信量：下行（server→client）+ 上行（client→server）
        comm_this_round        = 2 * self.model_bytes * n_selected
        self.total_comm_bytes += comm_this_round
        round_time             = time.time() - t_round_start

        metrics = {
            "round":            round_idx,
            "global_loss":      global_loss,
            "global_acc":       global_acc,
            "avg_client_loss":  avg_client_loss,
            "round_time":       round_time,
            "avg_client_time":  avg_client_time,
            "comm_bytes_round": comm_this_round,
            "comm_bytes_total": self.total_comm_bytes,
            "comm_mb_round":    comm_this_round / 1024 / 1024,
            "comm_mb_total":    self.total_comm_bytes / 1024 / 1024,
            "n_selected":       n_selected,
            "pm_acc":  avg_pm_acc,
            "pm_loss": avg_pm_loss,
        }

        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)

        print(
            f"  [Server] acc={global_acc:.4f} | loss={global_loss:.4f} | "
            f" PM={avg_pm_acc:.4f}" if avg_pm_acc is not None else ""
            f"time={round_time:.1f}s | "
            f"comm={comm_this_round / 1024 / 1024:.1f}MB "
            f"(total={self.total_comm_bytes / 1024 / 1024:.0f}MB)"
        )

        return metrics

    def run(self, logger=None):
        """
        执行完整实验，循环 n_rounds 轮通信。

        Args:
            logger: FLLogger 实例，为 None 时跳过 wandb logging

        Returns:
            history: dict，包含所有轮次的指标
        """
        n_rounds = self.config["federation"]["n_rounds"]

        for r in range(1, n_rounds + 1):
            metrics = self.run_round(r)
            if logger is not None:
                logger.log_round(r, metrics)

        print("\n[Server] Training complete.")
        return self.history
