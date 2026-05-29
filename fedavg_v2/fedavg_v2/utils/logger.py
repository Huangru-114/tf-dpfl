import wandb


class FLLogger:
    def __init__(self, config: dict):
        """
        初始化 wandb run。
        把整个 config 传入，自动记录所有超参数。
        """
        wandb.init(
            entity=config["wandb"]["entity"],
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config
        )

    def log_round(self, round_idx: int, metrics: dict):
        """
        每轮通信结束后调用，记录所有指标。
        用 / 分组命名，wandb 面板里自动归类到
        perf / time / comm 三个分组。
        """
        wandb.log({
            "round": round_idx,

            # 模型性能
            "perf/global_acc":       metrics["global_acc"],
            "perf/global_loss":      metrics["global_loss"],
            "perf/avg_client_loss":  metrics["avg_client_loss"],

            "perf/pm_acc":  metrics["pm_acc"] if metrics.get("pm_acc") is not None else None,
            "perf/pm_loss": metrics["pm_loss"] if metrics.get("pm_loss") is not None else None,

            # 时间成本
            "time/round_total_s":    metrics["round_time"],
            "time/client_avg_s":     metrics["avg_client_time"],

            # 通信量
            "comm/this_round_mb":    metrics["comm_mb_round"],
            "comm/cumulative_mb":    metrics["comm_mb_total"],

            # 参与情况
            "federation/n_selected": metrics["n_selected"],
        })

    def log_client(self, round_idx: int, client_id: int, loss: float):
        """记录单个客户端的本地训练 loss（可选）。"""
        wandb.log({
            f"client_{client_id}/train_loss": loss,
            "round": round_idx
        })

    def finish(self):
        """实验结束后关闭 wandb run。"""
        wandb.finish()
