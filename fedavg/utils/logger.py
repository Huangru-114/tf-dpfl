import wandb


class FLLogger:
    def __init__(self, config: dict):
        """
        初始化 wandb run。
        把整个 config 传入 wandb，自动记录所有超参数，
        方便在 wandb UI 里对比不同实验的配置。
        """
        wandb.init(
            entity=config["wandb"]["entity"],    
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config
        )

    def log_round(self, round_idx: int, metrics: dict):
        log_dict = {
            "total_epoch":          round_idx * metrics["edge_rounds"] * metrics["local_epochs"],
            "perf/gm_acc":          metrics["global_acc"],
            "perf/gm_loss":         metrics["global_loss"],
            "perf/avg_edge_loss":   metrics["avg_edge_loss"],
            "time/round_total_s":   metrics["round_time"],
            "time/client_avg_s":    metrics["avg_client_time"],
            "comm/total_mb":        metrics["comm_mb_total"],
            "comm/this_round_mb":   metrics["comm_mb_round"],
            "comm/cloud_edge_mb":   metrics["comm_mb_cloud_edge"],
            "comm/client_edge_mb":  metrics["comm_mb_client_edge"],
        }
        # em/pm 仅在 eval_interval 轮次存在，None 时跳过避免 wandb 断点
        if metrics.get("em_acc") is not None:
            log_dict["perf/em_acc"]  = metrics["em_acc"]
            log_dict["perf/em_loss"] = metrics["em_loss"]
            log_dict["perf/pm_acc"]  = metrics["pm_acc"]
            log_dict["perf/pm_loss"] = metrics["pm_loss"]
        wandb.log(log_dict, step=round_idx)
    def log_client(self, round_idx: int, client_id: int, loss: float):
        """记录单个客户端的本地训练 loss（可选，数据量大时可以关掉）。"""
        wandb.log({f"client_{client_id}/train_loss": loss}, step=round_idx)

    def finish(self):
        """实验结束后关闭 wandb run。"""
        wandb.finish()

class CentralizedLogger:
    """集中式训练的 wandb logger，和 FLLogger 用同一个 project，方便曲线叠加对比。"""

    def __init__(self, config: dict):
        wandb.init(
            entity=config["wandb"]["entity"],
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config
        )

    def log_epoch(self, epoch: int, metrics: dict):
        wandb.log({
            "epoch":        epoch,
            "total_epochs": metrics["total_epochs"],

            # 用和 FL 相同的命名空间，这样在 wandb 里可以直接叠加
            "perf/global_acc":  metrics["test_acc"],
            "perf/global_loss": metrics["test_loss"],
            "perf/train_loss":  metrics["train_loss"],
            "time/epoch_s":     metrics["epoch_time"],
        })

    def finish(self):
        wandb.finish()