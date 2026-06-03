"""
server/backdoor_server.py  –  后门感知的 Cloud 服务器

继承 CloudServer，不改其核心训练/聚合逻辑；只在每 bd_eval_interval 轮（及最后一轮）
额外计算并记录**分层后门指标**（任务2/3）：
  - global / edge / local 三层 ASR 与 ACC
  - local 模型 same_edge vs diff_edge ASR（判断后门是否溢出到其他边缘节点）
  - 特征空间分离度（global + 抽样 local）
  - 后门遗忘曲线（仅个性化/微调方法、仅评估轮、抽样 1 个良性客户端）
所有指标通过 FLLogger.log_round_metrics 记入 wandb。
"""

import numpy as np

from server.server import CloudServer
from attack.backdoor_eval import (evaluate_hierarchical_asr,
                                   evaluate_forgetting_curve,
                                   evaluate_feature_separation)
from utils.logger import FLLogger


# 个性化 / 微调方法：遗忘曲线对这些方法有意义（client.model 训练后停在个性化模型）
_PERSONALIZED_METHODS = {
    "hierpfedme", "pfedme", "hier_ditto", "hier_ditto_rep", "hier_pfedme_rep",
    "hier_perfedavg", "hier_fedrep",
}


class BackdoorCloudServer(CloudServer):

    def __init__(self, *args, bd_cfg=None, x_test=None, y_test=None,
                 trigger_fn=None, malicious_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bd_cfg = bd_cfg or {}
        self.x_test = x_test
        self.y_test = np.asarray(y_test).reshape(-1) if y_test is not None else None
        self.trigger_fn = trigger_fn
        self.malicious_ids = set(int(i) for i in (malicious_ids or set()))
        self.bd_eval_interval = int(self.bd_cfg.get("eval_interval", 50))
        self.bd_asr_max = int(self.bd_cfg.get("asr_max_samples", 0))  # 0 = 用全部非目标类
        self.bd_target = int(self.bd_cfg.get("target_label", 9))
        self.bd_verbose = bool(self.bd_cfg.get("verbose_clients", True))
        self._all_clients = [c for e in self.edge_servers for c in e.clients]
        self.history.setdefault("bd_c_acc", [])
        self.history.setdefault("bd_asr", [])

        # 特征分离度 / 遗忘曲线相关配置
        self.feature_eval = bool(self.bd_cfg.get("feature_eval", True))
        self.feature_eval_samples = int(self.bd_cfg.get("feature_eval_samples", 200))
        self.feature_clean_per_class = int(self.bd_cfg.get("feature_clean_per_class", 50))
        self.forgetting_epochs = int(self.bd_cfg.get(
            "forgetting_epochs", self.config["training"].get("local_epochs", 5)))
        self._feat_data = None       # 懒惰构建并缓存

    def run_round(self, round_idx: int):
        metrics = super().run_round(round_idx)
        n_rounds = int(self.config["federation"]["n_rounds"])
        if (round_idx % self.bd_eval_interval == 0) or (round_idx == n_rounds):
            self._backdoor_eval(round_idx)
        return metrics

    # ── 特征评估用数据：固定一次，避免每轮重建 ──────────────────────────────
    def _prepare_feature_data(self):
        if self._feat_data is not None:
            return self._feat_data
        x, y = self.x_test, self.y_test
        # 干净样本按类抽样
        clean_by_class = {}
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) > self.feature_clean_per_class:
                idx = idx[:self.feature_clean_per_class]
            clean_by_class[int(c)] = x[idx]
        # 后门样本：非目标类抽样后加 trigger，保留真实标签
        non_target = np.where(y != self.bd_target)[0]
        if len(non_target) > self.feature_eval_samples:
            non_target = non_target[:self.feature_eval_samples]
        poisoned = self.trigger_fn(x[non_target])
        poisoned_true = y[non_target]
        self._feat_data = (poisoned, poisoned_true, clean_by_class)
        return self._feat_data

    def _backdoor_eval(self, round_idx: int):
        xt, yt = self.x_test, self.y_test
        if self.bd_asr_max and yt is not None and len(yt) > self.bd_asr_max:
            idx = np.random.choice(len(yt), self.bd_asr_max, replace=False)
            xt, yt = xt[idx], yt[idx]

        print(f"\n[Backdoor] ===== Round {round_idx} hierarchical backdoor eval =====")

        # ── 任务2：分层 ASR（global/edge/local 三层 + same/diff edge） ────────
        metrics = evaluate_hierarchical_asr(
            self.global_model, self.edge_servers, self._all_clients,
            xt, yt, self.trigger_fn, self.bd_target,
            self.malicious_ids, fallback_test_ds=self.test_dataset,
        )

        # ── 任务2：特征空间分离度（global + 抽样 1 个良性 local） ─────────────
        if self.feature_eval:
            poisoned, poisoned_true, clean_by_class = self._prepare_feature_data()
            sep_g = evaluate_feature_separation(
                self.global_model, poisoned, poisoned_true,
                clean_by_class, self.bd_target)
            metrics["sep_score_global"] = sep_g["separation_score"]

            benign = [c for c in self._all_clients
                      if int(c.client_id) not in self.malicious_ids]
            if benign:
                sep_l = evaluate_feature_separation(
                    benign[0].model, poisoned, poisoned_true,
                    clean_by_class, self.bd_target)
                metrics["sep_score_local"] = sep_l["separation_score"]

        # ── 任务2：后门遗忘曲线（仅个性化/微调方法、抽样 1 个良性客户端） ─────
        method = self.config["training"].get("drift_correction", "fedavg")
        do_forget = (method in _PERSONALIZED_METHODS
                     or self.config.get("evaluation", {}).get("final_finetune", False))
        if do_forget:
            benign = [c for c in self._all_clients
                      if int(c.client_id) not in self.malicious_ids]
            if benign:
                c0 = benign[0]
                lr = float(self.config["training"].get("learning_rate", 0.02))
                metrics["forgetting_curve"] = evaluate_forgetting_curve(
                    c0.model, self.forgetting_epochs, c0.dataset,
                    xt, yt, self.trigger_fn, self.bd_target, lr=lr)

        # ── 历史 + 控制台 ─────────────────────────────────────────────────
        self.history["bd_c_acc"].append(metrics["local_acc_mean"])
        self.history["bd_asr"].append(metrics["local_asr_benign_mean"])
        print(f"[Backdoor] Round {round_idx} | "
              f"GM_ASR={metrics['global_asr']:.3f} | "
              f"EM_ASR={metrics['edge_asr_mean']:.3f} | "
              f"local_benign={metrics['local_asr_benign_mean']:.3f} "
              f"(same_edge={metrics['local_asr_same_edge']:.3f}, "
              f"diff_edge={metrics['local_asr_diff_edge']:.3f}) | "
              f"local_malicious={metrics['local_asr_malicious_mean']:.3f}\n")

        # ── 任务3：wandb 记录全部分层指标 ─────────────────────────────────
        FLLogger.log_round_metrics(round_idx, metrics)
        # 兼容旧看板字段
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"backdoor/c_acc": metrics["local_acc_mean"],
                           "backdoor/asr": metrics["local_asr_benign_mean"]},
                          step=round_idx)
        except Exception:
            pass
