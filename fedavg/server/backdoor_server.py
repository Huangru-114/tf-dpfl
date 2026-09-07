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

import time

import numpy as np

from server.server import CloudServer
from attack.backdoor_eval import (evaluate_hierarchical_asr,
                                   evaluate_forgetting_curve,
                                   evaluate_feature_separation,
                                   evaluate_drift,
                                   dataset_to_numpy)
from models.cnn import get_base_head_indices
from defense import create_post_hoc_defense
from utils.logger import FLLogger


# 个性化 / 微调方法：遗忘曲线对这些方法有意义（client.model 训练后停在个性化模型）
_PERSONALIZED_METHODS = {
    "hierpfedme", "pfedme", "hier_ditto", "hier_ditto_rep", "hier_pfedme_rep",
    "hier_fedrep",
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
        # 漂移测量（Experiment 3C）：默认关闭，零开销；3C 的 config 打开。
        self.drift_eval = bool(self.bd_cfg.get("drift_eval", False))
        self.drift_probe_samples = int(self.bd_cfg.get("drift_probe_samples", 256))
        self._drift_base_idx = None   # 懒缓存 backbone 索引

        self.feature_eval = bool(self.bd_cfg.get("feature_eval", True))
        self.feature_eval_samples = int(self.bd_cfg.get("feature_eval_samples", 200))
        self.feature_clean_per_class = int(self.bd_cfg.get("feature_clean_per_class", 50))
        self.forgetting_epochs = int(self.bd_cfg.get(
            "forgetting_epochs", self.config["training"].get("local_epochs", 5)))
        self._feat_data = None       # 懒惰构建并缓存

        # ── 后处理防御（Simple-Tuning）：对每个良性 client 的本地模型评估前做后处理。──
        # name 非后处理类时为 None；为 None 时分层评估直接用 client.model（行为不变）。
        self.post_defense = create_post_hoc_defense(
            self.config,
            num_classes=int(self.config["data"]["num_classes"]),
            lr_default=float(self.config["training"].get("learning_rate", 0.02)))

    def run_round(self, round_idx: int):
        n_rounds = int(self.config["federation"]["n_rounds"])
        do_eval  = (round_idx % self.bd_eval_interval == 0) or (round_idx == n_rounds)
        # 漂移的 anchor = 本轮起点（broadcast/训练之前）的全局权重。必须在 super() 前深拷贝，
        # 因为 super().run_round 会就地更新 self.global_model。
        anchor = None
        if self.drift_eval and do_eval:
            anchor = [w.copy() for w in self.global_model.get_weights()]

        metrics = super().run_round(round_idx)
        # CerP：cloud 层协调器，按上一轮快照给恶意客户端互相分发 peer 权重
        self._coordinate_cerp_peers()
        if do_eval:
            self._backdoor_eval(round_idx, anchor=anchor)
        return metrics

    def _coordinate_cerp_peers(self):
        """收集 CerP 恶意客户端最新权重，互相分发（排除自身），供下一轮 cos 正则使用。"""
        cerp = [c for c in self._all_clients
                if getattr(c, "is_malicious", False)
                and hasattr(c, "set_peer_malicious_weights")]
        if not cerp:
            return
        snaps = [(c, c.model.get_weights()) for c in cerp]
        for c in cerp:
            peers = [w for oc, w in snaps if oc is not c]
            c.set_peer_malicious_weights(peers)

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
        # 评估侧 trigger 约定 (model, x, y)；静态触发器忽略 model/y，这里用 global_model。
        # （Phase 2 的 model-dependent 触发器若需 per-model 特征样本，再单独处理。）
        poisoned = self.trigger_fn(self.global_model, x[non_target], y[non_target])
        poisoned_true = y[non_target]
        self._feat_data = (poisoned, poisoned_true, clean_by_class)
        return self._feat_data

    def _backdoor_eval(self, round_idx: int, anchor=None):
        # ASR 的探针是**留出分片**，不是原始 x_test。
        #
        # 本仓库按 PFLlib 口径把 train+test 合并后分区，所以官方 test split 里的图
        # 已经分给客户端、其中约 1−per_client_test_ratio 进了训练集。再拿同一份
        # x_test 当探针，等于在训练过的图上测攻击成功率。
        # 而分区使每个 index 只归属一个客户端、再在客户端内部切 train/test，
        # 于是「所有留出分片的并集」与「所有训练数据」**全局不相交** —— 那才是
        # 真正的留出集，而且 ASR 与 pm_acc 这下测在同一个 population 上。
        #
        #   global ASR ← self.test_dataset（= merge_test_datasets(所有 edge)）
        #   edge   ASR ← edge.get_test_dataset()
        #   client ASR ← client.test_dataset
        #
        # 子采样改为「取前 N 个合格样本」：测试集的 dataset 是 shuffle=False 建的，
        # 顺序固定，所以这是确定性的，不需要像旧路径那样缓存一份随机索引。
        # （x_test/y_test 只留给可选的 feature_eval，见 _prepare_feature_data；
        #  那条路径在 exp3 的全部配置里都是关的。）

        print(f"\n[Backdoor] ===== Round {round_idx} hierarchical backdoor eval =====")

        # ── 分阶段计时 ───────────────────────────────────────────────────
        # 为什么需要：`[Cloud] … time=Xs` 测的是 CloudServer.run_round 的
        # t0→elapsed，而 _backdoor_eval 是在 super().run_round() **返回之后**
        # 才调用的（见 run_round），所以那个数**不含**后门评估。
        # 于是「一轮到底花了多少、其中多少花在评估上」在 metrics.json 里
        # 此前完全看不到 —— 标定 (local_epochs, n_rounds, eval_interval)
        # 时这正是要读的那个数。
        # 未启用的阶段记 None → 打 "n/a"，不是 0.0（陷阱 #13 的同一约定：
        # 「没跑」与「跑了但是很快」不能在数值上长得一样）。
        _t_all = time.perf_counter()
        t_feature = t_forget = t_drift = None

        # ── Simple-Tuning 防御：良性 client 本地模型评估前做「重置头 + 干净微调」──
        # （恶意 client 不设防，保持 c.model 原样，仅用于报告 local_asr_malicious）
        local_model_fn = None
        if self.post_defense is not None:
            def local_model_fn(c):
                if int(c.client_id) in self.malicious_ids:
                    return c.model
                return self.post_defense.tune(c.model, getattr(c, "dataset", None))
            print(f"[Defense] applying Simple-Tuning to benign local models "
                  f"before ASR eval (round {round_idx})")

        # ── 任务2：分层 ASR（global/edge/local 三层 + same/diff edge） ────────
        _t0 = time.perf_counter()
        metrics = evaluate_hierarchical_asr(
            self.global_model, self.edge_servers, self._all_clients,
            self.test_dataset, self.trigger_fn, self.bd_target,
            self.malicious_ids, fallback_test_ds=self.test_dataset,
            local_model_fn=local_model_fn,
            asr_max_samples=self.bd_asr_max,
        )
        t_asr = time.perf_counter() - _t0

        # ── 任务2：特征空间分离度（global + 抽样 1 个良性 local） ─────────────
        if self.feature_eval:
            _t0 = time.perf_counter()
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
            t_feature = time.perf_counter() - _t0

        # ── 任务2：后门遗忘曲线（仅个性化/微调方法、抽样 1 个良性客户端） ─────
        method = self.config["training"].get("drift_correction", "fedavg")
        do_forget = (method in _PERSONALIZED_METHODS
                     or self.config.get("evaluation", {}).get("final_finetune", False))
        if do_forget:
            _t0 = time.perf_counter()
            benign = [c for c in self._all_clients
                      if int(c.client_id) not in self.malicious_ids]
            if benign:
                c0 = benign[0]
                lr = float(self.config["training"].get("learning_rate", 0.02))
                # 探针同样用**这个客户端自己**的留出分片。
                # evaluate_forgetting_curve 还吃 numpy（内部自己调 compute_asr、
                # 自己做目标类过滤），所以这里摊平一份。
                fx, fy = dataset_to_numpy(
                    c0.test_dataset if getattr(c0, "test_dataset", None) is not None
                    else self.test_dataset, self.bd_asr_max)
                if fx is not None:
                    metrics["forgetting_curve"] = evaluate_forgetting_curve(
                        c0.model, self.forgetting_epochs, c0.dataset,
                        fx, fy, self.trigger_fn, self.bd_target, lr=lr)
            t_forget = time.perf_counter() - _t0

        # ── 历史 + 控制台 ─────────────────────────────────────────────────
        self.history["bd_c_acc"].append(metrics["local_acc_mean"])
        self.history["bd_asr"].append(metrics["local_asr_benign_mean"])
        # 无定义的分组打 "n/a"（不是 0.000）——见 backdoor_eval._mean 的说明。
        # 回程解析器 harness/collect_metrics.py 认这个记号并写成 JSON 的 null。
        def _f3(v):
            return "n/a" if v is None else f"{v:.3f}"

        print(f"[Backdoor] Round {round_idx} | "
              f"GM_ASR={_f3(metrics['global_asr'])} | "
              f"EM_ASR={_f3(metrics['edge_asr_mean'])} | "
              f"local_benign={_f3(metrics['local_asr_benign_mean'])} "
              f"(same_edge={_f3(metrics['local_asr_same_edge'])}, "
              f"diff_edge={_f3(metrics['local_asr_diff_edge'])}) | "
              f"local_malicious={_f3(metrics['local_asr_malicious_mean'])}\n")

        # 逐 edge 面板（Experiment 3：per-edge 传播路径。聚合均值会抹平 amplification/
        # dilution/cross-edge cancellation，所以每个 edge 单独打一条可解析行）。
        for pe in metrics.get("per_edge", []):
            print(f"[Backdoor] Round {round_idx} | edge{pe['edge_id']} | "
                  f"edge_asr={_f3(pe['edge_asr'])} | "
                  f"client_benign={_f3(pe['client_benign'])} | "
                  f"client_malicious={_f3(pe['client_malicious'])} | "
                  f"n_benign={pe['n_benign']} | n_malicious={pe['n_malicious']} | "
                  f"has_malicious={pe['has_malicious']}")

        # ── 漂移测量（Experiment 3C）：edge 相对本轮起点全局的参数/表示漂移 ──────
        if self.drift_eval and anchor is not None:
            _t0 = time.perf_counter()
            self._eval_drift(round_idx, anchor)
            t_drift = time.perf_counter() - _t0

        # ── 计时行（可解析；harness/collect_metrics.py 的 RE_TIMING 认它）─────
        def _t3(v):
            return "n/a" if v is None else f"{v:.1f}"

        print(f"[Timing] Round {round_idx} | asr={_t3(t_asr)}s | "
              f"feature={_t3(t_feature)}s | forgetting={_t3(t_forget)}s | "
              f"drift={_t3(t_drift)}s | total={_t3(time.perf_counter() - _t_all)}s")

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

    def _eval_drift(self, round_idx: int, anchor):
        """Experiment 3C：每 edge 相对 anchor（本轮起点全局）的参数/表示漂移，打一条可解析行。"""
        if self.x_test is None or not self.edge_servers:
            return
        if self._drift_base_idx is None:
            num_classes = int(self.config["data"]["num_classes"])
            split = get_base_head_indices(self.global_model, num_classes)
            self._drift_base_idx = split["base_weight_indices"]
        n = min(self.drift_probe_samples, len(self.x_test))
        x_probe = self.x_test[:n]                       # 固定干净探针（确定性、跨轮跨格可比）
        d = evaluate_drift(self.edge_servers, self.global_model, anchor,
                           self._drift_base_idx, x_probe)
        print(f"[Drift] Round {round_idx} | "
              f"param_abs={d['param_abs']:.4f} | param_rel={d['param_rel']:.4f} | "
              f"repr_mean={d['repr_mean']:.4f} | repr_median={d['repr_median']:.4f} | "
              f"n_edges={len(self.edge_servers)}")
