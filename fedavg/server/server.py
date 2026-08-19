"""
server.py  –  层级 FL Cloud 服务器
支持所有 FL 方法的全局层聚合，通过 config 统一切换。

全局聚合策略
─────────────
  fedavg / fedprox / pfedme / perfedavg / scaffold
      → 标准样本加权 FedAvg（差异在 edge 层及以下）
  feddyn
      → FedDyn 去偏聚合（维护 h_g，unweighted mean + h_g 校正）
  scaffold + scaffold_cloud: true
      → SCAFFOLD 在 cloud 层也做控制变量聚合（维护 c_g）
"""

import numpy as np
import tensorflow as tf
import time

from models.model_utils   import get_model_bytes
from aggregation.fedavg   import aggregate
from aggregation.feddyn   import feddyn_aggregate,   init_h
from aggregation.scaffold import scaffold_aggregate, init_cv
from .robust_aggregation  import RobustAggregationMixin


class CloudServer(RobustAggregationMixin):
    def __init__(self, global_model: tf.keras.Model,
                 edge_servers: list,
                 test_dataset: tf.data.Dataset,
                 config: dict):
        self.global_model = global_model
        self.edge_servers = edge_servers
        self.test_dataset = test_dataset
        self.config       = config
        self.loss_fn      = tf.keras.losses.SparseCategoricalCrossentropy()

        # ── cloud 层防御（接口，默认关）─────────────────────────────────
        # defense.layers 缺省为 ["edge"] → 这里拿到 None → robust_mean 回退
        # aggregation.fedavg.aggregate，与改动前的 _aggregate_global 逐元素相同。
        # 守卫：tests/test_cloud_aggregate_default.py
        self._init_defense(config, layer="cloud")

        self.model_bytes      = get_model_bytes(global_model)
        self.total_comm_bytes = 0

        # ── FedDyn 全局校正状态 h_g ────────────────────────────────────
        # 把 edge server 视为"客户端"，做同级的 FedDyn 去偏校正。
        # h_g ← h_g − (α_g/E)·Σ_{e∈active}(θ_e^new − θ_g^prev)
        # θ_g^new = mean(θ_e^new) − (1/α_g)·h_g
        self.h_g = None   # List[np.ndarray]，懒惰初始化

        # ── SCAFFOLD 全局控制变量 c_g（可选）───────────────────────────
        # 仅当 config["training"]["scaffold_cloud"] == True 时使用。
        # c_g ← c_g + (1/E)·Σ_{e∈active} Δc_e
        # 此时 EdgeServer 需上传其 c_delta（待扩展）。
        self.c_g = None   # List[np.ndarray]，懒惰初始化

        print(f"[CloudServer] {len(edge_servers)} edges | "
              f"model: {self.model_bytes / 1024 / 1024:.2f} MB")

        self.history = {
            "round": [], "global_loss": [], "global_acc": [],
            "avg_edge_loss": [], "round_time": [],
            "avg_client_time": [], "comm_bytes_round": [], "comm_bytes_total": [],
        }

    # ══════════════════════════════════════════════════════════════════════
    # 初始化辅助
    # ══════════════════════════════════════════════════════════════════════

    def _init_h_g(self):
        if self.h_g is None:
            self.h_g = init_h(self.global_model.get_weights())

    def _init_c_g(self):
        if self.c_g is None:
            self.c_g = init_cv(self.global_model.get_weights())

    # ══════════════════════════════════════════════════════════════════════
    # 广播
    # ══════════════════════════════════════════════════════════════════════

    def broadcast_to_edges(self, selected_edges: list = None):
        """广播全局权重给所有（或选定的）Edge Server。"""
        if selected_edges is None:
            selected_edges = self.edge_servers
        gw = self.global_model.get_weights()
        for edge in selected_edges:
            edge.set_weights(gw)
            edge.set_global_ref(gw)

    # ══════════════════════════════════════════════════════════════════════
    # 全局聚合
    # ══════════════════════════════════════════════════════════════════════

    def _aggregate_global(self, edge_updates: list,
                          prev_global_weights: list) -> list:
        """
        在 Cloud 层聚合 edge 模型。

        FedDyn 路径
        ────────────
        α_g 可单独配置（alpha_feddyn_global），默认回退到 alpha_feddyn。
        通常 α_g > α_client（跨 cluster 异质性更高）。

        SCAFFOLD 全局路径（scaffold_cloud: true）
        ─────────────────────────────────────────
        Edge 上传的 c_delta 尚未在当前接口中传递，
        可扩展 edge.run() 返回值以支持。此处预留框架。

        其他方法
        ─────────
        样本加权 FedAvg（标准）。

        **当前实际行为**：下面三个分支全部被注释掉，函数体只剩最后一行
        `return self.aggregate_edges(...)` —— 也就是说不管 drift_correction 是什么，
        cloud 层永远是朴素样本加权 FedAvg。config 里的 `beta_hier`（Hier-pFedMe 式 9
        的 β）、`alpha_feddyn_global` 都是**死配置，读都没读**。
        本会话只把聚合入口收敛到 aggregate_edges（留接口），不改这个行为。
        """
        method = self.config["training"].get("drift_correction", "fedavg")

        # if method == "feddyn":
        #     self._init_h_g()
        #     alpha_g = self.config["training"].get(
        #         "alpha_feddyn_global",
        #         self.config["training"].get("alpha_feddyn", 0.01),
        #     )
        #     return feddyn_aggregate(
        #         h_state        = self.h_g,
        #         active_weights = [w for w, *_ in edge_updates],
        #         prev_anchor    = prev_global_weights,
        #         alpha          = alpha_g,
        #         total_n        = len(self.edge_servers),   # ← 全部 edge
        #     )

        # if method == "hierpfedme":
        #     # 式 (9)：W^{t+1} = (1−β)·W^t + β·(1/N)·Σ W_n^{t,I}
        #     # edge_updates 中每个 w 是 W_n^{t,I}（EdgeServer.run 已正确上传）
        #     # β=1 时退化为标准 FedAvg，与论文默认设置一致
        #     print("  [Cloud] Using Hier-pFedMe global aggregation (式 9).")
        #     beta           = self.config["training"].get("beta_hier", 1.0)
        #     active_weights = [w for w, *_ in edge_updates]
        #     mean_w = [
        #         np.mean([aw[j] for aw in active_weights], axis=0)
        #         for j in range(len(active_weights[0]))
        #     ]
        #     return [
        #         (1.0 - beta) * pw + beta * mw
        #         for pw, mw in zip(prev_global_weights, mean_w)
        #     ]

        # elif method == "scaffold" and self.config["training"].get("scaffold_cloud", False):
        #     # ── 预留：Cloud 层 SCAFFOLD（需 edge 上传 c_delta）────────────
        #     # 当前 edge.run() 不返回 c_delta；扩展后取消注释：
        #     #   self._init_c_g()
        #     #   c_deltas = [ed.get("c_delta") for ed in edge_aux if ed.get("c_delta")]
        #     #   return scaffold_aggregate(self.c_g, active_weights, c_deltas, len(self.edge_servers))
        #     # 暂时退化为 FedAvg（等价于 SCAFFOLD 仅在 edge 层生效）
        #     return aggregate(edge_updates)

        # else:
            # fedavg / fedprox / pfedme / perfedavg / scaffold(无cloud级) → FedAvg
        return self.aggregate_edges(edge_updates, prev_global_weights)

    # ══════════════════════════════════════════════════════════════════════
    # cloud 层聚合入口（防御轴的 cloud 侧接口）
    # ══════════════════════════════════════════════════════════════════════

    def aggregate_edges(self, edge_updates: list,
                        prev_global_weights: list) -> list:
        """
        把各 edge 上传的模型聚合成新的全局模型。

        `defense.layers` 不含 "cloud" 时（默认），`self.defense is None`，
        `robust_mean` 回退 `aggregation.fedavg.aggregate` —— 与改动前的
        `return aggregate(edge_updates)` **逐元素相同**。
        配了 `layers: [edge, cloud]` 就会在这里也跑一遍鲁棒聚合，
        参考点是广播前的全局模型快照 prev_global_weights。

        注意：edge 上传物不带 client_id（它们是 edge 级的 4-元组），
        所以 cloud 层的防御拿不到客户端身份，`record_decision` 会把
        last_admitted_ids 记成 edge 的位置索引。真要在 cloud 层做客户端级
        TPR/FPR 统计，需要 edge 往上透传身份 —— 本会话不做。
        """
        return self.robust_mean(edge_updates, prev_global_weights)

    def _default_ref_weights(self) -> list:
        """本层聚合前的模型权重（广播点）= 当前全局模型权重。"""
        return self.global_model.get_weights()

    # ══════════════════════════════════════════════════════════════════════
    # 主训练流程
    # ══════════════════════════════════════════════════════════════════════

    def collect_and_aggregate(self, round_idx: int,
                               prev_global_weights: list):
        """
        触发各 edge 执行内部聚合，收集 edge 模型，做全局聚合。

        Args:
            round_idx           : 当前全局轮次
            prev_global_weights : 广播前的全局模型快照（FedDyn anchor）
        """
        edge_updates, comm_ce = [], 0
        for edge in self.edge_servers:
            w, n, loss, t, comm = edge.run(round_idx)
            edge_updates.append((w, n, loss, t))
            comm_ce += comm

        new_w = self._aggregate_global(edge_updates, prev_global_weights)
        self.global_model.set_weights(new_w)

        total_n  = sum(n for _, n, *_ in edge_updates)
        avg_loss = sum(l * n / total_n for _, n, l, _ in edge_updates)
        # edge_updates: (w, n, loss, time) → 最后一个是 time
        avg_t    = float(np.mean([eu[3] for eu in edge_updates]))
        return float(avg_loss), avg_t, comm_ce

    def evaluate_global(self):
        """在全局测试集上评估全局模型。"""
        tl = tc = tn = 0
        for images, labels in self.test_dataset:
            pred  = self.global_model(images, training=False)
            tl   += self.loss_fn(labels, pred).numpy() * images.shape[0]
            tc   += np.sum(np.argmax(pred.numpy(), 1) == labels.numpy())
            tn   += images.shape[0]
        return tl / tn, tc / tn

    def run_round(self, round_idx: int):
        """
        一轮完整通信：

        Phase 0  快照全局 anchor（必须在 broadcast 前）
        Phase 1  广播 → Edge 内部聚合（含 client 训练）
        Phase 2  全局聚合（FedDyn/FedAvg/…）
        Phase 3  评估
        """
        t0 = time.time()

        # ── Phase 0：anchor 快照 ──────────────────────────────────────────
        # 必须在 broadcast_to_edges() 之前完成深拷贝。
        prev_gw = [w.copy() for w in self.global_model.get_weights()]

        print(f"\n[Round {round_idx:>3}] Broadcasting to {len(self.edge_servers)} edges...")
        self.broadcast_to_edges()

        # ── Phase 1+2：edge 聚合 + 全局聚合 ──────────────────────────────
        avg_edge_loss, avg_ct, comm_ce = self.collect_and_aggregate(round_idx, prev_gw)

        # ── Phase 3：评估 ────────────────────────────────────────────────

        global_loss, global_acc = self.evaluate_global()

        eval_interval = int(self.config.get("evaluation", {}).get("eval_interval", 10))
        n_rounds      = int(self.config["federation"]["n_rounds"])
        do_pm_eval    = (round_idx % eval_interval == 0) or (round_idx == n_rounds)


        # Edge model：每轮都测，使用 per-edge 测试集（若已设置），按样本数加权
        em_losses, em_accs, em_ns = [], [], []
        for edge in self.edge_servers:
            e_loss, e_acc = edge.evaluate_on(fallback_dataset=self.test_dataset)
            em_losses.append(e_loss)
            em_accs.append(e_acc)
            em_ns.append(edge.n_samples)
        total_em_n  = sum(em_ns)
        avg_em_acc  = float(sum(a * n / total_em_n for a, n in zip(em_accs, em_ns)))
        avg_em_loss = float(sum(l * n / total_em_n for l, n in zip(em_losses, em_ns)))

        # Personalized model：每 eval_interval 轮测一次，按样本数加权
        avg_pm_acc = avg_pm_loss = None
        if do_pm_eval:
            pm_losses, pm_accs, pm_ns = [], [], []
            for edge in self.edge_servers:
                for client in edge.clients:
                    c_loss, c_acc = client.evaluate_on(
                        fallback_dataset=self.test_dataset
                    )
                    pm_losses.append(c_loss)
                    pm_accs.append(c_acc)
                    pm_ns.append(client.n_samples)
            total_pm_n  = sum(pm_ns)
            avg_pm_acc  = float(sum(a * n / total_pm_n for a, n in zip(pm_accs, pm_ns)))
            avg_pm_loss = float(sum(l * n / total_pm_n for l, n in zip(pm_losses, pm_ns)))

        comm_ce_total   = 2 * self.model_bytes * len(self.edge_servers)
        comm_round      = comm_ce_total + comm_ce
        self.total_comm_bytes += comm_round
        elapsed = time.time() - t0

        metrics = {
            "round":            round_idx,
            "edge_rounds":      self.config["federation"]["edge_rounds"],
            "local_epochs":     self.config["training"]["local_epochs"],
            "global_loss":      global_loss,
            "global_acc":       global_acc,
            "avg_edge_loss":    avg_edge_loss,
            "round_time":       elapsed,
            "avg_client_time":  avg_ct,
            "comm_bytes_round": comm_round,
            "comm_bytes_total": self.total_comm_bytes,
            "comm_mb_round":    comm_round          / 1024 / 1024,
            "comm_mb_total":    self.total_comm_bytes / 1024 / 1024,
            "comm_mb_cloud_edge":  comm_ce_total / 1024 / 1024,
            "comm_mb_client_edge": comm_ce       / 1024 / 1024,
            "em_acc":  avg_em_acc,
            "em_loss": avg_em_loss,
            "pm_acc":  avg_pm_acc,
            "pm_loss": avg_pm_loss,
        }
        
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)

        pm_str = (f" PM={avg_pm_acc:.4f}" if avg_pm_acc is not None else "")
        print(f"  [Cloud] GM={global_acc:.4f} | EM={avg_em_acc:.4f}{pm_str} | "
            f"loss={global_loss:.4f} | time={elapsed:.1f}s | "
            f"comm={comm_round/1024/1024:.1f}MB "
            f"(total={self.total_comm_bytes/1024/1024:.0f}MB)")
        return metrics

    def run(self, logger=None):
        """执行完整实验（n_rounds 轮全局通信）。"""
        for r in range(1, self.config["federation"]["n_rounds"] + 1):
            m = self.run_round(r)
            if logger:
                logger.log_round(r, m)
        print("\n[Cloud] Training complete.")
        return self.history