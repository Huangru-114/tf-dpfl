"""
edge_server.py  –  层级 FL 边缘服务器
支持所有 FL 方法的边缘层聚合，通过 config 统一切换。

聚合策略汇总
────────────
  fedavg / fedprox / pfedme / perfedavg
      → 标准样本加权 FedAvg（客户端侧差异，服务端相同）
  feddyn
      → FedDyn 去偏聚合（维护 h_e，unweighted mean + h 校正）
  scaffold
      → SCAFFOLD 聚合（维护 c_e，unweighted mean + c_e 更新）
"""

import time
import numpy as np
import tensorflow as tf

from aggregation.fedavg  import aggregate
from aggregation.feddyn  import feddyn_aggregate,  init_h
from aggregation.scaffold import scaffold_aggregate, init_cv
from models.model_utils  import get_model_bytes


class EdgeServer:
    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        self.edge_id  = edge_id
        self.clients  = clients
        self.model    = model
        self.config   = config
        self.n_samples = sum(c.n_samples for c in clients)

        # ── FedDyn 校正状态 h_e ────────────────────────────────────────────
        # 更新：h_e ← h_e − (α/m_e)·Σ_{k∈P_t}(θ_k^new − θ_e^prev)
        # 聚合：θ_e^new = mean(θ_k^new) − (1/α)·h_e
        # 分母 m_e 是全部 client 数，不是活跃数！
        self.h_e = None   # List[np.ndarray]，懒惰初始化

        # ── SCAFFOLD 边缘控制变量 c_e ────────────────────────────────────
        # 更新：c_e ← c_e + (1/m_e)·Σ_{k∈P_t} Δc_k
        # 广播给 client 作为 c_server，纠正梯度漂移
        self.c_e = None   # List[np.ndarray]，懒惰初始化

        # ── 其他状态 ──────────────────────────────────────────────────────
        self._global_weights_ref    = None
        self._client_gradient_cache = {}
        self._cluster_assignments   = {}
        self._warmup_done           = False
        self.model_bytes            = get_model_bytes(model)

        # ── Hier-pFedMe：W_n —— edge 保存的全局模型副本 ────────────────────
        # 对应 Algorithm 1 中的 W_n^{t,i}。
        # 每个 global round 开始时由 set_global_ref() 重置为 W^t，
        # 之后在每个 edge round 内通过式 (8) 更新：
        #   W_n^{t,i+1} = W_n^{t,i} − η1·λ1·(W_n^{t,i} − θ_n^{t,i+1})
        # edge round 结束后上传 W_n^{t,I} 给 Cloud。
        self.W_n = None

        print(f"[EdgeServer {edge_id}] "
              f"{len(clients)} clients | {self.n_samples} samples")

    # ══════════════════════════════════════════════════════════════════════
    # 初始化辅助
    # ══════════════════════════════════════════════════════════════════════

    def _init_h_e(self):
        if self.h_e is None:
            self.h_e = init_h(self.model.get_weights())

    def _init_c_e(self):
        if self.c_e is None:
            self.c_e = init_cv(self.model.get_weights())

    # ══════════════════════════════════════════════════════════════════════
    # 权重管理
    # ══════════════════════════════════════════════════════════════════════

    def set_weights(self, global_weights: list):
        """接收 Cloud 广播的全局权重，覆盖本地 edge 模型。"""
        self.model.set_weights(global_weights)

    def set_global_ref(self, global_weights: list):
        """存储 Cloud 全局权重快照，转发给 client 用作远端参考。
        Hier-pFedMe：同时将 W_n^{t,0} 重置为 W^t（每个 global round 执行一次）。
        """
        self._global_weights_ref = [w.copy() for w in global_weights]
        # Hier-pFedMe：W_n^{t,0} ← W^t（Algorithm 1 第 4 行）
        if self.config["training"].get("drift_correction") == "hierpfedme":
            self.W_n = [w.copy() for w in global_weights]

    # ══════════════════════════════════════════════════════════════════════
    # 客户端选取与广播
    # ══════════════════════════════════════════════════════════════════════

    def select_clients(self, round_idx: int) -> list:
        frac     = self.config["federation"]["client_fraction"]
        n_select = max(1, int(len(self.clients) * frac))
        return np.random.choice(self.clients, n_select, replace=False).tolist()

    def broadcast_to_clients(self, selected: list, global_weights: list = None):
        """
        广播 edge 模型权重给选中客户端。
        • global_weights：同时转发给 client，供 FedProx(global)、FedDyn 等使用。
        • SCAFFOLD：额外注入边缘控制变量 c_e。
        """
        method      = self.config["training"].get("drift_correction", "fedavg")
        edge_weights = self.model.get_weights()
        gw           = global_weights if global_weights is not None else edge_weights

        for client in selected:
            client.set_weights(global_weights=gw, edge_weights=edge_weights)
            if method == "scaffold":
                self._init_c_e()
                client.set_scaffold_c(self.c_e)

    # ══════════════════════════════════════════════════════════════════════
    # 聚合核心
    # ══════════════════════════════════════════════════════════════════════

    def _aggregate(self, client_updates: list,
                   prev_edge_weights: list,
                   active_clients: list) -> list:
        """
        根据配置策略聚合 client 模型，返回新的 edge 权重。

        Args:
            client_updates   : [(weights, n_samples, loss, time), ...]
            prev_edge_weights: 广播前的 edge 模型深拷贝（FedDyn/SCAFFOLD anchor）
            active_clients   : 本轮参与训练的 FLClient 实例列表
        """
        method = self.config["training"].get("drift_correction", "fedavg")

        # ── FedDyn ──────────────────────────────────────────────────────
        if method == "feddyn":
            self._init_h_e()
            alpha = self.config["training"].get("alpha_feddyn", 0.01)
            return feddyn_aggregate(
                h_state        = self.h_e,
                active_weights = [w for w, *_ in client_updates],
                prev_anchor    = prev_edge_weights,
                alpha          = alpha,
                total_n        = len(self.clients),   # ← 全部 client，非活跃数
            )

        # ── SCAFFOLD ─────────────────────────────────────────────────────
        elif method == "scaffold":
            self._init_c_e()
            # 收集 c_delta（None 表示该 client 未参与或未产生 delta）
            c_deltas = [
                c.c_delta for c in active_clients
                if c.c_delta is not None
            ]
            return scaffold_aggregate(
                c_server       = self.c_e,
                active_weights = [w for w, *_ in client_updates],
                c_deltas       = c_deltas,
                total_n        = len(self.clients),   # ← 全部 client，非活跃数
            )

        # ── FedAvg / FedProx / pFedMe / Per-FedAvg ───────────────────
        # 以上方法客户端侧各有差异，但服务端聚合均为样本加权平均
        else:
            return aggregate(client_updates)

    # ══════════════════════════════════════════════════════════════════════
    # Warm-up 聚类（辅助）
    # ══════════════════════════════════════════════════════════════════════

    def _collect_gradient_update(self, client, prev_edge_weights: list):
        """记录 client 伪梯度（用于 warm-up 结束后的统计聚类）。"""
        flat = np.concatenate([
            (n - p).ravel()
            for n, p in zip(client.model.get_weights(), prev_edge_weights)
        ])
        self._client_gradient_cache[client.client_id] = flat

    # ══════════════════════════════════════════════════════════════════════
    # Hier-pFedMe 边缘轮次
    # ══════════════════════════════════════════════════════════════════════

    def run_edge_round_hierpfedme(self, global_round_idx: int,
                                   edge_round_idx: int):
        """
        Hier-pFedMe 单轮 edge 内部通信（对应 Algorithm 1 第 5-18 行）：

        1. 广播 θ_n^{t,i}（edge 模型）给选中 client
        2. client 执行内层+外层 Moreau 优化，上传 Θ_{n,m}^{t,i,R}
        3. edge 更新（式 7）：
               θ_n^{t,i+1} = mean(Θ_{n,m}) − η2·λ1·(θ_n^{t,i} − W_n^{t,i})
        4. W_n 更新（式 8）：
               W_n^{t,i+1} = W_n^{t,i} − η1·λ1·(W_n^{t,i} − θ_n^{t,i+1})
        """
        lam1 = self.config["training"].get("lambda1_hier", 35.0)
        eta2 = self.config["training"]["learning_rate"]
        eta1 = self.config["training"].get("eta1_hier", eta2)

        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Hier-pFedMe | Selected {len(selected)}/{len(self.clients)}")

        # 步骤 1：广播 θ_n^{t,i}（edge 模型）给 client
        edge_w = self.model.get_weights()
        for client in selected:
            client.set_weights(global_weights=self._global_weights_ref,
                               edge_weights=edge_w)

        # 步骤 2：client 本地训练，收集 Θ_{n,m}^{t,i,R}
        client_updates = []
        for client in selected:
            w, n, loss, t = client.local_train(global_round_idx)
            client_updates.append((w, n, loss, t))

        # 步骤 3：edge 模型更新（式 7）
        # mean(Θ_{n,m}^{t,i,R})
        M         = len(client_updates)
        n_layers  = len(client_updates[0][0])
        mean_Theta = [
            np.mean([upd[0][j] for upd in client_updates], axis=0)
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
    # Hier-pFedAvg 边缘轮次（辅助）
    # ══════════════════════════════════════════════════════════════════════        

    def run_edge_round_hier_perfedavg(self, global_round_idx: int,
                                    edge_round_idx: int):
        """
        Hier-PerFedAvg 单轮边缘元聚合：
        1. 广播当前 edge 元参数 θ_E 给选中 client
        2. 每个 client 计算元梯度 g_i = ∇_{θ_E} L_i(θ_E - α∇L_i(θ_E))
        3. NaN 过滤
        4. 加权聚合元梯度
        5. 边缘元参数更新：θ_E ← θ_E - β * Σ(n_i/N * g_i)
        """
        beta = self.config["training"].get("beta_outer", 0.01)

        selected     = self.select_clients(global_round_idx)
        edge_weights = self.model.get_weights()

        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
            f"Hier-PerFedAvg | Selected {len(selected)}/{len(self.clients)}")

        # 广播 edge 元参数
        self.broadcast_to_clients(selected, global_weights=self._global_weights_ref)

        # 收集元梯度
        meta_grads_collected, n_total, losses = [], 0, []
        for client in selected:
            grads, n, loss = client.compute_meta_gradient(edge_weights, global_round_idx)

            # NaN / inf 过滤（解决 Dirichlet α=0.1 的崩溃问题）
            if any(np.isnan(g).any() or np.isinf(g).any() for g in grads):
                print(f"    [SKIP] Client {client.client_id}: "
                    f"NaN/inf in meta-gradient")
                continue

            meta_grads_collected.append((grads, n))
            n_total += n
            losses.append(loss)

        if not meta_grads_collected:
            print(f"  [Edge {self.edge_id}] WARNING: all clients skipped, "
                f"keeping current edge weights")
            return float('nan'), 0.0

        # 加权平均元梯度
        n_params   = len(edge_weights)
        agg_grads  = [
            sum((n / n_total) * grads[i]
                for grads, n in meta_grads_collected)
            for i in range(n_params)
        ]

        # 边缘元参数更新
        new_edge_w = [w - beta * g for w, g in zip(edge_weights, agg_grads)]
        self.model.set_weights(new_edge_w)

        avg_loss = float(np.mean(losses))
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
            f"loss={avg_loss:.4f}")
        return avg_loss, 0.0
    # ══════════════════════════════════════════════════════════════════════
    # 评估
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_on(self, test_dataset: tf.data.Dataset):
        """
        在全局测试集上评估 edge 模型（Hier-pFedMe EM 评估用）。
        self.model 持有 θ_n^{t,I}（最后一个 edge round 结束后的 edge 模型）。
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        tl = tc = tn = 0
        for x, y in test_dataset:
            p  = self.model(x, training=False)
            tl += loss_fn(y, p).numpy() * x.shape[0]
            tc += np.sum(np.argmax(p.numpy(), 1) == y.numpy())
            tn += x.shape[0]
        if tn == 0:
            return 0.0, 0.0
        return float(tl / tn), float(tc / tn)

    # ══════════════════════════════════════════════════════════════════════
    # 轮次执行
    # ══════════════════════════════════════════════════════════════════════

    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """
        单轮 edge 内部通信：
          1. 选取活跃 client
          2. 快照 edge 模型（FedDyn/SCAFFOLD anchor）
          3. 广播（含 SCAFFOLD c_e 注入）
          4. 客户端本地训练
          5. 聚合并更新 edge 模型
        """
        selected = self.select_clients(global_round_idx)
        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"Selected {len(selected)}/{len(self.clients)}")

        # 步骤 2：深拷贝 anchor（必须在 broadcast 前）
        prev_edge_w = [w.copy() for w in self.model.get_weights()]

        # 步骤 3：广播
        self.broadcast_to_clients(selected, global_weights=self._global_weights_ref)

        # 步骤 4：本地训练 + warm-up 缓存
        client_updates = []
        warmup = self.config["federation"].get("warmup_rounds", 5)
        do_cache = (self.config["federation"].get("client_selection") == "clustered"
                    and global_round_idx <= warmup)
        for client in selected:
            w, n, loss, t = client.local_train(global_round_idx)
            client_updates.append((w, n, loss, t))
            if do_cache:
                self._collect_gradient_update(client, prev_edge_w)

        # 步骤 5：聚合
        new_w = self._aggregate(client_updates, prev_edge_w, selected)
        self.model.set_weights(new_w)

        total_n  = sum(n for _, n, *_ in client_updates)
        avg_loss = sum(l * n / total_n for _, n, l, _ in client_updates)
        avg_time = float(np.mean([t for *_, t in client_updates]))

        print(f"  [Edge {self.edge_id}] G{global_round_idx} | E{edge_round_idx} | "
              f"loss={avg_loss:.4f}")
        return float(avg_loss), avg_time

    def run(self, global_round_idx: int):
        """执行 edge_rounds 轮内部聚合后上传最终 edge 模型（或 W_n）。"""
        edge_rounds  = self.config["federation"]["edge_rounds"]
        method       = self.config["training"].get("drift_correction", "fedavg")
        round_losses, round_times = [], []
        for er in range(1, edge_rounds + 1):
            if method == "hierpfedme":
                loss, t = self.run_edge_round_hierpfedme(global_round_idx, er)
            elif method == "hier_perfedavg":          # ← 新增这一行
                loss, t = self.run_edge_round_hier_perfedavg(global_round_idx, er)
            else:
                loss, t = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(t)
        comm = 2 * self.model_bytes * len(self.clients) * edge_rounds
        # Hier-pFedMe 上传 W_n^{t,I}（不是 θ_n），其他方法上传 edge 模型
        upload_w = (self.W_n if method == "hierpfedme" and self.W_n is not None
                    else self.model.get_weights())
        return (upload_w, self.n_samples,
                float(np.mean(round_losses)), float(np.mean(round_times)), comm)