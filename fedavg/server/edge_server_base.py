"""
edges/base.py  –  FL 边缘服务器基类

职责：纯基础设施，不含任何聚合/训练算法逻辑。
  - 客户端管理（select_clients / broadcast_to_clients）
  - 并行 / 串行收集 client 更新
  - 权重管理（set_weights / set_global_ref / set_test_dataset）
  - 评估（evaluate_on）
  - run 声明为抽象接口，强制子类实现

子类选择：
  FedAvgEdgeServer  → edges/fedavg.py
  PFedMeEdgeServer  → edges/pfedme.py
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

from models.model_utils import get_model_bytes
from aggregation.fedavg import aggregate as fedavg_aggregate
from aggregation.client_update import as_client_update
from defense import create_defense


class EdgeServerBase(ABC):

    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        self.edge_id   = edge_id
        self.clients   = clients
        self.model     = model
        self.config    = config
        self.n_samples = sum(c.n_samples for c in clients)

        # ── 后门防御（鲁棒聚合）。None=不设防 → robust_mean 回退普通加权平均。──
        self.defense = create_defense(config)

        # ── 独立播种的 RNG：客户端选取必须可复现，否则矩阵里同一格重跑对不上。
        # 每个 edge 一条独立流（seed, edge_id），互不干扰全局 np.random。
        self.rng = np.random.default_rng([int(config.get("seed", 42)), int(edge_id)])

        # ── 辅助状态 ──────────────────────────────────────────────────────
        self._global_weights_ref    = None   # Cloud 广播的全局权重快照
        self._client_gradient_cache = {}     # warm-up 聚类用的伪梯度缓存
        self._cluster_assignments   = {}
        self._warmup_done           = False
        self.model_bytes            = get_model_bytes(model)

        # ── per-edge 测试集（EM 评估用，由 main.py 调用 set_test_dataset 注入）──
        self.test_dataset = None

        # ── 最近一轮被丢弃（本地训练抛异常）的客户端 id，供指标核对 ──────────
        self.last_failed_ids = []

        print(f"[EdgeServer {edge_id}] "
              f"{len(clients)} clients | {self.n_samples} samples")

    # ══════════════════════════════════════════════════════════════════════
    # 权重 / 数据集管理
    # ══════════════════════════════════════════════════════════════════════

    def set_weights(self, global_weights: list):
        """接收 Cloud 广播的全局权重，覆盖本地 edge 模型。"""
        self.model.set_weights(global_weights)

    def set_test_dataset(self, test_dataset: tf.data.Dataset):
        """注入 per-edge 同分布测试集（由 main.py 在初始化后调用）。"""
        self.test_dataset = test_dataset

    def get_test_dataset(self) -> tf.data.Dataset | None:
        """返回 per-edge 同分布测试集。"""
        return self.test_dataset

    def set_global_ref(self, global_weights: list):
        """
        存储 Cloud 全局权重快照，转发给 client 用作远端参考。

        子类（PFedMeEdgeServer）可 override 此方法以追加算法专属逻辑
        （例如 W_n 的重置），但必须调用 super().set_global_ref()。
        """
        self._global_weights_ref = [w.copy() for w in global_weights]

    # ══════════════════════════════════════════════════════════════════════
    # 客户端选取与广播
    # ══════════════════════════════════════════════════════════════════════

    def select_clients(self, round_idx: int) -> list:
        """
        用本 edge 的 seeded RNG 抽取参与客户端。

        用索引抽样（而不是 np.random.choice(self.clients)）：对象数组抽样在 numpy
        新版本里会告警，且索引抽样让选取结果只依赖 (seed, edge_id, 调用次数)，
        与客户端对象的内存布局无关 → 可复现。
        """
        frac     = self.config["federation"]["client_fraction"]
        n_select = max(1, int(len(self.clients) * frac))
        idx      = self.rng.choice(len(self.clients), n_select, replace=False)
        return [self.clients[int(i)] for i in idx]

    def broadcast_to_clients(self, selected: list, global_weights: list = None):
        """
        广播 edge 模型权重给选中客户端。

        global_weights：同时转发给 client，供 FedProx(global)、FedDyn 等使用；
                        为 None 时退化为 edge 权重。
        """
        edge_weights = self.model.get_weights()
        gw           = global_weights if global_weights is not None else edge_weights
        for client in selected:
            client.set_weights(global_weights=gw, edge_weights=edge_weights)

    # ══════════════════════════════════════════════════════════════════════
    # 并行 / 串行收集 client 更新
    # ══════════════════════════════════════════════════════════════════════

    def _collect_updates_parallel(self, selected: list,
                                  global_round_idx: int,
                                  mode: str = "fedavg") -> list:
        """
        并行执行所有选中客户端的本地训练。

        两条不变量（防御轴依赖它们，见 aggregation/client_update.py）：
          1. **顺序确定**：结果严格按 `selected` 的顺序返回，不按完成顺序。
             旧实现用 as_completed → 每次运行顺序不同，防御报出的「接纳索引」
             无法复现、也无法映射回客户端。
          2. **带身份**：每条结果包成 ClientUpdate，携带 client_id。

        失败的客户端不进结果（与旧行为一致），但会记进 self.last_failed_ids。

        Args:
            mode: 仅支持 "fedavg"（client.local_train）。meta_grad 模式随
                  Hier-PerFedAvg 一起移除——它聚合的是元梯度而非权重，
                  与防御接口语义不兼容。
        Returns:
            [ClientUpdate(weights, n, loss, t, client_id), ...]
        """
        if mode != "fedavg":
            raise ValueError(
                f"_collect_updates_parallel 只支持 mode='fedavg'，收到 {mode!r}。"
                f"meta_grad 模式已随 Hier-PerFedAvg 移除。")

        n_workers = self.config["federation"].get("n_workers", 4)

        def run_one(client):
            # 每轮按 global round 更新学习率（per-round 衰减，对齐 PFLlib）
            if hasattr(client, "apply_round_lr"):
                client.apply_round_lr(global_round_idx)
            return client.local_train(global_round_idx)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(run_one, c) for c in selected]   # 提交顺序 = selected 顺序
            raw = []
            for client, future in zip(selected, futures):               # 按提交顺序取结果
                try:
                    raw.append((client, future.result()))
                except Exception as e:
                    print(f"    [ERROR] Client {client.client_id}: {e}")
                    raw.append((client, None))

        return self._finalize_updates(raw)

    def _collect_updates_serial(self, selected: list,
                                global_round_idx: int,
                                mode: str = "fedavg") -> list:
        """串行版本。不变量与返回值同 _collect_updates_parallel。"""
        if mode != "fedavg":
            raise ValueError(
                f"_collect_updates_serial 只支持 mode='fedavg'，收到 {mode!r}。")

        raw = []
        for client in selected:
            try:
                if hasattr(client, "apply_round_lr"):
                    client.apply_round_lr(global_round_idx)
                raw.append((client, client.local_train(global_round_idx)))
            except Exception as e:
                print(f"    [ERROR] Client {client.client_id}: {e}")
                raw.append((client, None))
        return self._finalize_updates(raw)

    def _finalize_updates(self, raw: list) -> list:
        """把 [(client, result|None)] 规范成 [ClientUpdate]，并记录失败的客户端。"""
        results, failed = [], []
        for client, result in raw:
            if result is None:
                failed.append(int(client.client_id))
                continue
            results.append(as_client_update(result, client.client_id))
        self.last_failed_ids = failed
        if failed:
            print(f"    [Edge {self.edge_id}] {len(failed)} 个客户端本轮失败并被丢弃: "
                  f"{failed}")
        return results


    # ══════════════════════════════════════════════════════════════════════
    # 鲁棒聚合（防御统一入口）
    # ══════════════════════════════════════════════════════════════════════

    def robust_mean(self, client_updates: list, ref_weights: list = None) -> list:
        """
        统一的「加权平均步」替身：有防御时走鲁棒聚合，否则回退普通样本加权 FedAvg。

        各 edge server 把自己原本的加权平均（FedAvg 整体 / pFedMe·Ditto 的 mean /
        FedRep 的 backbone mean）改调此方法，即可让一套防御覆盖所有 PFL 方法。

        Args:
            client_updates: [(weights, n_samples, loss, train_time), ...]
            ref_weights:    聚合前 edge 模型权重（广播点）；None 时取当前 edge 模型权重。
                            FLAME/DnC 用作更新增量参考；坐标类防御忽略。
        Returns:
            List[np.ndarray]：聚合后的完整权重列表（与 aggregate 同形）。
        """
        if self.defense is None:
            return fedavg_aggregate(client_updates)
        if ref_weights is None:
            ref_weights = self.model.get_weights()
        agg = self.defense.aggregate(client_updates, ref_weights)
        # 防御接纳/剔除的判决按 client_id 记账（不是位置索引），供 TPR/FPR 统计
        self.defense.record_decision(client_updates)
        return agg

    # ══════════════════════════════════════════════════════════════════════
    # 评估
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_on(self, fallback_dataset: tf.data.Dataset = None):
        """
        评估 edge 模型。

        优先使用 self.test_dataset（per-edge 同分布测试集）；
        未注入时退化为 fallback_dataset（全体测试集）。
        """
        if self.test_dataset is None and fallback_dataset is None:
            raise ValueError("No test dataset for evaluation.")
        if self.test_dataset is None:
            print(f"  [Edge {self.edge_id}] No per-edge test dataset, "
                  f"using fallback dataset.")
        ds = self.test_dataset if self.test_dataset is not None else fallback_dataset

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        tl = tc = tn = 0
        for x, y in ds:
            p   = self.model(x, training=False)
            tl += loss_fn(y, p).numpy() * x.shape[0]
            tc += np.sum(np.argmax(p.numpy(), 1) == y.numpy())
            tn += x.shape[0]
        if tn == 0:
            return 0.0, 0.0
        return float(tl / tn), float(tc / tn)

    # ══════════════════════════════════════════════════════════════════════
    # 轮次接口（子类必须实现）
    # ══════════════════════════════════════════════════════════════════════

    @abstractmethod
    def run(self, global_round_idx: int):
        """
        执行 edge_rounds 轮内部聚合后上传最终权重。

        Returns:
            (upload_weights, n_samples, avg_loss, avg_time, comm_bytes)
        """