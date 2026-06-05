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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf

from models.model_utils import get_model_bytes
from aggregation.fedavg import aggregate as fedavg_aggregate
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

        # ── 辅助状态 ──────────────────────────────────────────────────────
        self._global_weights_ref    = None   # Cloud 广播的全局权重快照
        self._client_gradient_cache = {}     # warm-up 聚类用的伪梯度缓存
        self._cluster_assignments   = {}
        self._warmup_done           = False
        self.model_bytes            = get_model_bytes(model)

        # ── per-edge 测试集（EM 评估用，由 main.py 调用 set_test_dataset 注入）──
        self.test_dataset = None

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
        frac     = self.config["federation"]["client_fraction"]
        n_select = max(1, int(len(self.clients) * frac))
        return np.random.choice(self.clients, n_select, replace=False).tolist()

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
        并行执行所有选中客户端的本地训练 / 元梯度计算。

        Args:
            mode: "fedavg"    → client.local_train，返回 (weights, n, loss, t)
                  "meta_grad" → client.compute_meta_gradient，NaN 已过滤
        Returns:
            results: mode=fedavg   → [(weights, n, loss, t), ...]
                     mode=meta_grad → [(grads, n, loss), ...]
        """
        n_workers    = self.config["federation"].get("n_workers", 4)
        edge_weights = self.model.get_weights()  # 拍快照，所有线程只读

        def run_one(client):
            if mode == "fedavg":
                return client.local_train(global_round_idx)
            return client.compute_meta_gradient(edge_weights, global_round_idx)

        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_one, c): c for c in selected}
            for future in as_completed(futures):
                client = futures[future]
                try:
                    result = future.result()
                    # if mode == "meta_grad":
                    #     grads, n, loss = result
                    #     if any(np.isnan(g).any() or np.isinf(g).any()
                    #            for g in grads):
                    #         print(f"    [SKIP] Client {client.client_id}: NaN")
                    #         continue
                    results.append(result)
                except Exception as e:
                    print(f"    [ERROR] Client {client.client_id}: {e}")
        return results

    def _collect_updates_serial(self, selected: list,
                                global_round_idx: int,
                                mode: str = "fedavg") -> list:
        """
        串行执行所有选中客户端的本地训练 / 元梯度计算。

        Args / Returns：同 _collect_updates_parallel。
        """
        edge_weights = self.model.get_weights()
        results      = []
        for client in selected:
            try:
                if mode == "fedavg":
                    result = client.local_train(global_round_idx)
                else:
                    result = client.compute_meta_gradient(edge_weights, global_round_idx)
                    grads, n, loss = result
                    if any(np.isnan(g).any() or np.isinf(g).any() for g in grads):
                        print(f"    [SKIP] Client {client.client_id}: NaN")
                        continue
                results.append(result)
            except Exception as e:
                print(f"    [ERROR] Client {client.client_id}: {e}")
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
        return self.defense.aggregate(client_updates, ref_weights)

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