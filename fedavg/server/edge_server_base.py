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
from aggregation.client_update import as_client_update
from .robust_aggregation import RobustAggregationMixin
from .participation import edge_quota, round_index
from alignment import get_switch


class EdgeServerBase(RobustAggregationMixin, ABC):

    def __init__(self, edge_id: int, clients: list,
                 model: tf.keras.Model, config: dict):
        self.edge_id   = edge_id
        self.clients   = clients
        self.model     = model
        self.config    = config
        self.n_samples = sum(c.n_samples for c in clients)

        # ── 后门防御（鲁棒聚合）。None=不设防 → robust_mean 回退普通加权平均。──
        # robust_mean 现在来自 RobustAggregationMixin（cloud 层共用同一份实现）。
        self._init_defense(config, layer="edge")

        # ── 独立播种的 RNG：客户端选取必须可复现，否则矩阵里同一格重跑对不上。
        # 每个 edge 一条独立流（seed, edge_id），互不干扰全局 np.random。
        self.rng = np.random.default_rng([int(config.get("seed", 42)), int(edge_id)])

        # ── 轮号轴（A08 lr / D02 配额；不写 = 云轮号，旧行为）──────────────
        self._edge_rounds = int(config["federation"].get("edge_rounds", 1) or 1)
        self._lr_axis     = get_switch(config, "training.lr_round_axis")
        self._quota_axis  = get_switch(config, "federation.quota_round_axis")

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

    def select_clients(self, round_idx: int, edge_round_idx: int = None) -> list:
        """
        用本 edge 的 seeded RNG 抽取参与客户端。

        用索引抽样（而不是 np.random.choice(self.clients)）：对象数组抽样在 numpy
        新版本里会告警，且索引抽样让选取结果只依赖 (seed, edge_id, 调用次数)，
        与客户端对象的内存布局无关 → 可复现。

        名额按哪个轮号轮转由 federation.quota_round_axis 决定（D02 / D-036）：
        cloud（旧）= 云轮号；effective = 有效轮（需要 edge_round_idx）。
        """
        n_select = self._n_select(
            round_index(round_idx, edge_round_idx, self._edge_rounds, self._quota_axis))
        idx      = self.rng.choice(len(self.clients), n_select, replace=False)
        return [self.clients[int(i)] for i in idx]

    def _n_select(self, round_idx: int) -> int:
        """
        本 edge 本轮的抽样名额 —— 算术在 `server/participation.edge_quota`。

        单独成模块是为了让它**不依赖 TF**、能在本地秒级测到：这条配额决定了
        Experiment 3 的拓扑轴干不干净（旧实现下 4-edge 的格子少训 20%）。
        """
        fed = self.config["federation"]
        return edge_quota(
            self.edge_id, round_idx,
            n_clients=int(fed.get("n_clients", 0) or 0),
            n_edges=int(fed.get("n_edges", 0) or 0),
            client_fraction=float(fed["client_fraction"]),
            n_local_clients=len(self.clients),
        )

    def broadcast_to_clients(self, selected: list, global_weights: list = None,
                             round_idx: int = 0):
        """
        广播 edge 模型权重给选中客户端 —— **下行的唯一通道**。

        global_weights：同时转发给 client，供 FedProx(global)、FedDyn 等使用；
                        为 None 时退化为 edge 权重。

        **为什么必须所有 edge server 都走这里**（曾经 6 个里只有 1 个走）：
        主动防御要给客户端下发额外载荷（裁剪上界、挑战样本、参考统计量），
        写在这里才能到达所有 PFL 方法。5/6 的 edge server 曾经内联
        `for client in selected: client.set_weights(...)` 绕开本方法 → 那些方法下
        任何下行载荷**静默送不到**，而日志照样打印 [Defense] enabled。
        守卫：tests/test_broadcast_coverage.py（AST 静态检查）。
        """
        edge_weights = self.model.get_weights()
        gw           = global_weights if global_weights is not None else edge_weights
        control      = self._make_control(round_idx)
        for client in selected:
            client.set_weights(global_weights=gw, edge_weights=edge_weights)
            client.set_control(control)

    def _make_control(self, round_idx: int) -> dict:
        """
        本轮下发给客户端的额外载荷。无防御或防御不需要客户端配合时为空 dict。

        空载荷时 `client.set_control({})` 只是把 `self._control` 置空，不产生任何
        数值影响 —— 良性、无防御路径与改动前逐元素相同。
        """
        if getattr(self, "defense", None) is None:
            return {}
        return self.defense.make_control(self, round_idx)

    # ══════════════════════════════════════════════════════════════════════
    # 并行 / 串行收集 client 更新
    # ══════════════════════════════════════════════════════════════════════

    def _collect_updates_parallel(self, selected: list,
                                  global_round_idx: int,
                                  mode: str = "fedavg",
                                  edge_round_idx: int = None) -> list:
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
        lr_idx = self._lr_round(global_round_idx, edge_round_idx)

        def run_one(client):
            # 每轮更新学习率（per-round 衰减，对齐 PFLlib）；轮号轴见 _lr_round（A08）
            if hasattr(client, "apply_round_lr"):
                client.apply_round_lr(lr_idx)
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
                                mode: str = "fedavg",
                                edge_round_idx: int = None) -> list:
        """串行版本。不变量与返回值同 _collect_updates_parallel。"""
        if mode != "fedavg":
            raise ValueError(
                f"_collect_updates_serial 只支持 mode='fedavg'，收到 {mode!r}。")

        lr_idx = self._lr_round(global_round_idx, edge_round_idx)
        raw = []
        for client in selected:
            try:
                if hasattr(client, "apply_round_lr"):
                    client.apply_round_lr(lr_idx)
                raw.append((client, client.local_train(global_round_idx)))
            except Exception as e:
                print(f"    [ERROR] Client {client.client_id}: {e}")
                raw.append((client, None))
        return self._finalize_updates(raw)

    def _lr_round(self, global_round_idx: int, edge_round_idx) -> int:
        """
        lr 衰减用的轮号（training.lr_round_axis，A08 / D-023）：
          cloud（旧）   = 云轮号 g → 同一有效轮上 flat 与 HFL 的 lr 不同（F-003 的混淆）
          effective     = 有效轮 t_eff = (g−1)·R + er；R=1 时等于 g，flat 逐字节不变
        攻击时间窗（attacking）仍按云轮，不受这里影响。
        """
        return round_index(global_round_idx, edge_round_idx, self._edge_rounds, self._lr_axis)

    def _finalize_updates(self, raw: list) -> list:
        """
        把 [(client, result|None)] 规范成 [ClientUpdate]，并记录失败的客户端。

        同时收集客户端的上行额外载荷（`client.get_aux()` → `ClientUpdate.aux`），
        供主动防御在聚合时读取。基类 get_aux 返回 {}，不影响任何现有路径。
        """
        results, failed = [], []
        for client, result in raw:
            if result is None:
                failed.append(int(client.client_id))
                continue
            results.append(as_client_update(result, client.client_id,
                                            aux=client.get_aux()))
        self.last_failed_ids = failed
        if failed:
            print(f"    [Edge {self.edge_id}] {len(failed)} 个客户端本轮失败并被丢弃: "
                  f"{failed}")
        return results


    # ══════════════════════════════════════════════════════════════════════
    # 鲁棒聚合（防御统一入口）
    # ══════════════════════════════════════════════════════════════════════
    #
    # `robust_mean` 来自 RobustAggregationMixin（server/robust_aggregation.py），
    # 与 CloudServer 共用同一份实现。各 edge server 把自己原本的加权平均
    # （FedAvg 整体 / pFedMe·Ditto 的 mean / FedRep 的 backbone mean）改调它，
    # 一套防御即覆盖所有 PFL 方法。守卫：tests/test_defense_coverage.py。

    def _default_ref_weights(self) -> list:
        """本层聚合前的模型权重（广播点）= 当前 edge 模型权重。"""
        return self.model.get_weights()

    def _defense_label(self) -> str:
        return f"edge{self.edge_id}"

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
    def run_edge_round(self, global_round_idx: int, edge_round_idx: int):
        """
        一个 edge 轮：选客户端 → 广播 → 本地训练 → 聚合。返回 (avg_loss, avg_time)。

        实现里 select_clients 与 _collect_updates_* **都必须**传 edge_round_idx
        （A08 / D02 的有效轮轴要它；守卫 tests/test_schedule_alignment.py）。
        """

    def upload_weights(self) -> list:
        """交给 cloud 的权重。默认 = edge 模型；Hier-pFedMe 覆写为 W_n（Algorithm 1 第 19 行）。"""
        return self.model.get_weights()

    def cloud_upload(self, round_losses: list, round_times: list):
        """一个云周期的 edge 轮都跑完之后，交给 cloud 的 5 元组（两种调度共用）。"""
        edge_rounds = self.config["federation"]["edge_rounds"]
        comm = 2 * self.model_bytes * len(self.clients) * edge_rounds
        return (self.upload_weights(), self.n_samples,
                float(np.mean(round_losses)), float(np.mean(round_times)), comm)

    def run(self, global_round_idx: int):
        """
        顺序调度（federation.edge_schedule=sequential，旧行为）：本 edge 连续跑完
        edge_rounds 轮再上传。交错调度由 CloudServer.collect_and_aggregate 直接驱动
        run_edge_round（D01），两者都经 cloud_upload 上传。

        Returns:
            (upload_weights, n_samples, avg_loss, avg_time, comm_bytes)
        """
        edge_rounds = self.config["federation"]["edge_rounds"]
        round_losses, round_times = [], []
        for er in range(1, edge_rounds + 1):
            loss, t = self.run_edge_round(global_round_idx, er)
            round_losses.append(loss)
            round_times.append(t)
        return self.cloud_upload(round_losses, round_times)