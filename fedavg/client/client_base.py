"""
clients/base.py  –  FL 客户端基类

职责：纯基础设施，不含任何训练逻辑。
  - 模型/数据/优化器初始化
  - 权重管理（set_weights / _trainable_ref）
  - 测试集注入与评估（evaluate / evaluate_on）
  - local_train 声明为抽象接口，强制子类实现

子类选择：
  FedAvgClient   → clients/fedavg.py
  PFedMeClient   → clients/pfedme.py
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class FLClientBase(ABC):

    def __init__(
        self,
        client_id: int,
        dataset: tf.data.Dataset,
        model: tf.keras.Model,
        config: dict,
        n_samples: int = None,
    ):
        self.client_id = client_id
        self.dataset   = dataset
        self.model     = model
        self.config    = config



        # ── 样本数 ─────────────────────────────────────────────────────────
        self.n_samples = n_samples if n_samples is not None else sum(
            x.shape[0] for x, _ in dataset
        )

        # ── per-client 测试集（外部通过 set_test_dataset 注入） ────────────
        self.test_dataset = None

        # ── 优化器（每轮衰减一次的学习率，对齐 PFLlib 的 scheduler.step()）──────
        # 旧实现用 ExponentialDecay 按 optimizer 累计 step 衰减，且 iterations 跨轮永不
        # 归零；FedRep 的 head 优化器每个 global round 走 edge_rounds×plocal×steps 步，
        # 约每轮触发 ~25 次衰减 → 第 ~20 轮 lr 已≈0、head 被冻死 → 客户端越多越收敛不动。
        # 改为 per-round 衰减：lr(round) = lr0 · lr_decay^round。
        # lr_schedule 是一个 tf.Variable，所有优化器（base 及子类 head/base_opt）共享它，
        # 由 EdgeServerBase 在每个 global round 调 apply_round_lr() 统一更新（幂等）。
        self.lr0        = float(config["training"]["learning_rate"])
        self.lr_gamma   = float(config["training"]["lr_decay"])
        self.lr_schedule = tf.Variable(self.lr0, trainable=False, dtype=tf.float32,
                                       name=f"lr_c{client_id}")
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.lr_schedule
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction="sum_over_batch_size"
        )

        # ── 双层参考权重（由 set_weights 填入） ───────────────────────────
        self.global_weights: list | None = None
        self.edge_weights:   list | None = None

    # ══════════════════════════════════════════════════════════════════════
    # 权重管理
    # ══════════════════════════════════════════════════════════════════════

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """接收广播权重并初始化本地模型（以 edge_weights 优先）。"""
        self.global_weights = [w.copy() for w in global_weights]
        # if edge_weights is not None:
        self.edge_weights = [w.copy() for w in edge_weights]
        self.model.set_weights(edge_weights)
        print(f"  [Client {self.client_id:>2}] Received edge weights.")
        # else:
        #     self.edge_weights = None
        #     self.model.set_weights(global_weights)



    # ══════════════════════════════════════════════════════════════════════
    # 测试集管理
    # ══════════════════════════════════════════════════════════════════════

    def apply_round_lr(self, round_idx: int):
        """
        按 global round 更新学习率：lr = lr0 · lr_decay^round（每轮一次，对齐 PFLlib）。

        所有优化器都引用同一个 self.lr_schedule（tf.Variable），故只需更新它一次，
        base 优化器与子类的 head/base 优化器都会读到新值。由 EdgeServerBase 在收集
        client 更新前调用；同一 global round 内（多个 edge round）多次调用幂等。
        """
        r = max(0, int(round_idx))
        self.lr_schedule.assign(self.lr0 * (self.lr_gamma ** r))

    def set_test_dataset(self, test_dataset: tf.data.Dataset):
        """注入 per-client 同分布测试集（partition 后由 main.py 调用）。"""
        self.test_dataset = test_dataset

    def get_test_dataset(self) -> tf.data.Dataset | None:
        """返回 per-client 同分布测试集。"""
        return self.test_dataset

    # ══════════════════════════════════════════════════════════════════════
    # 训练接口（子类必须实现）
    # ══════════════════════════════════════════════════════════════════════

    @abstractmethod
    def local_train(self, round_idx: int):
        """
        执行本地训练，返回 (upload_weights, n_samples, avg_loss, elapsed_time)。

        子类在此方法内直接实现算法逻辑，不再通过 config 做运行时分发。
        """

    # ══════════════════════════════════════════════════════════════════════
    # 评估
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(self):
        """在训练集上评估当前模型，返回 (loss, accuracy)。"""
        tl = tc = tn = 0
        for x, y in self.dataset:
            p   = self.model(x, training=False)
            tl += self.loss_fn(y, p).numpy() * x.shape[0]
            tc += np.sum(np.argmax(p.numpy(), 1) == y.numpy())
            tn += x.shape[0]
        return tl / tn, tc / tn

    def evaluate_on(self, fallback_dataset: tf.data.Dataset = None):
        """
        在测试集上评估当前模型，返回 (loss, accuracy)。

        优先使用 self.test_dataset（per-client 同分布测试集）；
        未注入时退化为 fallback_dataset（全体测试集）。
        """
        # w = self.model.get_weights()
        # print(
        #     f"[C{self.client_id}] local model [0] mean={np.mean(w[0]):.6f} | "
        #     f"edge mean={np.mean(self.edge_weights[0]):.6f}"
        # )
        # self.model.set_weights(self._personalized_weights)  # 确保评估时使用当前模型权重

        # v = self.model.get_weights()
        # print(
        #     f"[C{self.client_id}] personalized model[0] mean={np.mean(v[0]):.6f} | "
        #     f"edge mean={np.mean(self.edge_weights[0]):.6f}"
        # )

        if self.test_dataset is None and fallback_dataset is None:
            raise ValueError("No test dataset for evaluation.")
        if self.test_dataset is None:
            print(
                f"  [Client {self.client_id}] No per-client test dataset, "
                f"falling back to global dataset."
            )
        ds = self.test_dataset if self.test_dataset is not None else fallback_dataset

        tl = tc = tn = 0
        all_preds, all_labels = [], []
        for x, y in ds:
            p   = self.model(x, training=False)
            tl += self.loss_fn(y, p).numpy() * x.shape[0]
            tc += np.sum(np.argmax(p.numpy(), 1) == y.numpy())
            tn += x.shape[0]
            all_preds.append(np.argmax(p.numpy(), 1))
            all_labels.append(y.numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        known_classes = sorted(set(all_labels.tolist()))

        per_class_acc = {
            c: np.mean(all_preds[all_labels == c] == c)
            for c in known_classes
        }
        print(
            f"[C{self.client_id}] n={tn} | classes={known_classes} | "
            f"acc={tc/tn:.4f} | "
            f"per_class={ {c: f'{a:.2f}' for c, a in per_class_acc.items()} }"
        )

        if tn == 0:
            return 0.0, 0.0
        return float(tl / tn), float(tc / tn)