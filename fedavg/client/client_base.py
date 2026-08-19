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

        # ── 独立播种的 RNG（seed, client_id）─────────────────────────────
        # 客户端侧的一切随机（投毒样本选取、触发器采样…）都走这条流，
        # 不碰全局 np.random。全局 RNG 的状态取决于此前所有调用的次数和顺序，
        # 在线程池里更是不确定 → 「固定种子重跑一致」根本不成立。
        self.rng = np.random.default_rng(
            [int((config or {}).get("seed", 42)), int(client_id)])



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

        # ── 行为组合的挂载点（见 client/compose.py 与下方「客户端钩子协议」）──
        # is_malicious 由 main.py 在**构造之后**赋值，故这里给 False 兜底，
        # 让 mixin 里可以直接读 self.is_malicious 而不必到处写 getattr(...)。
        self.is_malicious = False
        # 服务器下发的额外载荷（主动防御用），由 set_control 填。
        self._control: dict = {}

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
    # 客户端钩子协议（默认全部无操作 —— 良性、无防御路径数值零变化）
    # ══════════════════════════════════════════════════════════════════════
    #
    # **为什么钩子名是中性的 `on_*` 而不是 `_atk_*`**：攻击轴和防御轴用的是
    # 同一套钩子。CerP 的 α/β 正则项和未来某个主动防御的近端项都落在
    # on_extra_loss 上；Neurotoxin 的掩码投影和「客户端侧范数裁剪」这类主动
    # 防御都落在 on_upload 上。按攻击专用命名会导致加防御时再改一遍全部方法文件。
    #
    # 组合方式见 client/compose.py：
    #     MRO = AttackMixin → DefenseMixin → MethodCls → FLClientBase
    # 防御 mixin 挂给**所有**客户端（防御方分不出谁是恶意的），攻击 mixin 只挂
    # 恶意客户端且在外层 —— 恶意客户端有权把防御要求的客户端侧动作做假或跳过。
    #
    # 每个 mixin 的实现都必须调 super()（除非它就是要「不守协议」），
    # 使多个 mixin 能沿 MRO 串起来。

    def set_control(self, control: dict):
        """
        接收服务器下发的额外载荷（主动防御用：裁剪上界、挑战样本、参考统计量…）。

        由 EdgeServerBase.broadcast_to_clients 在每次广播时调用 —— 这也是
        「下行必须走 broadcast_to_clients」的理由（守卫见
        tests/test_broadcast_coverage.py）。基类只存下来，不解释内容。
        """
        self._control = dict(control or {})

    def on_round_start(self, round_idx: int):
        """本地训练开始前（已收到 edge 权重、尚未训练）。默认无操作。"""

    def on_batch(self, x, y):
        """
        每个训练 batch 送进 @tf.function 步骤**之前**的变换点。默认恒等。

        动态投毒（Bad-PFL / CerP）挂在这里：它们要对输入求梯度、要更新触发器
        变量，必须留在 eager，所以钩子在 Python 循环体里、不在 @tf.function 内。
        """
        return x, y

    def on_extra_loss(self):
        """
        叠加到「产出 upload 的那个训练步」的 tape 内的额外损失项（标量）。默认 0。

        只挂产出 upload 的那一步（而非每一步）：CerP 的两个正则项是为了让
        **上传物**躲过鲁棒聚合，加在个性化阶段没有意义。
        返回 0.0 时 `loss + 0.0` 对 float32 是精确恒等，良性路径数值不变。
        """
        return tf.constant(0.0, dtype=tf.float32)

    def on_upload(self, upload: list, round_idx: int):
        """
        返回给 edge 之前，对**上传权重列表**的最后一次变换。默认恒等。

        **必须是纯函数**：只返回新列表，不许 self.model.set_weights(...)。
        六个 PFL 方法里有五个的 upload ≠ self.model（pFedMe 传锚点 Θ、Ditto 传
        阶段 A 的 w_k、Rep 家族传 backbone+私有 head），写回会污染个性化模型，
        破坏 PM 评估 / 遗忘曲线 / local ASR 三处指标。
        守卫见 tests/test_client_hooks.py::test_on_upload_is_pure。
        """
        return upload

    def get_aux(self) -> dict:
        """
        随权重一起交给服务器的额外载荷（主动防御用：自检结果、签名、探针损失…）。

        由 EdgeServerBase._finalize_updates 收集进 ClientUpdate.aux。默认空。
        """
        return {}

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