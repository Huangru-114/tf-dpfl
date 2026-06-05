"""
defense/simple_tuning.py  –  Simple-Tuning 防御（训练后、客户端侧）

来自 Qin et al., KDD'23《Revisiting Personalized Federated Learning: Robustness
Against Backdoor Attacks》。与 5 个聚合层防御不同，Simple-Tuning 不改训练/聚合，
而是在**训练结束后**对每个（良性）客户端的最终模型做后处理：

  1. 冻结特征提取器（backbone，含卷积 / BN / head 前的 Dense）。
  2. 用 Kaiming(He) 重新初始化分类头（最后一个 Dense(num_classes) 的 kernel，bias 置 0）。
  3. 仅在本地**干净**数据上微调分类头若干 epoch。

直觉：后门主要靠「trigger 特征 → 目标类」的分类头映射实现；重置并在干净数据上重训头部
即可打断该关联，而冻结的（基本干净的）特征保持分类精度。评估侧 ASR 仍在加 trigger 的
测试集上测，得到「防御后」的本地 ASR。

实现说明：
  - 论文/官方代码（alibaba/FederatedScope, backdoor-bench 分支）抓取受限，确切
    epoch/lr 未能在线核对；这里做成可配置（默认 epochs=10、lr=本地学习率、SGD），
    与论文「lightweight」描述一致，可经 config.defense.params 调整。
  - 前向用 training=False，确保 BN 用滑动统计、Dropout 关闭 → 特征提取器真正冻结，
    梯度只回传到分类头变量。
  - 复用 models.cnn.get_base_head_indices 定位 head（最后一个 Dense(num_classes)）。
  - 用一个复用的 scratch 模型承载每个 client 的权重，避免反复构图（评估顺序串行，
    取出后立即用于 compute_asr/acc，再处理下一个 client，复用安全）。
"""

import tensorflow as tf

from models.model_utils import clone_model
from models.cnn import get_base_head_indices


class SimpleTuningDefense:

    name = "simple_tuning"

    def __init__(self, config: dict | None = None,
                 num_classes: int = 10, lr_default: float = 0.02):
        params = ((config or {}).get("defense", {}) or {}).get("params", {}) or {}
        self.num_classes = int(num_classes)
        self.epochs = int(params.get("st_epochs", 10))
        st_lr = float(params.get("st_lr", 0.0) or 0.0)
        self.lr = st_lr if st_lr > 0 else float(lr_default)   # 0/缺省 → 回退本地学习率
        self._scratch = None
        self._split   = None
        print(f"[Defense] post-hoc Simple-Tuning | epochs={self.epochs} lr={self.lr}")

    def _ensure_scratch(self, ref_model):
        if self._scratch is None:
            self._scratch = clone_model(ref_model)
            self._split = get_base_head_indices(self._scratch, self.num_classes)

    def tune(self, model: tf.keras.Model, clean_dataset) -> tf.keras.Model:
        """
        返回对 model 做 Simple-Tuning 后的模型（复用 scratch，非破坏 model 本身）。

        clean_dataset：该 client 的本地（干净）训练数据，batched (x, y) tf.data.Dataset。
        """
        self._ensure_scratch(model)
        scratch = self._scratch
        scratch.set_weights(model.get_weights())

        # ── 1+2. Kaiming 重置分类头 kernel，bias 置 0 ────────────────────────
        w = scratch.get_weights()
        he = tf.keras.initializers.HeNormal()
        for idx in self._split["head_weight_indices"]:
            arr = w[idx]
            if arr.ndim == 2:                       # kernel (in_dim, num_classes)
                w[idx] = he(arr.shape).numpy().astype(arr.dtype)
            else:                                   # bias (num_classes,)
                w[idx] = tf.zeros_like(arr).numpy()
        scratch.set_weights(w)

        # ── 3. 仅微调分类头（backbone 冻结）───────────────────────────────────
        head_vars = [scratch.trainable_variables[j]
                     for j in self._split["head_trainable_indices"]]
        if not head_vars or clean_dataset is None:
            return scratch
        opt = tf.keras.optimizers.SGD(learning_rate=self.lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        for _ in range(self.epochs):
            for x, y in clean_dataset:
                with tf.GradientTape() as tape:
                    # training=False：BN 用滑动统计、Dropout 关闭，特征提取器冻结
                    logits = scratch(x, training=False)
                    loss = loss_fn(y, logits)
                grads = tape.gradient(loss, head_vars)
                opt.apply_gradients(zip(grads, head_vars))
        return scratch
