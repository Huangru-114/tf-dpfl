"""
client/client_iba.py  –  IBA（Irreversible Backdoor Attack, NeurIPS 2023）攻击 mixin

官方参考：sail-research/iba，`fl_trainer.py` + `iba_helpers.py` + `attack_models/autoencoders.py`。
IBA 基于 LIRA 的**可学习触发器**：触发器由一个 autoencoder 生成器 G 产生，
    noise = G(x) · atk_eps ;  x_poison = clip(x + noise, MIN, MAX)
G 输出经 tanh ∈ [-1,1]，乘 atk_eps 得到有界扰动；隐蔽性全靠这个 eps 预算。
「irreversible」= 生成器跨轮持久演化、后门逐步焊进全局模型（靠 set_shared_generator
+ 客户端跨轮存活累积，机制已在 Bad-PFL exp007 上验证）。

与 Bad-PFL 的关键区别（务必分清）
──────────────────────────────
  - **图像空间**：IBA 直接在**标准化空间**操作，eps 就是标准化空间里的量
    （官方 `iba_helpers.py:get_clip_image` 把图 clip 到 IMAGENET_MIN/MAX）。
    因此 **eps 不做** Bad-PFL 那个 ÷STD 换算。这里 MIN/MAX 逐通道由数据管线的
    归一化常数算出：MIN=(0-mean)/std，MAX=(1-mean)/std（data/dataset.py:load_cifar*）。
  - **target-network 双缓冲**（LIRA）：两个生成器 atkmodel / tgtmodel。投毒用 tgtmodel
    （冻结、稳定目标）；每轮把 atk←tgt 训练一次再 tgt←atk 回写（官方
    `fl_trainer.py:1066-1070`）。避免用同一个正在更新的生成器投毒导致不稳。
  - **无 FGSM 破坏噪声**：IBA 触发器就是纯 G(x)（model-agnostic），不含 Bad-PFL 的 ξ。
    → eval 侧触发器与被评估模型无关，迁移性天然更好。

本实现要点
──────────
  - **是 mixin，不是 Client 子类**（CLAUDE.md 陷阱 #1）：不继承 FedAvgClient，
    与任意 PFL 方法类组合（client/compose.py），恶意端仍跑该方法的 local_train。
  - 复用 `models/autoencoder.py:build_autoencoder`（与官方 CIFAR `Autoencoder` 同族，
    tanh 有界、img_size≥16），不新写网络结构。
  - `on_round_start`（收到 edge 模型、未训练时）训练生成器；`on_batch`（@tf.function
    步之外、eager）动态投毒；`on_upload`（纯函数）做净增1 的固定系数缩放。

第一版为「打通端到端」做的忠实性妥协（见 plan / current-focus.md，待回补）
─────────────────────────────────────────────────────────────────────
  - **决策 C①**：`on_batch` 批内混合、单前向（poison_ratio ≈ 1-α），不做官方
    clean/poison 双前向的 α 加权组合损失（`fl_trainer.py:457`）。
  - **决策 D①**：eps 固定 `atk_eps`、直接投毒，不做两阶段 started_poisoning 门 /
    指数退火 / alternative_training 交替（`fl_trainer.py:1034-1041`）。
  - **净增1**：`iba_scale_weights_poison` 固定系数（默认 1.0=关），锚 edge_weights、
    只替换 edge 层；自适应 γ=N/n、全局替换、eps 门 = 后续单变量。
"""

import numpy as np
import tensorflow as tf

from models.autoencoder import build_autoencoder
from data.dataset import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD


class IBAMixin:

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = (config or {}).get("backdoor", {})
        self._atk_target_label = int(bd.get("target_label", 9))
        self._atk_poison_ratio = float(bd.get("poison_ratio", 0.5))
        self._atk_eps          = float(bd.get("iba_eps", 0.01))     # 标准化空间预算
        self._atk_gen_steps    = int(bd.get("iba_gen_steps", 30))
        self._atk_gen_lr       = float(bd.get("iba_gen_lr", 0.01))
        # 净增1：固定系数 model-replacement（默认 1.0 = 关 = 恒等）
        self._atk_scale_poison = float(bd.get("iba_scale_weights_poison", 1.0))
        self._atk_n_poisoned_batches = 0                            # 日志/诊断用

        # ── clip 边界（标准化空间的合法像素范围，逐通道）─────────────────────
        # 数据管线：x = (x/255 - mean) / std → 合法范围 [(0-mean)/std, (1-mean)/std]。
        # 与 data/dataset.py 的归一化常数保持一致（cifar10/cifar100 各取各的）。
        _ds = (config or {}).get("data", {}).get("dataset", "cifar10")
        mean = np.asarray(CIFAR100_MEAN if _ds == "cifar100" else CIFAR10_MEAN, np.float32)
        std  = np.asarray(CIFAR100_STD  if _ds == "cifar100" else CIFAR10_STD,  np.float32)
        self._clip_min = tf.constant(((0.0 - mean) / std), tf.float32)   # (C,)
        self._clip_max = tf.constant(((1.0 - mean) / std), tf.float32)   # (C,)

        # ── 生成器双缓冲：懒构建（is_malicious 在构造后才赋值）────────────────
        self._atk_atkmodel = None    # 被优化的生成器
        self._atk_tgtmodel = None    # 投毒用的冻结目标生成器（target-network）
        self._atk_gen_opt  = None

    # ── P4：共享生成器注入（对齐官方：adversary 持有单一 atk/tgt/optimizer）────
    def set_shared_generator(self, atkmodel, tgtmodel, optimizer):
        """
        所有恶意端共享同一 (atkmodel, tgtmodel, optimizer)，跨轮持久累积
        （官方 `use_our_attack` 里 atk/tgt 只建一次、全恶意端共享）。由 main.py 在
        build_clients 时对每个恶意端注入**同一组对象**；badpfl 同款机制、n_workers=1
        串行故无线程安全问题。默认不调用（每端独立），只有
        config.backdoor.iba_shared_generator=true 时 main.py 才注入。
        """
        self._atk_atkmodel = atkmodel
        self._atk_tgtmodel = tgtmodel
        self._atk_gen_opt  = optimizer

    # ── 生成器懒构建（未注入共享时才自建，per-client 回退）──────────────────
    def _atk_ensure_models(self):
        if self._atk_tgtmodel is None:
            img = int(self.config["data"]["img_size"])
            self._atk_atkmodel = build_autoencoder(img_size=img, channels=3)
            self._atk_tgtmodel = build_autoencoder(img_size=img, channels=3)
            self._atk_tgtmodel.set_weights(self._atk_atkmodel.get_weights())  # 初始对齐
            self._atk_gen_opt  = tf.keras.optimizers.Adam(learning_rate=self._atk_gen_lr)

    def _atk_clip(self, x):
        """裁剪到标准化空间的逐通道合法范围。"""
        return tf.clip_by_value(x, self._clip_min, self._clip_max)

    def _atk_apply_trigger(self, generator, x):
        """x_poison = clip(x + G(x)·atk_eps)，G 输出 ∈ [-1,1]（tanh）。"""
        noise = generator(x, training=False) * self._atk_eps
        return self._atk_clip(x + noise)

    # ── 生成器再训练（LIRA target-network：atk←tgt，训 atk，tgt←atk）────────
    def _atk_train_generator(self):
        """在当前（冻结的）分类模型上训练生成器，让加触发器样本被判为 target。"""
        self._atk_ensure_models()
        batches = [(x, y) for x, y in self.dataset]
        if not batches:
            return
        # atk ← tgt：从稳定目标出发再优化（官方 fl_trainer.py:1067）
        self._atk_atkmodel.set_weights(self._atk_tgtmodel.get_weights())
        for step in range(self._atk_gen_steps):
            x, _ = batches[step % len(batches)]
            x = tf.convert_to_tensor(x, tf.float32)
            tgt = tf.fill([tf.shape(x)[0]], self._atk_target_label)
            with tf.GradientTape() as tape:
                noise = self._atk_atkmodel(x, training=True) * self._atk_eps
                pred  = self.model(self._atk_clip(x + noise), training=False)
                loss  = self.loss_fn(tgt, pred)
            grads = tape.gradient(loss, self._atk_atkmodel.trainable_variables)
            self._atk_gen_opt.apply_gradients(
                zip(grads, self._atk_atkmodel.trainable_variables))
        # tgt ← atk：回写目标网络（官方 fl_trainer.py:1070）
        self._atk_tgtmodel.set_weights(self._atk_atkmodel.get_weights())

    # ══════════════════════════════════════════════════════════════════════
    # 钩子
    # ══════════════════════════════════════════════════════════════════════

    def on_round_start(self, round_idx: int):
        """在收到的 edge 模型上训练生成器（分类模型冻结），更新 target-network。"""
        super().on_round_start(round_idx)
        if not self.is_malicious:
            return
        self._atk_ensure_models()
        self._atk_n_poisoned_batches = 0
        self._atk_train_generator()

    def on_batch(self, x, y):
        """按 poison_ratio 批内混合 clean / poisoned（决策 C①：单前向，eager）。"""
        x, y = super().on_batch(x, y)
        if not self.is_malicious:
            return x, y

        self._atk_ensure_models()
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        k = int(round(n * self._atk_poison_ratio))
        if k <= 0:
            return x, y

        x_poison = self._atk_apply_trigger(self._atk_tgtmodel, x)   # 投毒用 tgtmodel
        y_poison = tf.fill([n], tf.cast(self._atk_target_label, y.dtype))
        # 投毒样本选取走客户端自己的 seeded RNG（陷阱 #2），不碰全局 np.random。
        mask = np.zeros(n, dtype=bool)
        mask[:k] = True
        self.rng.shuffle(mask)
        m4 = tf.constant(mask.reshape(-1, 1, 1, 1))
        self._atk_n_poisoned_batches += 1
        return tf.where(m4, x_poison, x), tf.where(tf.constant(mask), y_poison, y)

    def on_upload(self, upload: list, round_idx: int):
        """
        净增1：固定系数 model-replacement。放大「本地更新量」γ 倍，锚在收到的 edge
        权重上：new = edge + (upload - edge)·γ（官方 fl_trainer.py:1129 同式）。
        γ=1.0（默认）时恒等，良性/关闭路径数值不变。

        纯函数 —— 只返回新列表，不碰 self.model（陷阱 #4）。第一版只替换 edge 层，
        缩放后的更新照常过 robust_mean；防御≠none 时被剔除是**预期研究观测**，非 bug。
        """
        upload = super().on_upload(upload, round_idx)               # 先走内层（防御 mixin）
        if not self.is_malicious or self._atk_scale_poison == 1.0:
            return upload
        ref = self.edge_weights
        if ref is None or len(ref) != len(upload):
            print(f"  [Client {self.client_id:>2}] IBA: edge_weights 不可用或长度不匹配，"
                  f"跳过缩放。")
            return upload
        g = self._atk_scale_poison
        scaled = [r + (u - r) * g for r, u in zip(ref, upload)]
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"IBA model-replacement (scale={g:g}, ref=edge_weights)")
        return scaled

    # ── 评估侧触发器：纯 G(x)（model-agnostic），用共享 tgtmodel 生成 ─────────
    def eval_trigger(self, model, x, y=None):
        """trigger_fn(model, x, y) -> 加触发器的 numpy x。IBA 触发器与被评估模型无关。"""
        self._atk_ensure_models()
        x_tf = tf.convert_to_tensor(x, tf.float32)
        return self._atk_apply_trigger(self._atk_tgtmodel, x_tf).numpy()
