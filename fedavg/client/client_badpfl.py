"""
client/client_badpfl.py  –  Bad-PFL（ICLR 2025）攻击 mixin

官方参考：fmy266/Bad-PFL，fba.py + generator.py。触发器 = 破坏性噪声 ξ（FGSM，抹掉
真实类特征）+ 生成器扰动 δ（Autoencoder，让样本更像目标类）：T(x) = x + ξ + δ。
生成器在每个恶意客户端**本地训练、不上传聚合**。

本实现要点
──────────
  - **是 mixin，不是 Client 子类**（CLAUDE.md 陷阱 #1）：不继承 FedAvgClient，
    与任意 PFL 方法类组合（见 client/compose.py）。恶意客户端仍跑该方法的
    local_train，动态投毒挂在 `on_batch` 钩子上。
  - `on_batch` 在 @tf.function 训练步**之外**被调用，所以 FGSM（对输入求梯度）与
    生成器前向都留在 eager，不与良性客户端的图模式互相污染。
  - 生成器训练挂在 `on_round_start`（模型此时还是收到的 edge 权重，与官方
    「先在当前模型上训触发器、再投毒训练」一致）。

与官方的实现差异（已适配本仓库）
──────────────────────────────
  - 本仓库工作在 CIFAR 标准化空间；官方 ε=σ=4/255 在 [0,1] 像素空间。
    逐通道换算：eps_norm = sigma_norm = (4/255) / STD，STD 按 data.dataset 选取
    （cifar10→CIFAR10_STD，cifar100→CIFAR100_STD），与数据管线实际归一化常数一致。
  - pgd_attack(num_iter=1) = FGSM：ξ = sigma_norm·sign(∇_x CE(model(x), y_true))。

决策 B（全阶段投毒）：`on_batch` 挂在每个方法的**每个**训练循环体上，所以 Ditto 的
阶段 A/B、FedRep 的阶段 1/2 都会被投毒。攻击者控制自己客户端的全部本地计算，这贴合
威胁模型；代价是不同方法经过的投毒 epoch 数不同，跨方法的攻击强度不可直接比较。
"""

import numpy as np
import tensorflow as tf

from models.autoencoder import build_autoencoder
from data.dataset import CIFAR10_STD, CIFAR100_STD


class BadPFLMixin:

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = (config or {}).get("backdoor", {})
        self._atk_target_label = int(bd.get("target_label", 9))
        self._atk_poison_ratio = float(bd.get("poison_ratio", 0.5))
        self._atk_gen_steps    = int(bd.get("badpfl_gen_steps", 30))
        self._atk_n_poisoned_batches = 0          # 日志/诊断用

        eps   = float(bd.get("badpfl_epsilon", 4.0 / 255.0))
        sigma = float(bd.get("badpfl_sigma",   4.0 / 255.0))
        # 归一化常数必须跟随数据管线实际使用的 STD（data/dataset.py:load_cifar*）：
        # 官方预算 4/255 定义在 [0,1] 空间，本仓库喂给模型的是标准化图，需 ÷STD 换算。
        # 硬编码 CIFAR10_STD 会让 dataset=cifar100 的 ε/σ 预算偏离约 8%（陷阱 #5）。
        _ds   = (config or {}).get("data", {}).get("dataset", "cifar10")
        std   = np.asarray(CIFAR100_STD if _ds == "cifar100" else CIFAR10_STD,
                           np.float32)                          # (C,)
        self._atk_eps_norm   = (eps   / std).astype(np.float32)
        self._atk_sigma_norm = (sigma / std).astype(np.float32)

        # 生成器懒构建：is_malicious 在构造之后才由 main.py 赋值。
        self._atk_generator = None
        self._atk_gen_opt   = None

    # ── P4：共享生成器注入（对齐官方 fba.py：adversary 持有单一 generator）────
    def set_shared_generator(self, generator, optimizer):
        """
        所有恶意端共享同一 generator + optimizer（官方 fba.py:27 用 closure 让全部
        poison client 引用同一个 trigger_gen）。语义 = adversary 持有生成模型，恶意端
        每轮 download-optimize-return：被选中时在这个共享对象上训 30 步、更新持久累积。

        由 main.py 在 build_clients 时对每个恶意端注入**同一对象**。因 badpfl 强制
        n_workers=1 串行收集，共享可变对象无线程安全问题。默认不调用（每端独立），
        只有 config.backdoor.badpfl_shared_generator=true 时 main.py 才注入。
        """
        self._atk_generator = generator
        self._atk_gen_opt   = optimizer

    # ── 生成器懒构建（未注入共享时才自建，per-client 回退）──────────────────
    def _atk_ensure_generator(self):
        if self._atk_generator is None:
            img = int(self.config["data"]["img_size"])
            self._atk_generator = build_autoencoder(img_size=img, channels=3)
            lr = float(self.config.get("backdoor", {}).get("badpfl_gen_lr", 0.01))
            self._atk_gen_opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # ── FGSM 破坏性噪声 ξ（对输入求梯度，单步）──────────────────────────────
    def _atk_fgsm_noise(self, model, x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.loss_fn(y, model(x, training=False))
        g = tape.gradient(loss, x)
        return self._atk_sigma_norm * tf.sign(g)          # ξ，(N,H,W,C)

    def _atk_gen_delta(self, x):
        """生成器扰动 δ = generator(x) · eps_norm（生成器输出 ∈ [-1,1]）。"""
        return self._atk_generator(x, training=False) * self._atk_eps_norm

    # ── 生成器训练（分类模型冻结，只更新 generator）────────────────────────
    def _atk_train_generator(self):
        self._atk_ensure_generator()
        batches = [(x, y) for x, y in self.dataset]
        if not batches:
            return
        for step in range(self._atk_gen_steps):
            x, y = batches[step % len(batches)]
            x = tf.convert_to_tensor(x, tf.float32)
            xi = self._atk_fgsm_noise(self.model, x, y)      # 破坏性噪声（模型冻结）
            tgt = tf.fill([tf.shape(x)[0]], self._atk_target_label)
            with tf.GradientTape() as tape:
                delta = self._atk_generator(x, training=True) * self._atk_eps_norm
                pred  = self.model(x + xi + delta, training=False)
                loss  = self.loss_fn(tgt, pred)              # 让加噪样本被判为 target
            grads = tape.gradient(loss, self._atk_generator.trainable_variables)
            self._atk_gen_opt.apply_gradients(
                zip(grads, self._atk_generator.trainable_variables))

    # ══════════════════════════════════════════════════════════════════════
    # 钩子
    # ══════════════════════════════════════════════════════════════════════

    def on_round_start(self, round_idx: int):
        """在收到的 edge 模型上训练触发器生成器（30 步，分类模型冻结）。"""
        super().on_round_start(round_idx)
        if not self.is_malicious:
            return
        self._atk_ensure_generator()
        self._atk_n_poisoned_batches = 0
        self._atk_train_generator()

    def on_batch(self, x, y):
        """按 poison_ratio 混合 clean / poisoned（动态投毒，eager）。"""
        x, y = super().on_batch(x, y)
        if not self.is_malicious:
            return x, y

        self._atk_ensure_generator()
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        k = int(round(n * self._atk_poison_ratio))
        if k <= 0:
            return x, y

        xi       = self._atk_fgsm_noise(self.model, x, y)
        delta    = self._atk_gen_delta(x)
        x_poison = x + xi + delta
        y_poison = tf.fill([n], tf.cast(self._atk_target_label, y.dtype))
        # 投毒样本的选取走客户端自己的 seeded RNG（self.rng，见 client_base）。
        # 不用全局 np.random —— 线程池里全局 RNG 的消耗顺序不确定，实验不可复现。
        mask = np.zeros(n, dtype=bool)
        mask[:k] = True
        self.rng.shuffle(mask)
        m4 = tf.constant(mask.reshape(-1, 1, 1, 1))
        self._atk_n_poisoned_batches += 1
        return tf.where(m4, x_poison, x), tf.where(tf.constant(mask), y_poison, y)

    # ── 评估侧触发器：用被评估模型自身做 FGSM + 本客户端 generator 生成 δ ───
    def eval_trigger(self, model, x, y=None):
        """trigger_fn(model, x, y) -> 加触发器的 numpy x（model-dependent）。"""
        self._atk_ensure_generator()
        x_tf = tf.convert_to_tensor(x, tf.float32)
        if y is None:                       # 无标签时退化为用预测类做 FGSM
            y = np.argmax(model(x_tf, training=False).numpy(), axis=1)
        xi    = self._atk_fgsm_noise(model, x_tf, y)
        delta = self._atk_gen_delta(x_tf)
        return (x_tf + xi + delta).numpy()
