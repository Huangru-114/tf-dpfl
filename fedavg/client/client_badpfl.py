"""
client/client_badpfl.py  –  Bad-PFL（ICLR 2025）攻击 mixin

官方参考：fmy266/Bad-PFL，fba.py + generator.py。触发器 = 破坏性噪声 ξ（抹掉
真实类特征）+ 生成器扰动 δ（Autoencoder，让样本更像目标类）：T(x) = x + ξ + δ。
生成器在恶意客户端**本地训练、不上传聚合**。

本实现要点
──────────
  - **是 mixin，不是 Client 子类**（CLAUDE.md 陷阱 #1）：不继承 FedAvgClient，
    与任意 PFL 方法类组合（见 client/compose.py）。恶意客户端仍跑该方法的
    local_train，动态投毒挂在 `on_batch` 钩子上。
  - `on_batch` 在 @tf.function 训练步**之外**被调用，所以求 ξ（对输入求梯度）与
    生成器前向都留在 eager，不与良性客户端的图模式互相污染。
  - 生成器训练挂在 `on_round_start`（模型此时还是收到的 edge 权重，与官方
    「先在当前模型上训触发器、再投毒训练」一致）。

对齐开关（AUDIT A1 那几行，fedavg/alignment.py；不写 = 旧行为）
────────────────────────────────────────────────────────────────
  backdoor.badpfl_xi        fgsm（旧）：ξ = σ·sign(∇ₓCE(F(x), y))，在 x 处、无随机起点、无 clamp
                            pgd（A01 / D-014，官方 fba.py:6-22）：
                              x₀ = clamp(x + u, lo, hi)，u ~ U(−σ, σ)（客户端 seeded rng）
                              x₁ = x₀ + σ·sign(∇CE(F(x₀), y))      ← 梯度在**起点**处求
                              adv = clamp(x + clamp(x₁ − x, −σ, σ), lo, hi)
                              ξ = adv − x；之后 +δ **不再** clamp（fba.py:55）
                            生成器训练 / 训练期投毒 / 评估三处共用。
  backdoor.poison_sampling  exact_k（旧）/ bernoulli（A03 / D-016），见 attack/poison_mask.py
  backdoor.badpfl_bn_mode   inference（旧）：G 与 F 在投毒 / 评估时都用推理模式
                            official（A05 / D-018）：
                              #1 训 G：G train                     （两种模式一致）
                              #2 训 G：F inference                  （两种模式一致）
                              #3 训练期投毒：G train（整批 batch 统计）
                              #4 训练期投毒：F train（求 ξ 时；与官方一样会顺带更新 F 的 BN 统计量）
                              #5 评估：G train，按 32 张分块，尾块用本探针自己的样本循环补足到 32，
                                 只取真实样本的输出（A4 会话用户拍板）
                              #6 评估：求 ξ 的攻击者 F 用 inference（**偏离官方**：官方此处取决于
                                 客户端列表顺序，是偶然行为）
  backdoor.badpfl_generator legacy（旧）/ official（A14 / D-020）：结构见 models/autoencoder.py；
                            official 下生成器吃 [0,1] 图（标准化图先反变换）。

输入空间（data/pixel_space.py）：官方 ε=σ=4/255 定义在 [0,1]；本仓库默认喂标准化图，
逐通道换算 σ_in = σ/s、合法范围 [(0−μ)/s, (1−μ)/s]（sign 在正仿射变换下不变）。
data.normalize=false（G7）时换算退化为恒等（F-027）。

决策 B（全阶段投毒）：`on_batch` 挂在每个方法的**每个**训练循环体上，所以 Ditto 的
阶段 A/B 都会被投毒。FedRep 例外：`training.fedrep_poison_phases=body`（A24 / D-021）
时 head 阶段不走 `on_batch`（见 client/hier_fedrep.py）。
"""

import numpy as np
import tensorflow as tf

from alignment import get_switch
from attack.poison_mask import poison_mask
from data.pixel_space import pixel_stats, to_input_space, valid_range
from models.autoencoder import build_generator

# 评估时生成器按多大的块用 batch 统计（官方测试 loader 的 batch_size=32，A05 #5）
EVAL_CHUNK = 32


class BadPFLMixin:

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = (config or {}).get("backdoor", {})
        self._atk_target_label = int(bd.get("target_label", 9))
        self._atk_poison_ratio = float(bd.get("poison_ratio", 0.5))
        self._atk_gen_steps    = int(bd.get("badpfl_gen_steps", 30))
        self._atk_n_poisoned_batches = 0          # 日志/诊断用

        # ── 对齐开关（不写 = 旧行为）──────────────────────────────────────
        self._atk_xi_mode   = get_switch(config, "backdoor.badpfl_xi")
        self._atk_sampling  = get_switch(config, "backdoor.poison_sampling")
        self._atk_bn_mode   = get_switch(config, "backdoor.badpfl_bn_mode")
        self._atk_gen_kind  = get_switch(config, "backdoor.badpfl_generator")

        eps   = float(bd.get("badpfl_epsilon", 4.0 / 255.0))
        sigma = float(bd.get("badpfl_sigma",   4.0 / 255.0))
        # 官方预算 4/255 定义在 [0,1] 像素空间；模型输入是标准化图时逐通道 ÷STD。
        # STD 跟随数据管线实际用的常数（dataset 对应的那一组；关标准化时是 1）。
        self._atk_eps_norm   = to_input_space(eps, config)
        self._atk_sigma_norm = to_input_space(sigma, config)
        self._atk_lo, self._atk_hi = valid_range(config)
        self._atk_mean, self._atk_std = pixel_stats(config)

        # 评估期 PGD 起点噪声的独立流：评估频率不能改变训练（不消耗 self.rng）。
        self._atk_eval_rng = np.random.default_rng(
            [int((config or {}).get("seed", 42)), int(client_id), 0xE7A1])

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
            self._atk_generator = build_generator(self.config, channels=3)
            lr = float(self.config.get("backdoor", {}).get("badpfl_gen_lr", 0.01))
            self._atk_gen_opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # ══════════════════════════════════════════════════════════════════════
    # ξ：破坏性噪声
    # ══════════════════════════════════════════════════════════════════════
    def _atk_fgsm_noise(self, model, x, y):
        """旧口径（badpfl_xi=fgsm）：在 x 处单步 FGSM，无随机起点、无 clamp。"""
        x = tf.convert_to_tensor(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.loss_fn(y, model(x, training=False))
        g = tape.gradient(loss, x)
        return self._atk_sigma_norm * tf.sign(g)          # ξ，(N,H,W,C)

    def _atk_pgd_noise(self, model, x, y, *, training, rng=None, u=None):
        """官方单步 PGD（A01 / D-014）。返回 ξ = adv − x。

        u：起点噪声（已在输入空间、|u| ≤ σ_in），测试可直接注入；缺省用 rng 采。
        training：F 前向的模式（A05：训练期投毒 official 下为 True，其余 False）。
        """
        x = tf.convert_to_tensor(x, tf.float32)
        sig = tf.constant(self._atk_sigma_norm)
        lo, hi = tf.constant(self._atk_lo), tf.constant(self._atk_hi)
        if u is None:
            r = rng if rng is not None else self.rng
            u = (r.uniform(-1.0, 1.0, size=tuple(x.shape)).astype(np.float32)
                 * self._atk_sigma_norm)
        x0 = tf.clip_by_value(x + tf.convert_to_tensor(u, tf.float32), lo, hi)
        with tf.GradientTape() as tape:
            tape.watch(x0)
            loss = self.loss_fn(y, model(x0, training=training))
        g = tape.gradient(loss, x0)
        x1 = x0 + sig * tf.sign(g)
        eta = tf.clip_by_value(x1 - x, -sig, sig)
        adv = tf.clip_by_value(x + eta, lo, hi)
        return adv - x

    def _atk_xi(self, model, x, y, *, training=False, rng=None):
        if self._atk_xi_mode == "pgd":
            return self._atk_pgd_noise(model, x, y, training=training, rng=rng)
        return self._atk_fgsm_noise(model, x, y)

    # ══════════════════════════════════════════════════════════════════════
    # δ：生成器扰动
    # ══════════════════════════════════════════════════════════════════════
    def _atk_gen_input(self, x):
        """official 生成器吃 [0,1] 图（A14）；legacy 直接吃模型输入。"""
        if self._atk_gen_kind == "official":
            return x * self._atk_std + self._atk_mean
        return x

    def _atk_gen_delta(self, x, training=False):
        """δ = generator(x) · eps_in（生成器输出 ∈ [-1,1]）。"""
        x = tf.convert_to_tensor(x, tf.float32)
        return (self._atk_generator(self._atk_gen_input(x), training=training)
                * self._atk_eps_norm)

    def eval_delta(self, x):
        """评估侧 δ（A05 #5）。

        inference（旧）：整个探针一次推理模式前向。
        official：G 用 batch 统计，按 32 张分块；尾块用本探针自己的样本循环补足到 32，
        只取真实样本的输出 → 每个样本都计入 ASR，batch 统计的规模恒为 32。
        δ 只依赖输入，不依赖受害者 / 攻击者模型 → 各评估列可复用同一份。
        """
        self._atk_ensure_generator()
        x = np.asarray(x, np.float32)
        if self._atk_bn_mode != "official":
            return self._atk_gen_delta(x, training=False).numpy()
        n = len(x)
        out = np.empty_like(x)
        for s in range(0, n, EVAL_CHUNK):
            idx = np.arange(s, s + EVAL_CHUNK) % n          # 尾块循环补足（探针 < 32 张也成立）
            real = min(EVAL_CHUNK, n - s)
            d = self._atk_gen_delta(x[idx], training=True).numpy()
            out[s:s + real] = d[:real]
        return out

    # ══════════════════════════════════════════════════════════════════════
    # 生成器训练（分类模型冻结，只更新 generator）
    # ══════════════════════════════════════════════════════════════════════
    def _atk_gen_batches(self):
        """生成器训练取 `badpfl_gen_steps` 个批。

        data.batch_pipeline=per_epoch（A25）：与本地训练走同一个取数函数
        （`epoch_batches`，一个 epoch 不够就接着下一个 epoch —— 对应官方的持久迭代器
        client.py:48-57）；legacy：旧行为，读一遍 tf.data 循环取批。
        """
        if self.uses_epoch_pipeline():
            out = []
            while len(out) < self._atk_gen_steps:
                got = list(self.epoch_batches())
                if not got:
                    break
                out.extend(got)
            return out[:self._atk_gen_steps]
        batches = [(x, y) for x, y in self.dataset]
        if not batches:
            return []
        return [batches[step % len(batches)] for step in range(self._atk_gen_steps)]

    def _atk_train_generator(self):
        self._atk_ensure_generator()
        # 干净批次、真实标签（A04 / D-017：对齐论文 Eq.7，偏离官方代码）——
        # on_batch 不在这条路径上，生成器训练永远看不到投毒样本。
        for x, y in self._atk_gen_batches():
            x = tf.convert_to_tensor(x, tf.float32)
            xi = self._atk_xi(self.model, x, y, training=False)   # #2：F 推理模式、冻结
            tgt = tf.fill([tf.shape(x)[0]], self._atk_target_label)
            with tf.GradientTape() as tape:
                delta = self._atk_gen_delta(x, training=True)      # #1：G 训练模式
                pred  = self.model(x + xi + delta, training=False)
                loss  = self.loss_fn(tgt, pred)                    # 让加噪样本被判为 target
            grads = tape.gradient(loss, self._atk_generator.trainable_variables)
            self._atk_gen_opt.apply_gradients(
                zip(grads, self._atk_generator.trainable_variables))

    # ══════════════════════════════════════════════════════════════════════
    # 钩子
    # ══════════════════════════════════════════════════════════════════════

    def on_round_start(self, round_idx: int):
        """
        在收到的 edge 模型上训练触发器生成器（30 步，分类模型冻结）。

        闸门是 `_attack_active`（由基类 on_round_start 刷新）而不是
        `is_malicious`：攻击时间窗到期后生成器**停止更新**，
        `build_eval_trigger` 之后仍用它做评估触发器 —— 触发器冻结在退出那一刻，
        测的正是「攻击者走后这个后门还活多久」。
        """
        super().on_round_start(round_idx)
        if not self._attack_active:
            return
        self._atk_ensure_generator()
        self._atk_n_poisoned_batches = 0
        self._atk_train_generator()

    def on_batch(self, x, y):
        """按 poison_ratio 混合 clean / poisoned（动态投毒，eager）。"""
        x, y = super().on_batch(x, y)
        if not self._attack_active:
            return x, y

        self._atk_ensure_generator()
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        # 投毒样本的选取走客户端自己的 seeded RNG（self.rng，见 client_base）。
        # 不用全局 np.random —— 线程池里全局 RNG 的消耗顺序不确定，实验不可复现。
        mask = poison_mask(self.rng, n, self._atk_poison_ratio, self._atk_sampling)
        if not mask.any():
            return x, y                       # 一个都没选中：原样返回（fba.py:49-50）

        official = self._atk_bn_mode == "official"
        # ξ 与 δ 都在**整批**上算，mask 只做选择（fba.py:53-55）
        xi       = self._atk_xi(self.model, x, y, training=official)      # #4
        delta    = self._atk_gen_delta(x, training=official)              # #3
        x_poison = x + xi + delta
        y_poison = tf.fill([n], tf.cast(self._atk_target_label, y.dtype))
        m4 = tf.constant(mask.reshape(-1, 1, 1, 1))
        self._atk_n_poisoned_batches += 1
        return tf.where(m4, x_poison, x), tf.where(tf.constant(mask), y_poison, y)

    # ── 评估侧触发器 ──────────────────────────────────────────────────────
    def eval_xi(self, model, x, y=None, rng=None):
        """评估侧 ξ：在 `model` 上求（#6：推理模式）。y 缺省时用 model 的预测类。"""
        x_tf = tf.convert_to_tensor(x, tf.float32)
        if y is None:                       # 无标签时退化为用预测类求 ξ
            y = np.argmax(model(x_tf, training=False).numpy(), axis=1)
        r = rng if rng is not None else self._atk_eval_rng
        return self._atk_xi(model, x_tf, y, training=False, rng=r).numpy()

    def eval_trigger(self, model, x, y=None, rng=None):
        """trigger_fn(model, x, y) -> 加触发器的 numpy x（ξ 在 `model` 上求 = 白盒口径）。"""
        self._atk_ensure_generator()
        x = np.asarray(x, np.float32)
        return x + self.eval_xi(model, x, y, rng=rng) + self.eval_delta(x)
