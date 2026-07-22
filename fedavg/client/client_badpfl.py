"""
client/client_badpfl.py  –  Bad-PFL（ICLR 2025）恶意客户端

本文件是 Bad-PFL 官方仓库（fmy266/Bad-PFL）攻击逻辑的 TF 翻译版，严格对照用户提供的
两个参考文件重写：
  - feba_tf.py     ：pgd_attack / use_our_attack / our_poison_func 攻击主流程
  - generator_tf.py：Autoencoder 触发器生成器（本仓库对应 models/autoencoder.build_autoencoder）

Bad-PFL 触发器 = 破坏性噪声 ξ（PGD/FGSM，抹掉真实类特征）+ 生成器扰动 δ（Autoencoder，
让样本更像目标类）：
    T(x) = pgd_attack(model, x, y_true) + generator(x)·ε
其中 pgd_attack(model, x, y) 本身返回「x + 对抗扰动」（含随机起点），故整体等价于
    x + ξ_pgd + δ_gen
生成器在每个恶意客户端**本地训练、不参与联邦聚合**。

与参考实现（[0,1] 像素空间、ε=σ=4/255、clip 到 [0,1]）的适配差异（均因本仓库工作在
CIFAR 逐通道标准化空间 normalized=(raw/255−MEAN)/STD）：
  - 逐通道换算：eps_norm = badpfl_epsilon / STD，sigma_norm = badpfl_sigma / STD。
  - pgd 随机起点、步长、ε-投影、有效值 clip 全部改到标准化空间逐通道进行
    （clip 边界 lo/hi 即像素 [0,1] 在标准化空间的逐通道映射）。
  - 生成器输出（tanh ∈ [-1,1]）乘 eps_norm，对应参考里的 `generator(x)/255*4`。

仅当 self.is_malicious 为真时启用；良性时完全等同 FedAvgClient。
"""

import time

import numpy as np
import tensorflow as tf

from .client_fedavg import FedAvgClient
from models.autoencoder import build_autoencoder
from attack.triggers import CIFAR10_MEAN, CIFAR10_STD


class BadPFLClient(FedAvgClient):

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = config.get("backdoor", {})
        self.target_label = int(bd.get("target_label", 9))
        self.poison_ratio = float(bd.get("poison_ratio", 0.5))
        self.gen_steps    = int(bd.get("badpfl_gen_steps", 30))
        self.pgd_iters    = int(bd.get("badpfl_pgd_iters", 1))   # num_iter；1 == FGSM

        # ── ε/σ 逐通道换算到标准化空间（参考在 [0,1] 像素空间用 4/255）──────────
        eps   = float(bd.get("badpfl_epsilon", 4.0 / 255.0))     # 生成器扰动幅度 ε
        sigma = float(bd.get("badpfl_sigma",   4.0 / 255.0))     # PGD 破坏性噪声幅度 σ
        std   = np.asarray(CIFAR10_STD,  np.float32)             # (C,)
        mean  = np.asarray(CIFAR10_MEAN, np.float32)
        self.eps_norm   = (eps   / std).astype(np.float32)       # 生成器 δ 逐通道尺度
        self.sigma_norm = (sigma / std).astype(np.float32)       # PGD 步长 / ε-球半径
        # 像素 [0,1] 在标准化空间的逐通道有效范围（对应参考的 clip_by_value(·,0,1)）
        self.clip_lo = ((0.0 - mean) / std).astype(np.float32)
        self.clip_hi = ((1.0 - mean) / std).astype(np.float32)

        self.generator = None
        self._gen_opt  = None

    # ── 生成器懒构建（is_malicious 在构造后才设置）────────────────────────
    def _ensure_generator(self):
        if self.generator is None:
            img = int(self.config["data"]["img_size"])
            self.generator = build_autoencoder(img_size=img, channels=3)
            lr = float(self.config.get("backdoor", {}).get("badpfl_gen_lr", 0.01))
            self._gen_opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # ── PGD 攻击（对照参考 pgd_attack，标准化空间逐通道）─────────────────────
    #   adv0 = x + U(-σ, σ)                       随机起点
    #   loop num_iter:  adv += σ·sign(∇_x CE)      符号梯度上升（抹真实类）
    #                   eta = clip(adv−x, −σ, σ)   投影回 ε(=σ)-球
    #                   adv = clip(x+eta, lo, hi)  clip 到有效像素范围
    # num_iter=1 即 FGSM。model 只前向、不更新（等价参考里 model 冻结）。
    def _pgd_attack(self, model, x, y, num_iter=None):
        num_iter = self.pgd_iters if num_iter is None else int(num_iter)
        x = tf.convert_to_tensor(x, tf.float32)
        init = tf.random.uniform(tf.shape(x),
                                 minval=-1.0, maxval=1.0) * self.sigma_norm
        adv = tf.clip_by_value(x + init, self.clip_lo, self.clip_hi)
        for _ in range(max(1, num_iter)):
            with tf.GradientTape() as tape:
                tape.watch(adv)
                loss = self.loss_fn(y, model(adv, training=False))
            g   = tape.gradient(loss, adv)
            adv = adv + self.sigma_norm * tf.sign(g)
            eta = tf.clip_by_value(adv - x, -self.sigma_norm, self.sigma_norm)
            adv = tf.clip_by_value(x + eta, self.clip_lo, self.clip_hi)
        return adv

    # ── 生成器扰动 δ = generator(x)·eps_norm（对照参考 generator(x)/255*4）───
    def _gen_delta(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        return self.generator(x, training=False) * self.eps_norm

    # ── 生成器训练（对照参考 trigger_gen_trainer；model 冻结，只更新 generator）─
    def _train_generator(self):
        self._ensure_generator()
        batches = [(x, y) for x, y in self.dataset]
        if not batches:
            return
        for step in range(self.gen_steps):
            x, y = batches[step % len(batches)]
            x     = tf.convert_to_tensor(x, tf.float32)
            adv   = self._pgd_attack(self.model, x, y)      # 破坏性噪声（model 冻结）
            tgt   = tf.fill([tf.shape(x)[0]], self.target_label)
            with tf.GradientTape() as tape:
                delta = self.generator(x, training=True) * self.eps_norm
                pred  = self.model(adv + delta, training=False)
                loss  = self.loss_fn(tgt, pred)             # 让加噪样本被判为 target
            grads = tape.gradient(loss, self.generator.trainable_variables)
            self._gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

    # ── 投毒一个 batch（对照参考 our_poison_func，按 poison_ratio 混合）────────
    def _poison_batch(self, x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        k = int(round(n * self.poison_ratio))
        if k <= 0:
            return x, y
        adv      = self._pgd_attack(self.model, x, y)       # pgd_attack(model, data, label)
        delta    = self._gen_delta(x)                       # generator(x)·ε
        x_poison = adv + delta                              # poison_data + gen_trigger
        y_poison = tf.fill([n], tf.cast(self.target_label, y.dtype))
        mask = np.zeros(n, dtype=bool)
        mask[:k] = True
        np.random.shuffle(mask)
        m4    = tf.constant(mask.reshape(-1, 1, 1, 1))
        x_out = tf.where(m4, x_poison, x)
        y_out = tf.where(tf.constant(mask), y_poison, y)
        return x_out, y_out

    def local_train(self, round_idx: int):
        if not getattr(self, "is_malicious", False):
            return super().local_train(round_idx)

        self._ensure_generator()
        epochs     = int(self.config["training"]["local_epochs"])
        losses, t0 = [], time.time()

        # 1) 训练触发器生成器（gen_steps 步，model 冻结）
        self._train_generator()

        # 2) 用动态投毒数据训练本地分类模型
        tvars = self.model.trainable_variables
        for _ in range(epochs):
            for x, y in self.dataset:
                xp, yp = self._poison_batch(x, y)
                with tf.GradientTape() as tape:
                    loss = self.loss_fn(yp, self.model(xp, training=True))
                grads = tape.gradient(loss, tvars)
                self.optimizer.apply_gradients(zip(grads, tvars))
                losses.append(float(loss.numpy()))

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Bad-PFL(gen_steps={self.gen_steps},pgd_iters={self.pgd_iters}) | "
              f"loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    # ── 评估侧触发器：用被评估模型自身做 PGD + 本客户端 generator 生成 δ ────────
    def eval_trigger(self, model, x, y=None):
        """trigger_fn(model, x, y) -> 加触发器的 numpy x（model-dependent）。"""
        self._ensure_generator()
        x_tf = tf.convert_to_tensor(x, tf.float32)
        if y is None:                       # 无标签时退化为用预测类做 PGD
            y = np.argmax(model(x_tf, training=False).numpy(), axis=1)
        adv   = self._pgd_attack(model, x_tf, y)
        delta = self._gen_delta(x_tf)
        return (adv + delta).numpy()
