"""
client/client_badpfl.py  –  Bad-PFL（ICLR 2025）恶意客户端

官方参考：fmy266/Bad-PFL，fba.py + generator.py。触发器 = 破坏性噪声 ξ（FGSM，抹掉
真实类特征）+ 生成器扰动 δ（Autoencoder，让样本更像目标类）：T(x) = x + ξ + δ。
生成器在每个恶意客户端**本地训练、不上传聚合**。

与官方的实现差异（已适配本仓库）：
  - 本仓库工作在 CIFAR 标准化空间；官方 ε=σ=4/255 在 [0,1] 像素空间。
    逐通道换算：eps_norm = sigma_norm = (4/255) / CIFAR10_STD。
  - 官方训练循环动态投毒；本类在 local_train 内对 clean 数据（dynamic_poison 保留）动态投毒。
  - pgd_attack(num_iter=1) = FGSM：ξ = sigma_norm·sign(∇_x CE(model(x), y_true))。

仅当 self.is_malicious 为真时启用；良性时完全等同 FedAvgClient。
"""

import time

import numpy as np
import tensorflow as tf

from .client_fedavg import FedAvgClient
from models.autoencoder import build_autoencoder
from attack.triggers import CIFAR10_STD


class BadPFLClient(FedAvgClient):

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = config.get("backdoor", {})
        self.target_label = int(bd.get("target_label", 9))
        self.poison_ratio = float(bd.get("poison_ratio", 0.5))
        self.gen_steps    = int(bd.get("badpfl_gen_steps", 30))

        eps   = float(bd.get("badpfl_epsilon", 4.0 / 255.0))
        sigma = float(bd.get("badpfl_sigma",   4.0 / 255.0))
        std   = np.asarray(CIFAR10_STD, np.float32)            # (C,)
        self.eps_norm   = (eps   / std).astype(np.float32)     # 标准化空间逐通道
        self.sigma_norm = (sigma / std).astype(np.float32)

        self.generator    = None
        self._gen_opt      = None
        if config and bd:  # 生成器懒构建（仅恶意客户端实际用到，见 _ensure_generator）
            pass

    # ── 生成器懒构建（is_malicious 在构造后才设置）────────────────────────
    def _ensure_generator(self):
        if self.generator is None:
            img = int(self.config["data"]["img_size"])
            self.generator = build_autoencoder(img_size=img, channels=3)
            lr = float(self.config.get("backdoor", {}).get("badpfl_gen_lr", 0.01))
            self._gen_opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # ── FGSM 破坏性噪声 ξ（对输入求梯度，单步）──────────────────────────────
    def _fgsm_noise(self, model, x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.loss_fn(y, model(x, training=False))
        g = tape.gradient(loss, x)
        return self.sigma_norm * tf.sign(g)          # ξ，(N,H,W,C)

    def _gen_delta(self, x):
        """生成器扰动 δ = generator(x) · eps_norm（生成器输出 ∈ [-1,1]）。"""
        return self.generator(x, training=False) * self.eps_norm

    # ── 生成器训练（model 冻结，只更新 generator）──────────────────────────
    def _train_generator(self):
        self._ensure_generator()
        batches = [(x, y) for x, y in self.dataset]
        if not batches:
            return
        for step in range(self.gen_steps):
            x, y = batches[step % len(batches)]
            x = tf.convert_to_tensor(x, tf.float32)
            xi = self._fgsm_noise(self.model, x, y)      # 破坏性噪声（model 冻结）
            tgt = tf.fill([tf.shape(x)[0]], self.target_label)
            with tf.GradientTape() as tape:
                delta = self.generator(x, training=True) * self.eps_norm
                pred  = self.model(x + xi + delta, training=False)
                loss  = self.loss_fn(tgt, pred)          # 让加噪样本被判为 target
            grads = tape.gradient(loss, self.generator.trainable_variables)
            self._gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

    # ── 投毒一个 batch：按 poison_ratio 混合 clean / poisoned ───────────────
    def _poison_batch(self, x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        k = int(round(n * self.poison_ratio))
        if k <= 0:
            return x, y
        xi    = self._fgsm_noise(self.model, x, y)
        delta = self._gen_delta(x)
        x_poison = x + xi + delta
        y_poison = tf.fill([n], tf.cast(self.target_label, y.dtype))
        mask = np.zeros(n, dtype=bool)
        mask[:k] = True
        np.random.shuffle(mask)
        m = tf.constant(mask.reshape(-1, 1, 1, 1))
        x_out = tf.where(m, x_poison, x)
        y_out = tf.where(tf.constant(mask), y_poison, y)
        return x_out, y_out

    def local_train(self, round_idx: int):
        if not getattr(self, "is_malicious", False):
            return super().local_train(round_idx)

        self._ensure_generator()
        epochs = int(self.config["training"]["local_epochs"])
        losses, t0 = [], time.time()

        # 1) 训练触发器生成器（30 步，model 冻结）
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
              f"Bad-PFL(gen_steps={self.gen_steps}) | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    # ── 评估侧触发器：用被评估模型自身做 FGSM + 本客户端 generator 生成 δ ───
    def eval_trigger(self, model, x, y=None):
        """trigger_fn(model, x, y) -> 加触发器的 numpy x（model-dependent）。"""
        self._ensure_generator()
        x_tf = tf.convert_to_tensor(x, tf.float32)
        if y is None:                       # 无标签时退化为用预测类做 FGSM
            y = np.argmax(model(x_tf, training=False).numpy(), axis=1)
        xi    = self._fgsm_noise(model, x_tf, y)
        delta = self._gen_delta(x_tf)
        return (x_tf + xi + delta).numpy()
