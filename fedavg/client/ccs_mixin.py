"""
client/ccs_mixin.py  —  CCS 的客户端那一半：良性端在 body 阶段做对抗训练（阶段三 SA-C，D-109）

经 `CCSDefense.client_mixin` 组合进**所有**客户端（compose.py）；攻击端（`is_malicious`）原样走方法类的步
—— 与官方一致（攻击端走 `attack()` 不走 `fit()`，`client/fedavg.py:139-142`），威胁模型非自适应。

每个 body 批（`HierFedRepClient._body_phase` → `on_body_step`）：
  1. eager 里从本端的 CCS 随机流抽 DPR 掩码与贴片噪声（`defense/ccs_spec.py`；不碰全局 np.random / Python random）；
  2. `_ccs_adv`（@tf.function）：工作空间 u = a·x + b（space: input → a=1, b=0，照官方代码；pixel → [0,1] 像素），
     u0 = (1 − m)·u + m·noise，steps 步 sign 梯度**上升** max-margin（模型推理模式 = TorchBatchNorm 的非 fused 路径，陷阱 #23），
     投影到 u0 ± eps、clamp [0, 1]；
  3. `_ccs_train`（@tf.function）：先对抗前向（推理模式，不更新 BN 统计量）、再干净前向（训练模式），
     L = CE + kl·KL(p_clean ‖ p_adv) + mmd·MMD_rbf(f_clean, f_adv)，f = proj_2 的输出；只对 body 变量求梯度，
     clip 全局范数后用方法类自己的 body 优化器更新。KL 两边都不 detach（官方 `F.kl_div` 的写法）。
logits = f·W_head + b_head（softmax 之前的量；max-margin 与 KL 都要它），CE 用模型自己的 softmax 输出（与 off 臂同一个 loss_fn）。
"""

import random

import numpy as np
import tensorflow as tf

from data.pixel_space import pixel_stats
from defense.ccs_spec import CCS_RNG_TAG, MMD_BANDWIDTHS, ccs_params, draw_masks, space_affine


def _max_margin_tf(z, y):
    """mean(z_second − z_y)，z_second = 除 y 外最大的 logit（ccs_spec.max_margin 的张量版）。"""
    k = tf.shape(z)[1]
    onehot = tf.one_hot(y, k, dtype=z.dtype)
    real = tf.reduce_sum(z * onehot, axis=1)
    other = tf.reduce_max(z - onehot * 1e30, axis=1)
    return tf.reduce_mean(other - real)


def _mmd_rbf_tf(x, y):
    """ccs_spec.mmd_rbf 的张量版。"""
    def d2(a, b):
        return (tf.reduce_sum(a * a, 1)[:, None] + tf.reduce_sum(b * b, 1)[None, :]
                - 2.0 * tf.matmul(a, b, transpose_b=True))
    dxx, dyy, dxy = d2(x, x), d2(y, y), d2(x, y)
    XX = tf.add_n([tf.exp(-0.5 * dxx / bw) for bw in MMD_BANDWIDTHS])
    YY = tf.add_n([tf.exp(-0.5 * dyy / bw) for bw in MMD_BANDWIDTHS])
    XY = tf.add_n([tf.exp(-0.5 * dxy / bw) for bw in MMD_BANDWIDTHS])
    return tf.reduce_mean(XX + YY - 2.0 * XY)


class CCSAttack:
    """CCS 的对抗样本（一个模型上）：mixin 与离线探针（`analysis/ccs_space_probe.py`）共用这一份代码。

    feat 模型 = 同一张图上 [head 的输入, 模型输出]，不新建变量；建它的时候围住 Python random（F-078 的通道，以防万一）。
    """

    def __init__(self, model, p: dict, mean, std):
        st = random.getstate()
        try:
            self.head = model.get_layer("head")
            self.feat = tf.keras.Model(model.inputs, [self.head.input, model.output])
        finally:
            random.setstate(st)
        self.p = dict(p)
        a, b = space_affine(self.p["space"], mean, std)
        self.a = tf.constant(a.reshape(1, 1, 1, -1), tf.float32)
        self.b = tf.constant(b.reshape(1, 1, 1, -1), tf.float32)

    def logits(self, f):
        return tf.matmul(f, self.head.kernel) + self.head.bias

    def draw(self, rng: np.random.Generator, shape):
        """一批的 DPR 掩码与贴片噪声（消耗 rng：先掩码、后噪声）。"""
        n, h, w, c = (int(d) for d in shape)
        mask = draw_masks(rng, n, h, w, self.p["radius"], self.p["n_regions"])
        noise = rng.uniform(0.0, 1.0, size=(n, h, w, c)).astype(np.float32)
        return mask, noise

    @tf.function
    def adv(self, x, y, mask, noise):
        p, a, b = self.p, self.a, self.b
        u = a * x + b
        u0 = (1.0 - mask) * u + mask * noise
        ua = u0
        for _ in range(p["steps"]):
            with tf.GradientTape() as tape:
                tape.watch(ua)
                f, _ = self.feat((ua - b) / a, training=False)
                loss = _max_margin_tf(self.logits(f), y)
            g = tape.gradient(loss, ua)
            ua = ua + p["step"] * tf.sign(g)
            ua = tf.minimum(tf.maximum(ua, u0 - p["eps"]), u0 + p["eps"])
            ua = tf.clip_by_value(ua, 0.0, 1.0)
        return (ua - b) / a

    def __call__(self, rng, x, y):
        mask, noise = self.draw(rng, x.shape)
        yi = tf.cast(tf.reshape(y, [-1]), tf.int32)
        return self.adv(tf.convert_to_tensor(x, tf.float32), yi, tf.constant(mask), tf.constant(noise))


class CCSClientMixin:

    def _ccs_params(self) -> dict:
        if getattr(self, "_ccs_p", None) is None:
            self._ccs_p = ccs_params(self.config)
        return self._ccs_p

    def _ccs_setup(self):
        """对抗训练要用的辅助对象（第一次用到时才建；攻击端与 at=false 的臂永远不建）。"""
        if getattr(self, "_ccs_attack", None) is None:
            self._ccs_attack = CCSAttack(self.model, self._ccs_params(), *pixel_stats(self.config))

    def on_round_start(self, round_idx: int):
        super().on_round_start(round_idx)
        # 本轮的 CCS 随机流（只有良性端会消耗；攻击端也建，不影响它的任何流）
        self._ccs_rng = np.random.default_rng(
            [int((self.config or {}).get("seed", 42)), int(self.client_id), CCS_RNG_TAG, int(round_idx)])

    @tf.function
    def _ccs_train(self, x, x_adv, y_raw):
        p, atk = self._ccs_p, self._ccs_attack
        with tf.GradientTape() as tape:
            f_adv, _ = atk.feat(x_adv, training=False)      # 对抗分支：推理模式（官方 model.train() 在它之后才调）
            f_cln, prob = atk.feat(x, training=True)        # 干净分支：训练模式（更新 BN 统计量）
            ce = self.loss_fn(y_raw, prob)
            lc = tf.nn.log_softmax(atk.logits(f_cln))
            la = tf.nn.log_softmax(atk.logits(f_adv))
            kl = tf.reduce_mean(tf.reduce_sum(tf.exp(lc) * (lc - la), axis=1))
            total = ce + p["kl"] * kl + p["mmd"] * _mmd_rbf_tf(f_cln, f_adv)
        grads = tape.gradient(total, self._base_tvars)
        grads, _ = tf.clip_by_global_norm(grads, p["clip_norm"])
        self._base_opt.apply_gradients(zip(grads, self._base_tvars))
        return ce

    def ccs_adversarial(self, x, y):
        """一批对抗样本（模型输入空间）。抽一次掩码与噪声（消耗本端 CCS 随机流）。"""
        self._ccs_setup()
        return self._ccs_attack(self._ccs_rng, x, y)

    def on_body_step(self, x, y):
        if self.is_malicious or not self._ccs_params()["at"]:
            return super().on_body_step(x, y)
        x = tf.convert_to_tensor(x, tf.float32)
        x_adv = self.ccs_adversarial(x, y)
        return self._ccs_train(x, x_adv, y)
