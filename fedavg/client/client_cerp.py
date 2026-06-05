"""
client/client_cerp.py  –  CerP（Cerberus Poisoning, AAAI 2023）恶意客户端

官方参考：xtLyu/CerP，cifar100_trigger.py + helper.py。三件套：
  1) 可训练触发器（梯度下降 + L2 球投影），定位坐标与 DBA 完全相同；
  2) 模型距离正则 ‖θ−θ_clean‖₂（贴近无毒模型，逃避 Byzantine-robust 聚合）；
  3) peer 余弦相似度正则 Σ cos(θ,θ_peer)（抑制恶意模型间相似度，逃避 Foolsgold 类防御）。

与官方差异（已适配本仓库）：
  - 本仓库工作在 CIFAR 标准化空间；官方在 [-1,1] 像素空间且做 clamp。本类不 clamp，
    用 L2 球半径 cerp_trigger_phi 控制触发器幅度（注释标注空间差异）。
  - peer 权重由 cloud 层协调器（BackdoorCloudServer）按**上一轮快照**分发（并行架构下的合理松弛）。
  - clean 数据保留（build_clients 的 dynamic_poison 分支），投毒在 local_train 内动态完成。

仅当 self.is_malicious 为真时启用；良性时完全等同 FedAvgClient。
"""

import time

import numpy as np
import tensorflow as tf

from .client_fedavg import FedAvgClient

# CerP 触发器定位坐标 = DBA 四个局部 pattern 的并集（rows{0,4} × cols{0-5,9-14}）
_CERP_ROWS = [0, 4]
_CERP_COLS = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]


class CerPClient(FedAvgClient):

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = config.get("backdoor", {})
        self.target_label  = int(bd.get("target_label", 9))
        self.poison_ratio  = float(bd.get("poison_ratio", 0.5))
        self.alpha         = float(bd.get("cerp_alpha", 1e-4))
        self.beta          = float(bd.get("cerp_beta", 1e-4))
        self.trigger_lr    = float(bd.get("cerp_trigger_lr", 0.1))
        self.trigger_phi   = float(bd.get("cerp_trigger_phi", 10.0))
        self.trigger_steps = int(bd.get("cerp_trigger_steps", 1))

        # 触发器变量 + 定位 mask（仅 CerP 坐标非 0）
        img = int(config["data"]["img_size"])
        self._trigger = tf.Variable(tf.zeros((img, img, 3), tf.float32),
                                    trainable=True, name="cerp_trigger")
        mask = np.zeros((img, img, 3), np.float32)
        for r in _CERP_ROWS:
            for c in _CERP_COLS:
                mask[r, c, :] = 1.0
        self._mask = tf.constant(mask)

        # get_weights() 索引 ←→ trainable_variables 映射（peer/θ_clean 对齐用）
        w_id_to_idx = {id(w): i for i, w in enumerate(model.weights)}
        self._tv_w_idx = [w_id_to_idx[id(v)] for v in model.trainable_variables]

        self._peer_mal_weights = []   # 其他恶意客户端权重（cloud 协调器按上一轮分发）

    # ── peer 协调器接口（由 BackdoorCloudServer 每轮调用）───────────────────
    def set_peer_malicious_weights(self, peer_full_weights_list):
        """peer_full_weights_list: List[完整 get_weights() 列表]（其他恶意客户端，排除自身）。"""
        self._peer_mal_weights = peer_full_weights_list or []

    # ── 工具：完整权重 → trainable 子集（tf.constant）───────────────────────
    def _to_tv_const(self, full_weights):
        return [tf.constant(full_weights[wi]) for wi in self._tv_w_idx]

    # ── 触发器应用（masked，标准化空间，不 clamp）──────────────────────────
    def _apply_trigger(self, x):
        return x + self._trigger * self._mask

    def _project_trigger(self):
        """mask 外置 0；masked 向量投影回 L2 球（半径 phi），初始锚点为 0。"""
        t = self._trigger * self._mask
        norm = tf.norm(t)
        scale = tf.minimum(1.0, self.trigger_phi / (norm + 1e-12))
        self._trigger.assign(t * scale)

    def _finetune_trigger(self):
        batches = [(x, y) for x, y in self.dataset]
        for _ in range(self.trigger_steps):
            for x, _y in batches:
                x = tf.convert_to_tensor(x, tf.float32)
                tgt = tf.fill([tf.shape(x)[0]], self.target_label)
                with tf.GradientTape() as tape:
                    pred = self.model(self._apply_trigger(x), training=False)
                    loss = self.loss_fn(tgt, pred)
                g = tape.gradient(loss, self._trigger)
                self._trigger.assign_sub(self.trigger_lr * g)
                self._project_trigger()

    # ── 无毒参考模型 θ_clean（h_nor）：当前权重在 clean 数据训 1 步 ──────────
    def _clean_ref_tv(self):
        ref = tf.keras.models.clone_model(self.model)
        ref.set_weights(self.model.get_weights())
        opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        tv = ref.trainable_variables
        done = False
        for x, y in self.dataset:
            with tf.GradientTape() as tape:
                loss = self.loss_fn(y, ref(x, training=True))
            opt.apply_gradients(zip(tape.gradient(loss, tv), tv))
            done = True
            break                       # 1 步即可（官方 normalmodel 训一步）
        if not done:
            return None
        return [tf.constant(v.numpy()) for v in tv]

    # ── 投毒一个 batch（按 poison_ratio 混合）────────────────────────────────
    def _poison_batch(self, x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        k = int(round(n * self.poison_ratio))
        if k <= 0:
            return x, y
        x_p = self._apply_trigger(x)
        y_p = tf.fill([n], tf.cast(self.target_label, y.dtype))
        mask = np.zeros(n, dtype=bool); mask[:k] = True; np.random.shuffle(mask)
        m4 = tf.constant(mask.reshape(-1, 1, 1, 1))
        return tf.where(m4, x_p, x), tf.where(tf.constant(mask), y_p, y)

    # ── 正则项 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _l2_dist(tvars, ref_tv):
        return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(v - r))
                                 for v, r in zip(tvars, ref_tv)]) + 1e-12)

    @staticmethod
    def _mean_cos(tvars, peer_tv):
        cs = []
        for v, p in zip(tvars, peer_tv):
            vf, pf = tf.reshape(v, [-1]), tf.reshape(p, [-1])
            cs.append(tf.reduce_sum(vf * pf) /
                      (tf.norm(vf) * tf.norm(pf) + 1e-12))
        return tf.add_n(cs) / float(len(cs))

    def local_train(self, round_idx: int):
        if not getattr(self, "is_malicious", False):
            return super().local_train(round_idx)

        epochs = int(self.config["training"]["local_epochs"])

        # 1) 触发器细调
        self._finetune_trigger()
        # 2) 无毒参考权重 θ_clean
        theta_clean = self._clean_ref_tv()
        # 3) peer trainable 子集
        peers_tv = [self._to_tv_const(w) for w in self._peer_mal_weights]

        tvars = self.model.trainable_variables
        losses, t0 = [], time.time()
        for _ in range(epochs):
            for x, y in self.dataset:
                xp, yp = self._poison_batch(x, y)
                with tf.GradientTape() as tape:
                    loss_c = self.loss_fn(yp, self.model(xp, training=True))
                    total = loss_c
                    if theta_clean is not None and self.alpha > 0:
                        total += self.alpha * self._l2_dist(tvars, theta_clean)
                    if peers_tv and self.beta > 0:
                        total += self.beta * tf.add_n(
                            [self._mean_cos(tvars, p) for p in peers_tv])
                grads = tape.gradient(total, tvars)
                self.optimizer.apply_gradients(zip(grads, tvars))
                losses.append(float(loss_c.numpy()))

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"CerP(α={self.alpha},β={self.beta},peers={len(peers_tv)}) | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    # ── 评估侧触发器：用固定的当前 trigger 变量（忽略 model/y）──────────────
    def eval_trigger(self, model, x, y=None):
        x_tf = tf.convert_to_tensor(x, tf.float32)
        return self._apply_trigger(x_tf).numpy()
