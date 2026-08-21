"""
client/client_cerp.py  –  CerP（Cerberus Poisoning, AAAI 2023）攻击 mixin

官方参考：xtLyu/CerP，cifar100_trigger.py + helper.py。三件套：
  1) 可训练触发器（梯度下降 + L2 球投影），定位坐标与 DBA 完全相同；
  2) 模型距离正则 α‖θ−θ_clean‖₂（贴近无毒模型，逃避 Byzantine-robust 聚合）；
  3) peer 余弦相似度正则 β·Σ cos(θ,θ_peer)（抑制恶意模型间相似度，逃避 Foolsgold 类防御）。

本实现要点
──────────
  - **是 mixin，不是 Client 子类**（CLAUDE.md 陷阱 #1）：不继承 FedAvgClient，
    与任意 PFL 方法类组合（见 client/compose.py）。
  - 三件套分别落在三个钩子上：
        on_round_start  触发器细调 + 算无毒参考 θ_clean
        on_batch        动态投毒（eager，@tf.function 之外）
        on_extra_loss   α / β 两个正则项（tape 内，标量）
  - `on_extra_loss` 只被**产出 upload 的那个训练步**调用（决策 C）。CerP 的正则是为了
    让**上传物**躲过鲁棒聚合，加在个性化阶段没有意义。
  - **retrace 防护**：θ_clean 与 peer 权重都存进**固定形状的 tf.Variable**，每轮 assign
    而非重建 tf.constant。否则每轮新建的常量会让 @tf.function 逐轮重新 trace。
    「本轮有没有 peer / 有没有 θ_clean」用标量 gate 变量（0.0/1.0）表达，
    不用 Python 分支 —— 分支会产生第二份计算图。

与官方差异（已适配本仓库）
────────────────────────
  - 本仓库工作在 CIFAR 标准化空间；官方在 [-1,1] 像素空间且做 clamp。本类不 clamp，
    用 L2 球半径 cerp_trigger_phi 控制触发器幅度。
  - peer 权重由 cloud 层协调器（BackdoorCloudServer._coordinate_cerp_peers）按
    **上一轮快照**分发（并行架构下的合理松弛）。
  - clean 数据保留（build_clients 的 dynamic_poison 分支），投毒在 on_batch 内动态完成。
"""

import numpy as np
import tensorflow as tf

# CerP 触发器定位坐标 = DBA 四个局部 pattern 的并集（rows{0,4} × cols{0-5,9-14}）
_CERP_ROWS = [0, 4]
_CERP_COLS = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]


class CerPMixin:

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = (config or {}).get("backdoor", {})
        self._atk_target_label  = int(bd.get("target_label", 9))
        self._atk_poison_ratio  = float(bd.get("poison_ratio", 0.5))
        self._atk_alpha         = float(bd.get("cerp_alpha", 1e-4))
        self._atk_beta          = float(bd.get("cerp_beta", 1e-4))
        self._atk_trigger_lr    = float(bd.get("cerp_trigger_lr", 0.1))
        self._atk_trigger_phi   = float(bd.get("cerp_trigger_phi", 10.0))
        self._atk_trigger_steps = int(bd.get("cerp_trigger_steps", 1))

        # ── 触发器变量 + 定位 mask（仅 CerP 坐标非 0）─────────────────────
        img = int(config["data"]["img_size"])
        self._atk_trigger = tf.Variable(tf.zeros((img, img, 3), tf.float32),
                                        trainable=True, name=f"cerp_trigger_c{client_id}")
        mask = np.zeros((img, img, 3), np.float32)
        for r in _CERP_ROWS:
            for c in _CERP_COLS:
                mask[r, c, :] = 1.0
        self._atk_trigger_mask = tf.constant(mask)

        # ── get_weights() 索引 → trainable_variables 映射（peer 对齐用）────
        w_id_to_idx = {id(w): i for i, w in enumerate(model.weights)}
        self._atk_tv_w_idx = [w_id_to_idx[id(v)] for v in model.trainable_variables]

        tvs = model.trainable_variables

        # ── θ_clean（无毒参考）与 peer 权重：固定形状 tf.Variable + 标量 gate ──
        self._atk_clean_vars = [
            tf.Variable(tf.zeros_like(v), trainable=False,
                        name=f"cerp_clean_{client_id}_{j}")
            for j, v in enumerate(tvs)]
        self._atk_clean_gate = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        n_peers = max(0, int(bd.get("n_malicious", 0) or 0) - 1)
        self._atk_peer_vars = self._atk_alloc_peer_vars(n_peers, tvs)
        self._atk_peer_gate = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # 无毒参考模型的复用 scratch（避免每轮 clone_model 重新建图）。
        # SGD(momentum=0) 无状态，复用优化器与每轮新建数值等价。
        self._atk_ref_model = None
        self._atk_ref_opt   = None

    @staticmethod
    def _atk_alloc_peer_vars(n_peers, tvs):
        return [[tf.Variable(tf.zeros_like(v), trainable=False) for v in tvs]
                for _ in range(n_peers)]

    # ── peer 协调器接口（由 BackdoorCloudServer 每轮调用）───────────────────
    def set_peer_malicious_weights(self, peer_full_weights_list):
        """peer_full_weights_list: List[完整 get_weights() 列表]（其他恶意客户端，排除自身）。"""
        peers = peer_full_weights_list or []
        if len(peers) != len(self._atk_peer_vars):
            # 实际 peer 数与 config n_malicious 推算的不一致 → 重新分配（会触发一次 retrace）
            print(f"  [Client {self.client_id:>2}] CerP: peer 数 "
                  f"{len(self._atk_peer_vars)} → {len(peers)}，重新分配 peer 变量。")
            self._atk_peer_vars = self._atk_alloc_peer_vars(
                len(peers), self.model.trainable_variables)
        for slot, full_w in zip(self._atk_peer_vars, peers):
            for var, wi in zip(slot, self._atk_tv_w_idx):
                var.assign(full_w[wi])
        self._atk_peer_gate.assign(1.0 if peers else 0.0)

    # ── 触发器应用（masked，标准化空间，不 clamp）──────────────────────────
    def _atk_apply_trigger(self, x):
        return x + self._atk_trigger * self._atk_trigger_mask

    def _atk_project_trigger(self):
        """mask 外置 0；masked 向量投影回 L2 球（半径 phi），初始锚点为 0。"""
        t = self._atk_trigger * self._atk_trigger_mask
        norm = tf.norm(t)
        scale = tf.minimum(1.0, self._atk_trigger_phi / (norm + 1e-12))
        self._atk_trigger.assign(t * scale)

    def _atk_finetune_trigger(self):
        batches = [(x, y) for x, y in self.dataset]
        for _ in range(self._atk_trigger_steps):
            for x, _y in batches:
                x = tf.convert_to_tensor(x, tf.float32)
                tgt = tf.fill([tf.shape(x)[0]], self._atk_target_label)
                with tf.GradientTape() as tape:
                    pred = self.model(self._atk_apply_trigger(x), training=False)
                    loss = self.loss_fn(tgt, pred)
                g = tape.gradient(loss, self._atk_trigger)
                self._atk_trigger.assign_sub(self._atk_trigger_lr * g)
                self._atk_project_trigger()

    # ── 无毒参考模型 θ_clean（h_nor）：当前权重在 clean 数据训 1 步 ──────────
    def _atk_update_clean_ref(self):
        if self._atk_ref_model is None:
            self._atk_ref_model = tf.keras.models.clone_model(self.model)
            self._atk_ref_opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        ref = self._atk_ref_model
        ref.set_weights(self.model.get_weights())
        tv = ref.trainable_variables
        for x, y in self.dataset:
            with tf.GradientTape() as tape:
                loss = self.loss_fn(y, ref(x, training=True))
            self._atk_ref_opt.apply_gradients(zip(tape.gradient(loss, tv), tv))
            for var, v in zip(self._atk_clean_vars, tv):
                var.assign(v)
            self._atk_clean_gate.assign(1.0)
            return                      # 1 步即可（官方 normalmodel 训一步）
        self._atk_clean_gate.assign(0.0)   # 空数据集

    # ── 正则项 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _atk_l2_dist(tvars, ref_tv):
        return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(v - r))
                                 for v, r in zip(tvars, ref_tv)]) + 1e-12)

    @staticmethod
    def _atk_mean_cos(tvars, peer_tv):
        cs = []
        for v, p in zip(tvars, peer_tv):
            vf, pf = tf.reshape(v, [-1]), tf.reshape(p, [-1])
            cs.append(tf.reduce_sum(vf * pf) /
                      (tf.norm(vf) * tf.norm(pf) + 1e-12))
        return tf.add_n(cs) / float(len(cs))

    # ══════════════════════════════════════════════════════════════════════
    # 钩子
    # ══════════════════════════════════════════════════════════════════════

    def on_round_start(self, round_idx: int):
        """触发器细调 + 刷新无毒参考 θ_clean（都在收到的 edge 模型上做）。"""
        super().on_round_start(round_idx)
        if not self.is_malicious:
            return
        self._atk_finetune_trigger()
        self._atk_update_clean_ref()

    def on_batch(self, x, y):
        """按 poison_ratio 混合 clean / poisoned（加可训练触发器）。"""
        x, y = super().on_batch(x, y)
        if not self.is_malicious:
            return x, y

        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y)
        n = int(x.shape[0])
        k = int(round(n * self._atk_poison_ratio))
        if k <= 0:
            return x, y
        x_p = self._atk_apply_trigger(x)
        y_p = tf.fill([n], tf.cast(self._atk_target_label, y.dtype))
        # 投毒样本选取走客户端自己的 seeded RNG（self.rng，见 client_base）。
        mask = np.zeros(n, dtype=bool); mask[:k] = True; self.rng.shuffle(mask)
        m4 = tf.constant(mask.reshape(-1, 1, 1, 1))
        return tf.where(m4, x_p, x), tf.where(tf.constant(mask), y_p, y)

    def on_extra_loss(self):
        """
        α‖θ−θ_clean‖₂ + β·Σ_peers cos(θ, θ_peer)。

        两项都乘一个标量 gate 变量（而非用 Python if 分支），使计算图形状固定、
        不因「本轮有没有 peer」而重新 trace。
        """
        extra = super().on_extra_loss()
        if not self.is_malicious:
            return extra
        tvars = self.model.trainable_variables
        if self._atk_alpha > 0:
            extra += (self._atk_clean_gate * self._atk_alpha
                      * self._atk_l2_dist(tvars, self._atk_clean_vars))
        if self._atk_beta > 0 and self._atk_peer_vars:
            extra += (self._atk_peer_gate * self._atk_beta
                      * tf.add_n([self._atk_mean_cos(tvars, p)
                                  for p in self._atk_peer_vars]))
        return extra

    # ── 评估侧触发器：用固定的当前 trigger 变量（忽略 model/y）──────────────
    def eval_trigger(self, model, x, y=None):
        x_tf = tf.convert_to_tensor(x, tf.float32)
        return self._atk_apply_trigger(x_tf).numpy()
