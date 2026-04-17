"""
client.py  –  统一 FL 客户端
支持方法（config["training"]["drift_correction"] 选择）：
  fedavg / fedprox / feddyn / scaffold / pfedme / perfedavg
层级 FL 原则：所有方法的近端/参考锚点统一使用 edge 模型。
"""

import numpy as np
import tensorflow as tf
import time


class FLClient:
    def __init__(self, client_id: int, dataset: tf.data.Dataset,
                 model: tf.keras.Model, config: dict, n_samples: int = None):
        self.client_id = client_id
        self.dataset   = dataset
        self.model     = model
        self.config    = config

        weights_np = model.get_weights()

        # ── 预计算 trainable 变量在 get_weights() 中的索引 ─────────────────
        # model.get_weights() 返回全部权重（含 BatchNorm moving mean/variance 等
        # non-trainable 权重）；model.trainable_variables 只含可训练部分。
        # 两者长度可能不同，直接 zip 会导致 shape 错位。
        # _trainable_indices 用于从完整权重列表中准确提取 trainable 子集。
        self._trainable_indices = [
            i for i, w in enumerate(model.weights) if w.trainable
        ]

        # ── FedDyn：梯度历史 ∇L_k(θ^t_k)，跨轮次累积，不重置 ────────────
        self.grad_history = [
            tf.Variable(tf.zeros_like(v), trainable=False, name=f"gh_{i}")
            for i, v in enumerate(model.trainable_variables)
        ]

        # ── SCAFFOLD：客户端控制变量 c_i，服务端控制变量 c_server ─────────
        # 修复：只对 trainable_variables 分配零向量，避免与 trainable_variables
        # zip 时因含 non-trainable 权重而 shape 错位。
        _trainable_np = [v.numpy() for v in model.trainable_variables]
        self.c_i      = [np.zeros_like(w, dtype=np.float32) for w in _trainable_np]
        self.c_server = [np.zeros_like(w, dtype=np.float32) for w in _trainable_np]
        self.c_delta  = None   # Δc_i，EdgeServer 聚合时读取

        # ── pFedMe：本地个性化模型（warm-start 用，不上传）────────────────
        self.pfedme_theta = None

        # ── 样本数 ─────────────────────────────────────────────────────────
        self.n_samples = n_samples if n_samples is not None else sum(
            x.shape[0] for x, _ in dataset
        )

        # ── 优化器 ─────────────────────────────────────────────────────────
        lr    = config["training"]["learning_rate"]
        decay = config["training"]["lr_decay"]
        steps = max(1, self.n_samples // config["data"]["batch_size"])
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=steps * config["training"]["local_epochs"],
            decay_rate=decay, staircase=True,
        )
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.lr_schedule, momentum=0.9
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # ── 双层参考权重 ───────────────────────────────────────────────────
        self.global_weights = None
        self.edge_weights   = None

    # ══════════════════════════════════════════════════════════════════════
    # 权重管理
    # ══════════════════════════════════════════════════════════════════════

    def set_weights(self, global_weights: list, edge_weights: list = None):
        """接收广播权重并初始化本地模型（以 edge_weights 优先）。"""
        self.global_weights = [w.copy() for w in global_weights]
        if edge_weights is not None:
            self.edge_weights = [w.copy() for w in edge_weights]
            self.model.set_weights(edge_weights)
        else:
            self.edge_weights = None
            self.model.set_weights(global_weights)

    def set_scaffold_c(self, c_server: list):
        """EdgeServer 在广播时注入边缘层控制变量 c_e（SCAFFOLD 专用）。"""
        self.c_server = [c.copy() for c in c_server]

    def _trainable_ref(self, weights: list) -> list:
        """
        从 model.get_weights() 的完整权重列表中提取 trainable_variables 对应子集。

        model.get_weights() 包含 non-trainable 权重（如 BatchNorm 的 moving_mean /
        moving_variance），直接与 model.trainable_variables zip 会导致 shape 错位。
        此方法利用 __init__ 中预计算的 _trainable_indices 安全地提取 trainable 子集。
        """
        return [weights[i] for i in self._trainable_indices]

    # ══════════════════════════════════════════════════════════════════════
    # 本地训练统一入口
    # ══════════════════════════════════════════════════════════════════════

    def local_train(self, round_idx: int):
        """根据 config 分发到对应训练函数。"""
        method = self.config["training"].get("drift_correction", "fedavg")
        # edge 模型优先，fallback 到 global
        ref = self.edge_weights if self.edge_weights is not None else self.global_weights

        if method == "fedavg":
            return self._local_train(round_idx)
        elif method == "fedprox":
            prox_ref = self.config["training"].get("proximal_ref", "edge")
            if prox_ref == "both":
                return self._local_train_fedprox_both(round_idx)
            anchor = self.edge_weights if prox_ref == "edge" and self.edge_weights is not None \
                else self.global_weights
            return self._local_train_fedprox(round_idx, anchor)
        elif method == "feddyn":
            return self._local_train_feddyn(round_idx, ref)
        elif method == "scaffold":
            return self._local_train_scaffold(round_idx)
        elif method == "pfedme":
            return self._local_train_pfedme(round_idx, ref)
        elif method == "perfedavg":
            return self._local_train_perfedavg(round_idx, ref)
        elif method == "hierpfedme":
            return self._local_train_hierpfedme(round_idx, ref)
        else:
            raise ValueError(f"Unknown drift_correction: '{method}'")

    # ══════════════════════════════════════════════════════════════════════
    # FedAvg
    # ══════════════════════════════════════════════════════════════════════

    def _local_train(self, round_idx: int):
        epochs, losses, t0 = self.config["training"]["local_epochs"], [], time.time()
        for _ in range(epochs):
            bl = [float(self._train_step(x, y).numpy()) for x, y in self.dataset]
            losses.append(np.mean(bl))
        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | FedAvg | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    @tf.function
    def _train_step(self, images, labels):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.model.trainable_variables),
                self.model.trainable_variables))
        return loss

    # ══════════════════════════════════════════════════════════════════════
    # FedProx  （Li et al., MLSys 2020）
    # 目标：F_k(θ) + (μ/2)·‖θ − θ_ref‖²
    # ══════════════════════════════════════════════════════════════════════

    def _local_train_fedprox(self, round_idx: int, ref_weights: list):
        mu, epochs = self.config["training"].get("mu", 0.01), \
                     self.config["training"]["local_epochs"]
        ref_tf = [tf.constant(w, dtype=tf.float32) for w in self._trainable_ref(ref_weights)]
        losses, t0 = [], time.time()
        for _ in range(epochs):
            bl = [float(self._train_step_fedprox(x, y, ref_tf, mu).numpy())
                  for x, y in self.dataset]
            losses.append(np.mean(bl))
        avg = float(np.mean(losses))
        tag = self.config["training"].get("proximal_ref", "edge")
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"FedProx(ref={tag},μ={mu}) | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    @tf.function
    def _train_step_fedprox(self, images, labels, ref_tf, mu):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            prox = tf.add_n([tf.reduce_sum(tf.square(w - r))
                             for w, r in zip(self.model.trainable_variables, ref_tf)])
            loss = loss + (mu / 2.0) * prox
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.model.trainable_variables),
                self.model.trainable_variables))
        return loss

    def _local_train_fedprox_both(self, round_idx: int):
        """双近端：(μ₁/2)·‖θ−θ_edge‖² + (μ₂/2)·‖θ−θ_global‖²"""
        mu1, mu2 = self.config["training"].get("mu_edge", 0.1), \
                   self.config["training"].get("mu_global", 0.01)
        epochs = self.config["training"]["local_epochs"]
        e_tf = [tf.constant(w, dtype=tf.float32) for w in self._trainable_ref(self.edge_weights)]
        g_tf = [tf.constant(w, dtype=tf.float32) for w in self._trainable_ref(self.global_weights)]
        losses, t0 = [], time.time()
        for _ in range(epochs):
            bl = [float(self._train_step_fedprox_both(x, y, e_tf, g_tf, mu1, mu2).numpy())
                  for x, y in self.dataset]
            losses.append(np.mean(bl))
        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | FedProx(both) | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    @tf.function
    def _train_step_fedprox_both(self, images, labels, e_tf, g_tf, mu1, mu2):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            pe = tf.add_n([tf.reduce_sum(tf.square(w - r))
                           for w, r in zip(self.model.trainable_variables, e_tf)])
            pg = tf.add_n([tf.reduce_sum(tf.square(w - r))
                           for w, r in zip(self.model.trainable_variables, g_tf)])
            loss = loss + (mu1 / 2.0) * pe + (mu2 / 2.0) * pg
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.model.trainable_variables),
                self.model.trainable_variables))
        return loss

    # ══════════════════════════════════════════════════════════════════════
    # FedDyn  （Acar et al., ICLR 2021）
    # 目标：L_k(θ) − ⟨h_k, θ⟩ + (α/2)·‖θ − θ_anchor‖²
    # ══════════════════════════════════════════════════════════════════════

    def _local_train_feddyn(self, round_idx: int, ref_weights: list):
        """
        θ_anchor = edge 模型（固定快照）。
        训练完成后：h_k ← h_k − α·(θ_new − θ_anchor)。
        """
        alpha  = self.config["training"].get("alpha_feddyn", 0.01)
        epochs = self.config["training"]["local_epochs"]
        a_tf   = tf.constant(alpha, dtype=tf.float32)
        ref_tf = [tf.constant(w, dtype=tf.float32) for w in self._trainable_ref(ref_weights)]
        losses, t0 = [], time.time()
        for _ in range(epochs):
            bl = [float(self._train_step_feddyn(x, y, ref_tf, a_tf).numpy())
                  for x, y in self.dataset]
            losses.append(np.mean(bl))
        # 一次性更新梯度历史
        for h, v, r in zip(self.grad_history, self.model.trainable_variables, ref_tf):
            h.assign(h - a_tf * (v - r))
        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"FedDyn(α={alpha}) | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    @tf.function
    def _train_step_feddyn(self, images, labels, ref_tf, alpha):
        with tf.GradientTape() as tape:
            task = self.loss_fn(labels, self.model(images, training=True))
            lin  = tf.add_n([tf.reduce_sum(h * v)
                             for h, v in zip(self.grad_history,
                                             self.model.trainable_variables)])
            quad = tf.add_n([tf.reduce_sum(tf.square(v - r))
                             for v, r in zip(self.model.trainable_variables, ref_tf)])
            loss = task - lin + (alpha / 2.0) * quad
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.model.trainable_variables),
                self.model.trainable_variables))
        return loss

    # ══════════════════════════════════════════════════════════════════════
    # SCAFFOLD  （Karimireddy et al., ICML 2020 – Option II）
    # 梯度校正：g_corrected = ∇F_k(y) − c_i + c_server
    # ══════════════════════════════════════════════════════════════════════

    def _local_train_scaffold(self, round_idx: int):
        """
        c_server 由 EdgeServer.broadcast_to_clients 通过 set_scaffold_c 注入。

        训练后更新控制变量：
            Δc_i = −c_server + (x_start − y_end) / (K·η)
            c_i  ← c_i + Δc_i
        self.c_delta = Δc_i，EdgeServer._aggregate 读取。
        """
        epochs = self.config["training"]["local_epochs"]
        eta    = self.config["training"]["learning_rate"]   # 近似步长
        x_start = [v.numpy().copy() for v in self.model.trainable_variables]
        c_i_tf  = [tf.constant(ci, dtype=tf.float32) for ci in self.c_i]
        c_s_tf  = [tf.constant(cs, dtype=tf.float32) for cs in self.c_server]
        K, losses, t0 = 0, [], time.time()
        for _ in range(epochs):
            bl = []
            for images, labels in self.dataset:
                bl.append(float(
                    self._train_step_scaffold(images, labels, c_i_tf, c_s_tf).numpy()))
                K += 1
            losses.append(np.mean(bl))
        y_end  = [v.numpy() for v in self.model.trainable_variables]
        K_eta  = max(K * eta, 1e-9)
        new_ci, c_delta = [], []
        for ci, cs, xs, ye in zip(self.c_i, self.c_server, x_start, y_end):
            nci = ci - cs + (xs - ye) / K_eta
            c_delta.append((nci - ci).copy())
            new_ci.append(nci.copy())
        self.c_i    = new_ci
        self.c_delta = c_delta
        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | SCAFFOLD | loss={avg:.4f}")
        return self.model.get_weights(), self.n_samples, avg, time.time() - t0

    @tf.function
    def _train_step_scaffold(self, images, labels, c_i_tf, c_s_tf):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
        raw = tape.gradient(loss, self.model.trainable_variables)
        corrected = [g - ci + cs for g, ci, cs in zip(raw, c_i_tf, c_s_tf)]
        self.optimizer.apply_gradients(
            zip(corrected, self.model.trainable_variables))
        return loss

    # ══════════════════════════════════════════════════════════════════════
    # pFedMe  （T.Li et al., NeurIPS 2020）
    # ══════════════════════════════════════════════════════════════════════

    def _local_train_pfedme(self, round_idx: int, ref_weights: list):
        """
        内层：解 FedProx 子问题得到个性化模型 θ*_k。
        外层：Moreau 包络梯度步：
            w_k^+ = w_ref − β·λ·(w_ref − θ*_k)
        上传 w_k^+，服务端做 FedAvg。
        θ*_k 仅本地保留（可选 warm-start）。
        """
        lam    = self.config["training"].get("lambda_pfedme", 15.0)
        beta   = self.config["training"].get("beta_pfedme",    1.0)
        epochs = self.config["training"]["local_epochs"]
        warm   = self.config["training"].get("pfedme_warm_start", False)
        # 内层初始化
        if warm and self.pfedme_theta is not None:
            self.model.set_weights(self.pfedme_theta)
        else:
            self.model.set_weights(ref_weights)
        ref_tf = [tf.constant(w, dtype=tf.float32) for w in self._trainable_ref(ref_weights)]
        losses, t0 = [], time.time()
        for _ in range(epochs):
            bl = [float(self._train_step_fedprox(x, y, ref_tf, lam).numpy())
                  for x, y in self.dataset]
            losses.append(np.mean(bl))
        theta_star = self.model.get_weights()
        self.pfedme_theta = [w.copy() for w in theta_star]
        # Moreau 梯度步
        w_plus = [r - beta * lam * (r - t) for r, t in zip(ref_weights, theta_star)]
        self.model.set_weights(w_plus)
        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"pFedMe(λ={lam},β={beta}) | loss={avg:.4f}")
        return w_plus, self.n_samples, avg, time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # Per-FedAvg  （Fallah et al., NeurIPS 2020 – FO-MAML）
    # ══════════════════════════════════════════════════════════════════════

    def _local_train_perfedavg(self, round_idx: int, ref_weights: list):
        """
        每 batch 分为 D1（上半）/D2（下半）：
          1. 内层：θ' = w_meta − α·∇F_k(w_meta, D1)
          2. 外层：meta_grad = ∇F_k(θ', D2)
          3. 用 meta_grad 通过 optimizer 更新 w_meta
        上传元参数 w_meta；服务端 FedAvg 聚合。

        修复：ig / mg 来自 trainable_variables（仅 trainable），
        meta_w 来自 get_weights()（含 non-trainable）。
        使用 _trainable_indices 只更新对应位置，non-trainable 保持不变。
        """
        alpha  = self.config["training"].get("alpha_inner", 0.01)
        epochs = self.config["training"]["local_epochs"]
        meta_w = [w.copy() for w in ref_weights]
        losses, t0 = [], time.time()
        for _ in range(epochs):
            bl = []
            for images, labels in self.dataset:
                half   = max(1, images.shape[0] // 2)
                D1x, D1y = images[:half],  labels[:half]
                D2x, D2y = images[half:],  labels[half:]
                if D2x.shape[0] == 0:
                    D2x, D2y = D1x, D1y

                # 内层适配：θ' = meta_w − α·∇F(D1)
                self.model.set_weights(meta_w)
                with tf.GradientTape() as it:
                    l1 = self.loss_fn(D1y, self.model(D1x, training=True))
                ig = it.gradient(l1, self.model.trainable_variables)
                # 只更新 trainable 权重，non-trainable 原样保留
                adapted = [w.copy() for w in meta_w]
                for idx, g in zip(self._trainable_indices, ig):
                    adapted[idx] = meta_w[idx] - alpha * g.numpy()

                # 外层元梯度：∇F(θ', D2)
                self.model.set_weights(adapted)
                with tf.GradientTape() as ot:
                    l2 = self.loss_fn(D2y, self.model(D2x, training=True))
                mg = ot.gradient(l2, self.model.trainable_variables)

                # 更新元参数：通过 optimizer 施加元梯度
                self.model.set_weights(meta_w)
                self.optimizer.apply_gradients(
                    zip(mg, self.model.trainable_variables))
                meta_w = self.model.get_weights()
                bl.append(float(l2.numpy()))
            losses.append(np.mean(bl))
        self.model.set_weights(meta_w)
        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Per-FedAvg(α={alpha}) | loss={avg:.4f}")
        return meta_w, self.n_samples, avg, time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # Hier-pFedMe  （Liu et al., ICME 2025）
    # 三层嵌套优化的客户端部分（内层 + Moreau 梯度步）
    #
    # 性能关键：用 tf.Variable 持有 Theta，整个内外层更新留在 TF 图内。
    # 避免热循环里的 set_weights / get_weights 以及 tf.function retrace。
    #
    # 原始慢版本的三个瓶颈：
    #   1. 每 batch 调用 set_weights / get_weights → CPU↔GPU 数据拷贝
    #   2. anchor_tf 每 batch 新建 tf.constant list → @tf.function 每 batch retrace
    #   3. Moreau 步在 Python 层做 numpy 运算 → 无法 GPU 加速
    # ══════════════════════════════════════════════════════════════════════

    def _local_train_hierpfedme(self, round_idx: int, ref_weights: list):
        lam2   = float(self.config["training"].get("lambda2_hier", 35.0))
        eta2   = float(self.config["training"]["learning_rate"])
        epochs = int(self.config["training"]["local_epochs"])

        # Theta 以 tf.Variable 形式持有，整轮训练只初始化一次
        # 所有更新（内层 FedProx + 外层 Moreau）全部在 _hierpfedme_step 的
        # tf.function 图内完成，不再跨越 Python↔TF 边界
        trainable_ref = self._trainable_ref(ref_weights)
        theta_vars = [
            tf.Variable(w, trainable=False, dtype=tf.float32)
            for w in trainable_ref
        ]
        # 将模型初始化为 Theta
        for v, t in zip(self.model.trainable_variables, theta_vars):
            v.assign(t)

        lam2_tf = tf.constant(lam2, dtype=tf.float32)
        eta2_tf = tf.constant(eta2, dtype=tf.float32)

        losses, t0 = [], time.time()
        for _ in range(epochs):
            bl = [float(self._hierpfedme_step(x, y, theta_vars,
                                               lam2_tf, eta2_tf).numpy())
                  for x, y in self.dataset]
            losses.append(np.mean(bl))

        # 重建完整权重（trainable 部分 = 最终 Theta，non-trainable 不变）
        final_weights = list(ref_weights)
        for idx, t in zip(self._trainable_indices, theta_vars):
            final_weights[idx] = t.numpy()

        avg = float(np.mean(losses))
        print(f"  [Client {self.client_id:>2}] Round {round_idx} | "
              f"Hier-pFedMe(λ2={lam2}) | loss={avg:.4f}")
        return final_weights, self.n_samples, avg, time.time() - t0

    @tf.function
    def _hierpfedme_step(self, images, labels, theta_vars, lam2, eta2):
        """
        单 batch 的 Hier-pFedMe 更新，全程在 TF 图内执行：
          内层：FedProx 一步，以 theta_vars 为 anchor
                loss = CE + (λ2/2)·‖v − Θ‖²
          外层：Moreau 梯度步，就地更新 theta_vars 并重置模型
                Θ ← Θ − η2·λ2·(Θ − θ̃)
                model ← Θ（为下一 batch 准备）
        theta_vars 是 tf.Variable，作为 Variable 传入不触发 retrace。
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(labels, self.model(images, training=True))
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - t))
                for v, t in zip(self.model.trainable_variables, theta_vars)
            ])
            total_loss = loss + (lam2 / 2.0) * prox
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        # θ̃ = 内层更新后的模型权重
        # Moreau 步：Θ ← Θ − η2·λ2·(Θ − θ̃)
        # 同时将模型重置为新 Theta，为下一 batch 准备
        for t, v in zip(theta_vars, self.model.trainable_variables):
            new_t = t - eta2 * lam2 * (t - v)
            t.assign(new_t)
            v.assign(new_t)

        return loss

    # ══════════════════════════════════════════════════════════════════════
    # 本地评估（调试用）
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(self):
        tl = tc = tn = 0
        for x, y in self.dataset:
            p  = self.model(x, training=False)
            tl += self.loss_fn(y, p).numpy() * x.shape[0]
            tc += np.sum(np.argmax(p.numpy(), 1) == y.numpy())
            tn += x.shape[0]
        return tl / tn, tc / tn