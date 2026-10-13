"""
hardening/core.py  —  edge 侧对抗训练（阶段三 SA0）的 TF 部分：一次「加固」= 在 edge 干净集上把上传前的 body 训几步

    h = EdgeHardener(template_model, config)
    new_weights, info = h.harden(edge_weights, clean_x, clean_y, cfg, rng=np.random.default_rng([...]), lr=…)

`cfg` 是 `hardening.spec.expand_grid` 的一项（p0_configs.yaml）。纯算术（网格、步长、探针拟合、记账、判定量）在 `hardening/spec.py`。
SA0 的离线驱动 `analysis/p0_snapshot.py` 用它；SA1 的在线接线以后复用同一个类。

与官方实现的逐条差异见 SA0 语义 diff（DECISIONS D-103 / D-104、FINDINGS F-095）。要点：
  · **只动 body 的可训练变量**（卷积核、BN γ/β）+ 一个临时线性探针 head；edge 模型自己的 head 从未训练（FedRep 下 edge 只聚合 body），
    探针在冻结 body 的倒数第二层特征上拟合（spec.fit_probe），加固时与 body 一起训、用完丢弃。
  · **BN moving 统计量不变**：bn=train 时前向用 batch 统计（Keras 会顺手改 moving 统计量）→ 结束后按索引写回；
    bn=frozen 时一律推理模式（`resnet10_torch` 的 `TorchBatchNorm` 推理路径，GPU 确定性下可对输入求梯度，陷阱 #23）。
  · 扰动一律在**模型输入空间**（逐通道标准化后）里做：像素单位的 ε / 步长 ÷ std，投影到 ε 球后再投影到 `valid_range`。
    投影只在更新之后做，**求梯度的前向里不裁剪**（tape 里的 clip 让超界坐标梯度为 0 → sign(0) = 0、卡死，审查 R1）。
    SAU 官方连合法范围都不投影（clamp 被注释），这里投影（与 Bad-PFL 的 ξ、ck_snapshot 同一套）。
  · SAU 照官方：干净集不增强、Adam(amsgrad)、参照模型推理模式 + 在推理模式特征上拟合的探针（审查 R2–R4）。
  · 模型末层是 softmax（陷阱 #6：from_logits=False）→ 各项 loss 用「特征 × 探针 W + b」的 logits，不从概率反推。
  · **不得改变任何外部状态**：所有随机数来自调用方给的 Generator；草稿模型在构造时创建（`clone_model` 会消耗 Python random，F-078），
    构造与每次 harden 都包在 `random.getstate()/setstate()` 里；传进来的 weights 不被修改。
"""

from __future__ import annotations

import random
import time

import numpy as np
import tensorflow as tf

from analysis import functional_score as FS
from analysis.ck_snapshot import ck_reached_batched, extract, feature_model
from data.epoch_pipeline import augment_batch
from data.pixel_space import pixel_stats
from hardening import spec
from models.cnn import get_base_head_indices, get_bn_stat_indices
from models.model_utils import clone_model


# ── loss 的纯 TF 小块（单独成函数，测试拿 numpy 手写公式逐项对）──────────────────────

def kl_sum(p_nat, logits_adv):
    """Σ_i KL(p_nat,i ‖ softmax(logits_adv,i)) = Σ p·log p − p·log_softmax（torch `KLDivLoss(reduction='sum')` 的同一量）。"""
    return tf.reduce_sum(tf.math.xlogy(p_nat, p_nat) - p_nat * tf.nn.log_softmax(logits_adv))


def js_rows(logits_m, logits_r):
    """SAU 的 JS 项（逐样本，对类求和）：概率先 clamp 到 [1e-8, 1]，mix = 两者均值。"""
    pm = tf.clip_by_value(tf.nn.softmax(logits_m), 1e-8, 1.0)
    pr = tf.clip_by_value(tf.nn.softmax(logits_r), 1e-8, 1.0)
    mix = (pm + pr) / 2.0
    return tf.reduce_sum(0.5 * (pm * tf.math.log(pm) + pr * tf.math.log(pr))
                         - 0.5 * (pm * tf.math.log(mix) + pr * tf.math.log(mix)), axis=1)


def sau_shared(logits_p, pert_label_ref, poison, n_classes):
    """SAU 反学习的共享项：参照模型被翻转的样本上 −½[log(1e-6 + min(neg, .999)) + log(1 + 1e-6 − max(pos, .001))]，求和后 /B。"""
    p = tf.nn.softmax(logits_p)
    oh = tf.one_hot(pert_label_ref, n_classes)
    pos = tf.reduce_sum(p * oh, axis=1)
    neg = tf.reduce_sum(p * (1.0 - oh), axis=1)
    term = (-tf.math.log(1e-6 + tf.minimum(neg, 0.999))
            - tf.math.log(1.0 + 1e-6 - tf.maximum(pos, 0.001)))
    B = tf.cast(tf.shape(logits_p)[0], tf.float32)
    return tf.reduce_sum(tf.where(poison, term, 0.0)) / 2.0 / B


def make_optimizer(cfg: dict, lr: float | None):
    """
    SGD（客户端 body 同款，无动量）或 SAU 的 Adam。SAU 照官方 torch.optim.Adam(lr, betas=(.9, .999), amsgrad=True, eps=1e-8)。
    jit_compile=False：每次加固都新建优化器，XLA 会每次重新编译更新步（GPU 上缺省开 XLA）。
    Keras 的 ε 加在未做偏差校正的 √v 上（torch 加在校正后的上）→ 等价于 torch 的 ε/√(1−β2^t)，t = 1 时 3.2e-7、
    随 t 收敛到 1e-8；`tests/test_hardening_tf.py` 在先 1e-2 后 1e-4 的梯度序列上与 torch 公式逐步对过（差 ≤ 3.8e-4 个 lr）。
    """
    if cfg["optimizer"] == "adam":
        s = cfg.get("sau") or {}
        return tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"]), beta_1=0.9, beta_2=0.999,
                                        epsilon=float(s.get("adam_epsilon", 1e-8)),
                                        amsgrad=bool(s.get("amsgrad", True)), jit_compile=False)
    if lr is None:
        raise ValueError("SGD 目标需要 lr（客户端 body 的 lr，spec.lr_at）")
    return tf.keras.optimizers.SGD(learning_rate=float(lr), jit_compile=False)


class EdgeHardener:
    """一套草稿模型（被加固的模型 + SAU 的冻结参照模型），复用于全部加固调用。"""

    def __init__(self, template_model, config: dict, *, probe: dict | None = None,
                 kstar: dict | None = None, batch: int = 32):
        self.config = config
        self.K = int(config["data"]["num_classes"])
        self.batch = int(batch)
        self.probe_cfg = dict(probe or {"steps": 300, "lr": 0.01, "l2": 1e-4})
        self.kstar_cfg = dict(kstar or {"n": 500, "eps_px": 4.0 / 255.0, "steps": 10})
        st = random.getstate()
        try:
            self.model = clone_model(template_model)
            self.ref = clone_model(template_model)
        finally:
            random.setstate(st)
        self.feat = feature_model(self.model)
        self.feat_ref = feature_model(self.ref)
        split = get_base_head_indices(self.model, self.K)
        self.head_idx = list(split["head_weight_indices"])
        self.stat_idx = list(get_bn_stat_indices(self.model))
        tv = self.model.trainable_variables
        self.body_vars = [tv[j] for j in split["base_trainable_indices"]]
        d = int(self.feat.output_shape[-1])
        self.W = tf.Variable(tf.zeros((d, self.K)), trainable=True, name="probe_W")
        self.b = tf.Variable(tf.zeros((self.K,)), trainable=True, name="probe_b")
        lo, hi = spec.valid_range(config)
        self.lo, self.hi = tf.constant(lo), tf.constant(hi)
        _, self.std = pixel_stats(config)

    # ── 单位换算 ────────────────────────────────────────────────────────────
    def to_in(self, v_px) -> tf.Tensor:
        """像素单位的幅度 → 输入空间（逐通道）。"""
        return tf.constant(spec.to_input_space(float(v_px), self.config))

    def _clip(self, x):
        return tf.clip_by_value(x, self.lo, self.hi)

    def _project(self, xa, x0, eps_in):
        return self._clip(tf.clip_by_value(xa, x0 - eps_in, x0 + eps_in))

    def _project_pert(self, pert, x0, eps_in):
        """扰动形式的同一个投影：先到 [−ε, ε]，再让 x0 + pert 落进合法范围（返回的仍是扰动）。"""
        return self._clip(x0 + tf.clip_by_value(pert, -eps_in, eps_in)) - x0

    # ── 前向 ────────────────────────────────────────────────────────────────
    def logits(self, x, training: bool):
        return tf.matmul(self.feat(x, training=training), self.W) + self.b

    def logits_ref(self, x, W, b):
        """SAU 的参照模型：上传前的 w_e + 拟合好的探针（冻结、推理模式 —— 官方 model_ref.eval()）。"""
        return tf.matmul(self.feat_ref(x, training=False), W) + b

    def features(self, x, training: bool, rng: np.random.Generator) -> np.ndarray:
        """探针拟合用的特征：推理模式整批算；训练模式按 rng 打乱后逐 batch 算（与加固时同一种 BN 统计）。"""
        x = np.asarray(x, np.float32)
        if not training:
            return extract(self.feat, x)
        out = np.zeros((len(x), int(self.feat.output_shape[-1])), np.float64)
        perm = rng.permutation(len(x))
        for s in range(0, len(x), self.batch):
            idx = perm[s:s + self.batch]
            out[idx] = self.feat(tf.constant(x[idx]), training=True).numpy()
        return out

    # ── c_k 与 k*（AT-tgt）────────────────────────────────────────────────────
    def body_ck(self, x, y, x_att, y_att) -> list:
        """当前 body 的 c_k：干净集全部图的 NCM 原型 + 对 x_att 的定向 PGD（ck_snapshot 的同一实现，无随机起点）。"""
        protos, _ = FS.class_prototypes(extract(self.feat, np.asarray(x, np.float32)),
                                        np.asarray(y).reshape(-1), self.K)
        reached = ck_reached_batched(self.feat, protos, np.asarray(x_att, np.float32),
                                     spec.to_input_space(spec.parse_px(self.kstar_cfg["eps_px"]), self.config),
                                     int(self.kstar_cfg["steps"]), self.lo.numpy(), self.hi.numpy(), self.K)
        c, _ = FS.ck_from_reached(np.asarray(y_att).reshape(-1), reached)
        return c

    @staticmethod
    def pick_kstar(c: list, rng: np.random.Generator) -> tuple:
        """k* = argmin c_k（无定义的类不参与）；并列时用 rng 在并列者里选（np.argmin 会偏向类 0 = 目标类，F-085 的 argmax 同理）。"""
        vals = [(k, v) for k, v in enumerate(c) if v is not None]
        if not vals:
            return None, []
        m = min(v for _, v in vals)
        ties = [k for k, v in vals if v == m]
        return int(ties[0] if len(ties) == 1 else rng.choice(ties)), ties

    # ── 对抗样本 ────────────────────────────────────────────────────────────
    def pgd_ce(self, x, y, eps_in, alpha_in, steps, training, *, rng=None, start="uniform",
               target=None):
        """
        L∞ PGD（输入空间）。target=None：无目标，沿 +sign(∇CE(y)) 上升（Madry）；
        target=k：定向，沿 −sign(∇CE(k)) 下降。start="uniform" = Madry 的均匀随机起点；"none" = 从 x 出发。
        """
        x0 = tf.convert_to_tensor(x, tf.float32)
        if start == "uniform":
            u = rng.uniform(-1.0, 1.0, size=tuple(x0.shape)).astype(np.float32) * eps_in.numpy()
            xa = self._clip(x0 + u)
        else:
            xa = x0
        lab = y if target is None else tf.fill(tf.shape(y), tf.cast(target, y.dtype))
        sgn = 1.0 if target is None else -1.0
        for _ in range(int(steps)):
            with tf.GradientTape() as tape:
                tape.watch(xa)
                loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=lab, logits=self.logits(xa, training)))
            g = tape.gradient(loss, xa)
            xa = self._project(xa + sgn * alpha_in * tf.sign(g), x0, eps_in)
        return tf.stop_gradient(xa)

    def trades_adv(self, x, eps_in, alpha_in, steps, training, rng):
        """TRADES 的内层：从 x + 0.001·randn（像素单位）出发，沿 +sign(∇KL(p_nat ‖ p_adv)) 上升，投影 + 裁剪。"""
        x0 = tf.convert_to_tensor(x, tf.float32)
        p_nat = tf.stop_gradient(tf.nn.softmax(self.logits(x0, training)))
        noise = (rng.standard_normal(size=tuple(x0.shape)).astype(np.float32)
                 * (spec.TRADES_START_STD_PX / self.std).astype(np.float32))
        xa = x0 + noise
        for _ in range(int(steps)):
            with tf.GradientTape() as tape:
                tape.watch(xa)
                kl = kl_sum(p_nat, self.logits(xa, training))
            g = tape.gradient(kl, xa)
            xa = self._project(xa + alpha_in * tf.sign(g), x0, eps_in)
        return tf.stop_gradient(xa)

    def sau_pert(self, x, y, eps_in, alpha_in, steps, training, W_ref, b_ref, beta_1, beta_2):
        """
        SAU 的 Shared_PGD（BackdoorBench defense/sau.py）：两个模型（被净化的 / 冻结参照）上同时找共享对抗样本。
        init max（全部 +ε）；loss = β1·loss_adv + β2·loss_cross，沿 −sign(∇) 更新扰动并投影到 [−ε, ε]；
        loss_adv = −(Σ 未翻转样本的 CE(y))/2/B（各模型用自己的掩码）；loss_cross = 非共享样本上两模型输出的 JS，/B。
        更新之后若（更新前的）共享掩码已全真 → 提前停（官方同）。返回扰动（输入空间）。
        与官方唯一的差别：每次投影之后再让 x0 + pert 落进合法范围（`_project_pert`）；前向直接用 x0 + pert、不裁剪，
        所以贴边坐标的梯度照常回传（审查 R1：旧实现在 tape 里裁剪，init max 下贴边坐标 5 步都卡在 +ε）。
        """
        x0 = tf.convert_to_tensor(x, tf.float32)
        B = tf.cast(tf.shape(x0)[0], tf.float32)
        ori = tf.argmax(self.logits(x0, training), axis=1, output_type=y.dtype)
        ori_ref = tf.argmax(self.logits_ref(x0, W_ref, b_ref), axis=1, output_type=y.dtype)
        pert = self._project_pert(tf.zeros_like(x0) + eps_in, x0, eps_in)
        for _ in range(int(steps)):
            with tf.GradientTape() as tape:
                tape.watch(pert)
                xa = x0 + pert
                lm = self.logits(xa, training)
                lr = self.logits_ref(xa, W_ref, b_ref)
                pl = tf.argmax(lm, axis=1, output_type=y.dtype)
                plr = tf.argmax(lr, axis=1, output_type=y.dtype)
                succ, succ_r = pl != ori, plr != ori_ref
                shared = succ & succ_r & (pl == plr)
                ce_m = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=lm)
                ce_r = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=lr)
                loss_adv = -(tf.reduce_sum(tf.where(succ, 0.0, ce_m))
                             + tf.reduce_sum(tf.where(succ_r, 0.0, ce_r))) / 2.0 / B
                loss_cross = tf.reduce_sum(tf.where(shared, 0.0, js_rows(lm, lr))) / B
                loss = beta_1 * loss_adv + beta_2 * loss_cross
            g = tape.gradient(loss, pert)
            pert = self._project_pert(pert - alpha_in * tf.sign(g), x0, eps_in)
            if bool(tf.reduce_all(shared)):
                break
        return tf.stop_gradient(pert)

    def tgt_adv(self, x, y, kstar, eps_in, alpha_in, steps, sigma_in, training):
        """
        AT-tgt：定向 PGD 推向 k*（模拟 δ，无随机起点）+ 一步无目标 FGSM（模拟 ξ），总扰动投影到 ε+σ 球。
        真实类已经是 k* 的样本不做定向步（定向下降会降低它们的 CE(y)；与 c_k「y ≠ k 才算推向」的口径一致，D-105），只做 FGSM。
        """
        x0 = tf.convert_to_tensor(x, tf.float32)
        xt = self.pgd_ce(x0, y, eps_in, alpha_in, steps, training, start="none", target=kstar)
        keep = tf.reshape(tf.cast(y, tf.int64) == int(kstar), (-1, 1, 1, 1))
        xa = tf.where(keep, x0, xt)
        with tf.GradientTape() as tape:
            tape.watch(xa)
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=self.logits(xa, training)))
        g = tape.gradient(loss, xa)
        return tf.stop_gradient(self._project(xa + sigma_in * tf.sign(g), x0, eps_in + sigma_in))

    # ── 一个训练 batch 的 loss ───────────────────────────────────────────────
    def batch_loss(self, cfg, xb, yb, training, *, rng, kstar=None, ref=None):
        obj = cfg["objective"]
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits
        if obj == "cft":
            return lambda: tf.reduce_mean(ce(labels=yb, logits=self.logits(xb, training)))
        eps_in = self.to_in(cfg["eps_px"])
        alpha_in = self.to_in(spec.step_px(cfg))
        k = int(cfg["pgd_steps"])
        if obj == "pgd":
            xa = self.pgd_ce(xb, yb, eps_in, alpha_in, k, training, rng=rng, start="uniform")
            return lambda: tf.reduce_mean(ce(labels=yb, logits=self.logits(xa, training)))
        if obj == "trades":
            xa = self.trades_adv(xb, eps_in, alpha_in, k, training, rng)
            beta = float(cfg["beta"])

            def f():
                ln = self.logits(xb, training)
                la = self.logits(xa, training)
                B = tf.cast(tf.shape(xb)[0], tf.float32)
                return (tf.reduce_mean(ce(labels=yb, logits=ln))
                        + beta * kl_sum(tf.nn.softmax(ln), la) / B)
            return f
        if obj == "sau":
            s = cfg["sau"]
            W_ref, b_ref = ref
            pert = self.sau_pert(xb, yb, eps_in, alpha_in, k, training, W_ref, b_ref,
                                 float(s["beta_1"]), float(s["beta_2"]))
            xp = self._clip(xb + pert)                                   # pert 已投影过；这里只防舍入

            def f():
                both = self.logits(tf.concat([xb, xp], axis=0), training)    # 官方：干净 + 扰动一次前向
                lc, lp = both[:tf.shape(xb)[0]], both[tf.shape(xb)[0]:]
                lrc = self.logits_ref(xb, W_ref, b_ref)
                lrp = self.logits_ref(xp, W_ref, b_ref)
                ori_ref = tf.argmax(lrc, axis=1, output_type=yb.dtype)
                plr = tf.argmax(lrp, axis=1, output_type=yb.dtype)
                pois = (plr != yb) & (plr != ori_ref)
                loss_cl = tf.reduce_mean(ce(labels=yb, logits=lc))
                loss_at = tf.reduce_mean(ce(labels=yb, logits=lp))
                loss_sh = sau_shared(lp, plr, pois, self.K)
                return (float(s["lmd_1"]) * loss_cl + float(s["lmd_2"]) * loss_at
                        + float(s["lmd_3"]) * loss_sh)
            return f
        if obj == "tgt":
            xa = self.tgt_adv(xb, yb, kstar, eps_in, alpha_in, k, self.to_in(cfg["sigma_px"]), training)
            return lambda: tf.reduce_mean(ce(labels=yb, logits=self.logits(xa, training)))
        raise ValueError(f"未知目标 {obj!r}")

    # ── 一次加固 ────────────────────────────────────────────────────────────
    def harden(self, weights: list, x, y, cfg: dict, *, rng: np.random.Generator, lr: float | None,
               head_init: tuple | None = None, augment: bool = True) -> tuple:
        """
        weights：edge 上传前的完整 get_weights() 列表（不被修改）。x / y：干净集（输入空间、整数标签）。
        lr：SGD 的学习率（= 该时刻客户端 body 的 lr，spec.lr_at）；SAU 用配置里的 Adam lr。
        head_init：(W, b) → 不拟合探针，用它当 head 的初值（诊断：本 edge 各端私有 head 的均值）。
        返回 (新权重, info)。新权重只在 body 的可训练变量上与输入不同；head 与 BN 统计量逐位不变。
        """
        st = random.getstate()
        t0 = time.perf_counter()
        try:
            x = np.asarray(x, np.float32)
            y = np.asarray(y).reshape(-1).astype(np.int64)
            training = cfg["bn"] == "train"
            self.model.set_weights([np.asarray(w) for w in weights])
            info = {"id": cfg["id"], "n_clean": int(len(y))}
            # 探针拟合的随机数用一个子种子：无论拟不拟合都先抽它 → head_init 诊断与主配置的其余随机数逐位配对
            prng = np.random.default_rng(int(rng.integers(2 ** 62)))
            if head_init is None:
                F = self.features(x, training, prng)
                p = spec.fit_probe(F, y, self.K, steps=int(self.probe_cfg["steps"]),
                                   lr=float(self.probe_cfg["lr"]), l2=float(self.probe_cfg.get("l2", 1e-4)))
                W0, b0 = p["W"], p["b"]
                info["probe_acc"] = p["train_acc"]
            else:
                W0, b0 = (np.asarray(head_init[0], np.float32), np.asarray(head_init[1], np.float32))
                info["probe_acc"] = None
                info["head"] = "given"
            self.W.assign(W0)
            self.b.assign(b0)
            # bn=train 时拟合探针的前向已经改了 moving 统计量 → 先写回，后面的推理模式前向（c_k、SAU 参照）才对
            self._restore(weights)
            ref = None
            if cfg["objective"] == "sau":
                # 参照模型跑推理模式（官方 model_ref.eval()）→ 它的探针要在**推理模式**特征上拟合（审查 R4）。
                # bn=frozen 时主模型的探针本来就是在同一份推理模式特征上拟合的 → 直接共用（重拟合结果逐位相同）。
                self.ref.set_weights([np.asarray(w) for w in weights])
                if head_init is None and training:
                    pr = spec.fit_probe(extract(self.feat_ref, x), y, self.K, steps=int(self.probe_cfg["steps"]),
                                        lr=float(self.probe_cfg["lr"]), l2=float(self.probe_cfg.get("l2", 1e-4)))
                    Wr, br, info["ref_acc"] = pr["W"], pr["b"], pr["train_acc"]
                else:
                    Wr, br = W0, b0
                    info["ref_acc"] = (info["probe_acc"] if head_init is None else
                                       float(np.mean(np.argmax(extract(self.feat_ref, x) @ W0 + b0, axis=1) == y)))
                ref = (tf.constant(Wr), tf.constant(br))
            kstar = None
            if cfg["objective"] == "tgt":
                att = spec.stratified_subset(y, int(self.kstar_cfg["n"]), rng)
                c = self.body_ck(x, y, x[att], y[att])
                kstar, ties = self.pick_kstar(c, rng)
                info.update({"ck": c, "kstar": kstar, "kstar_ties": ties,
                             "frac_y_is_kstar": None if kstar is None else float(np.mean(y == kstar))})
                if kstar is None:
                    raise ValueError("AT-tgt：所有类的 c_k 都无定义，选不出 k*")
            opt = make_optimizer(cfg, lr)
            info["lr"] = float(cfg["lr"]) if cfg["optimizer"] == "adam" else float(lr)
            aug = bool(augment) and spec.augments(cfg)                 # SAU 不增强（审查 R2）
            info["augment"] = aug
            var = self.body_vars + [self.W, self.b]
            n_steps, losses = 0, []
            for _ in range(int(cfg["epochs"])):
                for idx in spec.epoch_batches(len(y), self.batch, rng):
                    xb = augment_batch(x[idx], rng) if aug else x[idx]
                    xb, yb = tf.constant(xb), tf.constant(y[idx])
                    loss_fn = self.batch_loss(cfg, xb, yb, training, rng=rng, kstar=kstar, ref=ref)
                    with tf.GradientTape() as tape:
                        loss = loss_fn()
                    grads = tape.gradient(loss, var)
                    opt.apply_gradients(zip(grads, var))
                    losses.append(float(loss))
                    n_steps += 1
            new = self._restore(weights, out=self.model.get_weights())
            info.update({"steps": n_steps, "loss_first": losses[0] if losses else None,
                         "loss_last": losses[-1] if losses else None,
                         "fb": spec.fb_harden(cfg, len(y), self.batch),
                         "seconds": round(time.perf_counter() - t0, 3)})
            return new, info
        finally:
            random.setstate(st)

    def _restore(self, weights, out=None):
        """把 BN moving 统计量与 head 写回输入的值。out=None → 直接改草稿模型；否则改 out 列表并返回。"""
        if out is None:
            cur = self.model.get_weights()
            for i in self.stat_idx + self.head_idx:
                cur[i] = np.asarray(weights[i])
            self.model.set_weights(cur)
            return cur
        for i in self.stat_idx + self.head_idx:
            out[i] = np.array(weights[i], copy=True)
        return out
