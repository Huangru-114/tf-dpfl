"""
server/update_ck.py  —  在线 c_k：逐上传更新的功能分数（S6b；DECISIONS D-087）

3-D 的第二类分数（第一类是 update_geometry 的几何分数），也是阶段三模块 B（edge 内软加权）的打分器。
原文 §7：功能分数 = 在 edge 的干净数据上，测客户端更新前后把样本推向每个类 k 的目标攻击代价 c_k。
Bad-PFL 的恶意训练会降低通往 y_t 的代价 → 用「某一个类的代价异常下降」打分：
    Δ_k = c_k(θ_before) − c_k(θ_i)，  s_i = (max_k Δ_k − median_k Δ_k) / (MAD_k Δ_k + ε)
**本模块只记原始 c_0…c_{K−1}**，s_i 由 `analysis/functional_score.update_score` 在 harness 里离线算
（以后换聚合方式不用重跑）。

做法（每个评分点；评分点 = 有效轮 eff = (g−1)·R + er 满足 eff % update_ck_every == 0 的 edge 轮）：
  · 挂在 `HierFedRepEdgeServer.run_edge_round` 收齐上传之后、`robust_mean` 之前，**只读**；
  · θ_before = 本 edge 轮下发的权重 edge_w；θ_i = edge_w 上换入上传的**可训练权重**（γ/β/卷积核），
    **BN moving 统计量保持 edge_w 的** —— 隔离「权重变化」与「客户端数据造成的统计量差」
    （S6a 的几何记录里这两者是混在一起的，见 update_geometry.py）；
  · 每个模型用本 edge 干净集（500 张，D-064）上倒数第二层特征的类均值做 NCM head，
    c_k = 前 n 张干净样本的定向 PGD（推向 k，ε = backdoor.badpfl_epsilon，steps 步，**无随机起点**）失败的比例；
    K 个类摞成一个批一次算（`analysis.ck_snapshot.ck_reached_batched`）。

**硬约束：记录不得改变训练**（同 S5 / S6a）：专用评分模型（第一次评分时才创建，创建推进 Python random，
整段包 `random.getstate()/setstate()`，F-078）；PGD 不用任何 RNG；不动权重、不动 `_eval_seq` / history。
GPU 确定性（陷阱 #23）：PGD 在推理模式的 BN 上求梯度，`resnet10_torch` 的 TorchBatchNorm 已规避。
守卫：tests/test_update_ck_tf.py。
"""

from __future__ import annotations

import random
import time

import numpy as np

from alignment import get_switch
from analysis import functional_score as FS
from analysis.ck_snapshot import ck_reached_batched, extract, feature_model
from data.pixel_space import to_input_space, valid_range
from models.model_utils import clone_model
from utils.kvline import fmt_list, format_kv


class UpdateCk:
    def __init__(self, config: dict, template_model, stat_idx=(), malicious_fn=None):
        self.config = config
        fed = config.get("federation", {}) or {}
        self.R = int(fed.get("edge_rounds", 1) or 1)
        self.every = int(get_switch(config, "evaluation.update_ck_every"))
        self.n = int(get_switch(config, "evaluation.update_ck_n"))
        self.steps = int(get_switch(config, "evaluation.update_ck_steps"))
        self.K = int(config["data"]["num_classes"])
        eps_px = float((config.get("backdoor") or {}).get("badpfl_epsilon", 4.0 / 255.0))
        self.eps_in = to_input_space(eps_px, config)
        self.lo, self.hi = valid_range(config)
        self.stat = set(int(i) for i in (stat_idx or ()))
        self.malicious_fn = malicious_fn or (lambda: set())
        self._template = template_model
        self._model = None            # 专用评分模型：第一次评分时才创建（在 random 围栏里）
        self._feat = None

    def due(self, global_round: int, edge_round: int):
        """(要不要评分, 有效轮)。"""
        eff = (int(global_round) - 1) * self.R + int(edge_round)
        return eff % self.every == 0, eff

    def _measure(self, weights, xs, ys, xa, ya):
        """装权重 → NCM 原型（整个干净集）→ 前 n 张的 c_k 与 NCM 干净精度。"""
        self._model.set_weights(weights)
        f_all = extract(self._feat, xs)
        protos, _ = FS.class_prototypes(f_all, ys, self.K)
        pred0 = FS.ncm_predict(f_all[:len(xa)], protos)
        reached = ck_reached_batched(self._feat, protos, xa, self.eps_in, self.steps,
                                     self.lo, self.hi, self.K)
        c, _ = FS.ck_from_reached(ya, reached)
        c = [None if np.isnan(protos[k, 0]) else c[k] for k in range(self.K)]   # 干净集里没有的类：无定义，不是 1.0
        return c, float(np.mean(pred0 == np.asarray(ya).reshape(-1)))

    def score(self, edge, global_round, edge_round, client_updates, edge_weights, base_idx):
        """edge 收齐上传后调用。不是评分点 → 什么也不做。只读（edge_weights 是 get_weights 的副本）。"""
        due, eff = self.due(global_round, edge_round)
        if not due:
            return
        xs, ys = getattr(edge, "clean_x", None), getattr(edge, "clean_y", None)
        if xs is None or ys is None or len(ys) == 0:
            raise ValueError(f"evaluation.update_ck：edge {edge.edge_id} 没有干净集（clean_x / clean_y）。"
                             f"main.py 应在开关开时挂上；不静默跳过，否则这个 edge 的分数静默缺失。")
        ys = np.asarray(ys).reshape(-1)
        n = min(self.n, len(ys))
        xa, ya = xs[:n], ys[:n]
        mal = self.malicious_fn()
        st = random.getstate()
        t0 = time.perf_counter()
        try:
            if self._model is None:                     # 创建消耗 Python random → 在围栏里（F-078）
                self._model = clone_model(self._template)
                self._feat = feature_model(self._model)
            before = [np.asarray(w) for w in edge_weights]
            c_b, acc_b = self._measure(before, xs, ys, xa, ya)
            rows = []
            for u in client_updates:
                w = list(before)
                for j in base_idx:
                    if int(j) not in self.stat:         # 可训练权重换成上传的；统计量保持 edge_w 的
                        w[j] = u[0][j]
                c, acc = self._measure(w, xs, ys, xa, ya)
                rows.append((int(u.client_id), c, acc))
        finally:
            random.setstate(st)
        elapsed = time.perf_counter() - t0
        base = {"edge_round": int(edge_round), "effective_round": int(eff)}
        print(format_kv("[CkBefore]", {**base, "n_proto": int(len(ys)), "n_attack": int(n),
                                       "ncm_acc": acc_b, "c": fmt_list(c_b)},
                        round_idx=global_round, edge_id=edge.edge_id))
        for cid, c, acc in rows:
            print(format_kv("[CkScore]", {**base, "cid": cid, "mal": int(cid in mal),
                                          "ncm_acc": acc, "c": fmt_list(c)},
                            round_idx=global_round, edge_id=edge.edge_id))
        print(format_kv("[TimingCk]", {**base, "n_updates": len(rows), "ck_s": round(elapsed, 2)},
                        round_idx=global_round, edge_id=edge.edge_id, digits=2))
