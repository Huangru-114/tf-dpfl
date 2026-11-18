"""
defense/edge_ref.py  —  方向 2（D-129 / D-131）：边缘服务器用自己的干净图作参照，给每个客户端上传打分

第一段 `mode: observe`：**只记录、全部接纳** —— 聚合与不设防同一个函数、同一个顺序（`aggregation.fedavg.aggregate`），
第二段 `mode: filter`（N-014 / D-132）：分数 > τ 的上传不进本次边缘平均（权重与 BN 统计量都不进），本轮全被剔则边缘模型不变；
打分本身与 observe 完全相同（缓存池对照仍收入全部上传 —— 它只是多数原则的读数，不参与剔除）。
所以开 / 关这个防御训练逐位相同（守卫 tests/test_edge_ref_tf.py；GPU 上由 ERP main-t0 两格 vs SNAP 同种子的 [Checksum] 验证）。
算术全部在 `defense/edge_ref_spec.py`（不 import TF）；这里只做：
  · 每个上传的**全部权重**装进草稿模型（BN 统计量在训练模式前向里不起作用，head 不在 BN 层输入的路径上），
    边缘干净图一整批、训练模式前向，取每个 BN 层**输入**的逐样本逐通道一阶 / 二阶矩；
  · 下发模型 θ_before（robust_mean 的 ref_weights = 本 edge 轮下发的 edge 权重）同样一次，给假客户端用；
  · 缓存池（本轮 ∪ 本 edge 其余端最近一次上传的统计量向量）→ `analysis/bn_cluster.detect`（CCS 式多数原则的对照）。
**不读恶意端身份**：只记 client_id，离线用 `run.malicious_ids` 对上（harness/edge_ref_probe.py）。

**硬约束：记录不得改变训练**（同 S5 / S6 / update_ck）：
  · 草稿模型与提取模型第一次用到才建；`clone_model` 的未播种初始化器会推进 Python random（F-078）→ 整段包 `random.getstate()/setstate()`；
  · 训练模式前向只改草稿模型自己的 moving 统计量（每次 set_weights 都整个覆盖）；不碰 edge 模型、不碰上传物；
  · 假客户端的随机数只来自 `default_rng([seed, 0xED6E, 有效轮, edge_id])`，不碰全局 np.random。
GPU 确定性：只有前向、没有梯度；训练模式的 fused BN 前向就是客户端训练每一步都在走的路径（陷阱 #23 只涉及推理模式 BN 的梯度）。
"""

from __future__ import annotations

import random
import time

import numpy as np

from aggregation.fedavg import aggregate as fedavg_aggregate
from analysis.bn_cluster import detect, stat_vector
from utils.kvline import fmt_list, format_kv

from . import edge_ref_spec as S
from .base_defense import BaseDefense


class EdgeRefDefense(BaseDefense):
    name = "edge_ref"
    layers = frozenset({"edge"})

    def __init__(self, params: dict | None = None, seed: int = 42, rule: dict | None = None):
        super().__init__(params)
        self.p = S.edge_ref_params({"defense": {"edge_ref": dict(params or {})}})
        self.seed = int(seed)
        self.rule = rule or {"kind": "dirichlet", "alpha_c": 0.5}
        self.stat_idx = None
        self.edge = None
        self.label = None
        self._cache: dict = {}          # client_id → 最近一次上传的统计量向量（本 edge）
        self._model = None              # 草稿模型（第一次打分才建，在 random 围栏里）
        self._extract = None
        self._layers = None             # 每个 BN 层 {"name", "stage", "eps", "mean_idx", "var_idx"}
        self.last_rows = []             # 最近一次打分的行（测试读）
        self.last_null = None

    @classmethod
    def from_config(cls, config: dict):
        dcfg = ((config or {}).get("defense") or {})
        return cls(dcfg.get("edge_ref") or {}, seed=int((config or {}).get("seed", 42)),
                   rule=S.client_rule(config))

    # ── 注入（EdgeServerBase.__init__，F-107：生产路径）─────────────────────
    def set_stat_indices(self, stat_idx, label=None):
        self.stat_idx = [int(i) for i in stat_idx]
        self.label = None if label is None else int(label)

    def set_edge(self, edge):
        self.edge = edge
        self.label = int(edge.edge_id)

    # ── TF 侧 ────────────────────────────────────────────────────────────
    def _build(self):
        import tensorflow as tf
        from models.model_utils import clone_model
        self._model = clone_model(self.edge.model)
        pos = {id(v): i for i, v in enumerate(self._model.weights)}
        bns = [l for l in self._model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        if not bns:
            raise ValueError("defense.name = edge_ref：模型里没有 BatchNormalization 层，没有统计量可比")
        self._layers = [{"name": l.name, "stage": S.stage_of(l.name), "eps": float(l.epsilon),
                         "mean_idx": pos[id(l.moving_mean)], "var_idx": pos[id(l.moving_variance)]}
                        for l in bns]
        self._extract = tf.keras.Model(self._model.inputs, [l.input for l in bns])

    def _moments(self, weights, x):
        """装权重 → 干净图一整批、训练模式前向 → 每个 BN 层输入的 (m1, m2)，形状 (N, C)，float64。"""
        import tensorflow as tf
        self._model.set_weights(weights)
        outs = self._extract(tf.convert_to_tensor(x), training=True)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        mom = []
        for o in outs:
            axes = list(range(1, len(o.shape) - 1))
            m1 = tf.reduce_mean(o, axis=axes) if axes else o
            m2 = tf.reduce_mean(tf.square(o), axis=axes) if axes else tf.square(o)
            mom.append((m1.numpy().astype(np.float64), m2.numpy().astype(np.float64)))
        return mom

    def _uploaded(self, weights):
        return [(np.asarray(weights[L["mean_idx"]], np.float64), np.asarray(weights[L["var_idx"]], np.float64))
                for L in self._layers]

    # ── 聚合（observe：全部接纳）────────────────────────────────────────
    def aggregate(self, client_updates: list, ref_weights: list) -> list:
        """observe：全部接纳，聚合 = 不设防的同一个函数。filter（N-014）：分数 > τ 的上传丢弃，
        其余照样 fedavg_aggregate；本轮全被剔 → 返回下发权重的副本（本台边缘模型这一轮不变）。"""
        self.score(client_updates, ref_weights)
        keep = S.admit(self.last_rows, self.p["score"], self.p["threshold"] if self.p["mode"] == "filter" else None)
        self.last_admitted = [j for j, k in enumerate(keep) if k]
        if len(self.last_admitted) == len(client_updates):
            return fedavg_aggregate(client_updates)
        if not self.last_admitted:
            return [np.array(w, copy=True) for w in ref_weights]
        return fedavg_aggregate([client_updates[j] for j in self.last_admitted])

    def score(self, client_updates, ref_weights):
        """打分并打印；不改任何传入的东西。"""
        edge = self.edge
        if edge is None or self.stat_idx is None:
            raise RuntimeError("EdgeRefDefense：没有注入 edge / 统计量索引（应由 EdgeServerBase.__init__ 注入，F-107）")
        xs, ys = getattr(edge, "clean_x", None), getattr(edge, "clean_y", None)
        if xs is None or ys is None or len(ys) == 0:
            raise ValueError(f"defense.name = edge_ref：edge {edge.edge_id} 没有干净集（clean_x / clean_y）。"
                             f"main.py 应在 defense.name = edge_ref 时挂上；不静默跳过。")
        ys = np.asarray(ys).reshape(-1)
        g, er = getattr(edge, "current_round", (None, None))
        R = int(edge.config["federation"].get("edge_rounds", 1) or 1)
        eff = None if g is None else (int(g) - 1) * R + int(er or 1)
        K = int(edge.config["data"]["num_classes"])
        props = np.bincount(ys, minlength=K).astype(np.float64)
        props = props / props.sum()
        ids = [int(getattr(u, "client_id", j)) for j, u in enumerate(client_updates)]

        st = random.getstate()
        t0 = time.perf_counter()
        try:
            if self._model is None:                                   # clone_model 推进 Python random（F-078）
                self._build()
            rows = []
            vecs = {}
            for cid, u in zip(ids, client_updates):
                w = [np.asarray(a) for a in u[0]]
                mom = self._moments(w, xs)
                up = self._uploaded(w)
                sc = S.score_one(mom, ys, K, up, self._layers, props)
                sc.update(S.bnguard4(*up[0]))
                rows.append({"cid": cid, "n": int(u[1]), **sc})
                vecs[cid] = stat_vector([w[i] for i in self.stat_idx])
            # θ_before 的逐样本矩 → 假客户端（零分布）
            mom0 = self._moments([np.asarray(a) for a in ref_weights], xs)
            rng = np.random.default_rng([self.seed, S.EDGE_REF_RNG_TAG, int(eff or 0), int(edge.edge_id)])
            P = S.fake_props(rng, props, self.p["n_fake"], self.rule)
            fakes = []
            for pk in P:
                idx = S.fake_indices(rng, ys, pk, self.p["fake_n"])
                up = [S.subset_moments(m1, m2, idx) for m1, m2 in mom0]
                fakes.append(S.score_one(mom0, ys, K, up, self._layers, props))
        finally:
            random.setstate(st)

        # 多数原则的对照（CCS 式缓存池；纯 numpy + scikit-learn，不在围栏里也不碰 random）
        pool = {k: v for k, v in self._cache.items() if k not in vecs}
        pool.update(vecs)
        self._cache.update(vecs)
        res = detect(pool)
        rej = set(res["rejected"])
        for r in rows:
            r["ccs_cos_med"] = S.cosine_median_to_others(pool, r["cid"])
            r["ccs_rej"] = int(r["cid"] in rej)
        elapsed = time.perf_counter() - t0

        base = {"effective_round": eff, "edge_round": er}
        keep = S.admit(rows, self.p["score"], self.p["threshold"] if self.p["mode"] == "filter" else None)
        for r, k in zip(rows, keep):
            r["admitted"] = int(k)
        for r in rows:
            print("    " + format_kv("[EdgeRefScore]", {**base, **{k: r[k] for k in S.SCORE_FIELDS[2:]}},
                                     round_idx=g, edge_id=edge.edge_id, digits=6))
        null = {**base, "n_clean": int(len(ys)), "ccs_pool": len(pool), "ccs_status": res["status"],
                **{k: fmt_list([f[k] for f in fakes], digits=6) for k in S.NULL_FIELDS[5:]}}
        print("    " + format_kv("[EdgeRefNull]", null, round_idx=g, edge_id=edge.edge_id))
        print("    " + format_kv("[TimingEdgeRef]", {**base, "n_updates": len(rows), "edge_ref_s": round(elapsed, 2)},
                                 round_idx=g, edge_id=edge.edge_id, digits=2))
        self.last_rows, self.last_null = rows, {"fakes": fakes, "pool": len(pool), "status": res["status"]}
        return rows
