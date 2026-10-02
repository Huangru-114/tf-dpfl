"""
server/update_geometry.py  —  逐更新几何日志（S6a ①；DECISIONS D-085）

问题：攻击者的 body 更新在几何上和良性更新分得开吗？3-D「edge 视角能否比全局视角更好地检测
恶意更新」的前提就是这个（附录 D：Bad-PFL 的更新在几何上可能根本分不出来 —— 没有证据）。

每个 edge 轮、每个 edge 收齐上传之后（`robust_mean` 之前），**只读**地记录每个上传的 body-only 更新
Δ_i = w_i[base] − edge_w[base]（edge_w = 本 edge 轮下发的权重）：
  · norm       ‖Δ_i‖
  · cos_edge   Δ_i 与「本 edge 其他更新之和」的余弦（留一；均值与和同向，余弦不变）
  · cos_global Δ_i 与「本 edge 轮全体 edge 的其他更新之和」的余弦（留一）
  · CountSketch 草图（固定维数，fp16 进 dumps，供离线任意重算内积）

留一余弦用「和」算：Δ_i·(S − Δ_i)，‖S − Δ_i‖² = ‖S‖² − 2 S·Δ_i + ‖Δ_i‖²，不需要两两内积。
float64 累加；Δ 本身存 float32。只有 1 个更新时无「其他」→ None（陷阱 #13：无定义不是 0）。

S6b（D-087）更正一个混杂：base 索引**包含 BN 的 moving_mean / moving_variance**，FedRep 下统计量私有（A27），
上传的 body Δ 里因此混着客户端自己的 BN 统计量（恶意端的投毒样本会改统计量）。现在把 Δ 拆成
「可训练权重部分」（γ/β/卷积核）与「BN 统计量部分」：多记 `norm_w` / `norm_s` / `cos_edge_w` / `cos_global_w`
（旧列 `norm` / `cos_edge` / `cos_global` 仍是整个 body，逐位不变）；草图另存权重部分 `sketch_w`
（统计量部分 = `sketch` − `sketch_w`，同一个哈希）。没有统计量索引（缺省）→ 新列为 None。

**硬约束：记录不得改变训练。** 本模块不碰任何全局 RNG（草图的哈希与符号来自常量种子的独立
Generator）、不修改传入的权重（只读 `get_weights()` 返回的副本）。**不 import TF。**
守卫：tests/test_update_geometry.py。
"""

from __future__ import annotations

import numpy as np

from utils.kvline import fmt_list, format_kv

SKETCH_SEED = 0x5C7E
DEFAULT_SKETCH_DIM = 4096


class CountSketch:
    """D 维向量 → dim 维草图：sketch[h[j]] += s[j]·x[j]。对 x 线性，<Sa,Sb> 对 <a,b> 无偏。"""

    def __init__(self, D: int, dim: int = DEFAULT_SKETCH_DIM, seed: int = SKETCH_SEED):
        rng = np.random.default_rng([int(seed)])          # 独立 Generator，不碰全局 RNG
        self.D, self.dim = int(D), int(dim)
        self.h = rng.integers(0, self.dim, size=self.D).astype(np.int32)
        self.s = (rng.integers(0, 2, size=self.D) * 2 - 1).astype(np.float32)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        if x.shape[0] != self.D:
            raise ValueError(f"向量长度 {x.shape[0]} ≠ 草图的 D={self.D}")
        return np.bincount(self.h, weights=self.s * x.astype(np.float64),
                           minlength=self.dim).astype(np.float32)


def body_delta(client_weights, edge_weights, base_idx) -> np.ndarray:
    """body-only 更新 Δ = w[base] − edge_w[base]，展平成 float32 一维向量。不修改入参。"""
    parts = [(np.asarray(client_weights[j], np.float64)
              - np.asarray(edge_weights[j], np.float64)).reshape(-1) for j in base_idx]
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def stat_mask(edge_weights, base_idx, stat_idx) -> np.ndarray:
    """body Δ（按 base_idx 顺序展平）里属于 BN 统计量的坐标 → 布尔掩码。stat_idx 为空 → 全 False。"""
    sset = set(int(i) for i in (stat_idx or ()))
    parts = [np.full(int(np.asarray(edge_weights[j]).size), int(j) in sset, dtype=bool)
             for j in base_idx]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def _loo_cos(delta_dot_S: float, norm2: float, S_norm2: float):
    """Δ 与 (S − Δ) 的余弦。S − Δ 为零向量或 Δ 为零 → None。"""
    other2 = S_norm2 - 2.0 * delta_dot_S + norm2
    if norm2 <= 0.0 or other2 <= 1e-30 * max(S_norm2, 1.0):
        return None
    return float((delta_dot_S - norm2) / (np.sqrt(norm2) * np.sqrt(other2)))


def geometry(groups: dict) -> dict:
    """
    groups: {edge_id: [Δ_i, …]}（同一个 edge 轮里各 edge 的更新）。
    返回 {edge_id: {"norm": [...], "cos_edge": [...], "cos_global": [...]}}，与输入同序。
    """
    S_e, norms2 = {}, {}
    for eid, ds in groups.items():
        norms2[eid] = [float(np.dot(d.astype(np.float64), d.astype(np.float64))) for d in ds]
        S_e[eid] = (np.sum([d.astype(np.float64) for d in ds], axis=0)
                    if len(ds) else None)
    present = [e for e in groups if S_e[e] is not None]
    S_all = np.sum([S_e[e] for e in present], axis=0) if present else None
    n_all = sum(len(groups[e]) for e in groups)
    out = {}
    for eid, ds in groups.items():
        rec = {"norm": [], "cos_edge": [], "cos_global": []}
        for i, d in enumerate(ds):
            d64 = d.astype(np.float64)
            n2 = norms2[eid][i]
            rec["norm"].append(float(np.sqrt(n2)))
            if len(ds) > 1:
                rec["cos_edge"].append(_loo_cos(float(np.dot(d64, S_e[eid])), n2,
                                                float(np.dot(S_e[eid], S_e[eid]))))
            else:
                rec["cos_edge"].append(None)
            if n_all > 1:
                rec["cos_global"].append(_loo_cos(float(np.dot(d64, S_all)), n2,
                                                  float(np.dot(S_all, S_all))))
            else:
                rec["cos_global"].append(None)
        out[eid] = rec
    return out


class UpdateGeometry:
    """
    边缘服务器在收齐上传后调 `observe`；云在「所有 edge 都跑完第 er 个 edge 轮」后调 `flush`
    （算几何、打 `[UpdateGeo]`、清缓冲）。缓冲只在一个 edge 轮内存在（约 10 个更新 × 19.6 MB）。
    """

    def __init__(self, malicious_ids=(), sketch_dim: int = DEFAULT_SKETCH_DIM, stat_idx=()):
        self.malicious_ids = set(int(i) for i in malicious_ids)
        self.sketch_dim = int(sketch_dim)
        self.stat_idx = tuple(int(i) for i in (stat_idx or ()))
        self._mask = None              # 统计量坐标掩码（第一次 observe 时按形状建一次）
        self._sketch = None
        self._buf = {}                 # edge_id → [(client_id, Δ)]
        self._round_sketches = []      # 本云轮累积：(edge_id, edge_round, [cid], [草图])

    def observe(self, edge_id, edge_round, client_updates, edge_weights, base_idx):
        """只读。client_updates 的 `.client_id` 必须存在（ClientUpdate）。"""
        if self._mask is None and self.stat_idx:
            self._mask = stat_mask(edge_weights, base_idx, self.stat_idx)
        rows = []
        for u in client_updates:
            rows.append((int(u.client_id), body_delta(u[0], edge_weights, base_idx)))
        self._buf[int(edge_id)] = rows

    def flush(self, round_idx, edge_round) -> list:
        """算几何、返回要打印的行，并把草图存进本云轮缓冲。缓冲为空 → []。"""
        if not self._buf:
            return []
        groups = {eid: [d for _, d in rows] for eid, rows in self._buf.items()}
        geo = geometry(groups)
        mask = self._mask
        has_split = mask is not None and bool(mask.any())
        if has_split:
            geo_w = geometry({eid: [d[~mask] for d in ds] for eid, ds in groups.items()})
        lines = []
        for eid in sorted(self._buf):
            rows = self._buf[eid]
            g = geo[eid]
            fields = {
                "edge_round": int(edge_round),
                "cid": fmt_list([c for c, _ in rows]),
                "mal": fmt_list([int(c in self.malicious_ids) for c, _ in rows]),
                "norm": fmt_list(g["norm"]),
                "cos_edge": fmt_list(g["cos_edge"]),
                "cos_global": fmt_list(g["cos_global"]),
            }
            if has_split:                       # S6b：权重部分 / 统计量部分（旧列不变）
                gw = geo_w[eid]
                fields["norm_w"] = fmt_list(gw["norm"])
                fields["norm_s"] = fmt_list([float(np.linalg.norm(d[mask].astype(np.float64)))
                                             for _, d in rows])
                fields["cos_edge_w"] = fmt_list(gw["cos_edge"])
                fields["cos_global_w"] = fmt_list(gw["cos_global"])
            lines.append(format_kv("[UpdateGeo]", fields, round_idx=round_idx, edge_id=eid))
            if rows:
                if self._sketch is None:
                    self._sketch = CountSketch(rows[0][1].shape[0], self.sketch_dim)
                self._round_sketches.append(
                    (eid, int(edge_round), [c for c, _ in rows],
                     [self._sketch.apply(d) for _, d in rows],
                     [self._sketch.apply(np.where(mask, 0.0, d)) for _, d in rows]
                     if has_split else None))
        self._buf = {}
        return lines

    def take_round_arrays(self):
        """本云轮累积的草图 → npz 数组字典（没有 → None），并清空。有统计量拆分时另存 sketch_w（权重部分）。"""
        if not self._round_sketches:
            return None
        edge, er, cid, sk, skw = [], [], [], [], []
        for eid, e_r, cids, sks, sksw in self._round_sketches:
            for i, (c, s_) in enumerate(zip(cids, sks)):
                edge.append(eid); er.append(e_r); cid.append(c); sk.append(s_)
                if sksw is not None:
                    skw.append(sksw[i])
        self._round_sketches = []
        out = {
            "sketch": np.stack(sk).astype(np.float16),
            "client_id": np.array(cid, np.int32),
            "edge_id": np.array(edge, np.int16),
            "edge_round": np.array(er, np.int16),
            "malicious": np.array([c in self.malicious_ids for c in cid], np.bool_),
            "sketch_dim": np.array(self.sketch_dim, np.int32),
        }
        if skw:
            out["sketch_w"] = np.stack(skw).astype(np.float16)
        return out
