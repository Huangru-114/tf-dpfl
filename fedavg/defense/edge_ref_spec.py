"""
defense/edge_ref_spec.py  —  方向 2（D-129）：边缘服务器用自己的干净图作参照、给每个客户端上传打分 —— 纯算术部分

**不 import TF**（L1 本地秒级覆盖）。TF 侧（`defense/edge_ref.py`）只做「装权重 → 干净图前向 → 取 BN 层输入的逐样本矩」，
其余全部在这里，数值以这里为准。语义对照与取舍：`experiments/defense/edge-native/DECISIONS.md` D-131、FINDINGS N-013。

做法（一个上传 i、一台边缘 e）：
  · 参照：把上传的**可训练权重**装进草稿模型，边缘干净图（500 张）**一整批、训练模式**前向 —— 训练模式下每个 BN 层用
    本批统计量归一化，结果与上传里的统计量无关，也与客户端本地训练时 BN 的算法相同。取每个 BN 层**输入**的逐样本逐通道
    一阶矩 m1[n, c] = mean_{h,w} x、二阶矩 m2[n, c] = mean_{h,w} x²。整集 / 逐类 / 子集的矩 = 对应样本的 m1、m2 平均
    （每张图的 h·w 相同 → 精确）。**近似**：子集的矩里，上游 BN 用的是整批 500 张的统计量，而不是子集自己的。
  · 上传的统计量：各 BN 层的 moving_mean / moving_variance（主干训练段末尾，动量 0.9 ≈ 最后 10 批的指数平均；F-123）。
  · 分数 1（逐客户端参照 KL）：逐通道 KL(N(μ_up, σ²_up + ε) ‖ N(μ_ref, σ²_ref + ε))，ε = BN 的 epsilon；
    层内取通道均值，再按层平均：全部层 / stem / stage1…4。来源：BNP（NeurIPS 2022）逐神经元比对 —— 这里改为给上传打分。
  · 分数 2（类混合残差，新设计、无文献）：干净图逐类矩 A_k = [μ_k / σ_ref ; E_k[x²] / σ²_ref]，在单纯形上拟合
    b = [μ_up / σ_ref ; (σ²_up + μ²_up) / σ²_ref]（二阶矩对类比例是线性的，所以混合也是线性的）；残差均方根 = 分数。
    另报 base 残差（π = 干净集的类比例，不拟合）与拟合比例的最大值 / 取到最大值的类。
    风险：Bad-PFL 的 δ 是目标类的自然特征，可能被「多一点目标类」解释掉；受害边缘有 4 类只有 12–13 张，逐类矩噪声大。
  · 对照 3（BNGuard 四数）：第一个 BN 层上传的 moving_mean、moving_variance 各自的均值与方差
    （BNGuard 用的是有偏还是无偏方差没有核对；对 K-means 的相对距离几乎无影响）。聚类留到离线。
  · 对照 4（多数原则，CCS 式）：本边缘缓存池（本轮 ∪ 其余端最近一次上传）的统计量向量 → `analysis/bn_cluster.detect`
    （在 `defense/edge_ref.py` 里调，它要 scikit-learn）+ 到池中其余端的余弦距离中位数（这里）。
  · 零分布（假客户端）：类比例按本划分的客户端生成规则抽（Dirichlet Dir(α_c·K·p_e)；固定画像第 j 个 = 画像 j mod 5），
    p_e = 本边缘干净集的类比例；按比例从干净集有放回抽 fake_n 张；用**下发模型** θ_before 的逐样本矩算同样的分数。
    只用干净数据、不用身份标签；只作参照，不进判定。随机数只来自调用方传入的 Generator。
"""

from __future__ import annotations

import math
import re

import numpy as np

DEFAULTS = {
    "mode": "observe",      # 第一段只记录、全部接纳（D-129）；过滤模式等第一段数据之后再加
    "n_fake": 8,            # 每个边缘轮的假客户端数（零分布）
    "fake_n": 375,          # 每个假客户端的样本数 = 一个真实客户端的训练分片（S3：500 = 375 训练 + 125 留出）
}
MODES = ("observe",)
EDGE_REF_RNG_TAG = 0xED6E    # 假客户端的随机流：default_rng([seed, 0xED6E, 有效轮, edge_id])
STAGES = ("stem", "stage1", "stage2", "stage3", "stage4")
SETTINGS_KEYS = ("mode", "n_fake", "fake_n")

# 每个上传一行 [EdgeRefScore] 的字段（collect_metrics 抄一份，守卫断言两份相同）
SCORE_FIELDS = ("effective_round", "edge_round", "cid", "n",
                "kl_all", "kl_stem", "kl_stage1", "kl_stage2", "kl_stage3", "kl_stage4",
                "mix_res", "mix_res_base", "mix_pmax", "mix_argmax",
                "bng_mm_mean", "bng_mm_var", "bng_mv_mean", "bng_mv_var",
                "ccs_cos_med", "ccs_rej")
# 每个边缘轮一行 [EdgeRefNull]：假客户端的分数（"/" 连接的 n_fake 个值）
NULL_FIELDS = ("effective_round", "edge_round", "n_clean", "ccs_pool", "ccs_status",
               "kl_all", "kl_stage4", "mix_res", "mix_res_base")


# ── 配置 ─────────────────────────────────────────────────────────────────────
def edge_ref_params(config: dict) -> dict:
    """`defense.edge_ref` 块（缺省 = DEFAULTS），校验后返回新 dict。单独成块，不放 defense.params（同 CCS，D-109）。"""
    raw = dict((((config or {}).get("defense") or {}).get("edge_ref") or {}))
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"defense.edge_ref 里有不认识的键：{sorted(unknown)}；合法：{sorted(DEFAULTS)}")
    p = dict(DEFAULTS)
    p.update(raw)
    p["mode"] = str(p["mode"]).lower()
    if p["mode"] not in MODES:
        raise ValueError(f"defense.edge_ref.mode ∈ {MODES}：{p['mode']!r}（过滤模式在第一段数据之后才实现）")
    for k in ("n_fake", "fake_n"):
        v = p[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"defense.edge_ref.{k} 必须是 ≥ 1 的整数：{v!r}")
    return p


def settings_row(config: dict) -> dict:
    """[设定14] 的内容。defense 不是 edge_ref → {"edge_ref": False}（与方向 2 之前的 run 是同一种 run）。"""
    name = str(((config or {}).get("defense") or {}).get("name", "none")).lower()
    if name != "edge_ref":
        return {"edge_ref": False}
    p = edge_ref_params(config)
    return {"edge_ref": True, **{k: p[k] for k in SETTINGS_KEYS}}


def client_rule(config: dict) -> dict:
    """本划分的客户端生成规则（假客户端照它抽类比例）。fixed → 固定画像；否则 Dirichlet（D-063 的社区口径）。"""
    d = (((config or {}).get("federation") or {}).get("design") or {})
    if d.get("client_profile") == "fixed":
        return {"kind": "fixed", "lam": float(d["profile_lambda"])}
    ac = d.get("alpha_client")
    return {"kind": "dirichlet", "alpha_c": 0.5 if ac is None else float(ac)}     # 0.5 = designed_partition 的缺省


# ── BN 层的名字 → 阶段 ────────────────────────────────────────────────────────
_STAGE_RE = re.compile(r"^stage(\d+)_")


def stage_of(layer_name: str) -> str:
    m = _STAGE_RE.match(str(layer_name))
    return f"stage{m.group(1)}" if m else "stem"


# ── 矩 ───────────────────────────────────────────────────────────────────────
def subset_moments(m1, m2, idx=None):
    """一层的逐样本矩 (N, C) → (μ, σ²)（逐通道）。idx = 样本下标（可重复 = 有放回）；None = 全部。"""
    m1 = np.asarray(m1, np.float64)
    m2 = np.asarray(m2, np.float64)
    if idx is not None:
        idx = np.asarray(idx, dtype=np.int64)
        m1, m2 = m1[idx], m2[idx]
    mu = m1.mean(0)
    var = np.maximum(m2.mean(0) - mu ** 2, 0.0)
    return mu, var


def class_moments(m1, m2, labels, n_classes):
    """逐类 (μ_k, E_k[x²])，形状 (K, C)；干净集里没有的类 → NaN 行。"""
    m1 = np.asarray(m1, np.float64)
    m2 = np.asarray(m2, np.float64)
    y = np.asarray(labels).reshape(-1)
    mu = np.full((n_classes, m1.shape[1]), np.nan)
    ex2 = np.full((n_classes, m1.shape[1]), np.nan)
    for k in range(n_classes):
        sel = y == k
        if sel.any():
            mu[k] = m1[sel].mean(0)
            ex2[k] = m2[sel].mean(0)
    return mu, ex2


# ── 分数 1：高斯 KL ───────────────────────────────────────────────────────────
def gauss_kl(mu_p, var_p, mu_q, var_q, eps):
    """逐通道 KL(N(μ_p, σ²_p + ε) ‖ N(μ_q, σ²_q + ε))。"""
    vp = np.asarray(var_p, np.float64) + eps
    vq = np.asarray(var_q, np.float64) + eps
    d = np.asarray(mu_p, np.float64) - np.asarray(mu_q, np.float64)
    return 0.5 * (np.log(vq / vp) + (vp + d ** 2) / vq - 1.0)


def kl_scores(up, ref, layers):
    """up / ref：每层 (μ, σ²) 的列表；layers：每层 {"stage", "eps"}。→ {"kl_all", "kl_stem", …, "kl_stage4"}。

    层内取通道均值，再在层之间平均（每层权重相同 —— 不让通道多的深层独占）。某阶段没有 BN 层 → None。"""
    per = [float(np.mean(gauss_kl(u[0], u[1], r[0], r[1], L["eps"]))) for u, r, L in zip(up, ref, layers)]
    out = {"kl_all": float(np.mean(per)) if per else None}
    for st in STAGES:
        v = [x for x, L in zip(per, layers) if L["stage"] == st]
        out[f"kl_{st}"] = float(np.mean(v)) if v else None
    return out


# ── 分数 2：单纯形上的类混合拟合 ──────────────────────────────────────────────
def simplex_project(v):
    """欧氏投影到概率单纯形（Duchi et al. 2008，精确，O(K log K)）。"""
    v = np.asarray(v, np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - (css - 1.0) / j > 0)[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def mixture_fit(A, b, iters: int = 3000, tol: float = 1e-12):
    """min_π ‖A π − b‖² s.t. π ∈ 单纯形。A：(d, K)；→ (π, 残差均方根)。

    加速投影梯度（FISTA），步长 1 / λ_max(AᵀA)，从均匀分布起步；确定性。K = 10、d 几千 → 毫秒级。"""
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    K = A.shape[1]
    G = A.T @ A
    h = A.T @ b
    L = float(np.linalg.eigvalsh(G)[-1])
    if not np.isfinite(L) or L <= 0:
        pi = np.full(K, 1.0 / K)
    else:
        pi = np.full(K, 1.0 / K)
        y, t = pi.copy(), 1.0
        for _ in range(int(iters)):
            new = simplex_project(y - (G @ y - h) / L)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = new + ((t - 1.0) / t_new) * (new - pi)
            done = float(np.max(np.abs(new - pi))) < tol
            pi, t = new, t_new
            if done:
                break
    r = A @ pi - b
    return pi, float(np.sqrt(np.mean(r ** 2)))


def mixture_design(class_mom, ref, up, layers):
    """拼出 (A, b)：逐层逐通道按参照标准差标准化（线性缩放，不破坏混合的线性）。

    class_mom：每层 (μ_k, E_k[x²])（K × C）；ref：每层 (μ_ref, σ²_ref)；up：每层 (μ_up, σ²_up)。
    返回 (A, b, present)：present = 干净集里有的类（A 只含这些列）。"""
    present = None
    cols, rhs = [], []
    for (mk, ek), (mr, vr), (mu, vu), L in zip(class_mom, ref, up, layers):
        if present is None:
            present = ~np.isnan(mk[:, 0])
        s2 = np.asarray(vr, np.float64) + L["eps"]
        s = np.sqrt(s2)
        cols.append(np.concatenate([(mk[present] / s).T, (ek[present] / s2).T], axis=0))
        rhs.append(np.concatenate([np.asarray(mu) / s, (np.asarray(vu) + np.asarray(mu) ** 2) / s2]))
    return np.concatenate(cols, axis=0), np.concatenate(rhs), present


def mixture_scores(class_mom, ref, up, layers, clean_props):
    """→ {"mix_res", "mix_res_base", "mix_pmax", "mix_argmax"}。base = π 取干净集的类比例（不拟合）。"""
    A, b, present = mixture_design(class_mom, ref, up, layers)
    pi, res = mixture_fit(A, b)
    base = np.asarray(clean_props, np.float64)[present]
    base = base / base.sum()
    res_base = float(np.sqrt(np.mean((A @ base - b) ** 2)))
    full = np.zeros(len(present))
    full[present] = pi
    k = int(np.argmax(full))
    return {"mix_res": res, "mix_res_base": res_base, "mix_pmax": float(full[k]), "mix_argmax": k}


# ── 对照 3：BNGuard 四数 ─────────────────────────────────────────────────────
def bnguard4(moving_mean, moving_var):
    """第一个 BN 层：[mean(running_mean), var(running_mean), mean(running_var), var(running_var)]（有偏方差）。"""
    mm = np.asarray(moving_mean, np.float64).reshape(-1)
    mv = np.asarray(moving_var, np.float64).reshape(-1)
    return {"bng_mm_mean": float(mm.mean()), "bng_mm_var": float(mm.var()),
            "bng_mv_mean": float(mv.mean()), "bng_mv_var": float(mv.var())}


# ── 对照 4：池内余弦距离中位数 ────────────────────────────────────────────────
def cosine_median_to_others(pool: dict, cid):
    """pool：{client_id: 统计量向量}；→ cid 到池中其余端的余弦距离中位数（池里只有它自己 → None）。"""
    others = [v for k, v in pool.items() if k != cid]
    if not others:
        return None
    a = np.asarray(pool[cid], np.float64)
    na = np.linalg.norm(a)
    ds = []
    for v in others:
        v = np.asarray(v, np.float64)
        nv = np.linalg.norm(v)
        ds.append(1.0 - float(a @ v) / (na * nv) if na > 0 and nv > 0 else 1.0)
    return float(np.median(ds))


# ── 零分布：假客户端 ──────────────────────────────────────────────────────────
def fake_props(rng: np.random.Generator, clean_props, n_fake: int, rule: dict) -> np.ndarray:
    """n_fake 个假客户端的类比例（n_fake × K，每行和 1）。只用传入的 rng。"""
    pe = np.asarray(clean_props, np.float64)
    pe = pe / pe.sum()
    K = len(pe)
    if rule["kind"] == "fixed":
        from data.designed_partition import profile_matrix
        P = profile_matrix(pe, n_fake, 1.0, rule["lam"])
        P = np.where(pe[None, :] > 0, P, 0.0)          # 干净集里没有的类抽不到（IPF 在真实划分里也会把它压到 0）
        return P / P.sum(1, keepdims=True)
    conc = np.maximum(rule["alpha_c"] * K * pe, 1e-3)
    P = rng.dirichlet(conc, size=int(n_fake))
    P = np.where(pe[None, :] > 0, P, 0.0)
    return P / P.sum(1, keepdims=True)


def fake_indices(rng: np.random.Generator, labels, props, n: int) -> np.ndarray:
    """按类比例从干净集有放回抽 n 张的下标（类计数 ~ Multinomial(n, props)）。"""
    y = np.asarray(labels).reshape(-1)
    counts = rng.multinomial(int(n), np.asarray(props, np.float64))
    out = []
    for k, c in enumerate(counts):
        if c:
            pool = np.flatnonzero(y == k)
            out.append(rng.choice(pool, size=int(c), replace=True))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# ── 一次打分（真实上传与假客户端共用）────────────────────────────────────────
def score_one(mom, labels, n_classes, up, layers, clean_props, idx=None):
    """mom：每层 (m1, m2)（逐样本，来自这个模型的前向）；up：每层 (μ_up, σ²_up)；idx：参照用的样本子集（None = 全部）。

    返回 KL 与混合两类分数。真实上传：up = 上传的统计量，参照 = 全部干净图。
    假客户端：调用方先用 subset_moments(m1, m2, 抽到的下标) 构造 up，参照仍是全部干净图（idx = None）。"""
    ref = [subset_moments(m1, m2, idx) for m1, m2 in mom]
    cm = [class_moments(m1, m2, labels, n_classes) for m1, m2 in mom]
    out = kl_scores(up, ref, layers)
    out.update(mixture_scores(cm, ref, up, layers, clean_props))
    return out
