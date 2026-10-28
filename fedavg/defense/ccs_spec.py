"""
defense/ccs_spec.py  —  CCS（Cui et al., ICASSP 2026）的纯算术部分：配置、DPR 掩码、对抗样本的逐步规则、KL / MMD 的 numpy 参照

**不 import TF**（L1 本地秒级覆盖）。TF 侧（`client/ccs_mixin.py`）只做张量版本，数值以这里为准。
语义对照表、用户的取舍：`experiments/defense/edge-native/DECISIONS.md` D-109；官方代码的读法：FINDINGS F-101。

官方代码 = chenjian0924/paper-code @ b4b8653，`CCS/`。**那份代码本身跑不起来**（F-101 ①：`perturb_input` 没有
`perturb_steps` 参数；`hdbscan_detect` 取到的统计量列表为空）—— 这里照的是它的**意图**，逐项注明出处：

  对抗样本（`src/defence/AT/AT.py` 的 `l_inf_patch` 分支，`client/fedavg.py:281` 的参数）：
    1. 掩码 = `SingleAnchorDPR`（`Patch.py`）：圆心 ∈ [radius, h − radius)、36 个扇区半径全取 radius → 一个 36 边形，
       边缘用 (tanh(cc·(ρ − 1)) + 1) / 2（cc = −100）软化；
    2. 初值 x0 = (1 − m)·x + m·U(0,1)（逐像素逐通道独立）；
    3. steps 步：x ← x + step·sign(∇_x L_mm)，投影到 x0 ± eps，再 clamp [0, 1]；
       L_mm = mean(z_second − z_y)（`max_margin_loss`，无目标，梯度**上升**）；模型 eval 模式。
  **空间**（D-109 / 用户 Q1）：官方在模型输入（已标准化）空间里做 U(0,1) / eps / clamp —— `space: input` 逐项照搬；
    `space: pixel` 把这三样解释成 [0,1] 像素（工作空间 u = x·σ + μ，模型看 (u − μ)/σ）。两者只差一个逐通道仿射。

  本地目标（`client/fedavg.py:283-290`）：L = CE(clean) + kl·KL(p_clean ‖ p_adv) + mmd·MMD_rbf(f_clean, f_adv)，
    f = 私有投影网络的输出（`resnet10_torch_proj` 的 proj_2）；clip 全局梯度范数 10。系数照原文 0.01 / 0.01（D-109 Q2；代码 defaults 0.1 / 0.01）。
  MMD：官方仓库里**没有**这个函数（F-093 / F-101）。调用签名 `MMD(x, y, 'rbf', device)` 与 DAN 流传的片段一致：
    带宽 {10, 15, 20, 50}，k(a, b) = exp(−‖a − b‖² / (2·bw))，mean(K_xx + K_yy − 2·K_xy)。**这一条是推断，没有证据**。

  聚类（服务端，`analysis/bn_cluster.detect` 已实现官方参数）：pool = round（官方：本轮上传就地聚）/ edge_cache
    （D-109 Q4：本轮上传 ∪ 本 edge 其余端最近一次上传的统计量，只对本轮上传剔除；池 < min_pool 视为退化、全部接纳）。
"""

from __future__ import annotations

import math

import numpy as np

DEFAULTS = {
    "at": True,             # 良性端对抗训练
    "cluster": True,        # 服务端按上传 BN 统计量 HDBSCAN 剔除
    "kl": 0.01,             # 原文 β（代码 AT_alpha 缺省 0.1）
    "mmd": 0.01,            # 原文 γ（代码 AT_beta 0.01）
    "steps": 15,            # client/fedavg.py:281 perturb_steps=15
    "step": 0.1,            # step_size=0.1
    "eps": 0.5,             # epsilon=0.5
    "radius": 10,           # AT.py: radius = 10
    "n_regions": 36,        # AT.py: num_regions = 36
    "space": "input",       # input = 照官方代码（标准化空间）；pixel = [0,1] 像素
    "clip_norm": 10.0,      # clip_grad_norm_(max_norm=10)
    "pool": "round",        # round（官方）/ edge_cache（D-109 Q4，分层用）
    "min_pool": 5,          # edge_cache：池里少于这么多端 → degenerate、不剔除
}
MMD_BANDWIDTHS = (10.0, 15.0, 20.0, 50.0)
ACT_CC = -100.0
SPACES = ("input", "pixel")
POOLS = ("round", "edge_cache")
CCS_RNG_TAG = 0xCC5          # 对抗样本的随机流：default_rng([seed, client_id, 0xCC5, round])


def ccs_params(config: dict) -> dict:
    """`defense.ccs` 块里的 CCS 参数（缺省 = DEFAULTS），校验后返回新 dict。

    单独成块、不放 `defense.params`：base.yaml 的 params 里堆着别的防御的键（trim_ratio / noise_scale …），
    混在一起就没法拒绝拼错的 CCS 键。
    """
    raw = dict((((config or {}).get("defense") or {}).get("ccs") or {}))
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"defense.ccs 里有 CCS 不认识的键：{sorted(unknown)}；合法：{sorted(DEFAULTS)}")
    p = dict(DEFAULTS)
    p.update({k: v for k, v in raw.items() if k in DEFAULTS})
    p["at"] = bool(p["at"])
    p["cluster"] = bool(p["cluster"])
    for k in ("kl", "mmd", "step", "eps", "clip_norm"):
        p[k] = float(p[k])
        if not (math.isfinite(p[k]) and p[k] >= 0):
            raise ValueError(f"defense.ccs.{k} 必须是有限的非负数：{p[k]!r}")
    for k in ("steps", "radius", "n_regions", "min_pool"):
        p[k] = int(p[k])
        if p[k] < 0:
            raise ValueError(f"defense.ccs.{k} 必须 ≥ 0：{p[k]!r}")
    p["space"] = str(p["space"]).lower()
    p["pool"] = str(p["pool"]).lower()
    if p["space"] not in SPACES:
        raise ValueError(f"defense.ccs.space ∈ {SPACES}：{p['space']!r}")
    if p["pool"] not in POOLS:
        raise ValueError(f"defense.ccs.pool ∈ {POOLS}：{p['pool']!r}")
    if not (p["at"] or p["cluster"]):
        raise ValueError("defense.name = ccs 但 at 与 cluster 都关 —— 这不是防御，写 defense.name: none")
    if p["n_regions"] < 3:
        raise ValueError("defense.ccs.n_regions 至少 3")
    return p


# ── DPR 掩码（Patch.py:SingleAnchorDPR 的 numpy 移植）──────────────────────
def _line(a, b):
    """过 a、b 两点的直线 A·x + B·y + C = 0（Patch.py:GaussianElimination）。"""
    A = b[..., 1] - a[..., 1]
    B = a[..., 0] - b[..., 0]
    C = b[..., 0] * a[..., 1] - a[..., 0] * b[..., 1]
    return A, B, C


def dpr_mask(cx: int, cy: int, radius, n_regions: int, h: int, w: int, cc: float = ACT_CC) -> np.ndarray:
    """单张图的 DPR 掩码，形状 (h, w)，值 ∈ [0, 1]。

    cx 是行坐标（第 0 维）、cy 是列坐标，与官方 `coordinate[..., 0] = 行号` 一致。
    radius：标量（官方：36 个扇区全取 10）或长度 n_regions 的数组。
    """
    r = np.broadcast_to(np.asarray(radius, np.float64), (n_regions,))
    rows, cols = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    rx, ry = rows - cx, cols - cy
    dis = np.sqrt(rx ** 2 + ry ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.degrees(np.arccos(ry / dis))
    ang = np.where(rx < 0, 360.0 - ang, ang)
    region = np.floor(ang / (360 // n_regions))
    region[int(cx), int(cy)] = n_regions
    theta = np.arange(n_regions) * (2.0 * np.pi / n_regions)
    pts = np.stack([cx + r * np.sin(theta), cy + r * np.cos(theta)], axis=1)        # (n_regions, 2)
    act = np.zeros((h, w), np.float64)
    o = np.array([cx, cy], np.float64)
    for i in range(n_regions):
        sel = region == i
        if not sel.any():
            continue
        if i == n_regions - 1:
            a, b = pts[0], pts[i]
        else:
            a, b = pts[i], pts[i + 1]
        A0, B0, C0 = _line(a, b)
        c = np.stack([rows[sel], cols[sel]], axis=1)
        A1, B1, C1 = _line(np.broadcast_to(o, c.shape), c)
        D = A0 * B1 - A1 * B0
        x = (B0 * C1 - B1 * C0) / D
        y = (A1 * C0 - A0 * C1) / D
        rho = dis[sel] / np.sqrt((o[0] - x) ** 2 + (o[1] - y) ** 2)
        act[sel] = (np.tanh(cc * (rho - 1.0)) + 1.0) / 2.0
    act[int(cx), int(cy)] = 1.0
    return act


def draw_masks(rng: np.random.Generator, n: int, h: int, w: int, radius: int, n_regions: int) -> np.ndarray:
    """一批掩码 (n, h, w, 1)，float32。圆心 ∈ [radius, h − radius) × [radius, w − radius)（AT.py: np.random.randint）。"""
    cx = rng.integers(radius, h - radius, size=n)
    cy = rng.integers(radius, w - radius, size=n)
    return np.stack([dpr_mask(int(a), int(b), radius, n_regions, h, w) for a, b in zip(cx, cy)]
                    )[..., None].astype(np.float32)


def space_affine(space: str, mean, std):
    """工作空间 u = a·x + b（x = 模型输入）。input：a = 1, b = 0；pixel：a = σ, b = μ（逐通道）。"""
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    if space == "input":
        return np.ones_like(std), np.zeros_like(mean)
    if space == "pixel":
        return std, mean
    raise ValueError(space)


# ── numpy 参照（L1 用；TF 侧与之对拍）───────────────────────────────────────
def adv_init(u, mask, noise):
    """初值（工作空间）：(1 − m)·u + m·noise。"""
    return (1.0 - mask) * u + mask * noise


def adv_step(u, u0, grad_sign, step, eps):
    """一步：u + step·sign(g) → 投影到 u0 ± eps → clamp [0, 1]（AT.py 的顺序）。"""
    u = u + step * grad_sign
    u = np.minimum(np.maximum(u, u0 - eps), u0 + eps)
    return np.clip(u, 0.0, 1.0)


def max_margin(logits, y):
    """AT.py:max_margin_loss（threshold=None）= mean(z_second − z_y)；z_second = 除 y 外最大的 logit。"""
    z = np.asarray(logits, np.float64)
    y = np.asarray(y).reshape(-1).astype(int)
    real = z[np.arange(len(y)), y]
    other = z.copy()
    other[np.arange(len(y)), y] = -np.inf
    return float(np.mean(other.max(axis=1) - real))


def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def kl_clean_adv(logits_clean, logits_adv):
    """F.kl_div(log_softmax(adv), softmax(clean), reduction='batchmean') = mean_i Σ_k p_c (log p_c − log p_a)。"""
    lc, la = log_softmax(logits_clean), log_softmax(logits_adv)
    return float(np.mean(np.sum(np.exp(lc) * (lc - la), axis=1)))


def mmd_rbf(x, y, bandwidths=MMD_BANDWIDTHS):
    """DAN 片段的 MMD（rbf）：mean(K_xx + K_yy − 2 K_xy)，K = Σ_bw exp(−0.5·d² / bw)。"""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    def d2(a, b):
        return (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T

    dxx, dyy, dxy = d2(x, x), d2(y, y), d2(x, y)
    K = lambda d: sum(np.exp(-0.5 * d / bw) for bw in bandwidths)    # noqa: E731
    return float(np.mean(K(dxx) + K(dyy) - 2.0 * K(dxy)))


# ── 自描述 ───────────────────────────────────────────────────────────────────
SETTINGS_KEYS = ("at", "cluster", "kl", "mmd", "steps", "step", "eps", "radius", "n_regions",
                 "space", "clip_norm", "pool", "min_pool")


def settings_row(config: dict) -> dict:
    """[设定12] 的内容。defense 不是 ccs → {"ccs": False}（与 SA-C 之前的 run 是同一种 run）。"""
    name = str(((config or {}).get("defense") or {}).get("name", "none")).lower()
    if name != "ccs":
        return {"ccs": False}
    p = ccs_params(config)
    return {"ccs": True, **{k: p[k] for k in SETTINGS_KEYS}}
