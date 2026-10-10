"""
hardening/spec.py  —  edge 侧对抗训练（阶段三 SA0）的纯算术部分（**不 import TF**）

阶段三 P0（`experiments/defense/edge-native/PLAN.md` §4；判定规则 FINDINGS N-008，D-101）问：
一次云聚合里，4 个 edge 都在上传前用自己的 500 张干净图做对抗训练（AT）再 FedAvg，受害 edge 的灌入能砍掉多少。
TF 部分（探针前向、PGD、训练步）在 `hardening/core.py`；离线驱动在 `analysis/p0_snapshot.py`；判定在 `harness/p0_verdict.py`。
这里放凡是能不用 TF 写的东西，好让 L1 在本地秒级覆盖它们：

  · 配置网格（`experiments/defense/edge-native/p0_configs.yaml` → 一串有确定 id 的配置）与每个目标的步长约定；
  · 像素单位 → 模型输入空间（复用 `data/pixel_space`，与 Bad-PFL 的 ξ、`ck_snapshot` 同一套换算）；
  · 干净集的分层子集与每 epoch 的打乱（干净集索引**按类排序**，F-095：不能取前 n 张）；
  · 临时线性探针 head 的拟合（numpy 的全批 softmax 回归，确定性）；
  · 前向反向次数的记账（P1 的开销换算要用）；
  · R_H、精度过滤、V0 的算术（N-008 P0 段）。

官方实现与本仓库的逐条差异见 SA0 语义 diff（计划文件，DECISIONS D-103 / D-104，FINDINGS F-095）。
随机性只来自调用方给的 `np.random.Generator`，不碰全局 np.random / Python random。
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from data.pixel_space import to_input_space, valid_range        # noqa: F401  （re-export：core / driver 只从这里取）

# ── N-008 P0 段的常数（D-101 确认；改动要留 DECISIONS 记录）──────────────────────
R_GO = 0.5            # go_online：R_H ≥ 0.5
R_KILL = 0.2          # kill_pre：R_H < 0.2
JUMP_MIN = 0.05       # J_t < 0.05 的快照不计
MTA_POOL_MAX = 0.02   # 精度过滤：池化 fresh ΔMTA ≤ 0.02
MTA_EDGE_MAX = 0.04   # 且每个 edge ≤ 0.04
V0_WEIGHT_TOL = 1e-5  # 离线 FedAvg 与快照 global 的 max |差|
V0_ASR_TOL = 0.01     # 离线重算与 run 记录的 ASR 差

OBJECTIVES = ("pgd", "trades", "sau", "tgt", "cft")
AT_OBJECTIVES = ("pgd", "trades", "sau", "tgt")        # C-ft 是归因对照，不参与选择（PLAN §3.2）
BN_MODES = ("train", "frozen")

# 每个目标的 PGD 步长约定（像素单位的 α = ratio · ε / 步数），都取自官方实现：
#   pgd    Madry cifar10_challenge：ε = 8、步长 2、10 步 → 2.5·ε/K
#   trades TRADES train_trades_cifar10.py：ε = 0.031、步长 0.007、10 步 → (0.007·10/0.031)·ε/K
#   sau    BackdoorBench sau：trigger_norm 0.2、adv_lr 0.2（每步 = ε）、5 步 → K·ε/K = ε
#   tgt    本计划自定（无官方实现）：定向部分沿用 ck_snapshot 的 2.5·ε/K
STEP_RATIO = {"pgd": 2.5, "trades": 0.007 * 10 / 0.031, "tgt": 2.5}
SAU_DEFAULTS = {"beta_1": 0.01, "beta_2": 1.0, "lmd_1": 1.0, "lmd_2": 0.0, "lmd_3": 1.0,
                "steps": 5, "optimizer": "adam", "lr": 1e-4}      # BackdoorBench config/defense/sau/cifar10.yaml
TRADES_START_STD_PX = 0.001                                     # TRADES：x_adv = x + 0.001·randn（[0,1] 像素）


# ══════════════════════════════════════════════════════════════════════════
# 配置网格
# ══════════════════════════════════════════════════════════════════════════

def parse_px(v) -> float:
    """像素单位的幅度：浮点数，或 "8/255" 这样的分数串。"""
    if isinstance(v, bool):
        raise ValueError(f"幅度不能是 bool：{v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if "/" in s:
        return float(Fraction(s))
    return float(s)


def px_label(v: float) -> str:
    """id 里用的短标签：k/255 的整数 k 写成 "e8"，其余写成 "e0.2"。"""
    k = v * 255.0
    if abs(k - round(k)) < 1e-9:
        return f"e{int(round(k))}"
    return f"e{v:g}"


def expand_grid(spec: dict) -> list:
    """
    p0_configs.yaml 的 `grid:` → 配置列表，顺序确定、id 唯一。每个配置：
      {id, objective, eps_px, sigma_px, beta, bn, budget, epochs, pgd_steps, optimizer, lr, sau}
    `budgets` 给出 {名: {epochs, pgd_steps}}；SAU 的 PGD 步数固定为官方的 5（预算只改 epoch 数）。
    """
    budgets = spec["budgets"]
    bn_modes = list(spec.get("bn", BN_MODES))
    for b in bn_modes:
        if b not in BN_MODES:
            raise ValueError(f"bn 只能是 {BN_MODES}：{b!r}")
    out = []
    for obj, o in spec["objectives"].items():
        if obj not in OBJECTIVES:
            raise ValueError(f"未知目标 {obj!r}（{OBJECTIVES}）")
        o = o or {}
        eps_list = [parse_px(e) for e in o.get("eps_px", [None])] if obj != "cft" else [None]
        betas = list(o.get("beta", [None])) if obj == "trades" else [None]
        sigma = parse_px(o["sigma_px"]) if obj == "tgt" else None
        for eps in eps_list:
            for beta in betas:
                for bn in bn_modes:
                    for bname, b in budgets.items():
                        steps = SAU_DEFAULTS["steps"] if obj == "sau" else int(b["pgd_steps"])
                        parts = [obj]
                        if eps is not None:
                            parts.append(px_label(eps))
                        if beta is not None:
                            parts.append(f"b{beta:g}")
                        parts += [bn, bname]
                        cfg = {"id": "-".join(parts), "objective": obj, "eps_px": eps,
                               "sigma_px": sigma, "beta": None if beta is None else float(beta),
                               "bn": bn, "budget": bname, "epochs": int(b["epochs"]),
                               "pgd_steps": 0 if obj == "cft" else steps,
                               "optimizer": SAU_DEFAULTS["optimizer"] if obj == "sau" else "sgd",
                               "lr": SAU_DEFAULTS["lr"] if obj == "sau" else None,   # None = 客户端 body 的 lr
                               "sau": dict(SAU_DEFAULTS) if obj == "sau" else None}
                        out.append(cfg)
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"配置 id 重复：{sorted(i for i in ids if ids.count(i) > 1)}")
    return out


def matched_cft(cfg: dict, grid: list) -> str | None:
    """同 bn、同预算的 C-ft（归因对照：同数据、同步数、同优化器，去掉对抗部分）。"""
    for c in grid:
        if c["objective"] == "cft" and c["bn"] == cfg["bn"] and c["budget"] == cfg["budget"]:
            return c["id"]
    return None


def step_px(cfg: dict) -> float:
    """PGD 一步的步长（像素单位）。"""
    obj, eps, k = cfg["objective"], cfg["eps_px"], int(cfg["pgd_steps"])
    if obj == "sau":
        return float(eps)
    if obj in STEP_RATIO:
        return STEP_RATIO[obj] * float(eps) / k
    raise ValueError(f"{obj} 没有 PGD 步长")


def lr_at(config: dict, t: int) -> float:
    """快照 t（云轮末）那一刻客户端 body 的 lr = lr0 · γ^round_index(t, R, R)（最后一个 edge 轮）。"""
    from alignment import get_switch
    from server.participation import round_index          # 纯 python，不 import TF
    R = int(config["federation"].get("edge_rounds", 1) or 1)
    tr = config["training"]
    r = round_index(int(t), R, R, get_switch(config, "training.lr_round_axis"))
    return float(tr["learning_rate"]) * float(tr.get("lr_decay", 1.0)) ** max(0, int(r))


# ══════════════════════════════════════════════════════════════════════════
# 干净集：分层子集、每 epoch 的批
# ══════════════════════════════════════════════════════════════════════════

def stratified_subset(y, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    从标签 y 里按类比例抽 n 个位置（不放回），返回升序的位置数组。
    名额 = 各类张数 × n / 总数，取整用最大余数法（余数并列时类号小的优先）；每类内由 rng 抽。
    n ≥ len(y) → 全部。干净集按类排序（F-095），所以「前 n 张」只覆盖前一两个类 —— 这里不那样取。
    """
    y = np.asarray(y).reshape(-1)
    n = int(n)
    if n >= len(y):
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    exact = counts * n / counts.sum()
    quota = np.floor(exact).astype(int)
    rem = n - int(quota.sum())
    order = sorted(range(len(classes)), key=lambda i: (-(exact[i] - quota[i]), i))
    for i in order[:rem]:
        quota[i] += 1
    picks = []
    for c, q in zip(classes, quota):
        pos = np.flatnonzero(y == c)
        if q:
            picks.append(rng.choice(pos, size=int(q), replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)


def epoch_batches(n: int, batch: int, rng: np.random.Generator) -> list:
    """一个 epoch 的批（位置数组），每次调用重新打乱；drop_last（BN 训练模式不吃小尾批）。"""
    n, bs = int(n), int(batch)
    perm = rng.permutation(n)
    return [perm[i * bs:(i + 1) * bs] for i in range(n // bs)]


# ══════════════════════════════════════════════════════════════════════════
# 临时线性探针 head（edge 模型的 head 从未训练，PLAN §3.2 / 语义 diff A2）
# ══════════════════════════════════════════════════════════════════════════

def fit_probe(features, y, n_classes: int, steps: int = 300, lr: float = 0.01,
              l2: float = 1e-4) -> dict:
    """
    在冻结 body 的倒数第二层特征上拟合 softmax 回归：零初始化、全批 Adam、float64 → 确定性。
    返回 {W [D, K] float32, b [K] float32, train_acc, loss}。没有样本的类照样有一列（只受 L2 约束）。
    """
    F = np.asarray(features, dtype=np.float64)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    n, d = F.shape
    K = int(n_classes)
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    W = np.zeros((d, K))
    b = np.zeros(K)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss = None
    for t in range(1, int(steps) + 1):
        Z = F @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(np.clip(P[np.arange(n), y], 1e-12, None))) + 0.5 * l2 * np.sum(W * W))
        G = (P - Y) / n
        gW = F.T @ G + l2 * W
        gb = G.sum(axis=0)
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW * gW
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * gb * gb
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    acc = float(np.mean(np.argmax(F @ W + b, axis=1) == y)) if n else None
    return {"W": W.astype(np.float32), "b": b.astype(np.float32), "train_acc": acc, "loss": loss}


# ══════════════════════════════════════════════════════════════════════════
# 记账：一次加固的前向反向次数（单位 = 一个 batch 的一次前向 + 一次反向）
# ══════════════════════════════════════════════════════════════════════════

def fb_per_batch(cfg: dict) -> float:
    """
    一个训练 batch 的前向 + 反向次数（近似：只前向的那一次记 0.5）。
      cft     1（训练步）
      pgd     K（生成）+ 1
      trades  0.5（干净分布）+ K + 1（训练步：干净 + 对抗两次前向、一次反向 → 记 1.5）→ K + 2
      sau     1（两个模型的干净前向）+ 2K（每步两个模型）+ 1.5（训练步）+ 1（参照模型两次前向）→ 2K + 3.5
      tgt     K（定向）+ 1（FGSM）+ 1
    """
    obj, k = cfg["objective"], int(cfg["pgd_steps"])
    return {"cft": 1.0, "pgd": k + 1.0, "trades": k + 2.0, "sau": 2.0 * k + 3.5,
            "tgt": k + 2.0}[obj]


def fb_harden(cfg: dict, n_clean: int, batch: int) -> float:
    return fb_per_batch(cfg) * int(cfg["epochs"]) * (int(n_clean) // int(batch))


def fb_clients_per_cloud_round(config: dict) -> float:
    """
    一个云周期里全部客户端本地训练的前向反向次数（同一单位），供 P1 的开销换算（PLAN §3.6）：
    每个 edge 轮被选中的端数 × (body epoch + head epoch) × 每 epoch 的 batch 数，再 × R。
    只是量级；恶意端的生成器训练与 PGD ξ 不计。
    """
    fed, tr, data = config["federation"], config["training"], config["data"]
    n_sel = round(int(fed["n_clients"]) * float(fed.get("client_fraction", 0.1)))
    design = fed.get("design") or {}
    n_train = int(design.get("n_per_client", 500)) - int(round(int(design.get("n_per_client", 500))
                                                                * float(data.get("per_client_test_ratio", 0.2))))
    per_epoch = n_train // int(data["batch_size"])
    epochs = int(tr.get("local_epochs", 1)) + int(tr.get("plocal_epochs", 0))
    return float(n_sel * epochs * per_epoch * int(fed.get("edge_rounds", 1) or 1))


# ══════════════════════════════════════════════════════════════════════════
# N-008 P0 段的算术
# ══════════════════════════════════════════════════════════════════════════

def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return math.fsum(vals) / len(vals) if vals else None


def r_h(a_g, a_gh, jump, jump_min: float = JUMP_MIN):
    """灌入削减率 R_H = [A(G) − A(G′_H)] / J_t；J_t < jump_min 或任一量缺失 → None（该快照不计）。"""
    if a_g is None or a_gh is None or jump is None or jump < jump_min:
        return None
    return (a_g - a_gh) / jump


def accuracy_ok(base: dict, hard: dict) -> dict:
    """
    精度过滤（N-008）：池化 fresh ΔMTA = MTA(G) − MTA(G′_H) ≤ 0.02 且每个 edge ≤ 0.04。
    base / hard：{"pm_acc": 池化, "per_edge": {e: {"pm_acc": …}}}。缺值 → 不通过（不能当 0）。
    """
    if base.get("pm_acc") is None or hard.get("pm_acc") is None:
        return {"ok": False, "d_pool": None, "d_edge_max": None, "reason": "pm_acc 缺失"}
    d_pool = base["pm_acc"] - hard["pm_acc"]
    d_edges = []
    for e, be in (base.get("per_edge") or {}).items():
        he = (hard.get("per_edge") or {}).get(e) or (hard.get("per_edge") or {}).get(str(e)) or {}
        if be.get("pm_acc") is None or he.get("pm_acc") is None:
            return {"ok": False, "d_pool": d_pool, "d_edge_max": None, "reason": f"edge {e} 的 pm_acc 缺失"}
        d_edges.append(be["pm_acc"] - he["pm_acc"])
    d_max = max(d_edges) if d_edges else None
    ok = d_pool <= MTA_POOL_MAX and (d_max is None or d_max <= MTA_EDGE_MAX)
    return {"ok": bool(ok), "d_pool": d_pool, "d_edge_max": d_max, "reason": None}


def max_abs_diff(a: list, b: list) -> float:
    """两组权重的逐元素 max |差|；形状不一致直接报错（不是「差很大」）。"""
    if len(a) != len(b):
        raise ValueError(f"权重个数不同：{len(a)} vs {len(b)}")
    m = 0.0
    for x, y in zip(a, b):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.shape != y.shape:
            raise ValueError(f"形状不同：{x.shape} vs {y.shape}")
        if x.size:
            m = max(m, float(np.max(np.abs(x - y))))
    return m


def v0_check(fedavg_max_abs, post_offline: dict, post_recorded: dict,
             full_offline: dict, full_recorded: dict) -> dict:
    """
    V0（N-008）：① 离线 FedAvg 与快照 global 的 max |差| ≤ 1e-5；
    ② 离线重算的云聚合后点（G）与 run 记录的第 t+1 轮 `per_edge_post_agg_rounds` 逐 edge 差 ≤ 0.01；
    ③ edge 模型不加固时与 run 记录的第 t 轮全量点（`per_edge_rounds`）逐 edge 差 ≤ 0.01。
    *_offline / *_recorded：{edge_id: client_benign}；None 的 edge 两边都得是 None。
    """
    reasons = []
    if fedavg_max_abs is None or fedavg_max_abs > V0_WEIGHT_TOL:
        reasons.append(f"FedAvg 与快照 global 的 max|差| = {fedavg_max_abs} > {V0_WEIGHT_TOL}")

    def cmp(name, off, rec):
        worst = 0.0
        for e in sorted(set(off) | set(rec), key=int):
            a, b = off.get(e), rec.get(e)
            if a is None or b is None:
                if (a is None) != (b is None):
                    reasons.append(f"{name} edge {e}：一边是 None（离线 {a}，记录 {b}）")
                continue
            worst = max(worst, abs(a - b))
        if worst > V0_ASR_TOL:
            reasons.append(f"{name}：逐 edge 最大差 {worst:.4f} > {V0_ASR_TOL}")
        return worst

    post_max = cmp("云聚合后点", post_offline, post_recorded)
    full_max = cmp("第 t 轮全量点", full_offline, full_recorded)
    return {"pass": not reasons, "reasons": reasons, "fedavg_max_abs": fedavg_max_abs,
            "post_max_abs": post_max, "full_max_abs": full_max}


def sha12(path) -> str:
    """文件的 sha256 前 12 位（= utils/dumps.write_npz 记进 manifest 的 sha）。"""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def victims(malicious_per_edge) -> list:
    """没有攻击者的 edge（集中布点 [10,0,0,0] → [1, 2, 3]；分散布点 → []）。"""
    return [e for e, m in enumerate(malicious_per_edge) if int(m) == 0]
