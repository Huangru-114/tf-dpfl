"""
data/designed_partition.py  —  S3：改版实验 3 的新划分（原文 §2；DECISIONS D-062 … D-068）

三种划分，共同的硬约束：**客户端等大小**（D-027，F-028）、**无放回**（有放回会让一个端的
留出图成为另一个端的训练图，陷阱 #11 同类）、每 edge 先切出一份与客户端不相交的**干净集**（D-064）。

  designed      固定设计 C1–C4（4 edge）。机构式比例表（D-062 / D-067）：E0 各类均衡；受害 E1–E3
                各偏重一对类（E1 automobile+truck、E2 cat+dog、E3 deer+horse，各 r），同组其余
                4 类各 (0.6 − 2r)/4；airplane / bird / ship / frog 在各 edge 均为 0.10。
                C1–C4 **只改 airplane（y_t）那一列**，非目标部分按 (1 − y) / 0.9 等比缩放。
  hdir          层级 Dirichlet（D-063，社区口径：每个类的参数 = α）：p_e ~ Dir(α_e, …, α_e)。
  equal_random  「edge 不对应机构」的对照（D-065）：每端类先验 Dir(α_c, …)，再按 block 分到 edge。

edge 内（designed / hdir）：每端 q_c ~ Dir(α_c · K · p_e) —— 均衡 edge 上即官方的 Dir(0.5, …)。

构造 =「名义比例 → 投影到供给可行集（IPF）→ 最大余数取整 → 各类索引里无放回取」。
名义 p_e 常常超出每类供给（F-057：hdir 名义 p_e 90–100% 超供给），投影后实测的 H 比名义平，两者都报。

随机性只来自 `np.random.default_rng([seed, RNG_TAG])` —— **不碰全局 np.random**（旧划分的全局流不受影响）。
纯 numpy，**不 import TF**（`data/__init__.py` 是空的），本地 L1 与 `harness/partition_preview.py` 直接用。
"""

from __future__ import annotations

import hashlib

import numpy as np

S3_PARTITIONS = ("designed", "hdir", "equal_random")
K = 10                                    # CIFAR-10
TARGET = 0                                # airplane；D-028 固定 target_label = 0
SUPPLY_PER_CLASS = 6000                   # 合并池（5000 训练 + 1000 测试）每类
SPECIALTY = {1: (1, 9), 2: (3, 5), 3: (4, 7)}     # 受害 edge → 偏重的一对类（D-067）
UNIFORM = (0, 2, 6, 8)                    # airplane / bird / ship / frog：各 edge 均为 0.10
CONDITIONS = {                            # 各 edge 的 airplane 占比（D-062）
    "C1": (0.10, 0.10, 0.10, 0.10),       # 均衡（差中差的基准）
    "C2": (0.30, 0.03, 0.03, 0.03),       # 攻击者富集
    "C3": (0.10, 0.10, 0.10, 0.005),      # 受害者 E3 缺失（3-B 判定格）
    "C4": (0.005, 0.10, 0.10, 0.10),      # 攻击者缺失
}
DESIGN_KEY = "design"                     # federation.design.*
DEFAULTS = {"condition": None, "strength": 0.25, "alpha_edge": None,
            "alpha_client": 0.5, "n_per_client": 500, "clean_per_edge": 500}
RNG_TAG = 0x533                           # "S3"


# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════

def partition_kind(config: dict) -> str:
    return str((config.get("federation") or {}).get("partition", "noniid"))


def design_params(config: dict) -> dict:
    fed = config.get("federation") or {}
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (fed.get(DESIGN_KEY) or {}).items() if v is not None})
    return p


def config_errors(config: dict) -> list:
    """S3 划分的配置错误（空列表 = 合法，或不是 S3 划分）。config_validate §4f 调用。"""
    kind = partition_kind(config)
    if kind not in S3_PARTITIONS:
        return []
    fed = config.get("federation") or {}
    data = config.get("data") or {}
    bd = config.get("backdoor") or {}
    p = design_params(config)
    errs = []
    if str(data.get("dataset", "cifar10")).lower() != "cifar10":
        errs.append(f"federation.partition={kind!r} 只为 CIFAR-10 设计（10 类、每类 6000 张），"
                    f"收到 data.dataset={data.get('dataset')!r}")
    n_clients, n_edges = int(fed.get("n_clients", 0) or 0), int(fed.get("n_edges", 0) or 0)
    if n_edges <= 0 or n_clients % n_edges:
        errs.append(f"n_clients={n_clients} 不能被 n_edges={n_edges} 整除 —— 各 edge 客户端数要相同")
    n, clean = p["n_per_client"], p["clean_per_edge"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        errs.append(f"federation.design.n_per_client 必须是正整数，收到 {n!r}")
        n = 0
    if isinstance(clean, bool) or not isinstance(clean, int) or clean < 0:
        errs.append(f"federation.design.clean_per_edge 必须是 ≥ 0 的整数，收到 {clean!r}")
        clean = 0
    ratio = float(data.get("per_client_test_ratio", 0.2))
    if n and abs(n * ratio - round(n * ratio)) > 1e-9:
        errs.append(f"n_per_client × per_client_test_ratio = {n} × {ratio} 不是整数 —— "
                    f"各端留出分片会差一张，等大小不成立")
    ac = p["alpha_client"]
    if not isinstance(ac, (int, float)) or isinstance(ac, bool) or ac <= 0:
        errs.append(f"federation.design.alpha_client 必须 > 0，收到 {ac!r}")
    if kind == "designed":
        if n_edges != 4:
            errs.append(f"partition=designed 的比例表是 4 edge 的，收到 n_edges={n_edges}")
        if p["condition"] not in CONDITIONS:
            errs.append(f"federation.design.condition={p['condition']!r}，合法取值 {sorted(CONDITIONS)}")
        r = p["strength"]
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not (0 < r <= 0.3):
            errs.append(f"federation.design.strength={r!r} 不在 (0, 0.3] —— 2r + 4q = 0.6 要求 q ≥ 0")
        if int(bd.get("target_label", TARGET)) != TARGET:
            errs.append(f"partition=designed 的特殊列是 class {TARGET}（airplane，D-028），"
                        f"收到 backdoor.target_label={bd.get('target_label')!r}")
    if kind == "hdir":
        a = p["alpha_edge"]
        if not isinstance(a, (int, float)) or isinstance(a, bool) or a <= 0:
            errs.append(f"partition=hdir 需要 federation.design.alpha_edge > 0，收到 {a!r}")
    if errs or not n:
        return errs
    total = n_clients * n + n_edges * clean
    if total > K * SUPPLY_PER_CLASS:
        errs.append(f"总需求 {total} > 合并池 {K * SUPPLY_PER_CLASS}")
    elif kind == "designed" and p["condition"] in CONDITIONS and n_edges == 4:
        cols = (condition_table(p["condition"], p["strength"])
                * (n_clients // n_edges * n + clean)).sum(0)
        over = [(c, round(v)) for c, v in enumerate(cols) if v > SUPPLY_PER_CLASS + 1e-6]
        if over:
            errs.append(f"比例表超出每类供给 {SUPPLY_PER_CLASS}：{over}（F-057；调小 n_per_client）")
    elif kind == "equal_random" and (n_clients * n) % K:
        errs.append(f"equal_random 要求 n_clients × n_per_client 能被 {K} 整除（各类总量相同）")
    return errs


# ══════════════════════════════════════════════════════════════════════════
# 比例表
# ══════════════════════════════════════════════════════════════════════════

def base_table(r: float = 0.25, n_edges: int = 4) -> np.ndarray:
    """C1 的机构式表（E × K，每行和为 1）。"""
    q = (0.6 - 2 * r) / 4
    if not (0 < r <= 0.3):
        raise ValueError(f"strength r={r} 不在 (0, 0.3]")
    het = sorted({c for pair in SPECIALTY.values() for c in pair})
    T = np.full((n_edges, K), 0.10)
    for e, pair in SPECIALTY.items():
        for c in het:
            T[e, c] = r if c in pair else q
    return T


def with_target_column(T: np.ndarray, y) -> np.ndarray:
    """只改 y_t 列；非目标部分按 (1 − y_e) / (1 − T[e, y_t]) 等比缩放（相对组成不变）。"""
    out = T.copy()
    for e, ye in enumerate(y):
        rest = [c for c in range(K) if c != TARGET]
        out[e, rest] = T[e, rest] * (1 - ye) / (1 - T[e, TARGET])
        out[e, TARGET] = ye
    return out


def condition_table(condition: str, r: float = 0.25) -> np.ndarray:
    return with_target_column(base_table(r), CONDITIONS[condition])


# ══════════════════════════════════════════════════════════════════════════
# 取整与投影
# ══════════════════════════════════════════════════════════════════════════

def lr_round(x, total: int) -> np.ndarray:
    """最大余数取整：非负、和恰为 total。"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    f = np.floor(x).astype(np.int64)
    rem = int(total) - int(f.sum())
    if rem > 0:
        order = np.argsort(-(x - np.floor(x)), kind="stable")
        f[order[:rem]] += 1
    elif rem < 0:
        order = np.argsort(x - np.floor(x), kind="stable")
        for j in order:
            if rem == 0:
                break
            take = min(int(f[j]), -rem)
            f[j] -= take
            rem += take
    return f


def ipf(M, rows, cols, cap: bool = False, iters: int = 1000) -> np.ndarray:
    """交替缩放：行和 = rows；列和 = cols（cap=True 时是列和 ≤ cols）。"""
    M = np.asarray(M, dtype=float) + 1e-12
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    for _ in range(iters):
        M *= (rows / M.sum(1))[:, None]
        cs = M.sum(0)
        s = np.where(cs > cols, cols / cs, 1.0) if cap else cols / np.maximum(cs, 1e-300)
        M *= s[None, :]
    M *= (rows / M.sum(1))[:, None]
    return M


def int_matrix(M, rows, cap) -> np.ndarray:
    """整数矩阵：行和恰为 rows、列和 ≤ cap（逐行最大余数，受剩余列容量约束）。
    sum(rows) == sum(cap) 时列和恰为 cap。"""
    M = np.asarray(M, dtype=float)
    rows = np.asarray(rows, dtype=np.int64)
    left = np.asarray(cap, dtype=np.int64).copy()
    out = np.zeros(M.shape, dtype=np.int64)
    if rows.sum() > left.sum():
        raise ValueError(f"行和 {rows.sum()} > 列容量 {left.sum()}")
    for i in range(M.shape[0]):
        x = np.minimum(M[i], left).astype(float)
        s = x.sum()
        x = x * rows[i] / s if s > 0 else left * rows[i] / max(left.sum(), 1)
        r = lr_round(x, rows[i])
        r = np.minimum(r, left)
        need = int(rows[i] - r.sum())
        while need > 0:
            room = left - r
            j = int(np.argmax(room))
            if room[j] <= 0:
                raise ValueError("列容量不足，无法凑满这一行")
            add = min(need, int(room[j]))
            r[j] += add
            need -= add
        out[i] = r
        left -= r
    return out


# ══════════════════════════════════════════════════════════════════════════
# 异质性度量（原文 §2）
# ══════════════════════════════════════════════════════════════════════════

def tv(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def h_inter(edge_dists) -> float | None:
    P = [np.asarray(p, float) for p in edge_dists]
    E = len(P)
    if E < 2:
        return None
    return float(np.mean([tv(P[i], P[j]) for i in range(E) for j in range(i + 1, E)]))


def h_inter_intra(client_counts, assignments):
    """client_counts：n_clients × K 计数；assignments：每端的 edge id。
    → (H_inter, H_intra, 各 edge 的类分布)。edge 分布 = 该 edge 全部客户端数据（不含干净集）。"""
    C = np.asarray(client_counts, dtype=float)
    a = np.asarray(assignments)
    edges = sorted(set(a.tolist()))
    pe = {e: C[a == e].sum(0) / C[a == e].sum() for e in edges}
    hi = h_inter([pe[e] for e in edges])
    ha = float(np.mean([np.mean([tv(row / row.sum(), pe[e]) for row in C[a == e]])
                        for e in edges]))
    return hi, ha, [pe[e] for e in edges]


def malicious_data_share(sizes, malicious_ids) -> float:
    sizes = np.asarray(sizes, dtype=float)
    ids = [int(i) for i in malicious_ids]
    return float(sizes[ids].sum() / sizes.sum()) if ids else 0.0


# ══════════════════════════════════════════════════════════════════════════
# 划分
# ══════════════════════════════════════════════════════════════════════════

def _edge_counts(kind, p, n_edges, edge_total, supply, rng):
    """→ (E × K 的 edge 级计数（含干净集）, 名义 H_inter)。"""
    if kind == "designed":
        T = condition_table(p["condition"], p["strength"])
        N = np.stack([lr_round(T[e] * edge_total, edge_total) for e in range(n_edges)])
        if (N.sum(0) > supply).any():
            raise ValueError(f"比例表超出供给：{N.sum(0).tolist()} > {supply.tolist()}")
        return N, h_inter(T)
    P = rng.dirichlet(np.full(K, float(p["alpha_edge"])), size=n_edges)
    if not np.isfinite(P).all():
        raise ValueError(f"Dir(α_e={p['alpha_edge']}) 抽样出现非有限值")
    M = ipf(P * edge_total, np.full(n_edges, edge_total), supply, cap=True)
    return int_matrix(M, np.full(n_edges, edge_total), supply), h_inter(P)


def _clients_in_edge(pool_counts, n_clients, n, alpha_c, rng):
    """edge 内：q_c ~ Dir(α_c · K · p_e) → 等式 IPF（行和 n、列和 = 池）→ 取整。"""
    pp = pool_counts / pool_counts.sum()
    Q = rng.dirichlet(np.maximum(alpha_c * K * pp, 1e-3), size=n_clients) * n
    M = ipf(Q, np.full(n_clients, n), pool_counts)
    return int_matrix(M, np.full(n_clients, n), pool_counts)


def designed_partition(labels, config: dict, seed: int | None = None) -> dict:
    """S3 划分。labels：合并池的标签（numpy，长度 60000）。

    Returns dict：
        train_indices / test_indices : list[np.ndarray]（每端，索引进 labels）
        assignments                  : list[int]（E0 = 前 clients_per_edge 个 id …）
        clean_indices                : list[np.ndarray]（每 edge）
        client_counts / clean_counts : 计数矩阵（每端完整分片 / 每 edge 干净集）
        info / per_edge              : `[Partition]` / `[PartitionEdge]` 行的字段（malicious_data_share 由调用方补）
    """
    kind = partition_kind(config)
    errs = config_errors(config)
    if kind not in S3_PARTITIONS:
        raise ValueError(f"federation.partition={kind!r} 不是 S3 划分 {S3_PARTITIONS}")
    if errs:
        raise ValueError("S3 划分配置不合法：\n" + "\n".join(errs))
    p = design_params(config)
    fed = config["federation"]
    n_clients, n_edges = int(fed["n_clients"]), int(fed["n_edges"])
    per, n, clean = n_clients // n_edges, int(p["n_per_client"]), int(p["clean_per_edge"])
    ratio = float((config.get("data") or {}).get("per_client_test_ratio", 0.2))
    n_test = int(round(n * ratio))
    seed = int(config.get("seed", 42) if seed is None else seed)
    rng = np.random.default_rng([seed, RNG_TAG])

    labels = np.asarray(labels).reshape(-1).astype(np.int64)      # keras 的 (N, 1) / uint8 也能用
    supply = np.bincount(labels, minlength=K).astype(np.int64)[:K]
    class_idx = [rng.permutation(np.flatnonzero(labels == k)) for k in range(K)]
    cursor = np.zeros(K, dtype=np.int64)

    def take(counts):
        out = []
        for k in range(K):
            c = int(counts[k])
            out.append(class_idx[k][cursor[k]:cursor[k] + c])
            cursor[k] += c
        return np.concatenate(out).astype(np.int64)

    nominal_h = None
    if kind == "equal_random":
        Q = rng.dirichlet(np.full(K, float(p["alpha_client"])), size=n_clients) * n
        target = np.full(K, n_clients * n // K)
        cc = int_matrix(ipf(Q, np.full(n_clients, n), target), np.full(n_clients, n), target)
        client_blocks = [cc[e * per:(e + 1) * per] for e in range(n_edges)]
        leftover = supply - cc.sum(0)
        pe = [b.sum(0) / b.sum() for b in client_blocks]
        clean_counts = int_matrix(np.stack(pe) * clean, np.full(n_edges, clean), leftover)
        client_idx = [take(row) for row in cc]
        clean_idx = [take(clean_counts[e]) for e in range(n_edges)]
    else:
        N, nominal_h = _edge_counts(kind, p, n_edges, per * n + clean, supply, rng)
        # 干净集按本 edge 分布 p_e（D-064）；行和恰为 clean、且不超过 edge 的各类计数
        clean_counts = np.stack([int_matrix((N[e] / N[e].sum() * clean)[None, :], [clean], N[e])[0]
                                 for e in range(n_edges)])
        client_blocks, client_idx, clean_idx = [], [], []
        for e in range(n_edges):
            pool = N[e] - clean_counts[e]
            if pool.sum() != per * n:
                raise ValueError(f"edge {e} 的客户端池 {pool.sum()} ≠ {per} × {n}")
            clean_idx.append(take(clean_counts[e]))
            block = _clients_in_edge(pool, per, n, float(p["alpha_client"]), rng)
            client_blocks.append(block)
            client_idx.extend(take(row) for row in block)
        cc = np.concatenate(client_blocks)

    train, test = [], []
    for idx in client_idx:
        perm = rng.permutation(len(idx))
        test.append(idx[perm[:n_test]])
        train.append(idx[perm[n_test:]])
    assignments = [e for e in range(n_edges) for _ in range(per)]

    hi, ha, pe = h_inter_intra(cc, assignments)
    info = {
        "partition": kind,
        "condition": p["condition"] if kind == "designed" else None,
        "strength": float(p["strength"]) if kind == "designed" else None,
        "alpha_edge": float(p["alpha_edge"]) if kind == "hdir" else None,
        "alpha_client": float(p["alpha_client"]),
        "n_per_client": n, "n_test_per_client": n_test, "clean_per_edge": clean,
        "n_edges": n_edges, "clients_per_edge": per,
        "h_inter": hi, "h_intra": ha,
        "h_inter_nominal": nominal_h,
        "client_size_min": int(cc.sum(1).min()), "client_size_max": int(cc.sum(1).max()),
        "max_class_use": int((cc.sum(0) + clean_counts.sum(0)).max()),
        "index_sha": index_sha(train, test, clean_idx),
        "rng_tag": RNG_TAG,
    }
    per_edge = []
    for e in range(n_edges):
        blk = client_blocks[e]
        per_edge.append({
            "edge_id": e, "n_clients": int(blk.shape[0]), "n_samples": int(blk.sum()),
            "yt_share": float(pe[e][TARGET]),
            "clean_n": int(clean_counts[e].sum()),
            "clean_yt": int(clean_counts[e][TARGET]),
            "class_counts": "/".join(str(int(v)) for v in blk.sum(0)),
            "clean_counts": "/".join(str(int(v)) for v in clean_counts[e]),
        })
    return {"train_indices": train, "test_indices": test, "assignments": assignments,
            "clean_indices": clean_idx, "client_counts": cc, "clean_counts": clean_counts,
            "info": info, "per_edge": per_edge}


def index_sha(train, test, clean) -> str:
    h = hashlib.sha256()
    for group in (train, test, clean):
        for idx in group:
            h.update(np.sort(np.asarray(idx, dtype=np.int64)).tobytes())
            h.update(b"|")
        h.update(b"#")
    return h.hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════
# 自描述行
# ══════════════════════════════════════════════════════════════════════════

def data_lines(split: dict, malicious_ids=None) -> list:
    """`[Partition]` 一行 + 每 edge 一行 `[PartitionEdge] edgeK | …`（utils/kvline，逐字段解析）。"""
    from utils.kvline import format_kv
    info = dict(split["info"])
    sizes = [len(t) for t in split["train_indices"]]
    info["malicious_data_share"] = (malicious_data_share(sizes, malicious_ids)
                                    if malicious_ids is not None else None)
    lines = [format_kv("[Partition]", info)]
    for row in split["per_edge"]:
        fields = {k: v for k, v in row.items() if k != "edge_id"}
        lines.append(format_kv("[PartitionEdge]", fields, edge_id=row["edge_id"]))
    return lines
