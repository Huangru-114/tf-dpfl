"""
attack/backdoor.py  –  恶意客户端接入与 fix-frequency 强制参与

设计（不修改现有训练逻辑）：
  - get_malicious_ids：从 config 读恶意客户端编号集合。
  - build_poisoned_dataset：构造投毒后的本地训练集（poison_ratio 比例样本加 trigger +
    改标签为 target_label），用于在 client 构造时替换原数据集。
    注意：恶意客户端数据集**不做数据增强**，以保证右下角 trigger 不被 flip/crop 破坏
    （benign 客户端不受影响，仍走原 make_client_dataset 的增强管线）。
  - install_forced_participation：运行时包装恶意客户端所在 edge 的 select_clients，
    实现 fix-frequency（round%Q==0 强制参与，其余轮换出），不改 edge 源码。
"""

import numpy as np
import tensorflow as tf

from attack.triggers import make_dba_local_trigger


def make_dba_local_triggers(bd_cfg: dict, malicious_ids) -> dict:
    """
    DBA：为每个恶意客户端分配一个**局部**触发器（按 client id 排序的 rank 循环取
    backdoor.dba_patterns）。本地投毒只用自己的局部 pattern；ASR 评估用全局触发器
    （build_trigger('dba')）。

    Returns:
        {client_id: trigger_fn(x)}  —— 投毒侧 1 参触发器
    """
    patterns = bd_cfg.get("dba_patterns") or []
    if not patterns:
        raise ValueError("trigger='dba' 需要 backdoor.dba_patterns（局部触发器坐标列表）。")
    value = float(bd_cfg.get("badnet_value", 1.0))
    out = {}
    for rank, cid in enumerate(sorted(int(i) for i in malicious_ids)):
        pattern = patterns[rank % len(patterns)]
        out[cid] = make_dba_local_trigger(pattern, value=value)
        print(f"  [DBA] client {cid} ← local pattern #{rank % len(patterns)} "
              f"({len(pattern)} px)")
    return out


def get_malicious_ids(bd_cfg: dict) -> set:
    """返回恶意客户端 id 集合；未启用后门时为空集。"""
    if not bd_cfg or not bd_cfg.get("enabled", False):
        return set()
    return set(int(i) for i in bd_cfg.get("malicious_ids", [15]))


def assign_malicious_across_edges(n_clients, n_malicious, assignments=None,
                                  seed=42):
    """
    把 n_malicious 个恶意客户端**尽量分散到不同边缘节点**（每 edge ≥1，
    当 n_malicious ≥ n_edges 时）。固定 seed 保证可复现。

    Args:
        n_clients    : 客户端总数
        n_malicious  : 恶意客户端数量
        assignments  : List[int] 或 None。client→edge 归属（baked 分区时可用）。
                       为 None 时退化为在 client id 空间上等距取点（统计上也能分散）。
        seed         : 随机种子
    Returns:
        set[int] 恶意客户端 id
    """
    rng = np.random.default_rng(seed)
    n_malicious = max(0, min(int(n_malicious), int(n_clients)))
    if n_malicious == 0:
        return set()

    if assignments is None:
        # 无 edge 归属信息：在 id 空间等距取点，随机分配后统计上也能跨 edge 分散
        picks = np.linspace(0, n_clients - 1, n_malicious).round().astype(int)
        return set(int(i) for i in np.unique(picks))

    # 按 edge 分组，轮转挑选，保证每 edge 至少 1 个（数量允许时）
    edges = {}
    for cid, e in enumerate(assignments):
        edges.setdefault(int(e), []).append(cid)
    for e in edges:
        rng.shuffle(edges[e])

    chosen, edge_keys = [], sorted(edges.keys())
    round_i = 0
    while len(chosen) < n_malicious:
        progressed = False
        for e in edge_keys:
            if round_i < len(edges[e]):
                chosen.append(edges[e][round_i])
                progressed = True
                if len(chosen) >= n_malicious:
                    break
        round_i += 1
        if not progressed:           # 所有 edge 都取尽
            break
    return set(int(i) for i in chosen)


def resolve_malicious_ids(bd_cfg: dict, n_clients: int,
                          assignments=None, seed=42) -> set:
    """
    统一解析恶意客户端 id，支持两种放置策略（默认 spread）：

      malicious_placement == "spread"       跨 edge 分散（assign_malicious_across_edges）
      malicious_placement == "concentrated" 直接用显式 malicious_ids（集中攻击实验）

    恶意数量优先取 bd_cfg["n_malicious"]，否则回退显式 malicious_ids 的长度。
    建议在 build_clients 解析一次后写回 bd_cfg["malicious_ids"]，保证后续调用一致。
    """
    if not bd_cfg or not bd_cfg.get("enabled", False):
        return set()

    placement = str(bd_cfg.get("malicious_placement", "spread")).lower()
    explicit = bd_cfg.get("malicious_ids", None)
    n_mal = int(bd_cfg.get("n_malicious", 0))

    if placement == "concentrated" or (n_mal <= 0 and explicit is not None):
        return set(int(i) for i in (explicit if explicit else [15]))

    if n_mal <= 0:
        n_mal = len(explicit) if explicit else 1
    return assign_malicious_across_edges(n_clients, n_mal, assignments, seed)


def build_poisoned_dataset(images_np, labels_np, train_idx, config,
                           trigger_fn, target_label, poison_ratio,
                           rng=None):
    """
    构造投毒后的本地训练 tf.data.Dataset（无增强）。

    Args:
        images_np, labels_np : 已标准化的全量图像 / 标签 numpy
        train_idx            : 该恶意客户端的训练样本索引
        trigger_fn           : 作用于标准化 numpy batch 的 trigger 函数
        target_label         : 后门目标类别
        poison_ratio         : 本地训练数据中被投毒的比例
        rng                  : np.random.Generator。哪些样本被投毒必须可复现，
                               否则同一格重跑的后门数据集都不一样。
                               None 时按 config["seed"] 现场造一条（仅兜底）。
    """
    if rng is None:
        rng = np.random.default_rng(int((config or {}).get("seed", 42)))
    x = np.array(images_np[train_idx], dtype=np.float32, copy=True)
    y = np.array(labels_np[train_idx], copy=True)
    n = len(train_idx)
    n_poison = int(round(n * float(poison_ratio)))

    if n_poison > 0:
        pos = rng.permutation(n)[:n_poison]
        x[pos] = trigger_fn(x[pos])
        y[pos] = int(target_label)

    bs = config["data"]["batch_size"]
    ds = (tf.data.Dataset.from_tensor_slices((x, y))
          .shuffle(max(1, n))
          .batch(bs)
          .prefetch(tf.data.AUTOTUNE))
    print(f"  [Backdoor] poisoned dataset built: n={n}, poisoned={n_poison} "
          f"(ratio={poison_ratio}), target_label={target_label} (no augmentation)")
    return ds


def install_forced_participation(edge_servers, malicious_client, Q: int,
                                 rng=None):
    """
    fix-frequency 强制参与：包装恶意客户端所在 edge 的 select_clients。

      round % Q == 0 ：保证 malicious_client 在选中集（必要时换掉一个良性以保持规模）
      其余轮         ：若 malicious_client 被随机选中则换出（补一个未选中的良性）

    通过给该 edge 实例重新赋值 select_clients 实现，不修改 edge 类源码。
    """
    target_edge = None
    for e in edge_servers:
        if any(c is malicious_client for c in e.clients):
            target_edge = e
            break
    if target_edge is None:
        print("[Backdoor] malicious client not found in any edge; skip forced participation.")
        return

    # 同一个 edge 上可能有**多个**恶意客户端。旧实现给每个都套一层包装器，
    # 后一层用「等量替换 others[-1]」补位时可能把前一层刚放进去的恶意客户端顶掉
    # （n_malicious / n_edges ≥ 2 时必然发生）。改为：每个 edge 只装一层包装器，
    # 用一张登记表统一处理该 edge 上所有被强制的客户端。
    registry = getattr(target_edge, "_forced_participation", None)
    if registry is None:
        registry = {}
        target_edge._forced_participation = registry
        orig_select = target_edge.select_clients
        # 强制参与的补位也必须可复现，走 edge 自己的 RNG（或显式传入的）
        pick_rng = rng if rng is not None else getattr(
            target_edge, "rng", np.random.default_rng(42))

        def wrapped(round_idx):
            base     = list(orig_select(round_idx))
            target_n = len(base)
            r        = int(round_idx)

            must_in  = [c for c, q in registry.items() if r % int(q) == 0]
            must_out = {c for c, q in registry.items() if r % int(q) != 0}

            # 1) 踢掉本轮不该在场的恶意客户端
            selected = [c for c in base if c not in must_out]
            # 2) 放入本轮必须在场的（去重，互不顶替）
            for mc in must_in:
                if mc not in selected:
                    selected.append(mc)
            # 3) 规模回到原本的选取数：多了只砍良性，少了只从良性池补
            benign_in = [c for c in selected if c not in registry]
            while len(selected) > target_n and benign_in:
                selected.remove(benign_in.pop())
            if len(selected) < target_n:
                pool = [c for c in target_edge.clients
                        if c not in registry and c not in selected]
                need = target_n - len(selected)
                order = pick_rng.permutation(len(pool))[:need]
                selected.extend(pool[int(i)] for i in order)
            return selected

        target_edge.select_clients = wrapped

    registry[malicious_client] = int(Q)
    print(f"[Backdoor] forced participation installed on edge {target_edge.edge_id} "
          f"for client {malicious_client.client_id} (Q={Q}); "
          f"该 edge 上已登记 {len(registry)} 个恶意客户端。")
