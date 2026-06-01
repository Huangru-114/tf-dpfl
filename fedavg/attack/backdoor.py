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


def get_malicious_ids(bd_cfg: dict) -> set:
    """返回恶意客户端 id 集合；未启用后门时为空集。"""
    if not bd_cfg or not bd_cfg.get("enabled", False):
        return set()
    return set(int(i) for i in bd_cfg.get("malicious_ids", [15]))


def build_poisoned_dataset(images_np, labels_np, train_idx, config,
                           trigger_fn, target_label, poison_ratio):
    """
    构造投毒后的本地训练 tf.data.Dataset（无增强）。

    Args:
        images_np, labels_np : 已标准化的全量图像 / 标签 numpy
        train_idx            : 该恶意客户端的训练样本索引
        trigger_fn           : 作用于标准化 numpy batch 的 trigger 函数
        target_label         : 后门目标类别
        poison_ratio         : 本地训练数据中被投毒的比例
    """
    x = np.array(images_np[train_idx], dtype=np.float32, copy=True)
    y = np.array(labels_np[train_idx], copy=True)
    n = len(train_idx)
    n_poison = int(round(n * float(poison_ratio)))

    if n_poison > 0:
        pos = np.random.permutation(n)[:n_poison]
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


def install_forced_participation(edge_servers, malicious_client, Q: int):
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

    orig_select = target_edge.select_clients

    def wrapped(round_idx):
        selected = list(orig_select(round_idx))
        attack_round = (int(round_idx) % int(Q) == 0)
        present = any(c is malicious_client for c in selected)

        if attack_round and not present:
            others = [c for c in selected if c is not malicious_client]
            if others:
                selected[selected.index(others[-1])] = malicious_client  # 等量替换
            else:
                selected.append(malicious_client)
        elif (not attack_round) and present:
            selected = [c for c in selected if c is not malicious_client]
            pool = [c for c in target_edge.clients
                    if c is not malicious_client and c not in selected]
            if pool:
                selected.append(pool[np.random.randint(len(pool))])
        return selected

    target_edge.select_clients = wrapped
    print(f"[Backdoor] forced participation installed on edge {target_edge.edge_id} "
          f"for client {malicious_client.client_id} (Q={Q}).")
