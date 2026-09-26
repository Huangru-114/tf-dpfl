"""
utils/pm.py  —  评估用的个性化模型（PM）怎么组装（AUDIT A28 / D-033）

D-033（用户提出的硬规则）：**主 ASR 与主 pm_acc 在同一个个性化模型上测**，两者都用

    fresh-PM = [评估时刻所在 edge 的当前 body, 自己的私有部分]

私有部分由客户端给出（`client.private_state()`）：
  · FedAvg：没有 → fresh-PM 就是当前 edge 模型
  · FedRep：私有 head（+ A27 私有时的 BN moving 统计量）
这与 PFLlib 的评估一致（serverrep.py:26-32：每轮先把 base 下发给全部客户端再评估）。
P1 的「陈旧 PM」（client.model = 上次参与时收到的 body + 在它上面训的 head）留作副列。

纯 python，不 import TF —— 组装规则本地秒级可测。
"""

from __future__ import annotations


def compose_pm(edge_w: list, private_w: list, private_idx: list) -> list:
    """edge 权重列表里，把 private_idx 位置换成 private_w（一一对应）。

    返回新列表；**不修改** edge_w（它是 edge 模型的权重，还要给别的客户端用）。
    未被替换的位置直接引用 edge_w 的数组（set_weights 会复制，不需要这里再拷一份）。
    """
    if len(private_w) != len(private_idx):
        raise ValueError(f"private_w 有 {len(private_w)} 项，private_idx 有 {len(private_idx)} 项")
    out = list(edge_w)
    for i, w in zip(private_idx, private_w):
        if not 0 <= int(i) < len(out):
            raise IndexError(f"私有索引 {i} 超出权重列表长度 {len(out)}")
        if tuple(getattr(w, "shape", ())) != tuple(getattr(out[i], "shape", ())):
            raise ValueError(f"私有索引 {i} 的形状 {getattr(w, 'shape', None)} 与 edge 权重 "
                             f"{getattr(out[i], 'shape', None)} 不一致")
        out[int(i)] = w
    return out
