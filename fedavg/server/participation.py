"""
server/participation.py  —  每轮参与名额的整数配额

单独成文件（**不 import tensorflow**）的理由：这里全是整数算术，是
Experiment 3 拓扑轴是否干净的判据，应该能在本地秒级测到，而不必等集群。
`edge_server_base.py` 只调用本模块。
"""


def edge_quota(edge_id: int, round_idx: int, *,
               n_clients: int, n_edges: int, client_fraction: float,
               n_local_clients: int) -> int:
    """
    edge `edge_id` 在第 `round_idx` 轮该抽多少个客户端。

    `client_fraction` 是**全局**参与率：无论把 N 个客户端分成 1/2/4/10 个 edge，
    每个 edge-round 全系统训练的客户端总数都应当是 round(N * frac)。

    旧实现在每个 edge 上各自向下取整 —— `max(1, int(len(clients) * frac))` ——
    于是（N=100, frac=0.1）：

        n_edges=2  → int(50*0.1)=5  × 2  = 10
        n_edges=4  → int(25*0.1)=2  × 4  = **8**   ← 比别人少训 20%
        n_edges=10 → int(10*0.1)=1  × 10 = 10

    Experiment 3A 的 README 声称「唯一自变量 = 恶意端布点」，但 4-edge 的三个
    格子整整少训 20% 的客户端-轮，拓扑轴并不干净。

    改用整数配额：edge e 在第 r 轮拿

        floor((s+1)·B / E) − floor(s·B / E),   s = (e + r) mod E

    其中 B = 全局预算、E = edge 总数。

      · 逐 edge 求和恒等于 B —— s 只是把同一个配额多重集做了个轮换；
      · 只依赖 (edge_id, n_edges, B, round_idx)，不需要知道别的 edge 有多少
        客户端 → 确定性、可复现；
      · 余数**按轮轮转**。这一条不是锦上添花：恶意端是按 edge 布点的
        （collocated 把 10 个全压在 E0），固定配额会让「布点」和「训练量」
        缠在一起 —— 那正是本函数要消除的那类混淆。

    拿不到全局 n_clients / n_edges 时回退旧公式，不改变既有单 edge 场景的行为。
    """
    if n_clients > 0 and n_edges > 0:
        budget = max(n_edges, int(round(n_clients * client_fraction)))
        s = (int(edge_id) + int(round_idx)) % n_edges
        quota = (s + 1) * budget // n_edges - s * budget // n_edges
    else:
        quota = int(n_local_clients * client_fraction)

    # 名额不能超过本 edge 实际有的客户端数；也至少要有 1 个，否则该 edge 整轮空转。
    return max(1, min(quota, n_local_clients))


# ══════════════════════════════════════════════════════════════════════════
# A4：有效轮与调度（AUDIT A08 / D-023，登记行 D01 / D02，D-036）
# ══════════════════════════════════════════════════════════════════════════
#
# 有效轮 t_eff = 截至本 edge 轮已完成的 edge 轮数（含本轮）= (g−1)·R + er。
# R = 1（flat）时 t_eff 恰好等于云轮号 g → 按有效轮的 lr / 配额在 flat 下与旧行为逐元素相同。

def effective_round(global_round_idx: int, edge_round_idx: int, edge_rounds: int) -> int:
    """第 g 个云轮（1 起）里的第 er 个 edge 轮（1 起）是第几个有效轮。"""
    g, er, R = int(global_round_idx), int(edge_round_idx), int(edge_rounds)
    if g < 1 or not 1 <= er <= R:
        raise ValueError(f"g={g}, er={er}, R={R}：要求 g ≥ 1 且 1 ≤ er ≤ R")
    return (g - 1) * R + er


def round_index(global_round_idx: int, edge_round_idx, edge_rounds: int, axis: str) -> int:
    """
    lr（training.lr_round_axis，A08）与参与配额（federation.quota_round_axis，D02）用哪个轮号：
      cloud     （旧）云轮号 g —— 同一云周期内 R 个 edge 轮拿同一个值
      effective 有效轮 t_eff —— 跨拓扑在同一有效轮上 lr 相同（F-003）、4 edge 的名额
                不再 3:2 偏斜（F-039）
    """
    if axis == "cloud":
        return int(global_round_idx)
    if axis == "effective":
        if edge_round_idx is None:
            raise ValueError("axis='effective' 需要 edge_round_idx（run_edge_round 必须把它传下来）")
        return effective_round(global_round_idx, edge_round_idx, edge_rounds)
    raise ValueError(f"未知的轮号轴 {axis!r}（cloud / effective）")


def edge_schedule_order(edge_rounds: int, edge_ids, mode: str) -> list:
    """
    一个云周期内 (edge_id, er) 的执行顺序（登记行 D01 / D-036）。

      sequential （旧）每个 edge 连续跑完 R 轮再轮到下一个
      interleaved HierFAVG 官方（hierfavg.py:299-325）：外层 edge 轮、内层 edge

    为什么要改（F-038）：共享生成器（badpfl_shared_generator，对齐官方 fba.py:27）
    是同一个对象，顺序执行时 E1 的攻击者第 1 轮拿到的生成器已在 E0 第 R 轮的模型上
    训过；交错执行下所有 edge 停在同一个有效轮。生成器仍是同一个对象。
    """
    R = int(edge_rounds)
    ids = list(edge_ids)
    if mode == "sequential":
        return [(e, er) for e in ids for er in range(1, R + 1)]
    if mode == "interleaved":
        return [(e, er) for er in range(1, R + 1) for e in ids]
    raise ValueError(f"未知的 edge_schedule {mode!r}（sequential / interleaved）")
