"""
tests/test_participation_quota.py  —  每轮训练的客户端总数必须与拓扑无关

**这条测试存在的理由**（让 Experiment 3A 的拓扑轴不干净的 bug）：
`EdgeServerBase.select_clients` 曾经在每个 edge 上各自向下取整 ——

    n_select = max(1, int(len(self.clients) * frac))

`client_fraction` 的语义是**全局**参与率，但这个写法是逐 edge 的。N=100、
frac=0.1 时：

    n_edges=2  → int(50*0.1)=5  × 2  = 10
    n_edges=4  → int(25*0.1)=2  × 4  =  8    ← 少训 20%
    n_edges=10 → int(10*0.1)=1  × 10 = 10

experiments/attack/hfl-propagation/README.md 声称「唯一自变量 = 恶意端在多少个
edge、怎么分布」，但 `4edge_collocated` / `4edge_distributed` / `4edge_mixed`
这三个格子同时还少训了 20% 的客户端-轮。拓扑效应和训练量混在一起，
两者都解释不清。

配额算术被抽到 `server/participation.py`（**不 import TF**），所以这条 L1
本地秒级可跑，不用等集群。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from server.participation import edge_quota   # noqa: E402


def _total_per_round(n_clients, n_edges, frac, round_idx):
    """把某一轮所有 edge 的名额加起来。假定 block 分配 → 各 edge 客户端数均等。"""
    per_edge_pop = n_clients // n_edges
    return sum(edge_quota(e, round_idx,
                          n_clients=n_clients, n_edges=n_edges,
                          client_fraction=frac, n_local_clients=per_edge_pop)
               for e in range(n_edges))


def test_total_participation_is_identical_across_topologies():
    """Experiment 3 的核心不变量：1/2/4/10 个 edge，每轮都训练同样多的客户端。"""
    N, FRAC = 100, 0.1
    expected = round(N * FRAC)          # 10
    for n_edges in (1, 2, 4, 5, 10):
        for r in range(1, 9):
            got = _total_per_round(N, n_edges, FRAC, r)
            assert got == expected, (
                f"n_edges={n_edges}, round={r}: 每轮训练 {got} 个客户端，"
                f"应为 {expected}。拓扑轴与训练量缠在一起了。")


def test_the_old_formula_really_was_broken():
    """
    反向锚点：确认上面那条不变量在旧公式下**不成立**。
    没有这条，test_total_participation 可能是在测一个本来就成立的东西。
    """
    def old(n_clients, n_edges, frac):
        return sum(max(1, int((n_clients // n_edges) * frac)) for _ in range(n_edges))

    assert old(100, 2, 0.1) == 10
    assert old(100, 10, 0.1) == 10
    assert old(100, 4, 0.1) == 8, "旧公式在 4-edge 下应当给出 8（比 10 少 20%）"


def test_remainder_rotates_so_no_edge_is_systematically_favoured():
    """
    B=10 / E=4 时配额是 {2,3,2,3} 的轮换。如果不轮转，E0（collocated 布点下
    10 个恶意端全在那里）会长期比 E1 少训一个客户端 —— 那就是用一个混淆
    换掉了另一个。
    """
    N, FRAC, E = 100, 0.1, 4
    per_edge_pop = N // E
    totals = [0] * E
    R = 4 * 25                     # 轮数取 E 的整数倍，轮换正好走完整数圈
    for r in range(R):
        for e in range(E):
            totals[e] += edge_quota(e, r, n_clients=N, n_edges=E,
                                    client_fraction=FRAC, n_local_clients=per_edge_pop)
    assert len(set(totals)) == 1, (
        f"{R} 轮后各 edge 的累计训练量是 {totals}，应当完全相同。"
        " 余数没有轮转 → 固定的某些 edge 长期多训。")


def test_quota_never_exceeds_the_edge_population():
    """名额不能超过本 edge 实际拥有的客户端数（否则无放回抽样直接抛异常）。"""
    # 极端：10 个 edge、frac=0.9 → 预算 90，但每个 edge 只有 10 个客户端
    for e in range(10):
        q = edge_quota(e, 3, n_clients=100, n_edges=10,
                       client_fraction=0.9, n_local_clients=10)
        assert 1 <= q <= 10, f"edge{e} 名额 {q} 越界"


def test_quota_is_at_least_one():
    """名额为 0 会让该 edge 整轮空转，且不会报错——必须挡住。"""
    q = edge_quota(0, 1, n_clients=100, n_edges=10,
                   client_fraction=0.01, n_local_clients=10)
    assert q >= 1


def test_falls_back_to_local_formula_without_global_context():
    """
    拿不到全局 n_clients / n_edges 时回退旧公式 —— 保证既有的单 edge 场景
    与测试桩行为不变。
    """
    assert edge_quota(0, 1, n_clients=0, n_edges=0,
                      client_fraction=0.5, n_local_clients=20) == 10


def test_quota_is_deterministic():
    """同样的 (edge_id, round_idx, 配置) 必须给同样的名额——不可复现的名额=不可复现的 run。"""
    kw = dict(n_clients=100, n_edges=4, client_fraction=0.1, n_local_clients=25)
    for e in range(4):
        for r in range(6):
            assert edge_quota(e, r, **kw) == edge_quota(e, r, **kw)
