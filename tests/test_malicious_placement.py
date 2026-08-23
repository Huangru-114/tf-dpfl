"""
tests/test_malicious_placement.py  —  确定性布点（Experiment 3A/3B enabling）

锁住两件事：
  1. block_assignment：确定性连续分块，client id 段 → edge，可从 id 直观预期。
  2. assign_malicious_by_edge / resolve_malicious_ids(by_edge)：按 per_edge 精确布点，
     collocated / distributed / mixed 三种拓扑落到正确的 edge，且可复现、越界报错。

clustering / backdoor 均在模块级 import tensorflow，故整体 importorskip（集群跑）。
布点逻辑本身是纯 numpy。
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from data.clustering import block_assignment            # noqa: E402
from attack.backdoor import (                            # noqa: E402
    assign_malicious_by_edge, resolve_malicious_ids,
)


class _C:
    def __init__(self, i):
        self.client_id = i


def _edge_of(assignments):
    """cid -> edge，供断言恶意端落在哪个 edge。"""
    return {cid: int(e) for cid, e in enumerate(assignments)}


# ══════════════════════════════════════════════════════════════════════════
# 1. block_assignment：确定性连续分块
# ══════════════════════════════════════════════════════════════════════════
def test_block_assignment_is_contiguous_and_deterministic():
    clients = [_C(i) for i in range(40)]
    a1 = block_assignment(clients, 4)
    a2 = block_assignment(clients, 4)
    assert a1 == a2, "block 分配必须确定（无洗牌）"
    # 40/4 = 10 连续一段
    assert a1[:10] == [0] * 10
    assert a1[10:20] == [1] * 10
    assert a1[20:30] == [2] * 10
    assert a1[30:40] == [3] * 10


def test_block_assignment_uneven_split():
    """不整除时用 array_split：前几段多 1，仍连续、仍覆盖全部 client。"""
    clients = [_C(i) for i in range(10)]
    a = block_assignment(clients, 3)          # 4,3,3
    assert a == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


# ══════════════════════════════════════════════════════════════════════════
# 2. assign_malicious_by_edge：三种拓扑
# ══════════════════════════════════════════════════════════════════════════
def test_collocated_all_in_one_edge():
    clients = [_C(i) for i in range(40)]
    a = block_assignment(clients, 4)
    mal = assign_malicious_by_edge(a, [4, 0, 0, 0], seed=1)
    e = _edge_of(a)
    assert len(mal) == 4
    assert all(e[m] == 0 for m in mal), "collocated：全部恶意端必须落在 edge0"


def test_distributed_one_per_edge():
    clients = [_C(i) for i in range(40)]
    a = block_assignment(clients, 4)
    mal = assign_malicious_by_edge(a, [1, 1, 1, 1], seed=1)
    e = _edge_of(a)
    assert sorted(e[m] for m in mal) == [0, 1, 2, 3], "distributed：每 edge 恰 1 个"


def test_mixed_topology():
    clients = [_C(i) for i in range(40)]
    a = block_assignment(clients, 4)
    mal = assign_malicious_by_edge(a, [3, 1, 0, 0], seed=1)
    e = _edge_of(a)
    per = {0: 0, 1: 0, 2: 0, 3: 0}
    for m in mal:
        per[e[m]] += 1
    assert per == {0: 3, 1: 1, 2: 0, 3: 0}


def test_by_edge_is_reproducible():
    clients = [_C(i) for i in range(40)]
    a = block_assignment(clients, 4)
    m1 = assign_malicious_by_edge(a, [2, 2, 0, 0], seed=7)
    m2 = assign_malicious_by_edge(a, [2, 2, 0, 0], seed=7)
    assert m1 == m2, "同 seed 布点必须一致（否则同格重跑不可比）"


def test_by_edge_capacity_and_length_errors():
    clients = [_C(i) for i in range(40)]
    a = block_assignment(clients, 4)          # 每 edge 10 个
    with pytest.raises(ValueError):
        assign_malicious_by_edge(a, [11, 0, 0, 0], seed=1)   # 放不下
    with pytest.raises(ValueError):
        assign_malicious_by_edge(a, [1, 1, 1], seed=1)       # 长度 != edge 数
    with pytest.raises(ValueError):
        assign_malicious_by_edge(None, [1, 1, 1, 1], seed=1)  # 无 assignments


# ══════════════════════════════════════════════════════════════════════════
# 3. resolve_malicious_ids 的 by_edge 分支（端到端）
# ══════════════════════════════════════════════════════════════════════════
def test_resolve_by_edge_end_to_end():
    clients = [_C(i) for i in range(40)]
    a = block_assignment(clients, 4)
    bd = {"enabled": True, "malicious_placement": "by_edge",
          "malicious_per_edge": [3, 1, 0, 0]}
    mal = resolve_malicious_ids(bd, n_clients=40, assignments=a, seed=1)
    e = _edge_of(a)
    assert sum(1 for m in mal if e[m] == 0) == 3
    assert sum(1 for m in mal if e[m] == 1) == 1
    assert all(e[m] in (0, 1) for m in mal)


def test_resolve_by_edge_missing_spec_raises():
    bd = {"enabled": True, "malicious_placement": "by_edge"}
    with pytest.raises(ValueError):
        resolve_malicious_ids(bd, n_clients=40, assignments=[0] * 40, seed=1)
