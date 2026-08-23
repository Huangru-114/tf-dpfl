"""
tests/test_forced_participation.py  —  install_forced_participation 的语义不变量

**为什么需要这条 L1**：exp006 的 pm_acc 崩溃根因就在这个包装器 ×「默认开启」的组合：
Q=1 时恶意端每轮被强制选入，而补位循环**只砍良性、绝不砍恶意** → 当某 edge 的恶意端数
≥ 该 edge 的抽样名额时，该 edge **一个良性端都进不来**（FedRep 下这些良性端私有 head
永不训练 → 输出退化）。本会话把强制参与改成**默认关闭**（config: backdoor.forced_participation，
默认 false = 所有客户端同概率被抽取），但保留这个接口给「投毒时间模式」实验。

这里锁住两件事：
  1. 关闭时（不安装包装器）—— 恶意端与良性端同概率被抽取，恶意端**不是**每轮都在。
  2. 打开时 —— 包装器的强制/换出语义精确成立，且**恶意密集 edge 不再饿死良性端**
     （base 抽中的良性全部保留；exp006 崩溃根因的修复守卫，改回旧逻辑会红）。

包装器逻辑是纯 numpy，但 attack.backdoor 在模块级 import tensorflow，故整体 importorskip。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")  # attack.backdoor 模块级 import tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from attack.backdoor import install_forced_participation   # noqa: E402


class _Client:
    def __init__(self, cid):
        self.client_id = cid


class _Edge:
    """最小 edge：复刻 EdgeServerBase.select_clients 的抽样公式（int(len·frac) 无放回随机）。"""
    def __init__(self, edge_id, clients, frac, seed=0):
        self.edge_id = edge_id
        self.clients = clients
        self._frac = frac
        self.rng = np.random.default_rng([seed, edge_id])

    def select_clients(self, round_idx):
        n_select = max(1, int(len(self.clients) * self._frac))
        idx = self.rng.choice(len(self.clients), n_select, replace=False)
        return [self.clients[int(i)] for i in idx]


def _ids(clients):
    return sorted(c.client_id for c in clients)


# ══════════════════════════════════════════════════════════════════════════
# 1. 关闭（不安装）—— 同概率抽样，恶意端不是每轮都在
# ══════════════════════════════════════════════════════════════════════════
def test_without_install_malicious_are_sampled_at_frac_not_forced():
    """
    forced_participation=false 的等价物：不安装包装器 → 恶意端和良性端一样按 frac 被抽。
    单个恶意端在 200 轮里的入选频率应≈frac(0.1)，**远不是 1.0**（这正是「同等对待」）。
    """
    clients = [_Client(i) for i in range(50)]     # 50 客户端
    edge = _Edge(0, clients, frac=0.1, seed=1)    # 每轮抽 5
    mal_id = 11
    hits = sum(1 for r in range(1, 201)
               if mal_id in _ids(edge.select_clients(r)))
    assert hits < 60, f"恶意端 {mal_id} 入选 {hits}/200，远超 frac=0.1 的期望(~20)，" \
                      f"说明它被变相强制了"
    assert hits > 0


# ══════════════════════════════════════════════════════════════════════════
# 2. 打开、Q=1 —— 恶意端每轮必在
# ══════════════════════════════════════════════════════════════════════════
def test_q1_forces_malicious_every_round():
    clients = [_Client(i) for i in range(50)]
    edge = _Edge(0, clients, frac=0.1, seed=2)
    mal = clients[11]
    install_forced_participation([edge], mal, Q=1)
    for r in range(1, 21):
        assert mal in edge.select_clients(r), f"Q=1 下恶意端在第 {r} 轮缺席"


# ══════════════════════════════════════════════════════════════════════════
# 3. 打开、Q=2 —— 只在 round%2==0 的轮强制在场，其余轮换出
# ══════════════════════════════════════════════════════════════════════════
def test_q2_participation_follows_schedule():
    clients = [_Client(i) for i in range(50)]
    edge = _Edge(0, clients, frac=0.2, seed=3)   # 每轮抽 10，够放下并换出
    mal = clients[11]
    install_forced_participation([edge], mal, Q=2)
    present = {r: (mal in edge.select_clients(r)) for r in range(1, 11)}
    assert all(present[r] for r in present if r % 2 == 0), "偶数轮恶意端应被强制在场"
    assert all(not present[r] for r in present if r % 2 == 1), "奇数轮恶意端应被换出"


# ══════════════════════════════════════════════════════════════════════════
# 4. 打开 + 恶意密集 edge —— 良性端**不再被饿死**（exp006 崩溃根因的修复守卫）
# ══════════════════════════════════════════════════════════════════════════
def test_dense_edge_keeps_benign_when_forced():
    """
    edge 有 7 个恶意端、base 抽样名额 int(50·0.1)=5。修复后：强制参与只保证恶意端在场，
    **base 抽中的良性端一个不砍**（selected = base良性 + must_in恶意）。所以恶意密集 edge
    每轮 = base 的全部良性 + 7 恶意端，良性不再被饿死（对照 exp006 的 0 良性）。

    用一个同种子的未包装 twin edge 复算 base，逐轮精确断言「base 的良性全部保留」。
    """
    mal_ids = [0, 7, 14, 21, 28, 35, 42]         # 7 个恶意端（均 < 50，落在本 edge 内）
    clients_f = [_Client(i) for i in range(50)]
    clients_r = [_Client(i) for i in range(50)]
    edge_f = _Edge(0, clients_f, frac=0.1, seed=4)   # 被强制
    edge_r = _Edge(0, clients_r, frac=0.1, seed=4)   # 同种子参照，不装包装器
    for mid in mal_ids:
        install_forced_participation([edge_f], clients_f[mid], Q=1)

    mal_set = set(mal_ids)
    saw_benign = False
    for r in range(1, 11):
        forced = set(_ids(edge_f.select_clients(r)))
        base   = set(_ids(edge_r.select_clients(r)))     # 同 RNG 序列 → 同 base
        base_benign = base - mal_set
        assert mal_set <= forced, f"第 {r} 轮恶意端未全部在场：{sorted(forced)}"
        assert base_benign <= forced, (
            f"第 {r} 轮 base 的良性 {sorted(base_benign)} 被砍掉了 —— 良性饿死回归")
        saw_benign = saw_benign or bool(base_benign)
    assert saw_benign, "10 轮里 base 从未抽到良性端，测试无区分力（应几乎必然抽到）"


def test_multiple_malicious_do_not_evict_each_other():
    """同 edge 多恶意端：一层包装器 + 登记表，Q=1 下彼此都在，不互相顶替。"""
    clients = [_Client(i) for i in range(50)]
    mal_ids = [11, 22]
    edge = _Edge(0, clients, frac=0.2, seed=5)   # 名额 10，放得下
    for mid in mal_ids:
        install_forced_participation([edge], clients[mid], Q=1)
    for r in range(1, 11):
        sel = set(_ids(edge.select_clients(r)))
        assert {11, 22} <= sel, f"第 {r} 轮 {sel} 缺了某个恶意端（互相顶替了）"
