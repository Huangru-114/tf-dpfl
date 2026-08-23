"""
tests/test_forced_participation.py  —  install_forced_participation 的语义不变量

**为什么需要这条 L1**：exp006 的 pm_acc 崩溃根因就在这个包装器 ×「默认开启」的组合：
Q=1 时恶意端每轮被强制选入，而补位循环**只砍良性、绝不砍恶意** → 当某 edge 的恶意端数
≥ 该 edge 的抽样名额时，该 edge **一个良性端都进不来**（FedRep 下这些良性端私有 head
永不训练 → 输出退化）。本会话把强制参与改成**默认关闭**（config: backdoor.forced_participation，
默认 false = 所有客户端同概率被抽取），但保留这个接口给「投毒时间模式」实验。

这里锁住两件事：
  1. 关闭时（不安装包装器）—— 恶意端与良性端同概率被抽取，恶意端**不是**每轮都在。
  2. 打开时 —— 包装器的强制/换出/补位语义精确成立，且**恶意密集 edge 会饿死良性端**
     （把 exp006 的失败模式钉成显式断言，改坏了会红）。

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
# 4. 打开 + 恶意密集 edge —— 良性端被饿死（exp006 的失败模式，显式钉住）
# ══════════════════════════════════════════════════════════════════════════
def test_dense_edge_starves_benign_when_forced():
    """
    edge 有 7 个恶意端、抽样名额 int(50·0.1)=5。强制参与只砍良性、不砍恶意 →
    该 edge 每轮选中集 = 全部 7 个恶意端、**0 个良性端**。这是 exp006 里 edge0 的真实行为，
    也正是「默认必须关闭」的原因。
    """
    clients = [_Client(i) for i in range(50)]
    mal_ids = [0, 11, 22, 33, 44, 55, 66]        # 7 个恶意端
    edge = _Edge(0, clients, frac=0.1, seed=4)   # 名额 = 5 < 7
    for mid in mal_ids:
        install_forced_participation([edge], clients[mid], Q=1)
    for r in range(1, 11):
        sel = _ids(edge.select_clients(r))
        assert sel == sorted(mal_ids), (
            f"第 {r} 轮选中 {sel}，期望恰为 7 个恶意端、0 良性 —— "
            f"恶意密集 edge 的良性饿死没有复现，说明补位逻辑变了")


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
