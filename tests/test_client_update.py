"""
tests/test_client_update.py  —  客户端上传的身份与防御判决记账

钉住两条：
  1. `ClientUpdate` 必须在**不破坏任何旧解包写法**的前提下带上 client_id。
     全仓库有大量 `for weights, n, loss, t in client_updates` / `upd[0]` /
     `sum(n for _, n, *_ in ...)`，这些必须一行不改地继续工作。
  2. 防御的判决必须能按 **client_id** 读出，而不是位置索引。
     位置索引只在本次调用内有意义；防御轴的核心指标（检出率 TPR / 误伤率 FPR）
     只能按 client_id 算。

纯 numpy，本地秒级可跑。
"""

import numpy as np
import pytest

from aggregation.client_update import ClientUpdate, as_client_update, client_ids
from aggregation.fedavg import aggregate as fedavg_aggregate
from defense.base_defense import flatten_weights
from defense.flame import FlameDefense
from defense.multi_krum import MultiKrumDefense


def _w(v, shape=(4, 3)):
    return [np.full(shape, float(v), dtype=np.float32), np.full((3,), float(v), np.float32)]


# ══════════════════════════════════════════════════════════════════════════
# 1. ClientUpdate 的向后兼容性
# ══════════════════════════════════════════════════════════════════════════
def test_unpacks_as_four_tuple():
    """旧写法 `weights, n, loss, t = upd` 必须继续有效。"""
    u = ClientUpdate(_w(1), 100, 0.5, 1.2, client_id=7)
    weights, n, loss, t = u
    assert n == 100 and loss == 0.5 and t == 1.2
    assert len(u) == 4, "长度必须是 4，否则所有解包点会 ValueError"
    assert u[0] is weights and u[1] == 100


def test_star_unpacking_and_equality():
    """`for _, n, *_ in updates` 和元组比较也必须照旧。"""
    u = ClientUpdate(_w(1), 100, 0.5, 1.2, client_id=7)
    _, n, *rest = u
    assert n == 100 and len(rest) == 2
    # client_id 不参与元组语义
    assert u == (u[0], 100, 0.5, 1.2)


def test_named_access_and_identity():
    u = ClientUpdate(_w(2), 50, 0.25, 0.1, client_id=3)
    assert u.client_id == 3
    assert u.n_samples == 50 and u.loss == 0.25 and u.train_time == 0.1


def test_as_client_update_wraps_plain_tuple():
    """local_train 返回裸元组时也要能补上身份。"""
    plain = (_w(1), 10, 0.1, 0.0)
    u = as_client_update(plain, client_id=42)
    assert isinstance(u, ClientUpdate) and u.client_id == 42
    # 已有身份的不被覆盖
    again = as_client_update(u, client_id=99)
    assert again.client_id == 42


def test_existing_aggregate_still_works():
    """现有的 aggregation.fedavg.aggregate 对 ClientUpdate 必须无感。"""
    ups = [ClientUpdate(_w(1), 100, 0.0, 0.0, 0),
           ClientUpdate(_w(3), 300, 0.0, 0.0, 1)]
    agg = fedavg_aggregate(ups)
    # 加权平均：(1*100 + 3*300) / 400 = 2.5
    np.testing.assert_allclose(agg[0], np.full((4, 3), 2.5, np.float32), rtol=1e-6)


def test_client_ids_helper():
    ups = [ClientUpdate(_w(1), 1, 0.0, 0.0, 5),
           ClientUpdate(_w(1), 1, 0.0, 0.0, 9)]
    assert client_ids(ups) == [5, 9]


# ══════════════════════════════════════════════════════════════════════════
# 2. 防御判决按 client_id 记账
# ══════════════════════════════════════════════════════════════════════════
def _updates_with_ids(rng, ids, directions):
    """按给定方向构造更新（ref=0，故 W_i = e_i）。"""
    ups = []
    for cid, d in zip(ids, directions):
        v = d + 0.02 * rng.normal(size=15)
        ups.append(ClientUpdate(
            [v[:12].reshape(4, 3).astype(np.float32), v[12:].astype(np.float32)],
            100, 0.0, 0.0, cid))
    return ups


def test_defense_reports_decision_by_client_id(rng):
    """
    这是防御轴能不能算 TPR/FPR 的前提：接纳/剔除必须能读成 client_id。
    客户端 id 故意用不连续的值（30/31/…），暴露「位置索引当 id 用」的错误。
    """
    ids = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    u = rng.normal(size=15); u /= np.linalg.norm(u)
    dirs = [u] * 9 + [-u] * 3                       # 后 3 个（id 39/40/41）方向相反
    ups = _updates_with_ids(rng, ids, dirs)
    ref = [np.zeros((4, 3), np.float32), np.zeros((3,), np.float32)]

    d = MultiKrumDefense({"num_malicious": 3})
    d.aggregate(ups, ref)
    d.record_decision(ups)

    assert d.last_admitted_ids is not None, (
        "筛选类防御必须报出 last_admitted_ids —— 否则无法回答「防御剔掉了谁」")
    assert set(d.last_admitted_ids) <= set(ids), "报出的必须是 client_id，不是位置索引"
    assert set(d.last_admitted_ids) | set(d.last_rejected_ids) == set(ids), (
        "接纳 ∪ 剔除 必须覆盖全部参与者")
    assert not (set(d.last_admitted_ids) & set(d.last_rejected_ids)), "两者不能重叠"
    # 真实攻击者（39/40/41）应当落在剔除集里
    assert {39, 40, 41} <= set(d.last_rejected_ids), (
        f"方向相反的攻击者未被剔除：admitted={d.last_admitted_ids}")


def test_coordinate_defense_reports_no_client_decision(rng):
    """坐标类防御（trimmed_mean/median）不做客户端级筛选 → 判决必须是 None，不能瞎报。"""
    from defense.trimmed_mean import TrimmedMeanDefense

    ids = [1, 2, 3, 4, 5, 6]
    u = rng.normal(size=15)
    ups = _updates_with_ids(rng, ids, [u] * 6)
    ref = [np.zeros((4, 3), np.float32), np.zeros((3,), np.float32)]

    d = TrimmedMeanDefense({"trim_ratio": 0.1})
    d.aggregate(ups, ref)
    d.record_decision(ups)
    assert d.last_admitted_ids is None and d.last_rejected_ids is None


def test_flame_decision_ids_align_with_admitted_indices(rng):
    """FLAME 的 last_admitted（位置）与 last_admitted_ids（身份）必须一一对应。"""
    ids = [100, 200, 300, 400, 500, 600, 700, 800]
    u = rng.normal(size=15); u /= np.linalg.norm(u)
    ups = _updates_with_ids(rng, ids, [u] * len(ids))
    ref = [np.zeros((4, 3), np.float32), np.zeros((3,), np.float32)]

    d = FlameDefense({"noise_scale": 0.0})
    d.aggregate(ups, ref)
    d.record_decision(ups)

    expected = [ids[i] for i in sorted(d.last_admitted)]
    assert d.last_admitted_ids == expected
