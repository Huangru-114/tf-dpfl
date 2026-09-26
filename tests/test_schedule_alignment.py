"""
tests/test_schedule_alignment.py  —  A4：调度（AUDIT A08 / A15，登记行 D01 / D02）

  A08 / D-023  lr 按有效轮衰减（training.lr_round_axis=effective）
  D02 / D-036  参与配额按有效轮轮转（federation.quota_round_axis=effective）
  D01 / D-036  edge 按 edge 轮交错执行（federation.edge_schedule=interleaved）
  A15 / D-028  enable_op_determinism() + [Checksum] 行

纯 python 部分本地秒级；TF 部分（CloudServer 真跑调度循环）本地无 TF 时 skip。
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from server.participation import (edge_quota, edge_schedule_order, effective_round,
                                  round_index)
from utils.checksum import weights_checksum

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"
LR0, GAMMA = 0.1, 0.992


def _lr(idx):
    return LR0 * GAMMA ** idx


def _g_er(t_eff, R):
    """第 t_eff 个有效轮落在第几个云轮的第几个 edge 轮。"""
    return (t_eff - 1) // R + 1, (t_eff - 1) % R + 1


# ══════════════════════════════════════════════════════════════════════════
# A08：lr 的轮号
# ══════════════════════════════════════════════════════════════════════════
def test_effective_round_formula():
    assert effective_round(1, 1, 5) == 1 and effective_round(1, 5, 5) == 5
    assert effective_round(2, 1, 5) == 6 and effective_round(30, 5, 5) == 150
    with pytest.raises(ValueError):
        effective_round(1, 6, 5)


def test_flat_lr_sequence_is_unchanged():
    """R=1（flat）：新旧开关给出的 lr 序列逐元素相同（flat 逐字节不变）。"""
    old = [_lr(round_index(g, 1, 1, "cloud")) for g in range(1, 301)]
    new = [_lr(round_index(g, 1, 1, "effective")) for g in range(1, 301)]
    assert old == new


@pytest.mark.parametrize("R", [5, 40])
def test_same_effective_round_same_lr_across_topologies(R):
    for t in range(1, 201):
        g, er = _g_er(t, R)
        flat_g, flat_er = _g_er(t, 1)
        assert _lr(round_index(g, er, R, "effective")) == _lr(round_index(flat_g, flat_er, 1, "effective"))


def test_cloud_axis_confounds_flat_and_hfl_reverse_anchor():
    """反向锚点：旧口径下同一有效轮 t=7，flat 用 lr(7)，R=5 用 lr(2)。"""
    g, er = _g_er(7, 5)
    assert round_index(g, er, 5, "cloud") == 2 != round_index(7, 1, 1, "cloud")


def test_effective_axis_requires_edge_round():
    with pytest.raises(ValueError):
        round_index(3, None, 5, "effective")
    assert round_index(3, None, 5, "cloud") == 3


# ══════════════════════════════════════════════════════════════════════════
# D02：配额
# ══════════════════════════════════════════════════════════════════════════
def _q(E, e, t):
    return edge_quota(e, t, n_clients=100, n_edges=E, client_fraction=0.1,
                      n_local_clients=100 // E)


def _cycle_totals(E, R, axis, g=1):
    return [sum(_q(E, e, round_index(g, er, R, axis)) for er in range(1, R + 1))
            for e in range(E)]


@pytest.mark.parametrize("E", [1, 2, 4, 10])
def test_r1_quota_sequence_is_unchanged(E):
    for g in range(1, 50):
        for e in range(E):
            assert _q(E, e, round_index(g, 1, 1, "effective")) == _q(E, e, round_index(g, 1, 1, "cloud"))


@pytest.mark.parametrize("E,R", [(1, 5), (2, 5), (4, 5), (4, 10), (4, 20), (10, 10)])
def test_every_edge_round_spends_the_whole_budget(E, R):
    for g in (1, 2, 7):
        for er in range(1, R + 1):
            t = round_index(g, er, R, "effective")
            assert sum(_q(E, e, t) for e in range(E)) == 10


def test_four_edges_get_equal_share_per_cycle():
    """F-039：4 edge、B=10。R=10 时 25/25/25/25（反向锚点：旧口径 30/20/30/20）。"""
    assert _cycle_totals(4, 10, "effective") == [25, 25, 25, 25]
    assert _cycle_totals(4, 10, "cloud") == [30, 20, 30, 20]
    assert _cycle_totals(4, 20, "effective") == [50, 50, 50, 50]
    tot5 = _cycle_totals(4, 5, "effective")
    assert max(tot5) - min(tot5) <= 1 and sum(tot5) == 50


@pytest.mark.parametrize("E", [1, 2, 10])
def test_divisible_topologies_are_elementwise_unchanged(E):
    for R in (2, 5, 10, 20):
        assert _cycle_totals(E, R, "effective") == _cycle_totals(E, R, "cloud")


# ══════════════════════════════════════════════════════════════════════════
# D01：调度顺序
# ══════════════════════════════════════════════════════════════════════════
def test_schedule_orders():
    assert edge_schedule_order(3, [0, 1], "sequential") == \
        [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    assert edge_schedule_order(3, [0, 1], "interleaved") == \
        [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    assert edge_schedule_order(1, [0, 1, 2], "interleaved") == \
        edge_schedule_order(1, [0, 1, 2], "sequential")
    with pytest.raises(ValueError):
        edge_schedule_order(2, [0], "random")


# ══════════════════════════════════════════════════════════════════════════
# AST 守卫：run_edge_round 把 edge_round_idx 传到配额与 lr 两处
# ══════════════════════════════════════════════════════════════════════════
def _run_edge_rounds():
    out = []
    for p in sorted((FEDAVG / "server").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_edge_round" \
                    and node.body and not (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)):
                out.append((p.name, node))
    return out


def _passes_er(call):
    return any(isinstance(a, ast.Name) and a.id == "edge_round_idx" for a in call.args) or \
        any(k.arg == "edge_round_idx" for k in call.keywords)


def test_every_run_edge_round_passes_edge_round_idx():
    found = _run_edge_rounds()
    assert len(found) >= 5, [f for f, _ in found]
    for fname, fn in found:
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Attribute)
                 and c.func.attr in ("select_clients", "_collect_updates_parallel",
                                     "_collect_updates_serial")]
        kinds = {c.func.attr for c in calls}
        assert "select_clients" in kinds and kinds & {"_collect_updates_parallel",
                                                     "_collect_updates_serial"}, fname
        for c in calls:
            assert _passes_er(c), f"{fname}: {c.func.attr} 没有传 edge_round_idx"


def test_determinism_is_enabled_before_any_model_is_built():
    tree = ast.parse((FEDAVG / "main.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "run_experiment")
    det = [c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
           and isinstance(c.func, ast.Attribute) and c.func.attr == "enable_op_determinism"]
    build = [c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == "build_model"]
    assert det and build and max(det) < min(build)
    src = ast.get_source_segment((FEDAVG / "main.py").read_text(encoding="utf-8"), fn)
    assert 'get_switch(config, "training.deterministic_ops")' in src


# ══════════════════════════════════════════════════════════════════════════
# A15：checksum
# ══════════════════════════════════════════════════════════════════════════
def test_checksum_is_stable_and_sensitive(rng):
    w = [rng.normal(size=(3, 4)).astype(np.float32), rng.normal(size=4).astype(np.float32)]
    a = weights_checksum(w)
    assert a == weights_checksum([x.copy() for x in w]) and len(a) == 12
    w2 = [x.copy() for x in w]
    w2[1][0] = np.nextafter(w2[1][0], np.float32(np.inf))
    assert weights_checksum(w2) != a
    assert weights_checksum([w[0].reshape(4, 3), w[1]]) != a            # 形状也进哈希


# ══════════════════════════════════════════════════════════════════════════
# TF：CloudServer 真跑调度循环；EdgeServerBase 把有效轮传给 lr
# ══════════════════════════════════════════════════════════════════════════
def _tiny_cloud(schedule, R=3, quota_axis="cloud"):
    tf = pytest.importorskip("tensorflow")
    from server.edge_server_fedavg import FedAvgEdgeServer
    from server.server import CloudServer

    class _C:
        def __init__(self, cid):
            self.client_id, self.n_samples = cid, 10

    cfg = {"seed": 42, "federation": {"n_clients": 8, "n_edges": 2, "client_fraction": 0.5,
                                      "edge_rounds": R, "n_rounds": 2,
                                      "edge_schedule": schedule,
                                      "quota_round_axis": quota_axis},
           "training": {"drift_correction": "hierfedavg"}, "defense": {"name": "none"}}
    m = tf.keras.Sequential([tf.keras.layers.Input((2,)), tf.keras.layers.Dense(2)])
    clients = [_C(i) for i in range(8)]
    edges = [FedAvgEdgeServer(e, clients[4 * e:4 * e + 4], tf.keras.models.clone_model(m), cfg)
             for e in range(2)]
    log = []
    for e in edges:
        def rer(g, er, _e=e):
            sel = _e.select_clients(g, er)
            log.append((_e.edge_id, er, tuple(c.client_id for c in sel)))
            return 0.0, 0.0
        e.run_edge_round = rer
    cloud = CloudServer(m, edges, None, cfg)
    return cloud, log


@pytest.mark.parametrize("quota_axis", ["cloud", "effective"])
def test_interleaved_changes_order_not_selections(quota_axis):
    seq, log_s = _tiny_cloud("sequential", quota_axis=quota_axis)
    itl, log_i = _tiny_cloud("interleaved", quota_axis=quota_axis)
    seq.collect_and_aggregate(1, None)
    itl.collect_and_aggregate(1, None)
    assert [(e, er) for e, er, _ in log_i] == edge_schedule_order(3, [0, 1], "interleaved")
    assert [(e, er) for e, er, _ in log_s] == edge_schedule_order(3, [0, 1], "sequential")
    assert sorted(log_s) == sorted(log_i)          # 每个 (edge, er) 选出的客户端完全相同


def test_collect_passes_effective_round_to_lr():
    tf = pytest.importorskip("tensorflow")
    from server.edge_server_fedavg import FedAvgEdgeServer

    class _C:
        def __init__(self, cid):
            self.client_id, self.n_samples, self.seen = cid, 10, []

        def apply_round_lr(self, r):
            self.seen.append(r)

        def local_train(self, r):
            return [np.zeros(1)], 10, 0.0, 0.0

        def get_aux(self):
            return {}

    for axis, want in (("cloud", 2), ("effective", 8)):
        cfg = {"seed": 1, "federation": {"n_clients": 2, "n_edges": 1, "client_fraction": 1.0,
                                         "edge_rounds": 5, "n_workers": 1},
               "training": {"lr_round_axis": axis}, "defense": {"name": "none"}}
        m = tf.keras.Sequential([tf.keras.layers.Input((2,)), tf.keras.layers.Dense(2)])
        cs = [_C(0), _C(1)]
        e = FedAvgEdgeServer(0, cs, m, cfg)
        e._collect_updates_parallel(cs, 2, edge_round_idx=3)           # g=2, er=3 → t_eff=8
        e._collect_updates_serial(cs, 2, edge_round_idx=3)
        assert all(c.seen == [want, want] for c in cs), (axis, [c.seen for c in cs])
