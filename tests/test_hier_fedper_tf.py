"""
阶段三 Hier-FedPer（`client/hier_fedper.py` / `server/hier_fedper.py`，D-113 / D-114）的 TF 部分。需要 TF。

照 CCS 官方代码的 FedPer（F-109）：每一步联合训练全部可训练变量；私有部分 = head + proj_*；BN 统计量随 body 共享。
  ① 一步联合训练：body、proj、head 都变了（对照 FedRep 的 body 步：proj / head 不动）；
  ② CCS 的对抗训练步在 FedPer 下梯度到达 proj / head，在 FedRep 下照旧只到 body（step_vars 接口，D-114）；
  ③ 一个 edge 轮：BN 统计量 = 上传物的样本加权平均、下发后客户端用的是聚合值（不私有）；私有层各端不同、edge 的不变；
  ④ 每个训练批都经过 on_batch（投毒入口），批数 = local_epochs × 批数；
  ⑤ 接线：resolve_client_classes 在 hier_fedper 下给出 FedPer 类（含 CCS mixin），edge 类是 HierFedPerEdgeServer。
小模型与夹具沿用 test_ccs_tf.py。
"""

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_ccs_tf as T                                     # noqa: E402  夹具：_cfg / _small_proj_model / _data
from client.compose import compose_client_class             # noqa: E402
from client.hier_fedper import HierFedPerClient             # noqa: E402
from client.hier_fedrep import HierFedRepClient             # noqa: E402
from defense.ccs import CCSDefense                          # noqa: E402
from models.cnn import get_bn_stat_indices                  # noqa: E402


def _cfg_fedper(ccs=True, **kw):
    cfg = T._cfg(**kw)
    cfg["training"]["drift_correction"] = "hier_fedper"
    if not ccs:
        cfg["defense"] = {"name": "none"}
    return cfg


def _client(cfg, cls, cid=1, model=None, ccs=True):
    Cls = compose_client_class(cls, CCSDefense.client_mixin) if ccs else cls
    x, y = T._data(cid)
    m = model if model is not None else T._small_proj_model()
    c = Cls(client_id=cid, dataset=tf.data.Dataset.from_tensor_slices((x, y)).batch(T.BS),
            model=m, config=cfg, n_samples=len(x))
    c.is_malicious = False
    with contextlib.redirect_stdout(io.StringIO()):
        c.set_weights(m.get_weights())
    return c, x, y


def _changed(w0, w1, idx):
    return {i: not np.array_equal(w0[i], w1[i]) for i in idx}


# ── ① 联合步 ──────────────────────────────────────────────────────────────
def test_joint_step_updates_body_and_private_layers():
    c, x, y = _client(_cfg_fedper(ccs=False), HierFedPerClient, ccs=False)
    tv = {id(v) for v in c.model.trainable_variables}
    train_idx = [i for i, v in enumerate(c.model.weights) if id(v) in tv]
    w0 = c.model.get_weights()
    c._train_joint_step(tf.constant(x[:T.BS]), tf.constant(y[:T.BS]))
    w1 = c.model.get_weights()
    ch = _changed(w0, w1, train_idx)
    head = set(c._head_w_idx)
    assert all(ch[i] for i in head)                                   # head + proj_* 都被训了
    assert any(ch[i] for i in train_idx if i not in head)             # body 也被训了
    assert len(c.step_vars()) == len(c.model.trainable_variables)
    # 对照：FedRep 的 body 步不动私有层
    r, _, _ = _client(T._cfg(), HierFedRepClient, ccs=False)
    w0 = r.model.get_weights()
    r._train_backbone_step(tf.constant(x[:T.BS]), tf.constant(y[:T.BS]))
    assert not any(_changed(w0, r.model.get_weights(), list(r._head_w_idx)).values())


# ── ② CCS 的训练步走 step_vars ───────────────────────────────────────────
def test_ccs_step_reaches_projection_under_fedper_but_not_fedrep():
    p, x, y = _client(_cfg_fedper(), HierFedPerClient)
    r, _, _ = _client(T._cfg(), HierFedRepClient)
    xb, yb = tf.constant(x[:T.BS]), tf.constant(y[:T.BS])
    for c in (p, r):
        c.on_round_start(1)
        w0 = c.model.get_weights()
        loss = c.on_body_step(xb, yb)
        assert loss is not None and np.isfinite(float(loss))
        c._w_change = _changed(w0, c.model.get_weights(), list(c._head_w_idx))
    assert all(p._w_change.values())                                  # FedPer：proj / head 收到 CE + KL + MMD 的梯度
    assert not any(r._w_change.values())                              # FedRep：照旧只到 body（D-109 的行为不变）
    assert r.step_vars() is r._base_tvars and r.step_optimizer() is r._base_opt


# ── ③ edge 轮：统计量共享、私有层私有 ─────────────────────────────────────
class _Grab:
    def observe(self, edge_id, er, updates, edge_w, base_idx):
        self.updates = [(list(u[0]), u[1]) for u in updates]


def test_edge_round_aggregates_bn_stats_and_keeps_private_layers():
    from server.hier_fedper import HierFedPerEdgeServer
    cfg = _cfg_fedper(ccs=False)
    g = T._small_proj_model()
    clients = []
    for cid in range(3):
        c, x, y = _client(cfg, HierFedPerClient, cid=cid, model=tf.keras.models.clone_model(g), ccs=False)
        c.model.set_weights(g.get_weights())
        c.set_train_source(x, y, np.arange(len(x)))
        c.assigned_edge = 0
        clients.append(c)
    em = tf.keras.models.clone_model(g)
    em.set_weights(g.get_weights())
    edge = HierFedPerEdgeServer(0, clients, em, cfg)
    edge.update_observer = grab = _Grab()
    with contextlib.redirect_stdout(io.StringIO()):
        edge.run_edge_round(1, 1)
    stat = get_bn_stat_indices(g)
    assert stat, "小模型要有 BN 统计量"
    n = np.array([w for _, w in grab.updates], np.float64)
    ew = em.get_weights()
    for i in stat:                                                    # 统计量 = 上传物的样本加权平均（随 body 聚合）
        want = sum(nk * u[i] for (u, _), nk in zip(grab.updates, n)) / n.sum()
        np.testing.assert_allclose(ew[i], want, rtol=1e-5, atol=1e-6)
    head = list(clients[0]._head_w_idx)
    g0 = g.get_weights()
    assert all(np.array_equal(ew[i], g0[i]) for i in head)            # edge 的私有层位置不参与聚合
    h = [c._head_weights for c in clients]
    assert not all(np.array_equal(h[0][k], h[1][k]) for k in range(len(head)))   # 各端私有层各自不同
    with contextlib.redirect_stdout(io.StringIO()):
        clients[0].set_weights(ew)
    cw = clients[0].model.get_weights()
    assert all(np.array_equal(cw[i], ew[i]) for i in stat)            # 下发后用聚合的统计量（不私有）
    assert all(np.array_equal(cw[i], clients[0]._head_weights[k]) for k, i in enumerate(head))


# ── ④ 每批都经过 on_batch ─────────────────────────────────────────────────
def test_every_training_batch_goes_through_on_batch():
    cfg = _cfg_fedper(ccs=False)
    cfg["training"]["local_epochs"] = 2
    c, x, y = _client(cfg, HierFedPerClient, ccs=False)
    c.set_train_source(x, y, np.arange(len(x)))
    seen = []
    orig = c.on_batch
    c.on_batch = lambda a, b: (seen.append(int(a.shape[0])), orig(a, b))[1]
    with contextlib.redirect_stdout(io.StringIO()):
        out = c.local_train(1)
    n_batches = len(list(c._epoch()))
    assert len(seen) == 2 * n_batches and n_batches > 0
    assert len(out[0]) == len(c.model.get_weights()) and np.isfinite(out[2])


# ── ⑤ 接线 ───────────────────────────────────────────────────────────────
def test_resolve_classes_for_hier_fedper():
    from client.ccs_mixin import CCSClientMixin
    from main import _select_method_classes, resolve_client_classes
    from server.hier_fedper import HierFedPerEdgeServer
    cfg = _cfg_fedper()
    ccls, ecls = _select_method_classes(cfg)
    assert ccls is HierFedPerClient and ecls is HierFedPerEdgeServer
    ben, _ = resolve_client_classes(copy.deepcopy(cfg))
    assert ben.__mro__[1] is CCSClientMixin and issubclass(ben, HierFedPerClient)
