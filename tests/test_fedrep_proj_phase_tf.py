"""
FedRep 下私有投影网络在哪一段训练（开关 training.fedrep_proj_phase，D-116）。需要 TF。

  ① 缺省 head：训练主干那一段的变量就是原来的 body 变量列表（同一个对象）→ 与改动前逐位相同；投影网络在这一段不动；
  ② both：训练主干那一段，主干与投影网络都变、分类头不变；CCS 的对抗训练步同样训练到投影网络、碰不到分类头；
  ③ both 的一整轮本地训练：私有部分的快照取主干那一段结束后的投影网络（不会被下一轮的 set_weights 冲掉）。
小模型与夹具沿用 test_ccs_tf.py（层名约定 proj_1 / proj_2 / head）。
"""

import contextlib
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

import test_ccs_tf as T                                     # noqa: E402
from client.compose import compose_client_class             # noqa: E402
from client.hier_fedrep import HierFedRepClient             # noqa: E402
from defense.ccs import CCSDefense                          # noqa: E402


def _cfg(phase=None, ccs=True):
    cfg = T._cfg()
    if phase is not None:
        cfg["training"]["fedrep_proj_phase"] = phase
    if not ccs:
        cfg["defense"] = {"name": "none"}
    return cfg


def _client(cfg, ccs=True, cid=1):
    Cls = compose_client_class(HierFedRepClient, CCSDefense.client_mixin) if ccs else HierFedRepClient
    x, y = T._data(cid)
    m = T._small_proj_model()
    c = Cls(client_id=cid, dataset=tf.data.Dataset.from_tensor_slices((x, y)).batch(T.BS),
            model=m, config=cfg, n_samples=len(x))
    c.is_malicious = False
    with contextlib.redirect_stdout(io.StringIO()):
        c.set_weights(m.get_weights())
    return c, x, y


def _index_groups(c):
    """get_weights 索引：分类头（最后的 Dense）、投影网络（proj_*）、主干（其余可训练）。"""
    names = [v.name for v in c.model.weights]
    tv = {id(v) for v in c.model.trainable_variables}
    head = [i for i, n in enumerate(names) if n.startswith("head/")]
    proj = [i for i, n in enumerate(names) if n.startswith("proj_")]
    body = [i for i, v in enumerate(c.model.weights) if id(v) in tv and i not in head and i not in proj]
    assert head and proj and body
    return head, proj, body


def _changed(w0, w1, idx):
    return [not np.array_equal(w0[i], w1[i]) for i in idx]


def test_default_is_the_old_body_variable_list():
    for phase in (None, "head"):
        c, x, y = _client(_cfg(phase, ccs=False), ccs=False)
        assert c._body_tvars is c._base_tvars and c.step_vars() is c._base_tvars
        head, proj, body = _index_groups(c)
        w0 = c.model.get_weights()
        c._train_backbone_step(tf.constant(x[:T.BS]), tf.constant(y[:T.BS]))
        w1 = c.model.get_weights()
        assert not any(_changed(w0, w1, proj + head)) and any(_changed(w0, w1, body))


def test_both_trains_projection_with_the_body_but_not_the_classifier():
    c, x, y = _client(_cfg("both", ccs=False), ccs=False)
    head, proj, body = _index_groups(c)
    assert len(c.step_vars()) == len(c._base_tvars) + len(proj)
    w0 = c.model.get_weights()
    c._train_backbone_step(tf.constant(x[:T.BS]), tf.constant(y[:T.BS]))
    w1 = c.model.get_weights()
    assert all(_changed(w0, w1, proj)) and any(_changed(w0, w1, body))
    assert not any(_changed(w0, w1, head))


def test_ccs_step_reaches_projection_only_when_both():
    for phase, expect in (("head", False), ("both", True)):
        c, x, y = _client(_cfg(phase))
        head, proj, _ = _index_groups(c)
        c.on_round_start(1)
        w0 = c.model.get_weights()
        loss = c.on_body_step(tf.constant(x[:T.BS]), tf.constant(y[:T.BS]))
        assert np.isfinite(float(loss))
        w1 = c.model.get_weights()
        assert all(_changed(w0, w1, proj)) is expect
        assert not any(_changed(w0, w1, head))


def test_both_keeps_the_body_phase_projection_in_the_private_snapshot():
    c, x, y = _client(_cfg("both", ccs=False), ccs=False)
    c.set_train_source(x, y, np.arange(len(x)))
    head, proj, _ = _index_groups(c)
    with contextlib.redirect_stdout(io.StringIO()):
        c.local_train(1)
    # 本轮结束时私有快照里的投影网络 = 模型里当前的投影网络（主干那一段训过之后的值）
    snap = dict(zip(c._head_w_idx, c._head_weights))
    cur = c.model.get_weights()
    assert all(np.array_equal(snap[i], cur[i]) for i in proj)
    # 下一轮收到 edge 权重后，私有投影网络是这份快照（没被 edge 下发的值冲掉）
    edge_w = [np.zeros_like(w) for w in cur]
    with contextlib.redirect_stdout(io.StringIO()):
        c.set_weights(edge_w)
    w2 = c.model.get_weights()
    assert all(np.array_equal(w2[i], snap[i]) for i in proj + head)


def test_validate_rejects_both_outside_fedrep_with_projection():
    from config_validate import ConfigError, validate_config
    import yaml
    cfg = yaml.safe_load((ROOT / "experiments/defense/edge-native/configs/CCSR30__fedrep-projboth-ccs__s42.yaml")
                         .read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        validate_config(cfg)
    bad = dict(cfg, model=dict(cfg["model"], arch="resnet10_torch"))
    bad["defense"] = {"name": "none"}
    with pytest.raises((ConfigError, SystemExit)), contextlib.redirect_stdout(io.StringIO()):
        validate_config(bad)
