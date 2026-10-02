"""
tests/test_ck_snapshot_tf.py  —  S6a：c_k 预检的 TF 部分（D-085）

小模型（conv → GAP → Dense(4) 特征 → softmax 头），断言：
  · ε = 0：PGD 什么也没动 → reached == (NCM 判成 k) → c_k 与 numpy 直接算的逐位相同；
  · ε 很大：c_k 的均值低于 ε = 0（PGD 真的在推；不逐类断言单调，非凸模型上不是不变量）；
  · 确定性，不碰全局 RNG（没有随机起点）；没有原型的类永远到达不了；
  · 快照往返：load_snapshot 读出的 edge 权重逐位等于写入的。
"""

import json
import random

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="需要 TF；本地无 TF 时在集群跑")

from analysis import ck_snapshot as CK                       # noqa: E402
from analysis import functional_score as FS                  # noqa: E402

K, IMG = 5, 8


def _model():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((IMG, IMG, 3))
    x = tf.keras.layers.Conv2D(6, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(K, activation="softmax")(x))


def _data(n, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, IMG, IMG, 3)).astype(np.float32)
    y = (np.arange(n) % K).astype(np.int64)
    return x, y


LO, HI = np.full(3, -50.0, np.float32), np.full(3, 50.0, np.float32)


def test_eps_zero_equals_the_plain_ncm_prediction():
    m = _model()
    feat = CK.feature_model(m)
    xp, yp = _data(40, 1)
    xa, ya = _data(30, 2)
    protos, _ = FS.class_prototypes(CK.extract(feat, xp), yp, K)
    pred = FS.ncm_predict(CK.extract(feat, xa), protos)
    reached = CK.ck_reached(feat, protos, xa, np.zeros(3, np.float32), 3, LO, HI, K)
    np.testing.assert_array_equal(reached, np.eye(K, dtype=bool)[pred])
    res = CK.edge_ck(m, m.get_weights(), xp, yp, xa, ya, np.zeros(3, np.float32), 3, LO, HI, K)
    want, _ = FS.ck_from_reached(ya, np.eye(K, dtype=bool)[pred])
    assert res["c"] == want and res["ncm_acc"] == pytest.approx(float(np.mean(pred == ya)))
    json.dumps(res)                                                    # 可 JSON 化


def test_large_eps_makes_classes_easier_to_reach_on_average():
    m = _model()
    xp, yp = _data(40, 1)
    xa, ya = _data(30, 2)
    c0 = CK.edge_ck(m, m.get_weights(), xp, yp, xa, ya, np.zeros(3, np.float32), 5, LO, HI, K)["c"]
    c1 = CK.edge_ck(m, m.get_weights(), xp, yp, xa, ya, np.full(3, 3.0, np.float32), 5, LO, HI, K)["c"]
    # 不逐类断言单调：非凸模型 + 有限步长，PGD 可能在个别 (类, 样本) 上冲过头；平均而言必须更容易到达
    assert np.mean(c1) < np.mean(c0)


def test_deterministic_and_does_not_touch_global_rngs():
    m = _model()
    xp, yp = _data(40, 1)
    xa, ya = _data(20, 2)
    np.random.seed(1)
    random.seed(2)
    st_np, st_py = np.random.get_state()[1].copy(), random.getstate()
    a = CK.edge_ck(m, m.get_weights(), xp, yp, xa, ya, np.full(3, 0.5, np.float32), 4, LO, HI, K)
    b = CK.edge_ck(m, m.get_weights(), xp, yp, xa, ya, np.full(3, 0.5, np.float32), 4, LO, HI, K)
    assert a == b
    np.testing.assert_array_equal(np.random.get_state()[1], st_np)
    assert random.getstate() == st_py


def test_class_without_prototype_is_never_reached():
    m = _model()
    feat = CK.feature_model(m)
    xp, yp = _data(40, 1)
    keep = yp != 3                                                     # 原型里没有类 3
    xa, ya = _data(20, 2)
    protos, _ = FS.class_prototypes(CK.extract(feat, xp[keep]), yp[keep], K)
    reached = CK.ck_reached(feat, protos, xa, np.full(3, 5.0, np.float32), 5, LO, HI, K)
    assert not reached[:, 3].any()


def test_snapshot_round_trip(tmp_path):
    m = _model()
    ws = m.get_weights()
    arrays = {f"edge0_{i:03d}": w for i, w in enumerate(ws)}
    arrays.update({f"edge1_{i:03d}": w + 1 for i, w in enumerate(ws)})
    arrays["meta_json"] = np.array(json.dumps({"malicious_ids": [0], "client_ids": [0, 1],
                                               "client_edge": [0, 1], "eval_attacker": 0}))
    np.savez(tmp_path / "s.npz", **arrays)
    meta, edges = CK.load_snapshot(tmp_path / "s.npz", len(ws), [0, 1])
    assert meta["eval_attacker"] == 0
    for a, b in zip(edges[1], ws):
        np.testing.assert_array_equal(a, b + 1)
