"""
tests/test_fedrep_resnet_alignment.py  —  A4：ResNet-10 与 FedRep 客户端（AUDIT A24 A25 A26 A27 A29）

需要 TF，本地无 TF 时 skip。每条的判据来自 AUDIT 该行的「怎么验证」列。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_badpfl import BadPFLMixin            # noqa: E402
from client.client_fedavg import FedAvgClient           # noqa: E402
from client.compose import compose_client_class         # noqa: E402
from client.hier_fedrep import HierFedRepClient         # noqa: E402
from models.cnn import (build_model, build_resnet10,     # noqa: E402
                        build_resnet10_torch, get_base_head_indices,
                        get_bn_stat_indices)

NCLS, IMG = 10, 16          # 16：Bad-PFL 生成器要求 img_size 是 16 的倍数


# ══════════════════════════════════════════════════════════════════════════
# A29：resnet10_torch
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def rt():
    return build_resnet10_torch((32, 32, 3), NCLS)


@pytest.fixture(scope="module")
def old():
    return build_resnet10((32, 32, 3), NCLS)


def _count(vs):
    return sum(int(np.prod(v.shape)) for v in vs)


def test_param_counts_match_torch(rt):
    assert _count(rt.trainable_variables) == 4_903_242
    assert _count(rt.non_trainable_variables) == 5_760


def test_registered_in_build_model():
    m = build_model((32, 32, 3), NCLS, arch="resnet10_torch")
    assert m.name == "resnet10_torch"


def _impulse_row(pad_layer, conv, in_row):
    """中心抽头的 3×3 卷积把输入第 in_row 行的冲激送到输出哪一行（没送到 → None）。"""
    imp = np.zeros((1, 32, 32, 1), np.float32)
    imp[0, in_row, in_row, 0] = 1.0
    kc = np.zeros((3, 3, 1, 1), np.float32)
    kc[1, 1, 0, 0] = 1.0
    x = pad_layer(imp) if pad_layer is not None else imp
    y = tf.nn.conv2d(x, kc, conv.strides[0], conv.padding.upper()).numpy()[0, :, :, 0]
    hit = np.argwhere(y)
    return tuple(hit[0]) if len(hit) else None


@pytest.mark.parametrize("stage", ["stage2", "stage3", "stage4"])
def test_stride2_conv_samples_even_rows_like_shortcut(rt, old, stage):
    """torch padding=1：输出 (5,5) 取输入 (10,10)，与 1×1/s2 shortcut 同一位置。"""
    pad = rt.get_layer(f"{stage}_pad1")
    conv = rt.get_layer(f"{stage}_conv1")
    assert conv.padding == "valid" and pad.padding == ((1, 1), (1, 1))
    assert _impulse_row(pad, conv, 10) == (5, 5)
    sc = rt.get_layer(f"{stage}_sc_conv")
    one = np.zeros((1, 1, 1, 1), np.float32)
    one[0, 0, 0, 0] = 1.0
    imp = np.zeros((1, 32, 32, 1), np.float32)
    imp[0, 10, 10, 0] = 1.0
    y = tf.nn.conv2d(imp, one, sc.strides[0], sc.padding.upper()).numpy()[0, :, :, 0]
    assert tuple(np.argwhere(y)[0]) == (5, 5)
    # 反向锚点：旧 resnet10 的 "same" 看不到第 10 行，看到的是第 11 行（错位 1 像素）
    oc = old.get_layer(f"{stage}_conv1")
    assert oc.padding == "same"
    assert _impulse_row(None, oc, 10) is None and _impulse_row(None, oc, 11) == (5, 5)


def test_bn_momentum_and_eps(rt):
    bns = [l for l in rt.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert len(bns) == 12
    assert all(l.momentum == 0.9 and l.epsilon == 1e-5 for l in bns)


def test_kernel_init_is_torch_default(rt):
    for name in ("stage4_conv1", "stage4_conv2", "stage3_conv2"):
        k = rt.get_layer(name).get_weights()[0]
        fan_in = k.shape[0] * k.shape[1] * k.shape[2]
        assert abs(k.var() * 3 * fan_in - 1) < 0.05, (name, k.var() * 3 * fan_in)
    kh, bh = rt.get_layer("head").get_weights()
    assert abs(kh.var() * 3 * 512 - 1) < 0.10
    assert np.all(np.abs(bh) <= 1 / np.sqrt(512) + 1e-7) and np.any(bh != 0)


def test_frozen_resnet10_is_untouched(old):
    bns = [l for l in old.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert all(l.momentum == 0.99 and l.epsilon == 1e-3 for l in bns)
    assert isinstance(old.get_layer("stage2_conv1").kernel_initializer,
                      tf.keras.initializers.GlorotUniform)
    assert _count(old.trainable_variables) == 4_903_242
    assert not any(l.name.endswith("_pad1") for l in old.layers)


def test_fedrep_split_still_works_on_resnet10_torch(rt):
    split = get_base_head_indices(rt, NCLS)
    w = rt.get_weights()
    assert [w[i].shape for i in split["head_weight_indices"]] == [(512, NCLS), (NCLS,)]
    assert len(get_bn_stat_indices(rt)) == 24


# ══════════════════════════════════════════════════════════════════════════
# FedRep 客户端：小模型（Conv + BN + Dense 头）
# ══════════════════════════════════════════════════════════════════════════
def _small_model(seed=0):
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input((IMG, IMG, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(NCLS, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def _cfg(**training):
    return {
        "seed": 42,
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1,
                     "plocal_epochs": 1, "head_lr_rep": 0.05, **training},
        "data": {"batch_size": 8, "img_size": IMG, "dataset": "cifar10",
                 "num_classes": NCLS},
        "backdoor": {"target_label": 0, "poison_ratio": 1.0, "badpfl_gen_steps": 1},
    }


def _data(n=24, seed=1):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, IMG, IMG, 3)).astype(np.float32)
    y = (np.arange(n) % NCLS).astype(np.int64)
    return x, y


def _fedrep(cfg, cls=HierFedRepClient, cid=0, n=24):
    x, y = _data(n)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(cfg["data"]["batch_size"])
    c = cls(client_id=cid, dataset=ds, model=_small_model(cid), config=cfg, n_samples=n)
    c.set_train_source(x, y, np.arange(n))
    return c


def _edge_weights(model, stat_val, gb_val=None):
    w = [v.copy() for v in model.get_weights()]
    for i in get_bn_stat_indices(model):
        w[i] = np.full_like(w[i], stat_val)
    if gb_val is not None:
        tv = {id(v) for v in model.trainable_variables}
        for i, v in enumerate(model.weights):
            if "batch_normalization" in v.name and id(v) in tv:
                w[i] = np.full_like(w[i], gb_val)
    return w


def _stats(c):
    w = c.model.get_weights()
    return [w[i] for i in get_bn_stat_indices(c.model)]


def _gamma_beta(c):
    tv = {id(v) for v in c.model.trainable_variables}
    return [v.numpy() for v in c.model.weights
            if "batch_normalization" in v.name and id(v) in tv]


def test_private_stats_first_receipt_adopts_broadcast():
    c = _fedrep(_cfg(fedrep_bn_stats="private"))
    ew = _edge_weights(c.model, 7.0)
    c.set_weights(ew, ew)
    assert all(np.all(s == 7.0) for s in _stats(c))


def test_private_stats_survive_later_broadcasts_but_gamma_beta_follow_edge():
    c = _fedrep(_cfg(fedrep_bn_stats="private"))
    ew = _edge_weights(c.model, 7.0)
    c.set_weights(ew, ew)
    c._stats = [np.full_like(s, 1.5) for s in c._stats]           # 哨兵：客户端自己的统计量
    h = [np.full_like(w, 0.25) for w in c._head_weights]
    c._head_weights = h
    ew2 = _edge_weights(c.model, 9.0, gb_val=3.0)
    c.set_weights(ew2, ew2)
    assert all(np.all(s == 1.5) for s in _stats(c))                # 统计量 = 自己的
    assert all(np.all(g == 3.0) for g in _gamma_beta(c))           # γ/β = edge 的
    w = c.model.get_weights()
    for k, i in enumerate(c._head_w_idx):                          # head = 自己的
        np.testing.assert_array_equal(w[i], h[k])


def test_shared_stats_are_overwritten_reverse_anchor():
    c = _fedrep(_cfg())
    ew = _edge_weights(c.model, 7.0)
    c.set_weights(ew, ew)
    c.local_train(1)
    ew2 = _edge_weights(c.model, 9.0)
    c.set_weights(ew2, ew2)
    assert all(np.all(s == 9.0) for s in _stats(c))


def test_pm_stats_are_the_head_phase_snapshot():
    """A27：陈旧 PM 的统计量 = head 阶段末的快照（body 阶段还会改它，上传带的是后者）。"""
    c = _fedrep(_cfg(fedrep_bn_stats="private"))
    ew = _edge_weights(c.model, 0.0)
    c.set_weights(ew, ew)
    snap = {}
    orig = c._body_phase

    def spy(epochs, losses):
        snap["stats"] = [s.copy() for s in _stats(c)]
        return orig(epochs, losses)
    c._body_phase = spy
    upload, *_ = c.local_train(1)
    for a, b in zip(_stats(c), snap["stats"]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(c._stats, snap["stats"]):
        np.testing.assert_array_equal(a, b)
    up_stats = [upload[i] for i in get_bn_stat_indices(c.model)]
    assert any(not np.array_equal(a, b) for a, b in zip(up_stats, snap["stats"]))


def test_private_state_exposes_head_and_stats():
    c = _fedrep(_cfg(fedrep_bn_stats="private"))
    assert c.private_state() == ([], [])                            # 还没参加过训练
    ew = _edge_weights(c.model, 0.0)
    c.set_weights(ew, ew)
    c.local_train(1)
    idx, val = c.private_state()
    assert idx == list(c._head_w_idx) + list(get_bn_stat_indices(c.model))
    assert len(val) == len(idx)
    shared = _fedrep(_cfg(), cid=1)
    shared.set_weights(ew, ew)
    shared.local_train(1)
    assert shared.private_state()[0] == list(shared._head_w_idx)


# ══════════════════════════════════════════════════════════════════════════
# A24 / A26：哪个阶段投毒、两个阶段的先后
# ══════════════════════════════════════════════════════════════════════════
def _event_log(c):
    log = []
    oh, ob, obt = c._train_head_step, c._train_backbone_step, c.on_batch

    def head(x, y):
        log.append("H")
        return oh(x, y)

    def body(x, y):
        log.append("B")
        return ob(x, y)

    def on_batch(x, y):
        log.append("p")
        return obt(x, y)
    c._train_head_step, c._train_backbone_step, c.on_batch = head, body, on_batch
    return log


MalFedRep = compose_client_class(HierFedRepClient, None, BadPFLMixin)


def _malicious(cfg):
    c = _fedrep(cfg, cls=MalFedRep)
    c.is_malicious = True
    ew = _edge_weights(c.model, 0.0)
    c.set_weights(ew, ew)
    return c


def _poisoned_before(log, step):
    """每个 step 事件之前紧挨着的是不是 on_batch（p）。"""
    return [log[i - 1] == "p" for i, e in enumerate(log) if e == step]


def test_body_only_poisoning_head_phase_sees_clean_batches():
    c = _malicious(_cfg(fedrep_poison_phases="body"))
    log = _event_log(c)
    c.local_train(1)
    assert log.count("H") == 3 and log.count("B") == 3
    assert not any(_poisoned_before(log, "H")) and all(_poisoned_before(log, "B"))


def test_both_phases_poisoned_reverse_anchor():
    c = _malicious(_cfg())
    log = _event_log(c)
    c.local_train(1)
    assert all(_poisoned_before(log, "H")) and all(_poisoned_before(log, "B"))


@pytest.mark.parametrize("order,first", [("head_first", "H"), ("body_first", "B")])
def test_training_order(order, first):
    c = _malicious(_cfg(fedrep_order=order, fedrep_poison_phases="body"))
    ev = _event_log(c)
    c.local_train(1)
    steps = [e for e in ev if e in ("H", "B")]
    k = steps.index("B" if first == "H" else "H")
    assert set(steps[:k]) == {first} and first not in steps[k:]
    assert not any(_poisoned_before(ev, "H"))                       # 两种顺序都只在 body 投毒


def test_body_first_upload_is_taken_before_head_phase():
    """body_first：上传 = body 阶段末；之后 head 阶段只动 head 与统计量 → 陈旧 PM = [w_k, h, s]。"""
    c = _malicious(_cfg(fedrep_order="body_first", fedrep_bn_stats="private"))
    upload, *_ = c.local_train(1)
    w = c.model.get_weights()
    stat = set(get_bn_stat_indices(c.model))
    head = set(c._head_w_idx)
    for i, (a, b) in enumerate(zip(upload, w)):
        if i not in stat and i not in head:
            np.testing.assert_array_equal(a, b)                     # body（w_k）一致
    assert any(not np.array_equal(upload[i], w[i]) for i in head)   # head 在之后才训


# ══════════════════════════════════════════════════════════════════════════
# A25：per_epoch 下两种客户端都走同一个取数函数、drop_last
# ══════════════════════════════════════════════════════════════════════════
def test_per_epoch_fedrep_uses_full_batches_only():
    c = _fedrep(_cfg(), n=27)
    c.config["data"]["batch_pipeline"] = "per_epoch"
    c._batch_list = []
    ew = _edge_weights(c.model, 0.0)
    c.set_weights(ew, ew)
    log = _event_log(c)
    c.local_train(1)
    assert log.count("H") == 27 // 8 and log.count("B") == 27 // 8


def test_per_epoch_fedavg_drops_the_tail_batch():
    cfg = _cfg()
    cfg["data"]["batch_pipeline"] = "per_epoch"
    x, y = _data(27)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
    c = FedAvgClient(client_id=0, dataset=ds, model=_small_model(), config=cfg, n_samples=27)
    c.set_train_source(x, y, np.arange(27))
    sizes = []
    orig = c._train_step
    c._train_step = lambda a, b: (sizes.append(int(a.shape[0])), orig(a, b))[1]
    c.local_train(1)
    assert sizes == [8, 8, 8]
    legacy = FedAvgClient(client_id=0, dataset=ds, model=_small_model(), config=_cfg(),
                          n_samples=27)
    sizes.clear()
    orig2 = legacy._train_step
    legacy._train_step = lambda a, b: (sizes.append(int(a.shape[0])), orig2(a, b))[1]
    legacy.local_train(1)
    assert sizes == [8, 8, 8, 3]                                    # 反向锚点：旧行为训尾批
