"""
阶段三 SA-C（D-109）：CCS 的 TF 部分 —— `resnet10_torch_proj`、客户端 mixin（`client/ccs_mixin.py`）、edge 上的接线。需要 TF。

  ① resnet10_torch_proj：前面逐层与冻结的 resnet10_torch 同形、base / 统计量索引相同；proj_* 归进私有 head；
  ② 造对抗样本：不改任何权重（对抗前向是推理模式，BN 统计量不动）、不碰 Python random / 全局 np.random、
     结果在 [0,1]（工作空间）里、掩码外离原图 ≤ eps；同 (seed, client, round) 可复现；
     一步的结果 = numpy 参照 `ccs_spec.adv_step`（梯度符号由独立的磁带算）；
  ③ 训练步：kl = mmd = 0、clip 不起作用 → 与方法类自己的 body 步**逐元素相近**（同一个 CE 梯度）；私有 head / 投影不动；
  ④ 攻击端 / at=false 的臂：on_body_step 返回 None（走方法类原来的步）；
  ⑤ 接线：resolve_client_classes 把 CCS mixin 组合进良性与恶意两类；edge 跑一轮发 [CCS] 与 [Decision]。
"""

import contextlib
import copy
import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))

from client.compose import compose_client_class            # noqa: E402
from client.hier_fedrep import HierFedRepClient            # noqa: E402
from defense import ccs_spec as S                          # noqa: E402
from defense.ccs import CCSDefense                         # noqa: E402
from models.cnn import (TorchBatchNorm, build_resnet10_torch, build_resnet10_torch_proj,   # noqa: E402
                        get_base_head_indices, get_bn_stat_indices)

IMG, NCLS, BS = 16, 10, 8


def _cfg(**ccs):
    return {
        "seed": 42,
        "data": {"dataset": "cifar10", "img_size": IMG, "batch_size": BS, "num_classes": NCLS},
        "federation": {"n_clients": 4, "n_edges": 1, "client_fraction": 1.0, "edge_rounds": 1,
                       "n_rounds": 1, "n_workers": 1},
        "training": {"drift_correction": "hier_fedrep", "learning_rate": 0.1, "lr_decay": 1.0,
                     "local_epochs": 1, "plocal_epochs": 1, "head_lr_rep": 0.05,
                     "fedrep_bn_stats": "private", "fedrep_poison_phases": "body"},
        "model": {"arch": "resnet10_torch_proj"},
        "defense": {"name": "ccs", "layers": ["client", "edge"],
                    "ccs": {"radius": 4, "steps": 2, **ccs}},
    }


def _small_proj_model():
    """小模型：同样的层名约定（BN = TorchBatchNorm、proj_1 / proj_2 / head）。"""
    tf.keras.utils.set_random_seed(0)
    L = tf.keras.layers
    inp = tf.keras.Input((IMG, IMG, 3))
    x = L.Conv2D(4, 3, padding="same", use_bias=False, name="stem_conv")(inp)
    x = TorchBatchNorm(momentum=0.9, epsilon=1e-5, name="stem_bn")(x)
    x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dense(6, activation="relu", name="proj_1")(x)
    x = L.Dense(5, name="proj_2")(x)
    return tf.keras.Model(inp, L.Dense(NCLS, activation="softmax", name="head")(x))


def _data(seed=0, n=16):
    r = np.random.default_rng(seed)
    return (r.normal(size=(n, IMG, IMG, 3)).astype(np.float32), (np.arange(n) % NCLS).astype(np.int64))


def _client(cfg, cid=1, malicious=False, model=None):
    Cls = compose_client_class(HierFedRepClient, CCSDefense.client_mixin)
    x, y = _data(cid)
    m = model if model is not None else _small_proj_model()
    c = Cls(client_id=cid, dataset=tf.data.Dataset.from_tensor_slices((x, y)).batch(BS),
            model=m, config=cfg, n_samples=len(x))
    c.is_malicious = malicious
    with contextlib.redirect_stdout(io.StringIO()):
        c.set_weights(m.get_weights())
    return c, x, y


# ── ① 模型 ───────────────────────────────────────────────────────────────
def test_proj_model_keeps_the_frozen_resnet10_torch_and_privatises_the_projection():
    a, b = build_resnet10_torch(), build_resnet10_torch_proj()
    wa, wb = a.get_weights(), b.get_weights()
    assert [w.shape for w in wa[:-2]] == [w.shape for w in wb[:len(wa) - 2]]
    assert [w.shape for w in wb[len(wa) - 2:]] == [(512, 256), (256,), (256, 512), (512,), (512, 10), (10,)]
    sa, sb = get_base_head_indices(a, 10), get_base_head_indices(b, 10)
    assert sa["base_weight_indices"] == sb["base_weight_indices"]
    assert sb["head_weight_indices"] == list(range(len(wa) - 2, len(wb)))
    assert len(sb["head_trainable_indices"]) == 6
    assert get_bn_stat_indices(a) == get_bn_stat_indices(b)
    assert sa["head_weight_indices"] == [len(wa) - 2, len(wa) - 1]          # 没有 proj_* 的模型不受影响


# ── ② 对抗样本 ───────────────────────────────────────────────────────────
def test_adversarial_batch_leaves_weights_and_global_rngs_alone():
    c, x, y = _client(_cfg())
    c.on_round_start(3)
    w0 = c.model.get_weights()
    st, gs = random.getstate(), np.random.get_state()[1].copy()
    xa = c.ccs_adversarial(x[:BS], y[:BS]).numpy()
    assert all(np.array_equal(p, q) for p, q in zip(w0, c.model.get_weights()))
    assert random.getstate() == st and np.array_equal(np.random.get_state()[1], gs)
    assert xa.min() >= 0.0 and xa.max() <= 1.0                               # space=input：照官方 clamp[0,1]


def test_adversarial_batch_is_reproducible_per_seed_client_round():
    c1, x, y = _client(_cfg(), cid=1)
    c2, _, _ = _client(_cfg(), cid=1)
    c1.on_round_start(2)
    c2.on_round_start(2)
    assert np.array_equal(c1.ccs_adversarial(x[:BS], y[:BS]).numpy(), c2.ccs_adversarial(x[:BS], y[:BS]).numpy())
    c2.on_round_start(5)
    assert not np.array_equal(c1.ccs_adversarial(x[:BS], y[:BS]).numpy(),
                              c2.ccs_adversarial(x[:BS], y[:BS]).numpy())


@pytest.mark.parametrize("space", ["input", "pixel"])
def test_one_step_equals_the_numpy_reference(space):
    cfg = _cfg(steps=1, space=space)
    c, x, y = _client(cfg)
    c.on_round_start(1)
    p = S.ccs_params(cfg)
    rng = np.random.default_rng([42, 1, S.CCS_RNG_TAG, 1])                  # 与 mixin 同一条流
    mask = S.draw_masks(rng, BS, IMG, IMG, p["radius"], p["n_regions"])
    noise = rng.uniform(0.0, 1.0, size=(BS, IMG, IMG, 3)).astype(np.float32)
    from data.pixel_space import pixel_stats
    a, b = S.space_affine(space, *pixel_stats(cfg))
    u = a * x[:BS] + b
    u0 = S.adv_init(u, mask, noise).astype(np.float32)
    t = tf.convert_to_tensor(u0)
    head = c.model.get_layer("head")
    feat = tf.keras.Model(c.model.inputs, head.input)
    with tf.GradientTape() as tape:
        tape.watch(t)
        z = tf.matmul(feat((t - b) / a, training=False), head.kernel) + head.bias
        zy = tf.gather(z, y[:BS], batch_dims=1)
        other = tf.reduce_max(z - tf.one_hot(y[:BS], NCLS) * 1e30, axis=1)
        loss = tf.reduce_mean(other - zy)
    g = tape.gradient(loss, t).numpy()
    want = (S.adv_step(u0, u0, np.sign(g), p["step"], p["eps"]) - b) / a
    got = c.ccs_adversarial(x[:BS], y[:BS]).numpy()
    assert np.allclose(got, want, atol=1e-5)
    # 掩码外：离 clamp 过的原图 ≤ eps（工作空间）。离**原图**可以远得多 —— space=input 下 clamp[0,1] 把标准化后
    # 不在 [0,1] 里的像素（比均值暗的、比 μ+σ 亮的）整个夹过去，这正是官方代码的行为（F-101 ②）。
    off = np.broadcast_to(mask < 1e-6, u.shape)
    du = np.abs(a * got + b - np.clip(u, 0.0, 1.0))[off]
    assert du.max() <= p["eps"] + 1e-5


# ── ③ 训练步 ─────────────────────────────────────────────────────────────
def test_zero_coefficients_reduce_to_the_methods_own_body_step():
    cfg = _cfg(kl=0.0, mmd=0.0, clip_norm=1e9)
    c1, x, y = _client(cfg)
    c2, _, _ = _client(cfg)                                                    # 同初值的第二个副本
    head = set(c1._head_w_idx)
    for c in (c1, c2):
        c.on_round_start(1)
    w0 = c1.model.get_weights()
    xb, yb = tf.constant(x[:BS]), tf.constant(y[:BS])
    ce = c1.on_body_step(xb, yb)
    c2._train_backbone_step(xb, yb)
    w1, w2 = c1.model.get_weights(), c2.model.get_weights()
    assert all(np.allclose(p, q, atol=1e-6) for p, q in zip(w1, w2))
    assert all(np.array_equal(w1[i], w0[i]) for i in head)                    # 私有 head / 投影不动
    assert np.isfinite(float(ce))


def test_full_step_changes_body_only_and_differs_from_plain_ce():
    c1, x, y = _client(_cfg())
    c2, _, _ = _client(_cfg(kl=0.0, mmd=0.0, clip_norm=1e9))
    for c in (c1, c2):
        c.on_round_start(1)
    w0 = c1.model.get_weights()
    xb, yb = tf.constant(x[:BS]), tf.constant(y[:BS])
    c1.on_body_step(xb, yb)
    c2.on_body_step(xb, yb)
    w1, w2 = c1.model.get_weights(), c2.model.get_weights()
    assert all(np.array_equal(w1[i], w0[i]) for i in c1._head_w_idx)
    assert any(not np.allclose(p, q) for p, q in zip(w1, w2))                 # KL / MMD 真的进了梯度


# ── ④ 不做的端 ─────────────────────────────────────────────────────────────
def test_attacker_and_cluster_only_arm_fall_through():
    c, x, y = _client(_cfg(), malicious=True)
    c.on_round_start(1)
    assert c.on_body_step(tf.constant(x[:BS]), tf.constant(y[:BS])) is None
    assert getattr(c, "_ccs_attack", None) is None                           # 攻击端不建辅助模型
    c, x, y = _client(_cfg(at=False))
    c.on_round_start(1)
    assert c.on_body_step(tf.constant(x[:BS]), tf.constant(y[:BS])) is None


# ── ⑤ 接线 ───────────────────────────────────────────────────────────────
def test_resolve_client_classes_composes_ccs_into_both_kinds():
    from client.ccs_mixin import CCSClientMixin
    from client.client_badpfl import BadPFLMixin
    from main import resolve_client_classes
    cfg = _cfg()
    cfg["backdoor"] = {"enabled": True, "malicious_strategy": "badpfl"}
    ben, mal = resolve_client_classes(cfg)
    assert ben.__mro__[1] is CCSClientMixin and issubclass(ben, HierFedRepClient)
    assert mal.__mro__[1] is BadPFLMixin and mal.__mro__[2] is CCSClientMixin


def test_edge_round_emits_ccs_and_decision_lines():
    from data.partition import merge_test_datasets
    from server.hier_fedrep import HierFedRepEdgeServer
    cfg = _cfg()
    g = _small_proj_model()
    clients = []
    for cid in range(4):
        c, _, _ = _client(cfg, cid=cid, model=tf.keras.models.clone_model(g))
        c.model.set_weights(g.get_weights())
        x, y = _data(cid)
        c.set_train_source(x, y, np.arange(len(x)))
        c.set_test_dataset(tf.data.Dataset.from_tensor_slices((x, y)).batch(BS))
        c.assigned_edge = 0
        clients.append(c)
    edge_model = tf.keras.models.clone_model(g)
    edge_model.set_weights(g.get_weights())
    edge = HierFedRepEdgeServer(0, clients, edge_model, cfg)
    assert isinstance(edge.defense, CCSDefense)
    edge.defense.set_stat_indices(get_bn_stat_indices(g), label=0)
    edge.set_test_dataset(merge_test_datasets(clients, BS))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        edge.run_edge_round(1, 1)
    out = buf.getvalue()
    ccs = [ln for ln in out.splitlines() if "[CCS]" in ln]
    dec = [ln for ln in out.splitlines() if "[Decision]" in ln]
    assert len(ccs) == 1 and "edge0" in ccs[0] and "cur=0/1/2/3" in ccs[0]
    assert len(dec) == 1 and "| ccs |" in dec[0]


# ── 离线空间探针（analysis/ccs_space_probe.py）的读数 ─────────────────────────
def test_space_probe_variant_stats_on_a_small_model():
    from analysis.ccs_space_probe import variant_stats
    from client.ccs_mixin import CCSAttack
    from data.pixel_space import pixel_stats
    cfg = _cfg()
    m = _small_proj_model()
    x, y = _data(3, n=BS)
    mean, std = (v.reshape(1, 1, 1, -1) for v in pixel_stats(cfg))
    x = ((np.clip(x * 0.25 + 0.5, 0, 1) - mean) / std).astype(np.float32)   # 像素 ∈ [0,1] 的标准化图
    p = S.ccs_params(cfg)
    out = {}
    for sp in ("input", "pixel"):
        atk = CCSAttack(m, {**p, "space": sp}, *pixel_stats(cfg))
        out[sp] = variant_stats(m, atk, np.random.default_rng(0), x, y, mean, std)
    assert out["pixel"]["clamp_frac"] == 0.0 and out["input"]["clamp_frac"] > 0.3     # 标准化后约一半像素 < 0
    for r in out.values():
        assert 0.0 <= r["clean_acc"] <= 1.0 and r["kl"] >= -1e-6 and r["dpx_out_mask"] >= 0.0
    # pixel 解释下 eps = 0.5 像素：掩码外最多能挪 0.5；input 解释下 eps = 0.5·σ ≈ 0.12 像素（外加 clamp 的位移）
    assert out["pixel"]["dpx_linf"] <= 1.0 + 1e-6
