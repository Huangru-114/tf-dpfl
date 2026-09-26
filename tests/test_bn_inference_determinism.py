"""
tests/test_bn_inference_determinism.py  —  resnet10_torch 的 BN 在推理模式下不走 fused 核（D-043）

第一轮 pilot（`e2ee9ca`）6 个 run 全部死于 GPU 上的
    UnimplementedError: A deterministic GPU implementation of fused batch-norm backprop,
    when training is disabled, is not currently available. [Op:FusedBatchNormGradV3]
即 `enable_op_determinism()`（A15）× 在推理模式的 F 上求梯度（Bad-PFL 的 PGD ξ、生成器训练、
评估侧 ξ）。CPU 上没有这个检查，所以这里不测「会不会抛」，而是测**触发条件本身**：
图里有没有 `is_training=False` 的 fused BN 算子（FINDINGS F-043）。

反向锚点：冻结的 resnet10（原生 Keras BN）在同一个探针下**有** FusedBatchNormGradV3
(is_training=False) —— 证明这个探针抓得住回归。

需要 TF，本地无 TF 时 skip。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from models.cnn import (TorchBatchNorm, build_resnet10,      # noqa: E402
                        build_resnet10_torch, get_base_head_indices,
                        get_bn_stat_indices)
from models.model_utils import clone_model                   # noqa: E402

NCLS = 10


@pytest.fixture(scope="module")
def rt():
    return build_resnet10_torch((32, 32, 3), NCLS)


@pytest.fixture(scope="module")
def old():
    return build_resnet10((32, 32, 3), NCLS)


def _fused_bn_ops(model, training):
    """对输入求梯度（= PGD ξ 的形状）的图里，全部 fused BN 算子 → [(type, is_training)]。"""

    @tf.autograph.experimental.do_not_convert
    def probe(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, model(x, training=training))
        return tape.gradient(loss, x)

    g = tf.function(probe).get_concrete_function(
        tf.TensorSpec((2, 32, 32, 3), tf.float32), tf.TensorSpec((2,), tf.int64)).graph
    return [(op.type, op.get_attr("is_training")) for op in g.get_operations()
            if op.type.startswith("FusedBatchNorm")]


def test_inference_gradient_has_no_fused_bn(rt):
    assert _fused_bn_ops(rt, training=False) == []


def test_reverse_anchor_native_bn_has_the_gpu_unsupported_op(old):
    ops = _fused_bn_ops(old, training=False)
    assert ("FusedBatchNormGradV3", False) in ops


def test_training_path_is_still_fused_with_is_training_true(rt):
    """训练模式交给父类：fused、is_training=True（GPU 上有确定性实现，良性端第 1 轮已跑通）。"""
    ops = _fused_bn_ops(rt, training=True)
    assert ("FusedBatchNormGradV3", True) in ops
    assert all(is_tr for _, is_tr in ops)


def test_clone_model_keeps_the_subclass(rt):
    """客户端 / edge / PM 草稿模型都是 clone_model 出来的 —— 退回原生 BN 就又会崩。"""
    c = clone_model(rt)
    bns = [l for l in c.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert len(bns) == 12 and all(type(l) is TorchBatchNorm for l in bns)
    assert _fused_bn_ops(c, training=False) == []


def _pair(seed=0, c=8):
    """同权重的 TorchBatchNorm 与原生 BN；moving 统计量取非平凡值。"""
    rng = np.random.default_rng(seed)
    kw = dict(momentum=0.9, epsilon=1e-5)
    a, b = TorchBatchNorm(**kw), tf.keras.layers.BatchNormalization(**kw)
    a.build((None, 4, 4, c))
    b.build((None, 4, 4, c))
    w = [rng.normal(1, .2, c), rng.normal(0, .2, c), rng.normal(0, .5, c),
         rng.uniform(.2, 2., c)]
    w = [v.astype(np.float32) for v in w]
    a.set_weights(w)
    b.set_weights(w)
    x = rng.normal(0, 1, (3, 4, 4, c)).astype(np.float32)
    return a, b, x


def test_training_output_and_stat_update_are_bitwise_the_parent():
    a, b, x = _pair()
    np.testing.assert_array_equal(a(x, training=True).numpy(), b(x, training=True).numpy())
    for va, vb in zip(a.weights, b.weights):
        np.testing.assert_array_equal(va.numpy(), vb.numpy())


def test_inference_output_matches_fused_within_float_rounding():
    a, b, x = _pair(1)
    np.testing.assert_allclose(a(x, training=False).numpy(), b(x, training=False).numpy(),
                               rtol=0, atol=1e-5)
    w0 = [v.numpy().copy() for v in a.weights]
    a(x, training=False)
    assert all(np.array_equal(u, v.numpy()) for u, v in zip(w0, a.weights))  # 推理不动统计量


def test_training_none_and_frozen_layer_follow_keras_semantics():
    a, b, x = _pair(2)
    np.testing.assert_allclose(a(x).numpy(), b(x, training=False).numpy(), atol=1e-5)
    a.trainable = False                     # Keras 2：不可训练的 BN 强制推理模式
    b.trainable = False
    np.testing.assert_allclose(a(x, training=True).numpy(), b(x, training=True).numpy(),
                               atol=1e-5)


def test_weights_layout_is_unchanged(rt, old):
    """变量由父类 build 创建：形状 / 顺序、FedRep 切分、A27 的统计量索引都和原生 BN 一样。"""
    assert [w.shape for w in rt.get_weights()] == [w.shape for w in old.get_weights()]
    assert get_bn_stat_indices(rt) == get_bn_stat_indices(old)
    assert get_base_head_indices(rt, NCLS) == get_base_head_indices(old, NCLS)


# ══════════════════════════════════════════════════════════════════════════
# 真实调用点：GPU 的检查在 CPU 上模拟出来
# ══════════════════════════════════════════════════════════════════════════
# eager 与 tf.function 的反向都经 TF 的梯度注册表取梯度函数；把 FusedBatchNorm* 的梯度
# 换成「is_training=False 就抛」—— 与 GPU kernel（fused_batch_norm_op.cc）在确定性模式下
# 的检查同一个条件。于是 Bad-PFL 的三处调用点在 CPU 上就能复现 pilot 第一轮的崩溃。
from tensorflow.python.framework import ops as _ops            # noqa: E402

from client.client_badpfl import BadPFLMixin                   # noqa: E402
from client.compose import compose_client_class                # noqa: E402
from client.hier_fedrep import HierFedRepClient                # noqa: E402

MalFedRep = compose_client_class(HierFedRepClient, None, BadPFLMixin)


@pytest.fixture
def gpu_determinism_check():
    reg = _ops._gradient_registry._registry
    saved = {}
    for name in ("FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"):
        saved[name] = orig = reg[name]["type"]

        def guarded(op, *grads, _orig=orig):
            if not op.get_attr("is_training"):
                raise tf.errors.UnimplementedError(
                    None, None, "emulated GPU determinism check: fused batch-norm backprop, "
                                "when training is disabled")
            return _orig(op, *grads)
        reg[name]["type"] = guarded
    yield
    for name, fn in saved.items():
        reg[name]["type"] = fn


def _mal_client(model, n=16):
    """P2 模板里与 Bad-PFL 相关的三项全开：PGD ξ（A01）、官方 BN 模式（A05）、官方生成器（A14）。"""
    cfg = {
        "seed": 42,
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1,
                     "plocal_epochs": 1, "head_lr_rep": 0.05,
                     "fedrep_poison_phases": "body", "fedrep_bn_stats": "private"},
        "data": {"batch_size": 8, "img_size": 32, "dataset": "cifar10", "num_classes": NCLS},
        "backdoor": {"target_label": 0, "poison_ratio": 1.0, "badpfl_gen_steps": 1,
                     "badpfl_xi": "pgd", "badpfl_bn_mode": "official",
                     "badpfl_generator": "official"},
    }
    r = np.random.default_rng(0)
    x = r.normal(size=(n, 32, 32, 3)).astype(np.float32)
    y = (np.arange(n) % NCLS).astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
    c = MalFedRep(client_id=0, dataset=ds, model=model, config=cfg, n_samples=n)
    c.set_train_source(x, y, np.arange(n))
    c.is_malicious = True
    ew = c.model.get_weights()
    c.set_weights(ew, ew)      # 收到 edge 模型（local_train 要用 edge_weights）
    c.on_round_start(1)        # 生成器训练：PGD ξ + 穿过冻结的 F 回传（pilot 训练侧崩在这里）
    return c, x, y


def test_badpfl_call_sites_survive_the_emulated_gpu_check(gpu_determinism_check):
    c, x, y = _mal_client(build_resnet10_torch((32, 32, 3), NCLS))
    assert c._atk_generator is not None
    c.local_train(1)                                           # on_batch：official 下 F 是训练模式
    xt = c.eval_trigger(c.model, x[:4], y[:4])                 # 评估侧 ξ（pilot 致命的那一处）
    assert xt.shape == (4, 32, 32, 3) and np.isfinite(xt).all()


def test_reverse_anchor_native_bn_reproduces_the_pilot_crash(gpu_determinism_check):
    with pytest.raises(tf.errors.UnimplementedError, match="training is disabled"):
        _mal_client(build_resnet10((32, 32, 3), NCLS))
