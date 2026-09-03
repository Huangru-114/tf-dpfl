"""
tests/test_iba_trigger.py  —  IBA (NeurIPS'23) 算法不变量

触发器 T(x) = clip(x + G(x)·eps, MIN, MAX)：
  G 输出经 tanh ∈ [-1,1] → |G(x)·eps| ≤ eps（隐蔽性全靠这个 eps 预算）。
  IBA 在**标准化空间**操作，eps 直接是标准化空间里的量（**不** ÷STD，与 Bad-PFL 相反），
  MIN/MAX = 数据管线归一化后的合法范围 (0-mean)/std、(1-mean)/std。

L1 只验「可解析算出的性质」：扰动预算、clip 边界、投毒比例/标签、可复现、
target-network 双缓冲同步、净增1 缩放不变量。
「生成器是否真的学会攻击」不是 L1 —— 那要训练，属于 L2 smoke（判据 = metrics.json 的 asr）。

需要 TF，本地自动 skip，在集群上跑。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_iba import IBAMixin                # noqa: E402
from client.client_fedavg import FedAvgClient         # noqa: E402
from client.compose import compose_client_class       # noqa: E402
from data.dataset import CIFAR10_MEAN, CIFAR10_STD    # noqa: E402


# IBA 是 mixin（陷阱 #1），与方法类组合而非替换；触发器逻辑与具体 PFL 方法无关。
IBAFedAvgClient = compose_client_class(FedAvgClient, None, IBAMixin)

# img_size 必须是 16 的倍数（4 层 AE bottleneck = img_size/16）。16 是最小合法值。
IMG, CH, NCLS = 16, 3, 10
EPS = 0.01
TARGET = 9
POISON_RATIO = 0.5


def _config(dataset="cifar10", scale=1.0):
    return {
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1},
        "data": {"batch_size": 8, "img_size": IMG, "dataset": dataset,
                 "num_classes": NCLS},
        "backdoor": {"target_label": TARGET, "poison_ratio": POISON_RATIO,
                     "iba_eps": EPS, "iba_gen_steps": 2, "iba_gen_lr": 0.01,
                     "iba_scale_weights_poison": scale},
    }


def _model():
    # 末层 softmax（陷阱 #6：from_logits=False，否则梯度恒 0 → 假绿）
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG, IMG, CH)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NCLS, activation="softmax"),
    ])


def _client(rng, dataset="cifar10", n=8, scale=1.0):
    x = rng.normal(size=(n, IMG, IMG, CH)).astype(np.float32)
    y = np.arange(n, dtype=np.int64) % NCLS
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(n)
    c = IBAFedAvgClient(client_id=0, dataset=ds, model=_model(),
                        config=_config(dataset, scale), n_samples=n)
    c.is_malicious = True
    return c, x, y


# ══════════════════════════════════════════════════════════════════════════
# 不变量 1：扰动预算 —— 隐蔽性的全部依据
# ══════════════════════════════════════════════════════════════════════════
def test_generator_noise_respects_eps_budget(rng):
    """noise = G(x)·eps，G 输出有界(tanh) → |noise| ≤ eps 逐元素成立。"""
    c, x, _ = _client(rng)
    c._atk_ensure_models()
    noise = (c._atk_tgtmodel(tf.convert_to_tensor(x), training=False) * c._atk_eps).numpy()
    m = float(np.abs(noise).max())
    assert m <= EPS + 1e-6, (
        f"|noise| 最大 {m:.6f} > eps={EPS} —— 生成器输出无界（末层缺 tanh？），"
        f"扰动预算失控、隐蔽性不成立")


def test_poisoned_image_within_clip_bounds(rng):
    """x_poison = clip(x+noise, MIN, MAX) 必须逐通道落在标准化合法范围内。"""
    c, x, _ = _client(rng)
    c._atk_ensure_models()
    xp = c._atk_apply_trigger(c._atk_tgtmodel, tf.convert_to_tensor(x)).numpy()
    lo = ((0.0 - CIFAR10_MEAN) / CIFAR10_STD).astype(np.float32)
    hi = ((1.0 - CIFAR10_MEAN) / CIFAR10_STD).astype(np.float32)
    for ch in range(CH):
        assert xp[..., ch].min() >= lo[ch] - 1e-5, f"通道{ch} 低于合法下界"
        assert xp[..., ch].max() <= hi[ch] + 1e-5, f"通道{ch} 高于合法上界"


def test_eps_is_not_std_divided(rng):
    """IBA eps 直接用于标准化空间，**不**做 Bad-PFL 的 ÷STD 换算（与 Bad-PFL 的区别）。"""
    c, _, _ = _client(rng)
    assert np.isclose(c._atk_eps, EPS), (
        "IBA eps 应等于 config 值本身（标准化空间），若被 ÷STD 处理则说明误抄了 Bad-PFL 语义")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 2：投毒比例与标签精确
# ══════════════════════════════════════════════════════════════════════════
def test_poison_batch_count_and_labels(rng):
    """恰好 round(n·ratio) 个样本被改成 target_label，其余逐位不动。"""
    n = 8
    c, x, y = _client(rng, n=n)
    c._atk_ensure_models()

    xp, yp = c.on_batch(x, y)
    xp, yp = xp.numpy(), yp.numpy()

    changed = np.flatnonzero(yp != y)
    assert len(changed) == int(round(n * POISON_RATIO)), (
        f"投毒 {len(changed)} 个，期望 {int(round(n * POISON_RATIO))}")
    assert np.all(yp[changed] == TARGET), "被投毒样本的标签不是 target_label"

    untouched = np.array([i for i in range(n) if i not in set(changed.tolist())])
    if len(untouched):
        np.testing.assert_array_equal(
            xp[untouched], x[untouched],
            err_msg="未投毒样本的像素被改动了 —— 投毒不是逐样本选择的")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 3：可复现（陷阱 #2）
# ══════════════════════════════════════════════════════════════════════════
def test_poison_selection_is_reproducible(rng):
    """投毒样本选择走客户端自己的 seeded RNG，不依赖全局 np.random（同 test_badpfl）。"""
    n = 8
    c1, x, y = _client(rng, n=n)
    c1._atk_ensure_models()
    c2, _, _ = _client(rng, n=n)               # 同 client_id=0、同 seed → self.rng 同序列
    c2._atk_ensure_models()

    np.random.seed(0)
    _, y1 = c1.on_batch(x, y)
    np.random.seed(999)
    _, y2 = c2.on_batch(x, y)
    assert np.array_equal(y1.numpy(), y2.numpy()), (
        "同 (seed, client_id) 在不同全局 np.random 状态下投毒选择不一致 —— 仍依赖全局 RNG")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 4：target-network 双缓冲同步（LIRA）
# ══════════════════════════════════════════════════════════════════════════
def test_target_network_synced_after_retrain(rng):
    """一轮再训后 tgtmodel ← atkmodel：两者全部权重逐一致（官方 fl_trainer.py:1070）。"""
    c, _, _ = _client(rng)
    c._atk_train_generator()
    for a, t in zip(c._atk_atkmodel.get_weights(), c._atk_tgtmodel.get_weights()):
        np.testing.assert_allclose(a, t, rtol=1e-6, atol=1e-6,
                                   err_msg="再训后 tgtmodel 未与 atkmodel 同步")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 5：净增1 —— 固定系数 model-replacement 缩放不变量
# ══════════════════════════════════════════════════════════════════════════
def test_scaling_identity_when_gamma_one(rng):
    """γ=1.0（默认）时 on_upload 恒等：上传物逐位不变。"""
    c, _, _ = _client(rng, scale=1.0)
    ref = [w.copy() for w in c.model.get_weights()]
    c.edge_weights = ref
    upload = [w + 0.5 for w in ref]                    # 任意非零更新
    out = c.on_upload([u.copy() for u in upload], round_idx=1)
    for o, u in zip(out, upload):
        np.testing.assert_array_equal(o, u, err_msg="γ=1 时 on_upload 应恒等")


def test_scaling_amplifies_update_exactly(rng):
    """γ=k 时更新量精确放大 k 倍，锚 edge_weights：new = edge + (upload-edge)·k。"""
    k = 3.0
    c, _, _ = _client(rng, scale=k)
    ref = [w.copy() for w in c.model.get_weights()]
    c.edge_weights = ref
    upload = [r + 0.2 for r in ref]
    out = c.on_upload([u.copy() for u in upload], round_idx=1)
    for o, r in zip(out, ref):
        np.testing.assert_allclose(o, r + 0.2 * k, rtol=1e-6,
                                   err_msg="缩放不是 edge + (upload-edge)·k")


def test_scaling_skipped_for_benign(rng):
    """非恶意端即便 γ≠1 也不缩放（缩放只作用于恶意端）。"""
    c, _, _ = _client(rng, scale=5.0)
    c.is_malicious = False
    ref = [w.copy() for w in c.model.get_weights()]
    c.edge_weights = ref
    upload = [r + 0.2 for r in ref]
    out = c.on_upload([u.copy() for u in upload], round_idx=1)
    for o, u in zip(out, upload):
        np.testing.assert_array_equal(o, u, err_msg="良性端不应被缩放")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 6：共享生成器注入（对齐官方：adversary 持单一 atk/tgt/optimizer）
# ══════════════════════════════════════════════════════════════════════════
def test_shared_generator_injection(rng):
    """注入同一 (atk,tgt,opt) 后两端引用同一对象，且 ensure 不重建。"""
    from models.autoencoder import build_autoencoder

    c1, _, _ = _client(rng)
    c2, _, _ = _client(rng)
    c1._atk_ensure_models()
    c2._atk_ensure_models()
    assert c1._atk_tgtmodel is not c2._atk_tgtmodel, "未注入时不应共享"

    atk = build_autoencoder(img_size=IMG, channels=3)
    tgt = build_autoencoder(img_size=IMG, channels=3)
    tgt.set_weights(atk.get_weights())
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    d1, _, _ = _client(rng)
    d2, _, _ = _client(rng)
    d1.set_shared_generator(atk, tgt, opt)
    d2.set_shared_generator(atk, tgt, opt)
    d1._atk_ensure_models()
    d2._atk_ensure_models()
    assert d1._atk_atkmodel is atk and d2._atk_atkmodel is atk, "注入后未引用同一 atkmodel"
    assert d1._atk_tgtmodel is tgt and d2._atk_tgtmodel is tgt, "注入后未引用同一 tgtmodel"
    assert d1._atk_gen_opt is d2._atk_gen_opt is opt, "注入后未共享同一 optimizer"
