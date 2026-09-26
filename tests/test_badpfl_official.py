"""
tests/test_badpfl_official.py  —  A4：Bad-PFL 攻击侧对齐官方代码（AUDIT A01 A03 A04 A05 A14 + G7 前提）

每条的「怎么验证」来自 AUDIT 该行；反向锚点 = 旧开关值下同一断言失败。
需要 TF，本地无 TF 时 skip，在集群（或 scratch 里的 TF CPU venv）跑。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_badpfl import BadPFLMixin, EVAL_CHUNK     # noqa: E402
from client.client_fedavg import FedAvgClient               # noqa: E402
from client.compose import compose_client_class             # noqa: E402
from data.pixel_space import (CIFAR10_MEAN, CIFAR10_STD,     # noqa: E402
                              normalize_images, valid_range)
from models.autoencoder import (build_autoencoder_official,  # noqa: E402
                                build_generator)

Cls = compose_client_class(FedAvgClient, None, BadPFLMixin)
IMG, CH, NCLS = 16, 3, 10
EPS = SIGMA = 4.0 / 255.0
TARGET = 0


def _cfg(**bd_extra):
    data = {"batch_size": 32, "img_size": IMG, "dataset": "cifar10", "num_classes": NCLS}
    for k in ("normalize", "augment"):
        if k in bd_extra:
            data[k] = bd_extra.pop(k)
    return {
        "seed": 42,
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1},
        "data": data,
        "backdoor": {"target_label": TARGET, "poison_ratio": 0.2,
                     "badpfl_epsilon": EPS, "badpfl_sigma": SIGMA,
                     "badpfl_gen_steps": 2, "badpfl_gen_lr": 0.01, **bd_extra},
    }


def _linear_softmax(seed=0):
    """输入梯度可解析的线性 softmax 模型：logits = flat(x)·W + b。"""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG, IMG, CH)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NCLS, activation="softmax"),
    ])
    r = np.random.default_rng(seed)
    W = r.normal(0, 0.05, size=(IMG * IMG * CH, NCLS)).astype(np.float32)
    b = r.normal(0, 0.1, size=(NCLS,)).astype(np.float32)
    m.layers[-1].set_weights([W, b])
    return m, W, b


def _client(cfg, n=32, seed=0, model=None):
    r = np.random.default_rng(seed)
    x_u8 = r.integers(0, 256, size=(n, IMG, IMG, CH), dtype=np.uint8)
    x = normalize_images(x_u8, cfg)
    y = (np.arange(n) % NCLS).astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(cfg["data"]["batch_size"])
    m = model if model is not None else _linear_softmax(seed)[0]
    c = Cls(client_id=0, dataset=ds, model=m, config=cfg, n_samples=n)
    c.is_malicious = True
    c._attack_active = True          # 直接调 on_batch：攻击时间窗要显式打开（c7a06c49）
    return c, x, y


def _pgd_reference(x, y, u, W, b, sigma, lo, hi):
    """numpy 版官方 pgd_attack（fba.py:6-22），在输入空间里。"""
    n = len(x)
    x0 = np.clip(x + u, lo, hi)
    z = x0.reshape(n, -1) @ W + b
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    oh = np.eye(NCLS, dtype=np.float32)[y]
    g = ((p - oh) @ W.T).reshape(x.shape)          # ∂CE/∂x（mean 归约只差正常数，不影响 sign）
    x1 = x0 + sigma * np.sign(g)
    eta = np.clip(x1 - x, -sigma, sigma)
    return np.clip(x + eta, lo, hi) - x


# ══════════════════════════════════════════════════════════════════════════
# A01：ξ = 官方单步 PGD
# ══════════════════════════════════════════════════════════════════════════
def test_pgd_matches_numpy_reference_with_injected_start():
    cfg = _cfg(badpfl_xi="pgd")
    m, W, b = _linear_softmax(1)
    c, x, y = _client(cfg, model=m)
    u = (np.random.default_rng(5).uniform(-1, 1, size=x.shape).astype(np.float32)
         * c._atk_sigma_norm)
    got = c._atk_pgd_noise(m, x, y, training=False, u=u).numpy()
    ref = _pgd_reference(x, y, u, W, b, c._atk_sigma_norm, c._atk_lo, c._atk_hi)
    np.testing.assert_allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("normalize", [True, False])
def test_pgd_stays_in_budget_and_valid_range(normalize):
    cfg = _cfg(badpfl_xi="pgd", normalize=normalize)
    c, x, y = _client(cfg, n=64)
    xi = c._atk_xi(c.model, x, y).numpy()
    lo, hi = valid_range(cfg)
    assert np.all(np.abs(xi) <= c._atk_sigma_norm * (1 + 1e-5))
    assert np.all(x + xi >= lo - 1e-5) and np.all(x + xi <= hi + 1e-5)


def _interior_full_ratio(mode):
    cfg = _cfg(badpfl_xi=mode)
    c, x, y = _client(cfg, n=64)
    xi = c._atk_xi(c.model, x, y).numpy()
    sig = c._atk_sigma_norm
    lo, hi = c._atk_lo, c._atk_hi
    interior = (x - 2 * sig > lo) & (x + 2 * sig < hi)
    full = np.abs(xi) >= sig * (1 - 1e-4)
    return full[interior].mean(), (np.abs(xi) / sig)[interior].mean()


def test_pgd_interior_saturation_is_about_half():
    """官方：内点约 50% 像素满幅、均值 0.75σ（F-019：u≥0 取满幅，u<0 取 U[0,σ)）。"""
    frac, mean = _interior_full_ratio("pgd")
    assert abs(frac - 0.5) < 0.02, frac
    assert abs(mean - 0.75) < 0.02, mean


def test_fgsm_interior_saturation_is_one_reverse_anchor():
    frac, _ = _interior_full_ratio("fgsm")
    assert frac == 1.0


def test_pgd_start_noise_uses_the_client_seeded_rng():
    cfg = _cfg(badpfl_xi="pgd")
    c1, x, y = _client(cfg)
    c2, _, _ = _client(cfg)
    np.random.seed(0)
    a = c1._atk_xi(c1.model, x, y).numpy()
    np.random.seed(123)
    b = c2._atk_xi(c2.model, x, y).numpy()
    np.testing.assert_array_equal(a, b)


# ══════════════════════════════════════════════════════════════════════════
# A03：伯努利投毒经过 on_batch
# ══════════════════════════════════════════════════════════════════════════
def test_bernoulli_rho_zero_leaves_batch_byte_identical():
    cfg = _cfg(poison_sampling="bernoulli")
    cfg["backdoor"]["poison_ratio"] = 0.0
    c, x, y = _client(cfg)
    xp, yp = c.on_batch(x, y)
    np.testing.assert_array_equal(np.asarray(xp), x)
    np.testing.assert_array_equal(np.asarray(yp), y)


def test_bernoulli_rho_one_poisons_everything():
    cfg = _cfg(poison_sampling="bernoulli")
    cfg["backdoor"]["poison_ratio"] = 1.0
    c, x, y = _client(cfg)
    _, yp = c.on_batch(x, y)
    assert np.all(yp.numpy() == TARGET)


# ══════════════════════════════════════════════════════════════════════════
# A05：6 个调用点的训练 / 推理模式
# ══════════════════════════════════════════════════════════════════════════
class _Rec:
    """包住一个 Keras 模型，记录每次调用的 (batch, training)，其余属性原样转发。"""

    def __init__(self, inner, log, tag):
        self._inner, self._log, self._tag = inner, log, tag

    def __call__(self, x, training=False):
        self._log.append((self._tag, int(x.shape[0]), bool(training),
                          float(tf.reduce_min(x)), float(tf.reduce_max(x))))
        return self._inner(x, training=training)

    def __getattr__(self, k):
        return getattr(self._inner, k)


def _recorded(cfg, n=32):
    c, x, y = _client(cfg, n=n)
    c._atk_ensure_generator()
    log = []
    c.model = _Rec(c.model, log, "F")
    c._atk_generator = _Rec(c._atk_generator, log, "G")
    return c, x, y, log


def _modes(log, tag):
    return {t for g, _, t, *_ in log if g == tag}


def test_bn_modes_official_training_side():
    c, x, y, log = _recorded(_cfg(badpfl_bn_mode="official", badpfl_xi="pgd"))
    c._atk_train_generator()
    assert _modes(log, "G") == {True} and _modes(log, "F") == {False}      # #1 #2
    log.clear()
    c.on_batch(x, y)
    assert _modes(log, "G") == {True} and _modes(log, "F") == {True}       # #3 #4
    assert {b for g, b, *_ in log} == {32}                                  # 整批


def test_bn_modes_legacy_training_side_reverse_anchor():
    c, x, y, log = _recorded(_cfg())
    c.on_batch(x, y)
    assert _modes(log, "G") == {False} and _modes(log, "F") == {False}


def test_bn_modes_official_eval_side_chunks_of_32():
    """#5：G 用 batch 统计、每块恰好 32 张（尾块循环补足）；#6：求 ξ 的 F 用推理模式。"""
    c, x, y, log = _recorded(_cfg(badpfl_bn_mode="official", badpfl_xi="pgd"), n=70)
    c.eval_trigger(c.model, x, y)
    g = [(b, t) for tag, b, t, *_ in log if tag == "G"]
    assert g and all(b == EVAL_CHUNK and t for b, t in g) and len(g) == 3   # 70 → 3 块
    assert _modes(log, "F") == {False}


def test_eval_delta_tail_padding_only_scores_real_samples():
    """尾块补足：每个真实样本都有 δ；与「把它单独放进一个完整块」的结果逐元素相同。"""
    c, x, y = _client(_cfg(badpfl_bn_mode="official"), n=40)
    c._atk_ensure_generator()
    d = c.eval_delta(x)
    assert d.shape == x.shape
    idx = np.arange(32, 64) % 40
    tail = c._atk_gen_delta(x[idx], training=True).numpy()[:8]
    np.testing.assert_allclose(d[32:40], tail, atol=1e-6)


def test_eval_delta_small_probe_cycles_to_32():
    c, x, y, log = _recorded(_cfg(badpfl_bn_mode="official"), n=5)
    d = c.eval_delta(x)
    assert d.shape == x.shape
    assert [b for tag, b, *_ in log if tag == "G"] == [32]


# ══════════════════════════════════════════════════════════════════════════
# A04（deviate，守现状）：生成器只在干净数据、真实标签上训
# ══════════════════════════════════════════════════════════════════════════
def test_generator_trains_on_clean_batches_with_true_labels():
    c, x, y = _client(_cfg(badpfl_xi="pgd", poison_sampling="bernoulli"))
    c._atk_ensure_generator()
    calls = []
    orig = c.on_batch
    c.on_batch = lambda *a: calls.append(1) or orig(*a)
    batches = c._atk_gen_batches()
    c._atk_train_generator()
    assert calls == [], "生成器训练路径调用了 on_batch（会看到投毒样本）"
    for xb, yb in batches:
        np.testing.assert_array_equal(np.asarray(yb), y[:len(yb)])
        np.testing.assert_array_equal(np.asarray(xb), x[:len(xb)])


# ══════════════════════════════════════════════════════════════════════════
# A14：生成器结构对齐官方 generator.py
# ══════════════════════════════════════════════════════════════════════════
def _torch_param_count(img_ch=3):
    """官方 Autoencoder 的 torch 参数量（解析值）：Conv/ConvT 全带 bias，前 7 层后接 BN。"""
    enc = [(img_ch, 16), (16, 32), (32, 64), (64, 128)]
    dec = [(128, 64), (64, 32), (32, 16), (16, img_ch)]
    conv = sum(i * o * 16 + o for i, o in enc + dec)
    bn = sum(2 * o for _, o in enc) + sum(2 * o for _, o in dec[:-1])
    return conv + bn


def test_official_generator_param_count_matches_torch():
    g = build_autoencoder_official(img_size=32)
    n = sum(int(np.prod(v.shape)) for v in g.trainable_variables)
    assert n == _torch_param_count() == 346_659


def test_official_generator_bn_bias_and_output():
    g = build_autoencoder_official(img_size=32)
    bns = [l for l in g.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    convs = [l for l in g.layers
             if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose))]
    assert len(bns) == 7 and all(l.epsilon == 1e-5 and l.momentum == 0.9 for l in bns)
    assert len(convs) == 8 and all(l.use_bias for l in convs)
    assert isinstance(g.layers[-1], tf.keras.layers.Activation)        # 末层 tanh，无 BN
    assert not isinstance(g.layers[-2], tf.keras.layers.BatchNormalization)
    out = g(np.random.default_rng(0).uniform(0, 1, (4, 32, 32, 3)).astype(np.float32),
            training=True).numpy()
    assert out.shape == (4, 32, 32, 3) and out.min() >= -1 and out.max() <= 1


def test_official_generator_init_is_torch_default():
    g = build_autoencoder_official(img_size=32)
    convs = [l for l in g.layers
             if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose))]
    for l in convs:
        k, b = l.get_weights()
        # Conv2D kernel (k,k,in,out) → fan_in = k²·in；ConvT kernel (k,k,out,in) → k²·out
        fan_in = 16 * k.shape[2]
        bound = 1 / np.sqrt(fan_in)
        assert np.all(np.abs(b) <= bound + 1e-7) and np.any(b != 0)
        if k.size >= 4096:                     # 样本够大时检方差（±10%）
            assert abs(k.var() * 3 * fan_in - 1) < 0.10, (l.name, k.var() * 3 * fan_in)


def test_same_padding_equals_torch_p1_for_k4_s2():
    """SAME ≡ ZeroPadding2D(1)+VALID（Conv2D）；SAME ≡ VALID+Cropping2D(1)（ConvT），逐元素。"""
    r = np.random.default_rng(0)
    x = r.normal(size=(2, 16, 16, 5)).astype(np.float32)
    k = r.normal(size=(4, 4, 5, 7)).astype(np.float32)
    same = tf.nn.conv2d(x, k, 2, "SAME").numpy()
    pad = tf.nn.conv2d(tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]]), k, 2, "VALID").numpy()
    np.testing.assert_allclose(same, pad, atol=1e-5)

    L = tf.keras.layers
    a = L.Conv2DTranspose(3, 4, strides=2, padding="same")
    v = L.Conv2DTranspose(3, 4, strides=2, padding="valid")
    xa = tf.constant(x)
    a(xa), v(xa)
    v.set_weights(a.get_weights())
    ya = a(xa).numpy()
    yv = L.Cropping2D(1)(v(xa)).numpy()
    np.testing.assert_allclose(ya, yv, atol=1e-5)


def test_official_generator_sees_unit_pixels():
    """official：生成器的实际输入 ∈ [0,1]（标准化图先反变换）；legacy：吃标准化图。"""
    c, x, y, log = _recorded(_cfg(badpfl_generator="official"))
    c.eval_delta(x)
    lo = min(v[3] for v in log if v[0] == "G")
    hi = max(v[4] for v in log if v[0] == "G")
    assert lo >= -1e-6 and hi <= 1 + 1e-6
    c2, x2, y2, log2 = _recorded(_cfg())
    c2.eval_delta(x2)
    assert min(v[3] for v in log2 if v[0] == "G") < 0          # 反向锚点：标准化空间有负值


def test_build_generator_follows_switch():
    assert build_generator(_cfg(badpfl_generator="official")).name == "badpfl_generator_official"
    assert build_generator(_cfg()).name == "badpfl_generator"


# ══════════════════════════════════════════════════════════════════════════
# G7 前提：关标准化时 ε 与静态触发器都回到 [0,1] 口径（F-027）
# ══════════════════════════════════════════════════════════════════════════
def test_badpfl_eps_follows_normalization_switch():
    on, _, _ = _client(_cfg())
    np.testing.assert_allclose(on._atk_eps_norm, EPS / CIFAR10_STD, rtol=1e-6)   # ≈ 0.0635
    off, _, _ = _client(_cfg(normalize=False))
    np.testing.assert_array_equal(off._atk_eps_norm, np.full(3, np.float32(EPS)))


def test_badnet_trigger_follows_normalization_switch():
    from attack.triggers import build_trigger
    bd = {"trigger": "badnet", "badnet_size": 3, "badnet_value": 1.0}
    x = np.zeros((1, 8, 8, 3), np.float32)
    off = build_trigger(bd, 8, config=_cfg(normalize=False))(x)
    assert np.all(off[0, -3:, -3:, :] == 1.0)
    on = build_trigger(bd, 8, config=_cfg())(x)
    np.testing.assert_allclose(on[0, -1, -1, :], (1 - CIFAR10_MEAN) / CIFAR10_STD, rtol=1e-6)
    legacy = build_trigger(bd, 8)(x)                          # 不传 config = 旧行为
    np.testing.assert_array_equal(legacy, on)
