"""
tests/test_badpfl_trigger.py  —  Bad-PFL (ICLR'25) 算法不变量

触发器 T(x) = x + ξ + δ：
  ξ = σ·sign(∇_x CE(f(x), y))       破坏性噪声（FGSM，单步）
  δ = G(x)·ε                        生成器扰动，G 输出应有界
官方 ε = σ = 4/255（[0,1] 像素空间）；本仓库工作在标准化空间，需逐通道换算
`ε_norm = ε / STD`，**STD 必须与数据管线实际使用的常数一致**。

L1 只验「可解析算出的性质」：扰动预算、投毒比例、标签、可复现性、归一化常数。
「生成器是否真的学会了攻击」不是 L1 —— 那要训练，属于 L2 smoke（见
experiments/attack/bad-pfl/smoke.yaml），判据是 metrics.json 里的 asr。

需要 TF，本地自动 skip，在集群上跑。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_badpfl import BadPFLClient      # noqa: E402


IMG, CH, NCLS = 8, 3, 10
EPS = SIGMA = 4.0 / 255.0
TARGET = 9
POISON_RATIO = 0.5


def _config(dataset="cifar10"):
    return {
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1},
        "data": {"batch_size": 8, "img_size": IMG, "dataset": dataset,
                 "num_classes": NCLS},
        "backdoor": {"target_label": TARGET, "poison_ratio": POISON_RATIO,
                     "badpfl_epsilon": EPS, "badpfl_sigma": SIGMA,
                     "badpfl_gen_steps": 2, "badpfl_gen_lr": 0.01},
    }


def _model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG, IMG, CH)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NCLS),
    ])


def _client(rng, dataset="cifar10", n=8):
    x = rng.normal(size=(n, IMG, IMG, CH)).astype(np.float32)
    y = np.arange(n, dtype=np.int64) % NCLS
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(n)
    c = BadPFLClient(client_id=0, dataset=ds, model=_model(),
                     config=_config(dataset), n_samples=n)
    c.is_malicious = True
    return c, x, y


# ══════════════════════════════════════════════════════════════════════════
# 不变量 1：扰动预算 —— 隐蔽性的全部依据
# ══════════════════════════════════════════════════════════════════════════
def test_fgsm_noise_respects_sigma_budget(rng):
    """ξ = σ·sign(g) → |ξ| 必须逐通道恰好等于 sigma_norm（sign 只取 ±1）。"""
    c, x, y = _client(rng)
    xi = c._fgsm_noise(c.model, x, y).numpy()

    for ch in range(CH):
        vals = np.unique(np.abs(xi[..., ch]).round(8))
        allowed = {0.0, round(float(c.sigma_norm[ch]), 8)}
        assert set(vals.tolist()) <= allowed, (
            f"通道 {ch} 的 |ξ| 取值 {vals[:5]}，超出预算 σ_norm={c.sigma_norm[ch]:.5f}")


def test_generator_delta_respects_eps_budget(rng):
    """δ = G(x)·ε_norm，G 输出须有界（tanh）→ |δ| ≤ eps_norm 逐通道成立。"""
    c, x, _ = _client(rng)
    c._ensure_generator()
    delta = c._gen_delta(tf.convert_to_tensor(x)).numpy()

    for ch in range(CH):
        m = float(np.abs(delta[..., ch]).max())
        assert m <= float(c.eps_norm[ch]) + 1e-6, (
            f"通道 {ch} 的 |δ| 最大 {m:.5f} > ε_norm={c.eps_norm[ch]:.5f} —— "
            f"生成器输出无界（输出层缺 tanh？），扰动预算失控、隐蔽性不成立")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 2：归一化常数必须跟随实际数据集（陷阱 #5）
# ══════════════════════════════════════════════════════════════════════════
def test_eps_conversion_uses_dataset_matching_std(rng):
    """
    data.dataset=cifar100 时，数据管线用 CIFAR100_STD 归一化
    （data/dataset.py:121），触发器换算却硬用 attack/triggers.py 的 CIFAR10_STD
    → ε/σ 的实际像素预算偏离官方设定。
    """
    from data.dataset import CIFAR100_STD

    c, _, _ = _client(rng, dataset="cifar100")
    expected = (EPS / np.asarray(CIFAR100_STD, np.float32))

    np.testing.assert_allclose(
        c.eps_norm, expected, rtol=1e-6,
        err_msg="dataset=cifar100 时仍用 CIFAR10_STD 换算 ε —— 扰动预算与数据管线不一致")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 3：投毒比例与标签精确
# ══════════════════════════════════════════════════════════════════════════
def test_poison_batch_count_and_labels(rng):
    """恰好 round(n·ratio) 个样本被改成 target_label，其余逐位不动。"""
    n = 8
    c, x, y = _client(rng, n=n)
    c._ensure_generator()

    xp, yp = c._poison_batch(x, y)
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
# 不变量 4：可复现（陷阱 #2）
# ══════════════════════════════════════════════════════════════════════════
def test_poison_selection_is_reproducible(rng):
    """
    `_poison_batch` 用全局 np.random.shuffle 选投毒样本 → 同一 seed 下不可复现，
    「固定种子两端跑出一致结果」不成立。投毒选择必须走客户端自己的 seeded RNG。
    """
    n = 8
    c, x, y = _client(rng, n=n)
    c._ensure_generator()

    np.random.seed(0)
    _, y1 = c._poison_batch(x, y)
    np.random.seed(0)
    _, y2 = c._poison_batch(x, y)
    assert np.array_equal(y1.numpy(), y2.numpy()), "重置全局 seed 后仍不可复现"

    # 真正的要求：不依赖全局 RNG 状态
    c2, _, _ = _client(rng, n=n)
    c2._ensure_generator()
    np.random.seed(12345)                      # 故意打乱全局状态
    _, y3 = c2._poison_batch(x, y)
    np.random.seed(999)
    _, y4 = c2._poison_batch(x, y)
    assert np.array_equal(y3.numpy(), y4.numpy()), (
        "投毒样本的选择随全局 np.random 状态变化 —— 实验不可复现，"
        "应改用客户端自己的 np.random.default_rng(seed + client_id)")
