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

from client.client_badpfl import BadPFLMixin       # noqa: E402
from client.client_fedavg import FedAvgClient      # noqa: E402
from client.compose import compose_client_class    # noqa: E402


# Bad-PFL 现在是 mixin（CLAUDE.md 陷阱 #1 已修复），与方法类组合而非替换。
# 触发器/投毒逻辑与具体 PFL 方法无关，这里用 FedAvgClient 作为方法类。
BadPFLFedAvgClient = compose_client_class(FedAvgClient, None, BadPFLMixin)


# img_size 必须是 16 的倍数：4 层 AE 的 bottleneck = img_size/16，非整数则 decoder
# 输出形状与输入不符（官方 torch 版在 8×8 同样会崩）。16 是最小合法值，本地仍秒级。
IMG, CH, NCLS = 16, 3, 10
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
    c = BadPFLFedAvgClient(client_id=0, dataset=ds, model=_model(),
                           config=_config(dataset), n_samples=n)
    c.is_malicious = True
    return c, x, y


# ══════════════════════════════════════════════════════════════════════════
# 不变量 1：扰动预算 —— 隐蔽性的全部依据
# ══════════════════════════════════════════════════════════════════════════
def test_fgsm_noise_respects_sigma_budget(rng):
    """ξ = σ·sign(g) → |ξ| 必须逐通道恰好等于 sigma_norm（sign 只取 ±1）。"""
    c, x, y = _client(rng)
    xi = c._atk_fgsm_noise(c.model, x, y).numpy()

    # sign() 只产生 ±1 或 0 → |ξ| 每个元素要么 0、要么恰好 sigma_norm[ch]。
    # 用浮点容差比较，而非 round(8) 后的精确集合：sigma_norm 是 float32，
    # np.round(float32,8) 与 round(float(float32),8) 在第 8 位可能不一致（float32↔float64
    # 提升差异），那会让本断言假红 —— 这是测试写法问题，与预算是否被遵守无关。
    for ch in range(CH):
        av = np.abs(xi[..., ch])
        budget = float(c._atk_sigma_norm[ch])
        ok = np.isclose(av, 0.0, atol=1e-6) | np.isclose(av, budget, atol=1e-6)
        assert ok.all(), (
            f"通道 {ch} 的 |ξ| 有超出预算的取值 {np.unique(av[~ok])[:5]}，"
            f"σ_norm={budget:.5f}")


def test_generator_delta_respects_eps_budget(rng):
    """δ = G(x)·ε_norm，G 输出须有界（tanh）→ |δ| ≤ eps_norm 逐通道成立。"""
    c, x, _ = _client(rng)
    c._atk_ensure_generator()
    delta = c._atk_gen_delta(tf.convert_to_tensor(x)).numpy()

    for ch in range(CH):
        m = float(np.abs(delta[..., ch]).max())
        assert m <= float(c._atk_eps_norm[ch]) + 1e-6, (
            f"通道 {ch} 的 |δ| 最大 {m:.5f} > ε_norm={c._atk_eps_norm[ch]:.5f} —— "
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
        c._atk_eps_norm, expected, rtol=1e-6,
        err_msg="dataset=cifar100 时仍用 CIFAR10_STD 换算 ε —— 扰动预算与数据管线不一致")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 3：投毒比例与标签精确
# ══════════════════════════════════════════════════════════════════════════
def test_poison_batch_count_and_labels(rng):
    """恰好 round(n·ratio) 个样本被改成 target_label，其余逐位不动。"""
    n = 8
    c, x, y = _client(rng, n=n)
    c._atk_ensure_generator()

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
# 不变量 4：可复现（陷阱 #2）
# ══════════════════════════════════════════════════════════════════════════
def test_poison_selection_is_reproducible(rng):
    """
    投毒样本的选择必须走客户端自己的 seeded RNG（client_base:
    `np.random.default_rng([seed, client_id])`），不依赖全局 np.random 状态。

    复现性 = 「两次独立运行、同 (seed, client_id) → 结果一致」。用两个**全新**
    客户端各调一次 on_batch 来模拟两次运行；self.rng 是有状态的，同一客户端连调
    两次本就不该相等（那是正常的随机推进，不是不可复现）。同时把全局 np.random
    置成不同状态，验证结果与全局 RNG 无关。只比较标签 y（投毒选取由 mask 决定；
    像素 x 还受未播种的 generator 权重影响，与本不变量无关）。
    """
    n = 8
    c1, x, y = _client(rng, n=n)
    c1._atk_ensure_generator()
    c2, _, _ = _client(rng, n=n)               # 同 client_id=0、同 seed → self.rng 同序列
    c2._atk_ensure_generator()

    np.random.seed(0)                          # 两次“运行”故意处于不同的全局状态
    _, y1 = c1.on_batch(x, y)
    np.random.seed(999)
    _, y2 = c2.on_batch(x, y)
    assert np.array_equal(y1.numpy(), y2.numpy()), (
        "同 (seed, client_id) 的两个客户端在不同全局 np.random 状态下投毒选择不一致 "
        "—— 说明仍依赖全局 RNG，实验不可复现；应走 np.random.default_rng([seed, client_id])")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 5：P4 共享生成器（对齐官方 fba.py：adversary 持有单一 generator）
# ══════════════════════════════════════════════════════════════════════════
def test_shared_generator_injection(rng):
    """
    未注入 → 每端 lazy build 各自的 generator（默认，per-client）。
    注入同一对象后 → 两端引用**同一** generator+optimizer，且 _atk_ensure_generator
    不再重建（P4 语义：adversary 持有单一共享生成器，恶意端 download-optimize-return）。
    """
    from models.autoencoder import build_autoencoder

    # 默认（未注入）：各自独立
    c1, _, _ = _client(rng)
    c2, _, _ = _client(rng)
    c1._atk_ensure_generator()
    c2._atk_ensure_generator()
    assert c1._atk_generator is not c2._atk_generator, "未注入时不应共享生成器"

    # 注入共享对象：同一 generator + 同一 optimizer，ensure 不重建
    shared = build_autoencoder(img_size=IMG, channels=3)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    d1, _, _ = _client(rng)
    d2, _, _ = _client(rng)
    d1.set_shared_generator(shared, opt)
    d2.set_shared_generator(shared, opt)
    d1._atk_ensure_generator()
    d2._atk_ensure_generator()
    assert d1._atk_generator is shared and d2._atk_generator is shared, (
        "注入后两端未引用同一共享 generator —— P4 语义未生效")
    assert d1._atk_gen_opt is d2._atk_gen_opt is opt, "注入后两端未共享同一 optimizer"
