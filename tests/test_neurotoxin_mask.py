"""
tests/test_neurotoxin_mask.py  —  Neurotoxin (Zhang et al., ICML'22) 算法不变量

核心思想：把后门投影到**良性训练几乎不更新**的坐标上，使其不被后续良性轮覆盖。
官方 `jhcknzzm/Federated-Learning-Backdoor` 的 `grad_mask_cv`：
对良性梯度取 |g| **最小**的 `ratio` 比例坐标，mask=1（**保留**恶意更新），
其余坐标 mask=0（丢弃恶意更新）；且阈值是**逐层**取的，不是全局。

> ⚠️ `ratio` 的语义（保留比例 vs 屏蔽比例）在 clone 官方实现前尚未证实。
>   本文件按「`neurotoxin_mask_ratio` = **保留**比例」写断言——这是文献语义。
>   当前实现是「屏蔽比例」（`mask_ratio=0.05` → 放行 95% 坐标 ≈ 退化成普通 BadNet）。
>   clone `reference/neurotoxin-src` 后第一件事：核对 `ratio` 的用法，
>   然后把结论写进 CLAUDE.md 陷阱 #4 并锁定本文件的断言。

需要 TF，本地自动 skip，在集群上跑：  pytest tests/test_neurotoxin_mask.py -v
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_neurotoxin import NeurotoxinClient      # noqa: E402


KEEP_RATIO = 0.05          # 文献语义：保留 |g| 最小的 5% 坐标


# ── 最小可跑装置：3 个坐标的线性模型 + 受控梯度 ────────────────────────────
def _tiny_config():
    return {
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1},
        "data": {"batch_size": 4, "img_size": 4},
        "backdoor": {"neurotoxin_mask_ratio": KEEP_RATIO, "neurotoxin_mask_batches": 0},
    }


def _tiny_model(n_features=100):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer="zeros"),
    ])
    return m


def _make_client(rng, n_features=100):
    x = rng.normal(size=(8, n_features)).astype(np.float32)
    y = np.array([0, 1] * 4, dtype=np.int64)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    c = NeurotoxinClient(client_id=0, dataset=ds, model=_tiny_model(n_features),
                         config=_tiny_config(), n_samples=8)
    c.is_malicious = True
    c.set_clean_dataset(ds)
    return c


def _fixed_grads(client, grad_values):
    """把 _benign_grads 换成返回一组已知梯度 → mask 完全可解析地算出。"""
    tensors = [tf.constant(g) for g in grad_values]
    client._benign_grads = lambda images, labels: tensors
    return tensors


# ══════════════════════════════════════════════════════════════════════════
# 不变量 1：保留集 = |benign grad| 最小的 ratio 比例坐标
# ══════════════════════════════════════════════════════════════════════════
def test_mask_keeps_smallest_benign_gradient_coords(rng):
    """
    这是 Neurotoxin 的**全部**理论依据：只在良性训练几乎不动的坐标上写后门。
    保留集必须精确等于 |g| 最小的那 ratio 比例坐标。
    """
    n_feat = 100
    c = _make_client(rng, n_feat)
    g = rng.normal(size=(n_feat, 2)).astype(np.float32)   # Dense kernel 形状
    _fixed_grads(c, [g])

    mask = c._compute_mask()
    assert mask is not None, "mask 为 None —— 掩码根本没生效"

    flat_mask = np.concatenate([m.reshape(-1) for m in mask])
    flat_grad = np.abs(g).reshape(-1)
    n_keep = int(round(flat_mask.sum()))

    expected_keep = int(len(flat_grad) * KEEP_RATIO)
    assert n_keep == pytest.approx(expected_keep, abs=1), (
        f"保留 {n_keep}/{len(flat_grad)} 个坐标（{n_keep / len(flat_grad):.0%}），"
        f"期望 {expected_keep}（{KEEP_RATIO:.0%}）—— mask_ratio 语义反了，"
        f"恶意更新几乎不受限制，Neurotoxin 退化成普通 BadNet。")

    kept = set(np.argsort(flat_grad)[:n_keep].tolist())
    actual = set(np.flatnonzero(flat_mask > 0.5).tolist())
    assert actual == kept, "保留集不是 |benign grad| 最小的那批坐标"


# ══════════════════════════════════════════════════════════════════════════
# 不变量 2：被屏蔽坐标上的上传更新必须**严格为 0**
# ══════════════════════════════════════════════════════════════════════════
def test_masked_coords_have_exactly_zero_update(rng):
    """投影必须是硬投影：mask=0 的坐标上，上传权重与收到的权重逐位相同。"""
    n_feat = 100
    c = _make_client(rng, n_feat)
    g = rng.normal(size=(n_feat, 2)).astype(np.float32)
    _fixed_grads(c, [g])

    w_before = [w.copy() for w in c.model.get_weights()]
    mask = c._compute_mask()
    upload, _, _, _ = c.local_train(round_idx=0)

    for wb, wa, m in zip(w_before, upload, mask):
        blocked = (m < 0.5)
        assert np.array_equal(wa[blocked], wb[blocked]), (
            "被屏蔽坐标上仍出现了非零更新 —— 投影不是硬投影，后门会写到高梯度坐标上")


# ══════════════════════════════════════════════════════════════════════════
# 不变量 3：阈值应逐层取，而非全局
# ══════════════════════════════════════════════════════════════════════════
def test_threshold_is_per_layer(rng):
    """
    官方按层取阈值。全局阈值下，梯度尺度小的层会被整层保留、尺度大的层整层屏蔽，
    掩码退化成「按层开关」而不是「层内选坐标」。
    构造两层量级相差 1000 倍的梯度来暴露这一点。
    """
    n_feat = 100
    c = _make_client(rng, n_feat)
    g_small = (rng.normal(size=(n_feat, 2)) * 1e-3).astype(np.float32)
    _fixed_grads(c, [g_small])
    mask_small_only = c._compute_mask()

    # 每层内部都应保留约 KEEP_RATIO 比例，与该层的绝对尺度无关
    for m in mask_small_only:
        frac = float(m.mean())
        assert frac == pytest.approx(KEEP_RATIO, abs=0.02), (
            f"该层保留比例 {frac:.0%}，与 ratio={KEEP_RATIO:.0%} 不符 —— "
            f"阈值受层间尺度影响，说明用的是全局阈值")
