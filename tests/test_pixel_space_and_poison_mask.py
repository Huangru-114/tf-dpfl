"""
tests/test_pixel_space_and_poison_mask.py  —  A4：G7 的前提（F-027）与 A03（D-016）

纯 numpy，不 import TF，本地秒级。

G7「官方预处理」关掉标准化时，凡是像素空间定义的量（Bad-PFL 的 ε/σ、PGD 的 clamp
范围）都必须跟着变回 [0,1] 口径 —— 否则 ξ 预算放大 1/std ≈ 3.8–4.1 倍。
A03：投毒量从「每批恰好 round(nρ) 个」改为逐样本伯努利(ρ)（官方 fba.py:48）。
"""

import numpy as np
import pytest

from data.pixel_space import (CIFAR10_MEAN, CIFAR10_STD, CIFAR100_STD,
                              normalize_images, pixel_stats, to_input_space,
                              to_unit, valid_range)
from attack.poison_mask import poison_mask

EPS = 4.0 / 255.0


def _cfg(normalize=None, dataset="cifar10"):
    c = {"data": {"dataset": dataset}}
    if normalize is not None:
        c["data"]["normalize"] = normalize
    return c


# ══════════════════════════════════════════════════════════════════════════
# G7 前提：标准化开关
# ══════════════════════════════════════════════════════════════════════════
def test_default_normalization_is_the_historical_expression(rng):
    """默认（不写 data.normalize）与 dataset.py 改动前的表达式逐字节相同。"""
    x = rng.integers(0, 256, size=(5, 4, 4, 3), dtype=np.uint8)
    old = (x.astype("float32") / 255.0 - CIFAR10_MEAN) / CIFAR10_STD
    np.testing.assert_array_equal(normalize_images(x, _cfg()), old)


def test_normalize_off_feeds_plain_unit_pixels(rng):
    x = rng.integers(0, 256, size=(5, 4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(normalize_images(x, _cfg(False)),
                                  x.astype("float32") / 255.0)


def test_eps_follows_the_normalization_switch():
    """反向锚点：开着标准化时 ε_in ≈ 0.0635（= 4/255 ÷ 0.247）；关掉必须精确回到 4/255。"""
    on = to_input_space(EPS, _cfg())
    assert np.allclose(on, EPS / CIFAR10_STD) and abs(float(on[0]) - 0.0635) < 1e-3
    off = to_input_space(EPS, _cfg(False))
    np.testing.assert_array_equal(off, np.full(3, np.float32(EPS)))


def test_eps_uses_the_dataset_matching_std():
    np.testing.assert_allclose(to_input_space(EPS, _cfg(dataset="cifar100")),
                               EPS / CIFAR100_STD, rtol=1e-6)


def test_valid_range_is_the_image_of_unit_interval():
    lo, hi = valid_range(_cfg())
    np.testing.assert_allclose(lo, (0 - CIFAR10_MEAN) / CIFAR10_STD, rtol=1e-6)
    np.testing.assert_allclose(hi, (1 - CIFAR10_MEAN) / CIFAR10_STD, rtol=1e-6)
    lo, hi = valid_range(_cfg(False))
    np.testing.assert_array_equal(lo, np.zeros(3, np.float32))
    np.testing.assert_array_equal(hi, np.ones(3, np.float32))


def test_to_unit_inverts_normalization(rng):
    x = rng.integers(0, 256, size=(3, 4, 4, 3), dtype=np.uint8)
    for cfg in (_cfg(), _cfg(False)):
        np.testing.assert_allclose(to_unit(normalize_images(x, cfg), cfg),
                                   x / 255.0, atol=1e-6)
    m, s = pixel_stats(_cfg(False))
    assert np.all(m == 0) and np.all(s == 1)


# ══════════════════════════════════════════════════════════════════════════
# A03：投毒掩码
# ══════════════════════════════════════════════════════════════════════════
def test_exact_k_reproduces_the_old_rng_sequence():
    """旧口径必须与改动前的代码消耗**同样的** rng 调用（冻结配置重跑逐字节不变）。"""
    r_new, r_old = np.random.default_rng([42, 3]), np.random.default_rng([42, 3])
    for _ in range(50):
        m = poison_mask(r_new, 32, 0.2, "exact_k")
        old = np.zeros(32, dtype=bool)
        old[:int(round(32 * 0.2))] = True
        r_old.shuffle(old)
        np.testing.assert_array_equal(m, old)


def test_exact_k_count_is_constant_reverse_anchor():
    """反向锚点：旧口径在 ρ=0.2、batch 32 下每批恒为 6 个（实际比例 0.1875）。"""
    r = np.random.default_rng(0)
    counts = {int(poison_mask(r, 32, 0.2, "exact_k").sum()) for _ in range(200)}
    assert counts == {6}


def test_bernoulli_rate_matches_rho_within_3_sigma():
    r = np.random.default_rng(1)
    n, B, rho = 32, 10_000, 0.2
    counts = np.array([poison_mask(r, n, rho, "bernoulli").sum() for _ in range(B)])
    frac = counts.sum() / (n * B)
    se = np.sqrt(rho * (1 - rho) / (n * B))
    assert abs(frac - rho) < 3 * se, f"比例 {frac:.5f} 偏离 ρ={rho} 超过 3σ={3*se:.5f}"
    assert counts.var() > 0, "伯努利下每批计数应当有方差"


@pytest.mark.parametrize("mode", ["exact_k", "bernoulli"])
def test_rho_zero_selects_nothing_and_rho_one_selects_all(mode):
    r = np.random.default_rng(2)
    assert not poison_mask(r, 32, 0.0, mode).any()
    assert poison_mask(r, 32, 1.0, mode).all()


def test_rho_zero_does_not_consume_rng():
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    poison_mask(r1, 32, 0.0, "bernoulli")
    poison_mask(r1, 32, 0.0, "exact_k")
    assert r1.random() == r2.random()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        poison_mask(np.random.default_rng(0), 8, 0.5, "per_batch")
