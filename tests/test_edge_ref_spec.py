"""
tests/test_edge_ref_spec.py  —  方向 2（D-131）参照打分的纯算术（defense/edge_ref_spec.py，不 import TF）

手工构造输入、解析算出正确值：
  · 高斯 KL 的闭式值（同分布 = 0、只差均值、只差方差）、分阶段汇总；
  · 子集矩 = 逐样本矩平均（有放回下标照算）、逐类矩、干净集里没有的类 → NaN；
  · 单纯形投影（Duchi）、混合拟合恢复已知比例且残差 ≈ 0、非混合目标残差 > 0；
  · BNGuard 四数、池内余弦距离中位数；
  · 假客户端只用传入的 Generator（不碰全局 np.random / random）、固定画像下不抽类比例；
  · 参数校验与 [设定14] 往返。
"""

import random

import numpy as np
import pytest

from defense import edge_ref_spec as S

EPS = 1e-5


# ── KL ───────────────────────────────────────────────────────────────────────
def test_gauss_kl_closed_form():
    assert np.allclose(S.gauss_kl([1.0, -2.0], [0.5, 3.0], [1.0, -2.0], [0.5, 3.0], EPS), 0.0)
    # 只差均值：KL = d² / (2 σ²)
    assert S.gauss_kl([1.0], [4.0 - EPS], [0.0], [4.0 - EPS], EPS)[0] == pytest.approx(1.0 / 8.0)
    # 只差方差：KL = ½ (log(q/p) + p/q − 1)
    p, q = 1.0, 2.0
    want = 0.5 * (np.log(q / p) + p / q - 1.0)
    assert S.gauss_kl([0.0], [p - EPS], [0.0], [q - EPS], EPS)[0] == pytest.approx(want)
    # 非对称：KL(p‖q) ≠ KL(q‖p)
    assert S.gauss_kl([0.0], [1.0], [0.0], [2.0], 0.0)[0] != pytest.approx(S.gauss_kl([0.0], [2.0], [0.0], [1.0], 0.0)[0])


def test_kl_scores_average_channels_then_layers_and_group_by_stage():
    layers = [{"stage": "stem", "eps": 0.0}, {"stage": "stage1", "eps": 0.0}, {"stage": "stage1", "eps": 0.0}]
    ref = [(np.zeros(2), np.ones(2)), (np.zeros(4), np.ones(4)), (np.zeros(1), np.ones(1))]
    up = [(np.array([1.0, 0.0]), np.ones(2)),           # 通道 KL 0.5 / 0 → 层 0.25
          (np.full(4, 2.0), np.ones(4)),                  # 每通道 2 → 层 2
          (np.zeros(1), np.ones(1))]                      # 0
    out = S.kl_scores(up, ref, layers)
    assert out["kl_stem"] == pytest.approx(0.25)
    assert out["kl_stage1"] == pytest.approx(1.0)
    assert out["kl_all"] == pytest.approx((0.25 + 2.0 + 0.0) / 3)          # 层等权，不按通道数加权
    assert out["kl_stage2"] is None and out["kl_stage4"] is None


def test_stage_of_uses_the_resnet_block_names():
    assert S.stage_of("stage4_b1_bn2") == "stage4" and S.stage_of("stage1_b0_sc_bn") == "stage1"
    assert S.stage_of("stem_bn") == "stem" and S.stage_of("batch_normalization") == "stem"


# ── 矩 ───────────────────────────────────────────────────────────────────────
def test_subset_moments_are_averages_of_per_sample_moments():
    r = np.random.default_rng(0)
    x = r.normal(size=(6, 5, 5, 3))                      # 6 张图、5×5、3 通道
    m1, m2 = x.mean((1, 2)), (x ** 2).mean((1, 2))
    mu, var = S.subset_moments(m1, m2)
    assert np.allclose(mu, x.mean((0, 1, 2)))
    assert np.allclose(var, x.var((0, 1, 2)))            # 每张图像素数相同 → 精确
    idx = [0, 0, 3]                                       # 有放回
    mu2, var2 = S.subset_moments(m1, m2, idx)
    sub = x[idx]
    assert np.allclose(mu2, sub.mean((0, 1, 2))) and np.allclose(var2, sub.var((0, 1, 2)))


def test_class_moments_and_missing_class_is_nan():
    m1 = np.array([[1.0], [3.0], [10.0]])
    m2 = np.array([[2.0], [10.0], [101.0]])
    mu, ex2 = S.class_moments(m1, m2, [0, 0, 2], 4)
    assert mu[0, 0] == 2.0 and ex2[0, 0] == 6.0 and mu[2, 0] == 10.0
    assert np.isnan(mu[1, 0]) and np.isnan(mu[3, 0])


# ── 混合拟合 ─────────────────────────────────────────────────────────────────
def test_simplex_projection():
    assert np.allclose(S.simplex_project([0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])
    assert np.allclose(S.simplex_project([2.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    assert np.allclose(S.simplex_project([1.0, 1.0]), [0.5, 0.5])
    p = S.simplex_project(np.array([0.9, -0.4, 0.8, 0.1]))
    assert p.min() >= 0 and p.sum() == pytest.approx(1.0)


def test_mixture_fit_recovers_known_proportions():
    r = np.random.default_rng(1)
    A = r.normal(size=(60, 5))
    pi = np.array([0.5, 0.0, 0.3, 0.2, 0.0])
    est, res = S.mixture_fit(A, A @ pi)
    assert np.allclose(est, pi, atol=1e-6) and res < 1e-6
    # 目标在单纯形外（不是任何混合）→ 残差 > 0、解仍在单纯形上
    est2, res2 = S.mixture_fit(A, A @ pi + 3.0)
    assert res2 > 0.1 and est2.min() >= 0 and est2.sum() == pytest.approx(1.0)


def _toy_world(seed=2, n=60, K=3, C=4):
    r = np.random.default_rng(seed)
    y = np.arange(n) % K
    centers = r.normal(size=(K, C)) * 2.0
    m1 = centers[y] + 0.1 * r.normal(size=(n, C))
    m2 = m1 ** 2 + 0.5                                    # 每张图内方差 0.5
    return y, m1, m2


def test_score_one_of_a_class_mixture_has_near_zero_residual_and_finds_the_mix():
    y, m1, m2 = _toy_world()
    layers = [{"stage": "stem", "eps": EPS}]
    props = np.bincount(y) / len(y)
    idx = np.concatenate([np.flatnonzero(y == 0), np.flatnonzero(y == 0), np.flatnonzero(y == 2)])  # 2/3 类 0、1/3 类 2
    up = [S.subset_moments(m1, m2, idx)]
    sc = S.score_one([(m1, m2)], y, 3, up, layers, props)
    assert sc["mix_res"] < 1e-5 and sc["mix_argmax"] == 0 and sc["mix_pmax"] == pytest.approx(2 / 3, abs=1e-4)
    assert sc["mix_res_base"] > sc["mix_res"]             # 按干净集比例（均匀）解释不了
    assert sc["kl_all"] > 0
    # 整个干净集本身：KL = 0、残差 0、base 残差 0
    whole = S.score_one([(m1, m2)], y, 3, [S.subset_moments(m1, m2)], layers, props)
    assert whole["kl_all"] == pytest.approx(0.0, abs=1e-12) and whole["mix_res_base"] < 1e-9


def test_a_shifted_upload_is_not_explained_by_any_class_mix():
    y, m1, m2 = _toy_world()
    layers = [{"stage": "stem", "eps": EPS}]
    props = np.bincount(y) / len(y)
    mu, var = S.subset_moments(m1, m2)
    sc = S.score_one([(m1, m2)], y, 3, [(mu + 5.0, var)], layers, props)
    clean = S.score_one([(m1, m2)], y, 3, [(mu, var)], layers, props)
    assert sc["mix_res"] > 1.0 > clean["mix_res"] + 0.9 and sc["kl_all"] > clean["kl_all"]


# ── 对照 ─────────────────────────────────────────────────────────────────────
def test_bnguard_four_numbers():
    out = S.bnguard4([1.0, 3.0], [2.0, 2.0])
    assert out == {"bng_mm_mean": 2.0, "bng_mm_var": 1.0, "bng_mv_mean": 2.0, "bng_mv_var": 0.0}


def test_cosine_median_to_others():
    pool = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([1.0, 1.0])}
    assert S.cosine_median_to_others(pool, 1) == pytest.approx(np.median([1.0, 1 - 1 / np.sqrt(2)]))
    assert S.cosine_median_to_others({7: np.ones(3)}, 7) is None


# ── 零分布 ───────────────────────────────────────────────────────────────────
def test_fake_clients_use_only_the_given_generator():
    np.random.seed(5)
    random.seed(5)
    g_np, g_py = np.random.get_state()[1].copy(), random.getstate()
    y = np.repeat(np.arange(4), [10, 20, 0, 5])
    pe = np.bincount(y, minlength=4) / len(y)
    a = S.fake_props(np.random.default_rng([1, 2]), pe, 6, {"kind": "dirichlet", "alpha_c": 0.5})
    b = S.fake_props(np.random.default_rng([1, 2]), pe, 6, {"kind": "dirichlet", "alpha_c": 0.5})
    assert np.array_equal(a, b) and np.allclose(a.sum(1), 1.0) and (a[:, 2] == 0).all()   # 没有的类抽不到
    idx = S.fake_indices(np.random.default_rng(3), y, a[0], 40)
    assert len(idx) == 40 and set(y[idx]) <= {0, 1, 3}
    assert np.array_equal(np.random.get_state()[1], g_np) and random.getstate() == g_py


def test_fixed_rule_fake_props_follow_the_profiles_without_randomness():
    pe = np.full(10, 0.1)
    rng = np.random.default_rng(0)
    st = rng.bit_generator.state
    P = S.fake_props(rng, pe, 6, {"kind": "fixed", "lam": 0.4})
    assert rng.bit_generator.state == st                  # 固定画像：不抽随机数
    assert P[0, 0] == pytest.approx(0.6 * 0.1 + 0.2) and np.allclose(P[5], P[0])


def test_client_rule_reads_the_partition():
    assert S.client_rule({"federation": {"design": {"client_profile": "fixed", "profile_lambda": 0.4}}}) == \
        {"kind": "fixed", "lam": 0.4}
    assert S.client_rule({"federation": {"design": {"alpha_client": 0.7}}}) == {"kind": "dirichlet", "alpha_c": 0.7}
    assert S.client_rule({}) == {"kind": "dirichlet", "alpha_c": 0.5}


# ── 参数 / 自描述 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("block,needle", [
    ({"mode": "filter"}, "mode"), ({"n_fake": 0}, "n_fake"), ({"fake_n": True}, "fake_n"),
    ({"cutoff": 1.0}, "不认识"),
])
def test_params_reject(block, needle):
    with pytest.raises(ValueError, match=needle):
        S.edge_ref_params({"defense": {"edge_ref": block}})


def test_settings_row():
    assert S.settings_row({"defense": {"name": "none"}}) == {"edge_ref": False}
    assert S.settings_row({"defense": {"name": "edge_ref", "edge_ref": {"mode": "observe"}}}) == \
        {"edge_ref": True, "mode": "observe", "n_fake": 8, "fake_n": 375, "score": None, "threshold": None}
    assert S.settings_row({"defense": {"name": "edge_ref", "edge_ref": {"mode": "filter", "score": "kl_all",
                                                                          "threshold": 0.004079}}}) == \
        {"edge_ref": True, "mode": "filter", "n_fake": 8, "fake_n": 375, "score": "kl_all", "threshold": 0.004079}


# ── 第二段：过滤（N-014 / D-132）─────────────────────────────────────────────
def test_admit_is_strictly_greater_than_threshold_and_none_is_admitted():
    rows = [{"kl_all": 0.003}, {"kl_all": 0.004079}, {"kl_all": 0.0041}, {"kl_all": None}]
    assert S.admit(rows, "kl_all", 0.004079) == [True, True, False, True]     # 等于 τ 接纳（与 N-013 回放同口径）
    assert S.admit(rows, "kl_all", None) == [True] * 4                          # observe：全部接纳
    assert S.admit([], "kl_all", 0.1) == []


@pytest.mark.parametrize("block,needle", [
    ({"mode": "filter"}, "score"),
    ({"mode": "filter", "score": "kl_all"}, "threshold"),
    ({"mode": "filter", "score": "bng_mm_mean", "threshold": 0.1}, "score"),     # BNGuard 四数不是一维可疑分
    ({"mode": "filter", "score": "kl_all", "threshold": -1.0}, "threshold"),
    ({"mode": "filter", "score": "kl_all", "threshold": float("nan")}, "threshold"),
    ({"mode": "filter", "score": "kl_all", "threshold": True}, "threshold"),
    ({"mode": "observe", "threshold": 0.1}, "只在 mode = filter"),
    ({"mode": "observe", "score": "kl_all"}, "只在 mode = filter"),
])
def test_filter_params_reject(block, needle):
    with pytest.raises(ValueError, match=needle):
        S.edge_ref_params({"defense": {"edge_ref": block}})


def test_filter_params_accept_and_cast():
    p = S.edge_ref_params({"defense": {"edge_ref": {"mode": "filter", "score": "mix_res", "threshold": 1}}})
    assert p["threshold"] == 1.0 and isinstance(p["threshold"], float) and p["score"] == "mix_res"
    assert "admitted" in S.SCORE_FIELDS and set(S.FILTER_SCORES) <= set(S.SCORE_FIELDS)


def test_spec_does_not_import_tensorflow():
    import ast
    import pathlib
    src = pathlib.Path(S.__file__).read_text(encoding="utf-8")
    mods = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    mods |= {(n.module or "").split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert "tensorflow" not in mods
