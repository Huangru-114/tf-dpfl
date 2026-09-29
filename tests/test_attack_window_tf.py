"""
tests/test_attack_window_tf.py  —  S4：投毒窗口起点 + 生成器语义在真 Bad-PFL 客户端上的行为（需要 TF）

纯 python 的真值表在 test_attack_window.py；这里验证接线：
  · window（A）：窗口外生成器不更新、不投毒、不耗投毒随机数；窗口内两者都有；
  · always（B）：窗口外生成器照样更新（= ρ=0 影子攻击者），但不投毒、不耗投毒随机数；
  · ρ=0 影子攻击者（S4 缺的两条守卫）：生成器确实被训了；评估触发器用的是生成器（δ 真的加上了）。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_badpfl_official as TBO                           # noqa: E402

START, STOP = 5, 9


def _gen(c):
    c._atk_ensure_generator()
    return [w.copy() for w in c._atk_generator.get_weights()]


def _changed(a, b):
    return any(not np.array_equal(u, v) for u, v in zip(a, b))


def _rng_state(c):
    return repr(c.rng.bit_generator.state)


@pytest.mark.parametrize("sched,round_idx,trains,poisons", [
    ("window", 4, False, False),       # 窗口前
    ("window", 5, True, True),         # 窗口第一轮（含）
    ("window", 8, True, True),
    ("window", 9, False, False),       # 停止轮（不含）
    ("always", 4, True, False),        # B：窗口外只是不投毒
    ("always", 5, True, True),
    ("always", 9, True, False),
])
def test_generator_and_poisoning_gates(sched, round_idx, trains, poisons):
    cfg = TBO._cfg(badpfl_xi="pgd", attack_start_round=START, attack_stop_round=STOP,
                   generator_schedule=sched)
    c, x, y = TBO._client(cfg)
    before = _gen(c)
    c.on_round_start(round_idx)
    assert _changed(before, _gen(c)) is trains
    assert c._attack_active is poisons and c._gen_active is trains

    rng_before = _rng_state(c)
    xb, yb = c.on_batch(x, y)
    if poisons:
        assert c._atk_n_poisoned_batches == 1
        assert not np.array_equal(np.asarray(yb), np.asarray(y))       # ρ=0.2 × 32 张，有样本被改标
    else:
        assert np.array_equal(np.asarray(xb), np.asarray(x))
        assert np.array_equal(np.asarray(yb), np.asarray(y))
        assert _rng_state(c) == rng_before, "窗口外不能消耗投毒随机数（否则 B 与 ρ=0 影子攻击者轨迹不同）"


def test_default_config_is_the_old_single_gate():
    """不写 S4 的两个键 → 生成器闸门 ≡ 投毒闸门（与 S4 之前逐字节一致）。"""
    cfg = TBO._cfg(badpfl_xi="pgd", attack_stop_round=3)
    c, _, _ = TBO._client(cfg)
    for r in (1, 2, 3, 4):
        c.on_round_start(r)
        assert c._gen_active is c._attack_active is (r < 3)


def test_rho0_shadow_attacker_trains_the_generator_but_never_poisons():
    """S4 缺的守卫 ①：ρ=0 时生成器确实被训了（FLR / G0 的 floor 靠它），而且一张都不投。"""
    cfg = TBO._cfg(badpfl_xi="pgd", poison_ratio=0.0)
    c, x, y = TBO._client(cfg)
    before = _gen(c)
    c.on_round_start(1)
    assert _changed(before, _gen(c))
    rng_before = _rng_state(c)
    xb, yb = c.on_batch(x, y)
    assert np.array_equal(np.asarray(xb), np.asarray(x)) and np.array_equal(np.asarray(yb), np.asarray(y))
    assert _rng_state(c) == rng_before and c._atk_n_poisoned_batches == 0


def test_rho0_eval_trigger_uses_the_generator(monkeypatch):
    """S4 缺的守卫 ②：评估触发器 = x + ξ + δ，δ 来自生成器（不回退静态 BadNet，F-006）。"""
    cfg = TBO._cfg(badpfl_xi="pgd", poison_ratio=0.0)
    c, x, y = TBO._client(cfg)
    c.on_round_start(1)
    deltas, xis = [], []
    orig_delta, orig_xi = c.eval_delta, c.eval_xi

    def spy_delta(xx):
        d = orig_delta(xx)
        deltas.append(np.asarray(d))
        return d

    def spy_xi(*a, **kw):          # 同一次调用里记下 ξ —— 不重算（GPU 上两次 PGD 可能在 sign 处不同，F-067）
        v = orig_xi(*a, **kw)
        xis.append(np.asarray(v))
        return v
    monkeypatch.setattr(c, "eval_delta", spy_delta)
    monkeypatch.setattr(c, "eval_xi", spy_xi)
    xt = c.eval_trigger(c.model, x, y, rng=np.random.default_rng(0))
    assert len(deltas) == 1 and len(xis) == 1 and np.abs(deltas[0]).max() > 0
    np.testing.assert_allclose(xt, np.asarray(x) + xis[0] + deltas[0], atol=1e-6)
