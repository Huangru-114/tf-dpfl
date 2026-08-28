"""
tests/test_probe_determinism.py  —  **Exp 0.3 的门禁测试**（需 TF）。

这个文件守的是整个实验的地基：Δθ_BD = Δθ_poison − Δθ_clean 只有在「两次训练除了
投毒开关之外完全相同」时才有意义。本仓库原本有三处泄漏（见 probe/determinism.py
的模块 docstring）。这里逐条钉死。

**如果这些测试红了，Exp 0.3 的所有数字都不可读** —— 而不可读的表现是
「‖Δθ_BD‖ ≈ ‖Δθ_stochastic‖」，与「后门无法定位」这个科学结论外观完全一样。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

# 本文件的断言是**逐比特**的，前提是跑在 CPU 上 —— 由 tests/conftest.py 统一钉死
# （那里写了为什么不能靠 TF 的确定性开关在 GPU 上救它）。这里加一条自检：设备没被
# 钉住时直接 skip，而不是产出一堆看起来像回归、实为环境问题的红。
if tf.config.get_visible_devices("GPU"):
    pytest.skip("本文件的逐比特断言要求 CPU；GPU 可见（conftest 的钉死没生效）→ skip",
                allow_module_level=True)

from client.client_badpfl import BadPFLMixin          # noqa: E402
from client.compose import compose_client_class       # noqa: E402
from client.hier_fedrep import HierFedRepClient       # noqa: E402
from probe.determinism import ClientSnapshot, FrozenOrder   # noqa: E402
from probe.paired import (assert_attack_is_hook_gated, probe_client,   # noqa: E402
                          single_trajectory)

IMG, NC, BS = 16, 3, 8      # img_size 必须是 16 的倍数：build_autoencoder 的断言


def _config():
    return {
        "seed": 42,
        "data": {"dataset": "cifar10", "img_size": IMG, "batch_size": BS,
                 "num_classes": NC},
        "training": {"learning_rate": 0.05, "lr_decay": 0.99, "local_epochs": 1,
                     "plocal_epochs": 1, "head_lr_rep": 0.01},
        "backdoor": {"target_label": 2, "poison_ratio": 0.5, "badpfl_gen_steps": 2},
    }


def _model():
    """
    末层必须带 softmax：本仓库是 `from_logits=False`（陷阱 #6）。忘了加 activation
    的话梯度恒为 0、训练什么都不做，而断言会在比较两个全零数组 → **假绿**。
    """
    L = tf.keras.layers
    inp = L.Input((IMG, IMG, 3))
    x = L.Conv2D(4, 3, padding="same", name="c1")(inp)
    x = L.BatchNormalization(name="bn1")(x)
    x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, L.Dense(NC, activation="softmax", name="head")(x))


def _dataset(seed=0, n=32):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, IMG, IMG, 3)).astype("float32")
    y = r.integers(0, NC, n).astype("int32")
    # 刻意复刻 data/dataset.py:175 的 reshuffle_each_iteration=True —— 那正是泄漏源之一
    return (tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(n, reshuffle_each_iteration=True)
            .batch(BS, drop_remainder=True))


def _client(malicious=True, cid=0):
    cls = compose_client_class(HierFedRepClient, None, BadPFLMixin if malicious else None)
    c = cls(client_id=cid, dataset=_dataset(cid), model=_model(), config=_config())
    c.is_malicious = malicious
    return c


def _theta():
    return [w.copy() for w in _model().get_weights()]


def _maxdiff(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def _l2(ws):
    return float(np.sqrt(sum(float((np.asarray(w) ** 2).sum()) for w in ws)))


# ══════════════════════════════════════════════════════════════════════════
# 判据 1：同 order_seed 的两条 clean 轨迹必须**逐比特**一致
# ══════════════════════════════════════════════════════════════════════════

def test_same_order_seed_gives_bitwise_identical_clean_trajectories():
    c, th = _client(), _theta()
    a = single_trajectory(c, th, 1, order_seed=7, attack=False)
    b = single_trajectory(c, th, 1, order_seed=7, attack=False)
    assert _maxdiff(a["upload"], b["upload"]) < 1e-6
    assert _maxdiff(a["personal"], b["personal"]) < 1e-6


def test_same_order_seed_is_reproducible_for_the_poison_trajectory_too():
    """poison twin 也必须可复现 —— 否则 Δθ_BD 每次重跑都不一样。"""
    c, th = _client(), _theta()
    a = single_trajectory(c, th, 1, order_seed=7, attack=True)
    b = single_trajectory(c, th, 1, order_seed=7, attack=True)
    assert _maxdiff(a["upload"], b["upload"]) < 1e-6


def test_frozen_order_is_not_a_no_op():
    """
    反向守卫：不同 order_seed **必须**给出不同轨迹。
    若 FrozenOrder 退化成「总是同一个顺序」，上面那条会平凡通过，而 clean-vs-clean
    control 会恒为 0 → 噪声地板被伪造成零，判据 2 的比值恒为 inf。
    """
    c, th = _client(), _theta()
    a = single_trajectory(c, th, 1, order_seed=7, attack=False)
    b = single_trajectory(c, th, 1, order_seed=999, attack=False)
    assert _maxdiff(a["upload"], b["upload"]) > 1e-9


def test_trajectory_actually_trains():
    """
    起点与终点必须不同。忘了 softmax 之类的失误会让梯度恒为 0，
    上面所有"一致性"断言都会在比较两个全零数组上平凡通过（假绿）。
    """
    c, th = _client(), _theta()
    a = single_trajectory(c, th, 1, order_seed=7, attack=False)
    assert _maxdiff(a["start"], a["upload"]) > 1e-9


# ══════════════════════════════════════════════════════════════════════════
# 判据 2/3：投毒开关确实是开关
# ══════════════════════════════════════════════════════════════════════════

def test_poison_trajectory_differs_from_clean_for_a_malicious_client():
    c, th = _client(malicious=True), _theta()
    a = single_trajectory(c, th, 1, order_seed=7, attack=False)
    p = single_trajectory(c, th, 1, order_seed=7, attack=True)
    assert _maxdiff(a["upload"], p["upload"]) > 1e-6


def test_benign_client_has_exactly_zero_bd_displacement():
    """
    没有攻击 mixin 的客户端，翻 `is_malicious` 是无操作 → 两条轨迹逐字节相同
    → Δθ_BD 必须**精确**为 0。非零就说明批序冻结或状态还原漏了东西（仪表坏了）。
    """
    c, th = _client(malicious=False, cid=1), _theta()
    res = probe_client(c, th, 1, seed_a=7, seed_b=8)
    for space in ("upload", "personal"):
        assert _l2(res["deltas"][space]["bd"]) == 0.0


def test_benign_client_still_has_nonzero_stochastic_control():
    """良性端的 Δθ_stochastic 是噪声地板本身，它必须非零才有比较的意义。"""
    c, th = _client(malicious=False, cid=1), _theta()
    res = probe_client(c, th, 1, seed_a=7, seed_b=8)
    assert _l2(res["deltas"]["upload"]["stochastic"]) > 0.0


def test_malicious_bd_displacement_exceeds_the_stochastic_floor():
    c, th = _client(malicious=True), _theta()
    res = probe_client(c, th, 1, seed_a=7, seed_b=8)
    d = res["deltas"]["upload"]
    assert _l2(d["bd"]) > _l2(d["stochastic"])


def test_probe_client_deltas_satisfy_their_definitions():
    """bd == poison − clean，stochastic == clean_A − clean_B，逐元素。"""
    c, th = _client(), _theta()
    res = probe_client(c, th, 1, seed_a=7, seed_b=8)
    for space in ("upload", "personal"):
        d = res["deltas"][space]
        for bd, p, cl in zip(d["bd"], d["poison"], d["clean"]):
            np.testing.assert_allclose(bd, p - cl, rtol=0, atol=0)


def test_probe_client_reports_the_original_role_not_the_toggled_one():
    c, th = _client(malicious=True), _theta()
    res = probe_client(c, th, 1, seed_a=7, seed_b=8)
    assert res["was_malicious"] is True
    assert c.is_malicious is True          # 探测完必须还原


# ══════════════════════════════════════════════════════════════════════════
# ClientSnapshot：漏一项就静默污染下一个 twin
# ══════════════════════════════════════════════════════════════════════════

def test_snapshot_restores_model_weights():
    c, th = _client(), _theta()
    before = [w.copy() for w in c.model.get_weights()]
    with ClientSnapshot(c):
        with FrozenOrder(c, 7):
            c.set_weights(th, th)
            c.local_train(1)
    assert _maxdiff(before, c.model.get_weights()) == 0.0


def test_snapshot_restores_the_shared_badpfl_generator():
    """
    共享 generator 是**跨轮累积的攻击者状态**（main.py 让全体恶意端共享同一对象）。
    poison twin 会在它上面训 badpfl_gen_steps 步；不还原就污染之后所有客户端的探测。
    """
    c, th = _client(malicious=True), _theta()
    c._atk_ensure_generator()
    before = [w.copy() for w in c._atk_generator.get_weights()]
    single_trajectory(c, th, 1, order_seed=7, attack=True)
    assert _maxdiff(before, c._atk_generator.get_weights()) == 0.0


def test_generator_really_is_trained_inside_the_poison_twin():
    """反向守卫：若生成器根本没被训练，上一条会平凡通过。"""
    c, th = _client(malicious=True), _theta()
    c._atk_ensure_generator()
    before = [w.copy() for w in c._atk_generator.get_weights()]
    with FrozenOrder(c, 7):
        c.set_weights(th, th)
        c.on_round_start(1)
    assert _maxdiff(before, c._atk_generator.get_weights()) > 0.0


def test_snapshot_restores_the_client_rng_stream():
    """Bad-PFL 的投毒 mask 消耗 self.rng；不还原，两条轨迹之后的随机流就错位。"""
    c, th = _client(malicious=True), _theta()
    before = c.rng.bit_generator.state
    single_trajectory(c, th, 1, order_seed=7, attack=True)
    assert c.rng.bit_generator.state == before


def test_snapshot_restores_is_malicious():
    c, th = _client(malicious=True), _theta()
    single_trajectory(c, th, 1, order_seed=7, attack=False)
    assert c.is_malicious is True


def test_snapshot_restores_the_private_head():
    """FedRep 的私有 head 是持久个性化状态，跨 twin 泄漏会让第二条轨迹起点不同。"""
    c, th = _client(), _theta()
    c.set_weights(th, th)
    c.local_train(1)
    before = [w.copy() for w in c._head_weights]
    single_trajectory(c, th, 2, order_seed=7, attack=False)
    assert _maxdiff(before, c._head_weights) == 0.0


def test_snapshot_restores_optimizer_state_even_when_built_inside():
    """
    探针启动时优化器 slot 还没建出来；第一条轨迹才建。按位置还原会对不上号，
    按名字还原并把「快照时不存在的变量」置零才等价于"从未训练"。
    """
    c, th = _client(), _theta()
    single_trajectory(c, th, 1, order_seed=7, attack=False)
    v = c._base_opt.variables
    iters = [x for x in (v() if callable(v) else v) if "iter" in x.name]
    assert iters, "SGD 应该至少有 iterations 变量"
    assert all(int(x.numpy()) == 0 for x in iters)


def test_frozen_order_restores_the_dataset_object():
    c = _client()
    ds = c.dataset
    with FrozenOrder(c, 7):
        assert c.dataset is not ds
    assert c.dataset is ds


def test_frozen_order_restores_shuffled_batches_method():
    c = _client()
    with FrozenOrder(c, 7):
        pass
    assert "_shuffled_batches" not in c.__dict__       # 实例属性遮蔽已撤掉
    assert callable(c._shuffled_batches)


# ══════════════════════════════════════════════════════════════════════════
# 入口守卫
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy", ["badpfl", "neurotoxin", "cerp"])
def test_hook_gated_strategies_are_accepted(strategy):
    assert_attack_is_hook_gated({"backdoor": {"malicious_strategy": strategy}})


@pytest.mark.parametrize("strategy", ["vanilla", "badnet", "blended", "dba"])
def test_dataset_baked_strategies_are_rejected(strategy):
    """
    vanilla 的投毒烘焙在 client.dataset 里，`is_malicious` 对它不是开关 →
    Δθ_BD 恒为 0。那个 0 与"后门无位移"外观相同，是最坏的一类假阴性，必须在入口拦掉。
    """
    with pytest.raises(ValueError, match="不经 on_batch"):
        assert_attack_is_hook_gated({"backdoor": {"malicious_strategy": strategy}})
