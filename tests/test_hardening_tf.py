"""
阶段三 SA0：`fedavg/hardening/core.py`（edge 侧对抗训练的 TF 部分）。需要 TF；本地无 TF 时 skip、集群上跑。

- 扰动在像素空间的 L∞ 预算内、在合法像素范围内（PGD / TRADES / SAU / AT-tgt）；
- 一次加固只改 body 的可训练变量：head 与 BN moving 统计量逐位不变、传入的权重不被改、Python random 状态不变；
- 同一个 rng 种子 → 逐位相同；
- loss 的三个小块 = numpy 手写公式（TRADES 的 KL、SAU 的 JS 与共享项）；
- k* 并列时不偏向类 0；
- GPU 确定性的模拟检查下（陷阱 #23）两种 BN 模式都跑得通（`resnet10_torch`）；
- 反向锚点：去掉 random 围栏，构造草稿模型就会推进 Python random（F-078）。
"""

import random

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                         # noqa: E402
from test_bn_inference_determinism import gpu_determinism_check   # noqa: E402,F401  （fixture）
from test_eval_grid_tf import _prod_seeding                  # noqa: E402

from hardening import core as C                              # noqa: E402
from hardening import spec                                   # noqa: E402
from models.cnn import build_resnet10_torch, get_bn_stat_indices, get_base_head_indices   # noqa: E402

NCLS = 10
CFG = {"seed": 42, "data": {"dataset": "cifar10", "num_classes": NCLS, "img_size": TEI.IMG}}
GRID = spec.expand_grid({"bn": ["train", "frozen"],
                         "budgets": {"low": {"epochs": 1, "pgd_steps": 2}},
                         "objectives": {"pgd": {"eps_px": ["8/255"]}, "trades": {"eps_px": ["8/255"], "beta": [6]},
                                        "sau": {"eps_px": ["8/255"]}, "tgt": {"eps_px": ["4/255"], "sigma_px": "4/255"},
                                        "cft": {}}})
BY = {c["id"]: c for c in GRID}


def _data(n=16, img=TEI.IMG, seed=0):
    r = np.random.default_rng(seed)
    lo, hi = spec.valid_range(CFG)
    x = np.clip(r.normal(size=(n, img, img, 3)), lo, hi).astype(np.float32)
    y = (np.arange(n) % 4).astype(np.int64)
    return x, y


def _hardener(model=None, batch=8):
    m = model if model is not None else TEI._model()
    return m, C.EdgeHardener(m, CFG, probe={"steps": 20, "lr": 0.05}, kstar={"n": 8, "eps_px": "4/255", "steps": 2},
                             batch=batch)


def _px(d):
    return np.max(np.abs(np.asarray(d) * spec.to_input_space(1.0, CFG) ** -1))


# ── 扰动预算 ─────────────────────────────────────────────────────────────────
def test_perturbations_stay_in_the_pixel_budget_and_valid_range():
    m, h = _hardener()
    h.model.set_weights(m.get_weights())
    x, y = _data()
    xt, yt = tf.constant(x), tf.constant(y)
    eps, eps4 = 8 / 255, 4 / 255
    lo, hi = spec.valid_range(CFG)
    rng = np.random.default_rng(0)
    cases = {
        "pgd": h.pgd_ce(xt, yt, h.to_in(eps), h.to_in(spec.step_px(BY["pgd-e8-train-low"])), 3, False,
                        rng=rng, start="uniform"),
        "trades": h.trades_adv(xt, h.to_in(eps), h.to_in(0.01), 3, False, rng),
        "tgt": h.tgt_adv(xt, yt, 1, h.to_in(eps4), h.to_in(0.01), 3, h.to_in(eps4), False),
        "sau": xt + h.sau_pert(xt, yt, h.to_in(eps), h.to_in(eps), 3, False, h.W, h.b, 0.01, 1.0),
    }
    budgets = {"pgd": eps, "trades": eps, "tgt": 2 * eps4, "sau": eps}
    for k, xa in cases.items():
        d = (xa.numpy() - x) * np.array([0.2470, 0.2435, 0.2616])
        assert np.max(np.abs(d)) <= budgets[k] + 1e-6, k
        if k != "sau":                                          # SAU 的扰动本身不裁剪；训练时 x + 扰动再裁剪
            assert np.all(xa.numpy() >= lo - 1e-6) and np.all(xa.numpy() <= hi + 1e-6), k


# ── 一次加固只改 body ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("cid", [c["id"] for c in GRID])
def test_harden_changes_only_body_trainables_and_no_outside_state(cid):
    m, h = _hardener()
    w0 = [np.array(w, copy=True) for w in m.get_weights()]
    keep = [np.array(w, copy=True) for w in w0]
    x, y = _data()
    st = random.getstate()
    new, info = h.harden(w0, x, y, BY[cid], rng=np.random.default_rng(1), lr=0.05)
    assert random.getstate() == st                              # 围栏：Python random 没被推进
    for a, b in zip(w0, keep):                                  # 传进来的权重没被改
        np.testing.assert_array_equal(a, b)
    split = get_base_head_indices(m, NCLS)
    for i in split["head_weight_indices"] + get_bn_stat_indices(m):
        np.testing.assert_array_equal(new[i], w0[i])            # head 与 BN 统计量逐位不变
    changed = [i for i in split["base_weight_indices"] if not np.array_equal(new[i], w0[i])]
    assert changed and set(changed) <= set(split["base_weight_indices"]) - set(get_bn_stat_indices(m))
    assert info["steps"] == 2 and np.isfinite(info["loss_last"])
    if cid.startswith("tgt"):
        assert info["kstar"] in range(NCLS)


def test_harden_is_deterministic_given_the_rng():
    m, h = _hardener()
    x, y = _data()
    cfg = BY["pgd-e8-train-low"]
    a, _ = h.harden(m.get_weights(), x, y, cfg, rng=np.random.default_rng(5), lr=0.05)
    b, _ = h.harden(m.get_weights(), x, y, cfg, rng=np.random.default_rng(5), lr=0.05)
    c, _ = h.harden(m.get_weights(), x, y, cfg, rng=np.random.default_rng(6), lr=0.05)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert any(not np.array_equal(u, v) for u, v in zip(a, c))


def test_head_init_skips_the_probe():
    m, h = _hardener()
    x, y = _data()
    d = int(h.feat.output_shape[-1])
    W, b = np.zeros((d, NCLS), np.float32), np.zeros(NCLS, np.float32)
    _, info = h.harden(m.get_weights(), x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=0.05,
                       head_init=(W, b))
    assert info["probe_acc"] is None and info["head"] == "given"


# ── loss 小块 = 手写公式 ─────────────────────────────────────────────────────
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_loss_pieces_match_numpy():
    r = np.random.default_rng(0)
    la, lb = r.normal(size=(5, NCLS)), r.normal(size=(5, NCLS))
    p, q = _softmax(la), _softmax(lb)
    want_kl = np.sum(p * (np.log(p) - np.log(q)))
    assert float(C.kl_sum(tf.constant(p, tf.float32), tf.constant(lb, tf.float32))) == pytest.approx(want_kl, rel=1e-4)
    pc, qc = np.clip(p, 1e-8, 1), np.clip(q, 1e-8, 1)
    mix = (pc + qc) / 2
    want_js = np.sum(0.5 * (pc * np.log(pc) + qc * np.log(qc)) - 0.5 * (pc * np.log(mix) + qc * np.log(mix)), axis=1)
    np.testing.assert_allclose(C.js_rows(tf.constant(la, tf.float32), tf.constant(lb, tf.float32)).numpy(),
                               want_js, rtol=1e-4, atol=1e-7)
    lab = np.array([0, 3, 3, 9, 1])
    pois = np.array([True, False, True, True, False])
    pos = q[np.arange(5), lab]
    neg = 1 - pos
    term = -np.log(1e-6 + np.minimum(neg, 0.999)) - np.log(1 + 1e-6 - np.maximum(pos, 0.001))
    want_sh = np.sum(term[pois]) / 2 / 5
    got = float(C.sau_shared(tf.constant(lb, tf.float32), tf.constant(lab), tf.constant(pois), NCLS))
    assert got == pytest.approx(want_sh, rel=1e-4)


def test_kstar_ties_are_broken_by_the_rng_not_by_index():
    assert C.EdgeHardener.pick_kstar([0.3, 0.1, 0.2], np.random.default_rng(0)) == (1, [1])
    picks = {C.EdgeHardener.pick_kstar([0.0, 0.0, 0.0, None], np.random.default_rng(s))[0] for s in range(30)}
    assert picks == {0, 1, 2}                                   # np.argmin 会永远选 0（= 目标类）
    assert C.EdgeHardener.pick_kstar([None, None], np.random.default_rng(0)) == (None, [])


# ── GPU 确定性（陷阱 #23）与 random 围栏的反向锚点 ─────────────────────────────
@pytest.mark.parametrize("cid", ["pgd-e8-frozen-low", "sau-e8-frozen-low", "tgt-e4-frozen-low",
                                 "trades-e8-b6-train-low"])
def test_resnet10_torch_survives_the_emulated_gpu_check(gpu_determinism_check, cid):
    m = build_resnet10_torch((32, 32, 3), NCLS)
    cfg = dict(CFG, data=dict(CFG["data"], img_size=32))
    h = C.EdgeHardener(m, cfg, probe={"steps": 5, "lr": 0.05}, kstar={"n": 8, "eps_px": "4/255", "steps": 1},
                       batch=8)
    x, y = _data(n=8, img=32)
    new, info = h.harden(m.get_weights(), x, y, BY[cid], rng=np.random.default_rng(0), lr=0.05)
    assert info["steps"] == 1 and all(np.isfinite(w).all() for w in new)


def test_without_the_fence_constructing_the_scratch_models_moves_python_random(monkeypatch):
    _prod_seeding(monkeypatch)                                  # main.py 的播种：clone_model 从 Python random 取种子
    m = TEI._model()
    _prod_seeding(monkeypatch)
    random.seed(7)
    st = random.getstate()
    C.EdgeHardener(m, CFG)
    assert random.getstate() == st
    monkeypatch.setattr(random, "setstate", lambda s: None)     # 去掉围栏
    C.EdgeHardener(m, CFG)
    assert random.getstate() != st
