"""
阶段三 SA0：`fedavg/hardening/core.py`（edge 侧对抗训练的 TF 部分）。需要 TF；本地无 TF 时 skip、集群上跑。

- 扰动在像素空间的 L∞ 预算内、在合法像素范围内（PGD / TRADES / SAU / AT-tgt）；
- SAU 的四条（审查 R1–R4）：贴边坐标照样有梯度（投影而不是在 tape 里裁剪）、不增强、Adam(amsgrad, ε 1e-8) 与 torch 公式逐步对上、
  参照模型的探针在推理模式特征上拟合；
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
        assert np.all(xa.numpy() >= lo - 1e-6) and np.all(xa.numpy() <= hi + 1e-6), k


# ── SAU：审查 R1–R4 ───────────────────────────────────────────────────────────
def _random_heads(h, seed=3):
    r = np.random.default_rng(seed)
    d = int(h.feat.output_shape[-1])
    h.W.assign(r.normal(size=(d, NCLS)).astype(np.float32))
    h.b.assign(np.zeros(NCLS, np.float32))
    return tf.constant(r.normal(size=(d, NCLS)).astype(np.float32)), tf.zeros(NCLS)


def test_sau_boundary_coordinates_still_move_r1():
    """
    init max 下 x0 + ε 超出合法范围的坐标：投影后起点 = hi − x0；一步之后，梯度要求下降的那些坐标必须走到 hi − x0 − ε，
    其余停在 hi − x0。旧实现在求梯度的前向里裁剪 → 这些坐标梯度恒为 0 → 一个都不动（改动前 moved = 0，已实测）。
    """
    m, h = _hardener()
    h.model.set_weights(m.get_weights())
    h.ref.set_weights(m.get_weights())
    Wr, br = _random_heads(h)
    eps_in = h.to_in(8 / 255)
    lo, hi = spec.valid_range(CFG)
    r = np.random.default_rng(4)
    x = (hi - r.uniform(0.05, 0.95, size=(16, TEI.IMG, TEI.IMG, 3)) * eps_in.numpy()).astype(np.float32)
    y = (np.arange(16) % 4).astype(np.int64)
    p1 = h.sau_pert(tf.constant(x), tf.constant(y), eps_in, eps_in, 1, False, Wr, br, 0.01, 1.0).numpy()
    room = hi - x                                               # 全部坐标都贴边：room < ε
    assert np.all(room < eps_in.numpy())
    moved = np.isclose(p1, room - eps_in.numpy(), atol=1e-6)
    stayed = np.isclose(p1, room, atol=1e-6)
    assert np.all(moved | stayed)
    assert moved.mean() > 0.1, moved.mean()
    xa = x + p1
    assert np.all(xa >= lo - 1e-6) and np.all(xa <= hi + 1e-6)


def test_sau_does_not_augment_but_the_others_do_r2(monkeypatch):
    calls = []
    orig = C.augment_batch
    monkeypatch.setattr(C, "augment_batch", lambda xb, rng: calls.append(len(xb)) or orig(xb, rng))
    m, h = _hardener()
    x, y = _data()
    _, info = h.harden(m.get_weights(), x, y, BY["sau-e8-frozen-low"], rng=np.random.default_rng(0), lr=0.05)
    assert calls == [] and info["augment"] is False
    _, info = h.harden(m.get_weights(), x, y, BY["pgd-e8-frozen-low"], rng=np.random.default_rng(0), lr=0.05)
    assert len(calls) == info["steps"] == 2 and info["augment"] is True


def _torch_adam(grads, lr, amsgrad, b1=0.9, b2=0.999, eps=1e-8):
    """torch.optim.Adam 的单变量更新（torch/optim/adam.py 的 _single_tensor_adam），float64。"""
    p = np.zeros_like(grads[0], dtype=np.float64)
    m, v, vmax, out = np.zeros_like(p), np.zeros_like(p), np.zeros_like(p), []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        vmax = np.maximum(vmax, v)
        denom = np.sqrt(vmax if amsgrad else v) / np.sqrt(1 - b2 ** t) + eps
        p = p - lr / (1 - b1 ** t) * m / denom
        out.append(p.copy())
    return out


def test_sau_optimizer_matches_torch_amsgrad_r3():
    cfg = BY["sau-e8-frozen-low"]
    c = C.make_optimizer(cfg, None).get_config()
    assert c["amsgrad"] is True and c["epsilon"] == 1e-8 and (c["beta_1"], c["beta_2"]) == (0.9, 0.999)
    assert float(c["learning_rate"]) == pytest.approx(1e-4)
    r = np.random.default_rng(0)
    # 前 10 步 |g| ~ 1e-2、之后 ~ 1e-4：v 掉下来 → AMSGrad 的 max 起作用（不开 AMSGrad 的 torch 公式在这组上差 0.035 lr）
    grads = [r.normal(size=6) * (1e-2 if t < 10 else 1e-4) for t in range(60)]
    want = _torch_adam(grads, 1e-4, amsgrad=True)

    def worst(opt):                                             # 以「一个名义步长 lr」为单位（相对误差在参数过 0 时没有意义）
        v, w_ = tf.Variable(np.zeros(6, np.float32)), 0.0
        for g, w in zip(grads, want):
            opt.apply_gradients([(tf.constant(g, tf.float32), v)])
            w_ = max(w_, float(np.max(np.abs(v.numpy() - w))) / 1e-4)
        return w_
    assert worst(C.make_optimizer(cfg, None)) < 2e-3            # 实测 3.8e-4 lr（Keras 的 ε 加在未校正的 √v 上）
    # 反向锚点：旧实现（Keras 缺省 Adam：无 AMSGrad、ε 1e-7）对不上（实测 0.035 lr）
    assert worst(tf.keras.optimizers.Adam(learning_rate=1e-4)) > 2e-2
    assert C.make_optimizer(BY["pgd-e8-frozen-low"], 0.05).get_config()["name"] == "SGD"
    with pytest.raises(ValueError):
        C.make_optimizer(BY["pgd-e8-frozen-low"], None)


def test_sau_reference_probe_is_fitted_on_inference_features_r4(monkeypatch):
    seen = []
    orig = C.spec.fit_probe
    monkeypatch.setattr(C.spec, "fit_probe", lambda F, *a, **k: seen.append(np.array(F, copy=True)) or orig(F, *a, **k))
    m, h = _hardener()
    x, y = _data()
    _, info = h.harden(m.get_weights(), x, y, BY["sau-e8-train-low"], rng=np.random.default_rng(0), lr=0.05)
    assert len(seen) == 2                                       # 主模型（训练模式特征）+ 参照模型（推理模式特征）
    inf = C.extract(h.feat_ref, x)                              # 参照模型的权重 = 输入的权重（从未被训练）
    np.testing.assert_array_equal(seen[1], inf)
    assert not np.allclose(seen[0], inf)                        # 训练模式用 batch 统计：与推理模式不同
    assert info["ref_acc"] == orig(inf, y, NCLS, steps=20, lr=0.05)["train_acc"]
    seen.clear()
    _, info = h.harden(m.get_weights(), x, y, BY["sau-e8-frozen-low"], rng=np.random.default_rng(0), lr=0.05)
    assert len(seen) == 1 and info["ref_acc"] == info["probe_acc"]     # bn=frozen：同一份推理模式特征，共用探针


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
