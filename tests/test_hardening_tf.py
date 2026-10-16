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
    """探针 head 用随机值（全零 head 的梯度为 0 → 扰动不动、预算断言假绿，第二轮审查）；另断言每种扰动真的动了。"""
    m, h = _hardener()
    h.model.set_weights(m.get_weights())
    h.ref.set_weights(m.get_weights())
    Wr, br = _random_heads(h)
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
        "sau": xt + h.sau_pert(xt, yt, h.to_in(eps), h.to_in(eps), 3, False, Wr, br, 0.01, 1.0),
    }
    # 不带随机起点 / 只靠梯度的那几种：起点之外真的走了
    base = {"pgd": None, "trades": None, "tgt": xt, "sau": xt + h._project_pert(tf.zeros_like(xt) + h.to_in(eps), xt,
                                                                                   h.to_in(eps))}
    budgets = {"pgd": eps, "trades": eps, "tgt": 2 * eps4, "sau": eps}
    for k, xa in cases.items():
        d = (xa.numpy() - x) * np.array([0.2470, 0.2435, 0.2616])
        assert np.max(np.abs(d)) <= budgets[k] + 1e-6, k
        assert np.all(xa.numpy() >= lo - 1e-6) and np.all(xa.numpy() <= hi + 1e-6), k
        assert np.max(np.abs(d)) > 0.5 / 255, k                 # 扰动不是零（梯度确实回传了）
        if base[k] is not None:
            assert not np.allclose(xa.numpy(), base[k].numpy()), k


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


# ── 第二轮审查（D-105）：各目标的训练 loss、AT-tgt 掩码、中途写回、lr、随机数配对 ─────────
def _ce(z, y):
    z = z - z.max(axis=1, keepdims=True)
    return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1)))


@pytest.mark.parametrize("cid", ["pgd-e8-frozen-low", "trades-e8-b6-frozen-low", "sau-e8-frozen-low",
                                 "tgt-e4-frozen-low", "cft-frozen-low"])
def test_training_loss_of_each_objective_is_the_documented_combination(cid, monkeypatch):
    """对抗样本换成固定张量 → 闭包的值 = 用 h.logits 手写的组合（pgd 不能退化成 C-ft、TRADES 的 β·KL/B、SAU 的 λ 与掩码）。"""
    m, h = _hardener()
    h.model.set_weights(m.get_weights())
    h.ref.set_weights(m.get_weights())
    Wr, br = _random_heads(h)
    x, y = _data(n=8)
    xb, yb = tf.constant(x), tf.constant(y)
    xa = tf.constant(_data(n=8, seed=11)[0])                     # 「对抗样本」= 另一批图（只为让两组 logits 明显不同）
    monkeypatch.setattr(h, "pgd_ce", lambda *a, **k: xa)
    monkeypatch.setattr(h, "trades_adv", lambda *a, **k: xa)
    monkeypatch.setattr(h, "tgt_adv", lambda *a, **k: xa)
    monkeypatch.setattr(h, "sau_pert", lambda *a, **k: xa - xb)
    cfg = BY[cid]
    got = float(h.batch_loss(cfg, xb, yb, False, rng=np.random.default_rng(0), kstar=1, ref=(Wr, br))())
    ln, la = h.logits(xb, False).numpy(), h.logits(xa, False).numpy()
    obj = cfg["objective"]
    if obj in ("pgd", "tgt"):
        want = _ce(la, y).mean()
        assert abs(want - _ce(ln, y).mean()) > 1e-4             # 能分辨「只在对抗样本上训」与 C-ft
    elif obj == "cft":
        want = _ce(ln, y).mean()
    elif obj == "trades":
        p, q = _softmax(ln), _softmax(la)
        want = _ce(ln, y).mean() + 6.0 * np.sum(p * (np.log(p) - np.log(q))) / len(y)
    else:
        s = cfg["sau"]
        lrc, lrp = h.logits_ref(xb, Wr, br).numpy(), h.logits_ref(xa, Wr, br).numpy()
        plr = lrp.argmax(1)
        pois = (plr != y) & (plr != lrc.argmax(1))
        q = _softmax(la)
        pos = q[np.arange(len(y)), plr]
        term = -np.log(1e-6 + np.minimum(1 - pos, 0.999)) - np.log(1 + 1e-6 - np.maximum(pos, 0.001))
        want = (s["lmd_1"] * _ce(ln, y).mean() + s["lmd_2"] * _ce(la, y).mean()
                + s["lmd_3"] * np.sum(term[pois]) / 2 / len(y))
    assert got == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_tgt_skips_the_targeted_step_for_samples_already_in_kstar_d105():
    """y == k* 的行只做 FGSM：扰动 = σ·sign(∇CE(y)) 投影（定向下降会降低它们的 CE(y)）。"""
    m, h = _hardener()
    h.model.set_weights(m.get_weights())
    _random_heads(h)
    x, y = _data(n=8)
    kstar = 1
    eps4, sig = h.to_in(4 / 255), h.to_in(4 / 255)
    xa = h.tgt_adv(tf.constant(x), tf.constant(y), kstar, eps4, h.to_in(0.01), 3, sig, False).numpy()
    rows = y == kstar
    assert rows.any() and (~rows).any()
    with tf.GradientTape() as tape:
        xt = tf.constant(x)
        tape.watch(xt)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.constant(y),
                                                                            logits=h.logits(xt, False)))
    want = np.clip(x + sig.numpy() * np.sign(tape.gradient(loss, xt).numpy()), *spec.valid_range(CFG))
    np.testing.assert_allclose(xa[rows], want[rows], atol=1e-6)
    assert not np.allclose(xa[~rows], want[~rows], atol=1e-6)   # 其余行做了定向步


def test_bn_train_restores_the_stats_before_ck_and_the_reference(monkeypatch):
    """bn=train：拟合探针的训练模式前向会改 moving 统计量 → 求 c_k 之前必须写回输入值（否则 k* 跟着变）。"""
    m, h = _hardener()
    w0 = m.get_weights()
    stat = get_bn_stat_indices(m)
    seen = []
    orig = h.body_ck

    def spy(*a, **k):
        cur = h.model.get_weights()
        seen.append(all(np.array_equal(cur[i], w0[i]) for i in stat))
        return orig(*a, **k)
    monkeypatch.setattr(h, "body_ck", spy)
    x, y = _data()
    h.harden(w0, x, y, BY["tgt-e4-train-low"], rng=np.random.default_rng(0), lr=0.05)
    assert seen == [True]


def test_lr_is_used_zero_lr_leaves_the_body_unchanged():
    m, h = _hardener()
    w0 = m.get_weights()
    x, y = _data()
    new0, _ = h.harden(w0, x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=0.0)
    for a, b in zip(new0, w0):
        np.testing.assert_array_equal(a, b)
    new1, _ = h.harden(w0, x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=0.05)
    assert any(not np.array_equal(a, b) for a, b in zip(new1, w0))


def test_head_init_run_differs_from_the_main_run_only_by_the_head(monkeypatch):
    """head_diag 的配对：harden 先抽探针子种子（拟不拟合都抽）→ 给 head_init = 主配置拟合出的探针时，两次结果逐位相同。"""
    m, h = _hardener()
    w0 = m.get_weights()
    x, y = _data()
    fitted = {}
    orig = C.spec.fit_probe

    def keep(F, *a, **k):
        r = orig(F, *a, **k)
        fitted.setdefault("p", r)
        return r
    monkeypatch.setattr(C.spec, "fit_probe", keep)
    cfg = BY["pgd-e8-train-low"]
    a, _ = h.harden(w0, x, y, cfg, rng=np.random.default_rng(5), lr=0.05)
    b, _ = h.harden(w0, x, y, cfg, rng=np.random.default_rng(5), lr=0.05,
                    head_init=(fitted["p"]["W"], fitted["p"]["b"]))
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def test_optimizers_do_not_jit_compile():
    """
    每次加固都新建优化器 → GPU 上缺省的 XLA 会每次重编译。Keras 在没有 GPU 时自己把 jit_compile 关掉，
    所以 get_config() 在 CPU 上分辨不出来 → 查源码：make_optimizer 里每个优化器构造都显式传 jit_compile=False。
    """
    import ast
    import inspect
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(C.make_optimizer).strip()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("Adam", "SGD")]
    assert len(calls) == 2
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert isinstance(kw.get("jit_compile"), ast.Constant) and kw["jit_compile"].value is False


# ── 第三轮审查：攻击方向、SAU 参照权重 / 掩码 / 提前停、head_init 真的被用了 ────────────
def _ce_t(h, x, lab, training=False):
    return float(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.constant(lab), logits=h.logits(tf.constant(x), training))))


def test_attack_directions():
    """符号翻了（上升 ↔ 下降）这些断言就红：无目标 PGD 抬高 CE(y)、TRADES 抬高 KL、定向 PGD 压低 CE(k)、AT-tgt 在非 k* 行压低 CE(k*)。"""
    m, h = _hardener()
    h.model.set_weights(m.get_weights())
    _random_heads(h)
    x, y = _data(n=16)
    xt, yt = tf.constant(x), tf.constant(y)
    eps = h.to_in(8 / 255)
    big = h.to_in(64 / 255)                                     # 放大预算，小模型上效果才明显
    a = h.to_in(16 / 255)
    # (a) 无目标 PGD：同一个均匀起点（steps=0 复现）→ 走 3 步后 CE(y) 变大
    x0 = h.pgd_ce(xt, yt, big, a, 0, False, rng=np.random.default_rng(1), start="uniform").numpy()
    x3 = h.pgd_ce(xt, yt, big, a, 3, False, rng=np.random.default_rng(1), start="uniform").numpy()
    assert _ce_t(h, x3, y) > _ce_t(h, x0, y)
    # (b) TRADES：KL(p_nat ‖ p_adv) 比起点（x + 0.001·randn）大
    p_nat = tf.nn.softmax(h.logits(xt, False))
    n0 = (np.random.default_rng(2).standard_normal(x.shape).astype(np.float32)
          * (spec.TRADES_START_STD_PX / h.std).astype(np.float32))
    # KL 在 x0 处最小，两个方向走都会变大 → 比大小分辨不出符号；改为钉住第一步 = 起点 + α·sign(∇KL)（再投影）
    xs = xt + n0
    with tf.GradientTape() as tape:
        tape.watch(xs)
        kl = C.kl_sum(p_nat, h.logits(xs, False))
    want = h._project(xs + a * tf.sign(tape.gradient(kl, xs)), xt, big).numpy()
    xa = h.trades_adv(xt, big, a, 1, False, np.random.default_rng(2)).numpy()
    np.testing.assert_allclose(xa, want, atol=1e-6)
    # (c) 定向 PGD：CE(k) 变小
    k = 5
    xk = h.pgd_ce(xt, yt, big, a, 3, False, start="none", target=k).numpy()
    assert _ce_t(h, xk, np.full_like(y, k)) < _ce_t(h, x, np.full_like(y, k))
    # (d) AT-tgt：非 k* 行的定向部分压低 CE(k*)（σ = 0 时只剩定向部分）
    rows = y != k
    xg = h.tgt_adv(xt, yt, k, big, a, 3, tf.zeros_like(eps), False).numpy()
    assert _ce_t(h, xg[rows], np.full(rows.sum(), k)) < _ce_t(h, x[rows], np.full(rows.sum(), k))


def test_sau_reference_is_the_given_weights_not_the_template(monkeypatch):
    """加固传进来的权重 B ≠ 构造时的模板 A：参照模型必须是 B，参照探针的特征 = B 的推理模式特征。"""
    m, h = _hardener()
    r = np.random.default_rng(7)
    wB = [w + (r.normal(scale=0.05, size=w.shape).astype(w.dtype) if w.dtype.kind == "f" else 0)
          for w in m.get_weights()]
    for i in get_bn_stat_indices(m):                             # 统计量保持合法（方差 > 0）
        wB[i] = m.get_weights()[i]
    seen = []
    orig = C.spec.fit_probe
    monkeypatch.setattr(C.spec, "fit_probe", lambda F, *a, **k: seen.append(np.array(F, copy=True)) or orig(F, *a, **k))
    x, y = _data()
    h.harden(wB, x, y, BY["sau-e8-train-low"], rng=np.random.default_rng(0), lr=0.05)
    for u, v in zip(h.ref.get_weights(), wB):
        np.testing.assert_array_equal(u, v)
    mB = TEI._model()
    mB.set_weights(wB)
    np.testing.assert_allclose(seen[1], C.extract(C.feature_model(mB), x), rtol=1e-6, atol=1e-7)


def test_head_init_is_actually_used(monkeypatch):
    m, h = _hardener()
    x, y = _data()
    d = int(h.feat.output_shape[-1])
    r = np.random.default_rng(9)
    W, b = r.normal(size=(d, NCLS)).astype(np.float32), r.normal(size=NCLS).astype(np.float32)
    first = []
    orig = h.batch_loss

    def spy(*a, **k):
        if not first:
            first.append((h.W.numpy().copy(), h.b.numpy().copy()))
        return orig(*a, **k)
    monkeypatch.setattr(h, "batch_loss", spy)
    a_, _ = h.harden(m.get_weights(), x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=0.05, head_init=(W, b))
    np.testing.assert_array_equal(first[0][0], W)
    np.testing.assert_array_equal(first[0][1], b)
    b_, _ = h.harden(m.get_weights(), x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=0.05)
    assert any(not np.array_equal(u, v) for u, v in zip(a_, b_))


def _sau_loss_ref(lm, lr, ori, ori_r, y, b1, b2):
    """Shared_PGD 的 loss（照 BackdoorBench defense/sau.py 独立重写一遍）：各模型只在**还没翻**的样本上算 CE；JS 只在**非共享**样本上。"""
    pl, plr = tf.argmax(lm, 1, output_type=y.dtype), tf.argmax(lr, 1, output_type=y.dtype)
    succ, succ_r = pl != ori, plr != ori_r
    shared = succ & succ_r & (pl == plr)
    B = tf.cast(tf.shape(lm)[0], tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits
    adv = -(tf.reduce_sum(tf.where(succ, 0.0, ce(labels=y, logits=lm)))
            + tf.reduce_sum(tf.where(succ_r, 0.0, ce(labels=y, logits=lr)))) / 2.0 / B
    cross = tf.reduce_sum(tf.where(shared, 0.0, C.js_rows(lm, lr))) / B
    return b1 * adv + b2 * cross, succ, succ_r, shared


@pytest.mark.parametrize("b1,b2", [(1.0, 0.0), (0.01, 1.0)])     # 只 CE 掩码 / 官方权重（JS 主导）
def test_sau_step_follows_the_reference_masks(monkeypatch, b1, b2):
    """
    一步 Shared_PGD = proj(pert0 − α·sign(∇ 参考 loss))。logits 换成逐行可控的函数，让四种情况都出现：
    都没翻 / 只主模型翻 / 只参照翻 / 两个都翻到同一类（共享）—— 掩码写错哪一个，梯度符号就对不上。
    """
    m, h = _hardener()
    x, y = _data(n=8)
    x = np.clip(x, -0.5, 0.5).astype(np.float32)
    X0 = tf.constant(x)
    r = np.random.default_rng(3)
    wpix = tf.constant(r.uniform(0.5, 1.5, size=x.shape[1:]).astype(np.float32))
    L0 = np.zeros((8, NCLS), np.float32)
    L0[:, 0] = 1.0                                                # 起点 argmax = 0（margin 1）
    L0[:, 2:] = r.normal(scale=0.1, size=(8, NCLS - 2))
    # 每行的扰动方向（s 越大越偏向哪些类）。行 0–1 都不翻；2 只主模型翻；3 两个都翻但到不同类；4–5 只参照翻；
    # 6–7 共享（同一类 1，但主模型还分质量给类 2 → JS 随 s 增大，掩码去掉就看得出来）
    Dm, Dr = np.zeros((8, NCLS), np.float32), np.zeros((8, NCLS), np.float32)
    Dm[[2, 3, 6, 7], 1] = 5.0
    Dm[[6, 7], 2] = 4.0
    Dr[[4, 5, 6, 7], 1] = 5.0
    Dr[3, 3] = 5.0
    Dm, Dr = tf.constant(Dm), tf.constant(Dr)

    def fake(xa, D):
        s = tf.reduce_mean(tf.reshape((xa - X0) * wpix, (8, -1)), axis=1, keepdims=True)
        return tf.constant(L0) + s * D + 0.3 * tf.reduce_mean(tf.reshape(xa * wpix, (8, -1)), 1, keepdims=True)
    monkeypatch.setattr(h, "logits", lambda xa, training: fake(xa, Dm))
    monkeypatch.setattr(h, "logits_ref", lambda xa, W, b: fake(xa, Dr))
    yt = tf.constant(y)
    eps = h.to_in(0.2)
    pert0 = h._project_pert(tf.zeros_like(X0) + eps, X0, eps)
    ori = tf.argmax(h.logits(X0, False), 1, output_type=yt.dtype)
    ori_r = tf.argmax(h.logits_ref(X0, None, None), 1, output_type=yt.dtype)
    with tf.GradientTape() as tape:
        tape.watch(pert0)
        xa = X0 + pert0
        loss, succ, succ_r, shared = _sau_loss_ref(h.logits(xa, False), h.logits_ref(xa, None, None), ori, ori_r, yt,
                                                   b1, b2)
    succ, succ_r, shared = succ.numpy(), succ_r.numpy(), shared.numpy()
    assert (~succ & ~succ_r).any() and (succ & ~succ_r).any() and (~succ & succ_r).any() and shared.any()
    want = h._project_pert(pert0 - eps * tf.sign(tape.gradient(loss, pert0)), X0, eps).numpy()
    got = h.sau_pert(X0, yt, eps, eps, 1, False, None, None, b1, b2).numpy()
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_sau_stops_early_when_every_sample_is_shared(monkeypatch):
    """更新前的共享掩码全真 → 更新一次就停（官方同）：起点之外只投影一次。"""
    m, h = _hardener()
    x, y = _data(n=8)
    x = np.clip(x, -0.5, 0.5).astype(np.float32)                  # 远离上界：init max 之后扰动全为正
    X0 = tf.constant(x)

    def fake(xa, *_a, **_k):                                      # 起点 argmax = 0；任何正扰动 → argmax = 1（两个模型相同 → 共享）
        s = tf.reduce_sum(tf.reshape(xa - X0, (tf.shape(xa)[0], -1)), axis=1, keepdims=True)
        return tf.concat([-s, s, tf.zeros((tf.shape(xa)[0], NCLS - 2))], axis=1)
    monkeypatch.setattr(h, "logits", lambda xa, training: fake(xa))
    monkeypatch.setattr(h, "logits_ref", lambda xa, W, b: fake(xa))
    calls = []
    orig = h._project_pert
    monkeypatch.setattr(h, "_project_pert", lambda *a: calls.append(1) or orig(*a))
    h.sau_pert(X0, tf.constant(y), h.to_in(8 / 255), h.to_in(8 / 255), 5, False, None, None, 0.01, 1.0)
    assert len(calls) == 2                                        # 起点 1 次 + 一次更新；不提前停会是 1 + 5


# ── GPU 探路（2026-10-16）：pgd-e8-frozen-low 在真实快照上 15 步内发散成 NaN → 发散要被拦下、要有逐步记录 ──────
def test_divergence_is_detected_and_traced():
    m, h = _hardener()
    x, y = _data()
    _, info = h.harden(m.get_weights(), x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=0.05)
    assert info["diverged_at"] is None and len(info["loss_trace"]) == len(info["gnorm_body"]) == info["steps"]
    assert all(np.isfinite(info["gnorm_body"])) and info["dw_body_absmax"] > 0
    _, info = h.harden(m.get_weights(), x, y, BY["cft-frozen-low"], rng=np.random.default_rng(0), lr=1e30)
    assert info["diverged_at"] is not None and info["dw_body_absmax"] is None
