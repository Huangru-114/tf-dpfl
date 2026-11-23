"""
阶段三 D3：`fedavg/analysis/d3_probes.py`（离线探针的 TF 部分）。需要 TF。

在一个很小的真 HFL（tests/test_eval_integration._setup：2 edge × 3 个 FedRep 客户端、2 个 Bad-PFL 恶意端；
模型换成层名带 stem / stage1–4 前缀的五段小网络，好让「只训 stage4」切得出来）上：
  1. 跑第 1 云轮、存快照，记下 run 自己第 1 轮全量点的逐 edge 良性 ASR 与 margin；
  2. 全新的世界从快照恢复 → 重现闸逐位通过；
  3. 精确重估 = 各批统计量的样本数加权均值、旧值不留、方差口径与 Keras 自己的更新一致；
  4. 子集重训只动选中的层；P-a / P-b / P-c 跑得通、不改任何外部状态；
     P-a 的 none 臂主列 = 固定 ξ 列（定义如此）；P-b 受害者就是攻击者时白盒列 = 主列；P-c 触发器为 0 时 Δ = 0。
"""

import copy
import json
import random

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                         # noqa: E402
import server.backdoor_server as BS                         # noqa: E402
from analysis import d3_probes as D3                         # noqa: E402
from analysis import d3_spec as S                            # noqa: E402
from analysis import p0_snapshot as P0                       # noqa: E402
from models.model_utils import clone_model                   # noqa: E402
from utils.pm import compose_pm                              # noqa: E402

from pathlib import Path                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "experiments/defense/edge-native"
EV = {"post_agg_eval": True, "snapshot_rounds": "1"}
FED = {"edge_schedule": "interleaved", "edge_rounds": 2, "n_rounds": 2}
STAGES5 = ("stem", "stage1", "stage2", "stage3", "stage4")


def _model5():
    """层名沿用 ResNet-10 的 stage 前缀，只有几百个参数，CPU 上秒级。"""
    tf.keras.utils.set_random_seed(0)
    L = tf.keras.layers
    inp = tf.keras.Input((TEI.IMG, TEI.IMG, 3))
    x = inp
    for name in STAGES5:
        conv = f"{name}_conv" if name == "stem" else f"{name}_conv1"
        bn = f"{name}_bn" if name == "stem" else f"{name}_bn1"
        x = L.Conv2D(4, 3, padding="same", use_bias=False, name=conv)(x)
        x = L.BatchNormalization(momentum=0.9, epsilon=1e-5, name=bn)(x)
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, L.Dense(TEI.NCLS, activation="softmax", name="head")(x))


@pytest.fixture(scope="module")
def mp_model():
    mp = pytest.MonkeyPatch()
    mp.setattr(TEI, "_model", _model5)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def source(mp_model, tmp_path_factory):
    """跑第 1 云轮的「原 run」：记下第 1 轮全量点主列的逐 edge 良性 ASR 与 margin，返回快照路径与假 metrics。"""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("dumps")
    mp.setenv("TFDPFL_DUMPDIR", str(tmp))
    random.seed(42)
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=dict(EV), federation=dict(FED))
    full = []
    orig_eha = BS.evaluate_hierarchical_asr

    def eha(*a, **k):
        m = orig_eha(*a, **k)
        if not k.get("light"):
            mg = P0.edge_margins(m, a[5], k.get("n_classes"))
            full.append({int(d["edge_id"]): {"client_benign": d.get("client_benign"), "margin_p50": mg.get(int(d["edge_id"]))}
                         for d in m.get("per_edge") or []})
        return m
    mp.setattr(BS, "evaluate_hierarchical_asr", eha)
    cloud.run_round(1)
    mp.undo()
    snaps = sorted(tmp.rglob("snapshot_r001.npz"))
    assert len(snaps) == 1 and full
    rec = full[0]
    metrics = {"per_edge_rounds": {"1": [{"edge_id": e, "client_benign": v["client_benign"]} for e, v in rec.items()]},
               "per_edge_detail_columns": ["edge_id", "margin_p50", "benign_asr_p90", "yt_clean_benign"],
               "per_edge_detail_rounds": {"1": [[e, v["margin_p50"], None, None] for e, v in rec.items()]}}
    return {"snap": snaps[0], "metrics": metrics, "rec": rec}


def _world():
    random.seed(42)
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=dict(EV), federation=dict(FED))
    w = P0.World()
    w.config, w.cloud, w.global_model = cfg, cloud, cloud.global_model
    w.edges, w.clients = list(cloud.edge_servers), clients
    w.malicious = set(TEI.MAL)
    w.R, w.seed = FED["edge_rounds"], 42
    E = int(cloud.eval_attacker.assigned_edge)
    w.victims = [int(e.edge_id) for e in w.edges if int(e.edge_id) != E]      # 测试里另一台 edge 也有攻击者，只当受害 edge 用
    w.placement = "collocated"
    r = np.random.default_rng(9)
    w.clean = {int(e.edge_id): (r.normal(size=(16, TEI.IMG, TEI.IMG, 3)).astype(np.float32),
                                (np.arange(16) % 4).astype(np.int64)) for e in w.edges}
    return w


@pytest.fixture(scope="module")
def restored(mp_model, source):
    w = _world()
    snap = P0.load_snapshot(source["snap"], w)
    P0.restore(w, snap)
    return w, snap


@pytest.fixture(scope="module")
def d3cfg():
    return S.load_config(HERE / "d3_configs.yaml")


@pytest.fixture(scope="module")
def hardener(restored, d3cfg):
    w, _ = restored
    return D3.make_hardener(w, d3cfg, HERE, batch=8)           # 16 张干净图、批 8 → 每 epoch 2 步 × 5 epoch


def _state(w):
    """外部状态的指纹：全局 / 各 edge 权重、各端私有部分、生成器、两套全局随机数。"""
    att = w.cloud.eval_attacker
    h = [np.concatenate([np.ravel(a) for a in w.global_model.get_weights()])]
    h += [np.concatenate([np.ravel(a) for a in e.model.get_weights()]) for e in w.edges]
    for c in w.clients:
        idx, vals = c.private_state()
        h += [np.ravel(np.asarray(v)) for v in vals]
    h += [np.ravel(a) for a in att._atk_generator.get_weights()]
    return h, random.getstate(), np.random.get_state()[1].copy()


def _same_state(a, b):
    return (len(a[0]) == len(b[0]) and all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
            and a[1] == b[1] and np.array_equal(a[2], b[2]))


# ── 精确重估 ─────────────────────────────────────────────────────────────────
def _bn_model():
    tf.keras.utils.set_random_seed(1)
    L = tf.keras.layers
    inp = tf.keras.Input((6, 6, 3))
    x = L.Conv2D(3, 3, padding="same", use_bias=False, name="c1")(inp)
    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5, name="b1")(x)
    x = L.ReLU()(x)
    x = L.Conv2D(2, 3, padding="same", use_bias=False, name="c2")(x)
    x = L.BatchNormalization(momentum=0.9, epsilon=1e-5, name="b2")(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, L.Dense(4, activation="softmax")(x))


def _stats(m):
    out = []
    for l in D3.bn_layers(m):
        out += [l.moving_mean.numpy().astype(np.float64), l.moving_variance.numpy().astype(np.float64)]
    return out


def test_precise_bn_single_batch_is_the_batch_statistic():
    """一批：moving_mean = 该批 BN 输入的均值（逐通道）；moving_variance = Keras 自己的口径 —— 这里钉住它是哪个。"""
    m = _bn_model()
    x = np.random.default_rng(0).normal(size=(10, 6, 6, 3)).astype(np.float32)
    ext = tf.keras.Model(m.inputs, [l.input for l in D3.bn_layers(m)])
    ins = [o.numpy().astype(np.float64) for o in ext(x, training=True)]
    D3.precise_bn(m, x, batch=64)
    got = _stats(m)
    for k, o in enumerate(ins):
        flat = o.reshape(-1, o.shape[-1])
        n = flat.shape[0]
        np.testing.assert_allclose(got[2 * k], flat.mean(0), rtol=1e-5, atol=1e-6)
        biased, unbiased = flat.var(0), flat.var(0) * n / (n - 1)
        # 实测（TF 2.15 / Keras 2.15，fused BN）：moving_variance 用的是这一个
        assert np.allclose(got[2 * k + 1], unbiased, rtol=1e-4, atol=1e-6) or \
            np.allclose(got[2 * k + 1], biased, rtol=1e-4, atol=1e-6)
        KERAS_VAR = "unbiased" if np.allclose(got[2 * k + 1], unbiased, rtol=1e-4, atol=1e-6) else "biased"
        assert KERAS_VAR == PINNED_VAR_CONVENTION


PINNED_VAR_CONVENTION = "unbiased"


def test_precise_bn_is_the_sample_weighted_mean_of_per_batch_statistics_and_forgets_old_values():
    x = np.random.default_rng(1).normal(size=(23, 6, 6, 3)).astype(np.float32)
    spans = S.seq_batches(len(x), 8)                          # 8 / 8 / 7
    per = []
    for s, e in spans:                                         # 每批单独（同一组初始权重）
        m1 = _bn_model()
        D3.precise_bn(m1, x[s:e], batch=64)
        per.append((e - s, _stats(m1)))
    m = _bn_model()
    for l in D3.bn_layers(m):                                  # 旧统计量是垃圾 → 结果不应受影响
        l.moving_mean.assign(np.full(l.moving_mean.shape, 123.0, np.float32))
        l.moving_variance.assign(np.full(l.moving_variance.shape, 77.0, np.float32))
    D3.precise_bn(m, x, batch=8)
    got = _stats(m)
    # 第 1 层 BN 的输入不依赖别的批 → 严格等于加权均值；第 2 层的输入经过第 1 层的 batch 归一化，各批相同（同一批数据）→ 也相等
    for k in range(len(got)):
        want = sum(n * st[k] for n, st in per) / sum(n for n, _ in per)
        np.testing.assert_allclose(got[k], want, rtol=1e-5, atol=1e-6)


def test_precise_bn_restores_momentum_and_touches_only_bn_statistics():
    m = _bn_model()
    before = m.get_weights()
    tv = {id(v) for v in m.trainable_variables}
    D3.precise_bn(m, np.random.default_rng(2).normal(size=(9, 6, 6, 3)).astype(np.float32), batch=4)
    assert all(l.momentum == 0.9 for l in D3.bn_layers(m))
    for v, b, a in zip(m.weights, before, m.get_weights()):
        if id(v) in tv:
            assert np.array_equal(a, b)                       # 可训练权重一位不动


# ── 子集重训 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("arm", ["stage4", "stem_stage3", "body"])
def test_harden_subset_changes_only_the_selected_layers(restored, hardener, d3cfg, arm):
    w, snap = restored
    h, base = hardener
    E = int(w.cloud.eval_attacker.assigned_edge)
    pre = S.arm_prefixes(d3cfg)[arm]
    x, y = w.clean[E]
    new, info = D3.harden_subset(h, snap["edges"][E], x, y, base, pre, rng=np.random.default_rng(0), lr=0.05)
    assert info["steps"] == 10 and info["diverged_at"] is None
    owners = D3.owner_names(h.model, h.model.weights)
    tv = {id(v) for v in h.model.trainable_variables}
    changed = {owners[i] for i, (a, b) in enumerate(zip(new, snap["edges"][E])) if not np.array_equal(a, b)}
    for i, v in enumerate(h.model.weights):
        if id(v) not in tv:
            assert np.array_equal(new[i], snap["edges"][E][i])  # BN 统计量写回
    assert "head" not in changed
    if pre is None:
        assert changed == {f"{s}_conv" if s == "stem" else f"{s}_conv1" for s in STAGES5} | \
            {f"{s}_bn" if s == "stem" else f"{s}_bn1" for s in STAGES5}
    else:
        assert changed and all(str(n).startswith(pre) for n in changed)
    assert len(h.body_vars) == 15                              # 全部主干的变量表原样恢复（5 段 × conv + γ + β）
    assert info["n_vars"] == {"stage4": 3, "stem_stage3": 12, "body": 15}[arm]


# ── 重现闸 ───────────────────────────────────────────────────────────────────
def test_reproduce_gate_is_bitwise_on_the_restored_world(source, restored):
    w, snap = restored
    full = P0.eval_full_main(w, snap, 1, detail=True)
    chk = S.reproduce_check(full, S.recorded_edges(source["metrics"], 1), 0.01)
    assert chk["pass"] and chk["max_abs_diff"] == 0.0


# ── P-a ─────────────────────────────────────────────────────────────────────
def test_pa_runs_restores_everything_and_none_main_equals_fixed(restored, hardener, d3cfg):
    w, snap = restored
    h, base = hardener
    before = _state(w)
    out = D3.probe_pa(w, snap, 1, d3cfg, h, base)
    assert _same_state(before, _state(w))
    assert set(out["arms"]) == {"none", "stage4", "stem_stage3", "body"}
    none = out["arms"]["none"]
    assert none["main"] == none["fixed"]                       # 固定 ξ = none 臂的 ξ（D-138）
    assert none["harden"] is None and out["arms"]["stage4"]["harden"]["n_vars"] == 3
    assert out["n_benign"] == sum(1 for c in w.clients if int(c.assigned_edge) == out["edge"]
                                  and int(c.client_id) not in w.malicious)
    for arm in ("none", "stage4"):
        a = out["arms"][arm]
        assert a["reest"]["n"] + len(a["reest"]["skipped"]) == 3       # E 上 3 个客户端
        assert a["main"]["acc"] is not None and a["main"]["margin_p50"] is not None
    json.dumps(S.jsonable(out))


def test_pa_is_repeatable(restored, hardener, d3cfg):
    w, snap = restored
    h, base = hardener
    a = D3.probe_pa(w, snap, 1, d3cfg, h, base)
    b = D3.probe_pa(w, snap, 1, d3cfg, h, base)
    for k in ("none", "stage4"):
        assert a["arms"][k]["main"] == b["arms"][k]["main"] and a["arms"][k]["fixed"] == b["arms"][k]["fixed"]


def test_reestimate_writes_the_precise_statistics(restored):
    w, snap = restored
    E = int(w.cloud.eval_attacker.assigned_edge)
    cs = [c for c in w.clients if int(c.assigned_edge) == E and c._stats is not None]
    assert cs, "测试世界里 E 上至少要有一个参训过的客户端"
    c = cs[0]
    orig = [np.array(s, copy=True) for s in c._stats]
    idx, vals = c.private_state()
    ref = clone_model(w.global_model)
    ref.set_weights(compose_pm(snap["edges"][E], vals, idx))
    want = D3.precise_bn(ref, D3.train_images(c), 32)
    try:
        acct = D3.reestimate(w, [c], snap["edges"][E], 32)
        assert acct == {"n": 1, "skipped": []}
        for k, i in enumerate(c._stat_w_idx):
            assert np.array_equal(c._stats[k], want[i])
        assert any(not np.array_equal(a, b) for a, b in zip(c._stats, orig))   # 真的重估了
    finally:
        c._stats = orig


def test_reestimate_skips_clients_without_private_statistics(restored):
    w, snap = restored
    c = w.clients[1]
    saved = (c._head_weights, c._stats)
    c._head_weights, c._stats = None, None
    try:
        assert D3.reestimate(w, [c], snap["edges"][int(c.assigned_edge)], 32) == {"n": 0, "skipped": [1]}
    finally:
        c._head_weights, c._stats = saved


# ── P-b ─────────────────────────────────────────────────────────────────────
def test_pb_whitebox_equals_main_when_the_victim_is_the_attacker(restored):
    w, snap = restored
    att = w.cloud.eval_attacker
    with D3.fence(w):
        P0.set_edges(w, snap["edges"])
        w.cloud.begin_pm_eval()
        m_att = D3.attacker_pm(w)
        recs = D3.pb_client(w, att, m_att, 1)
    assert recs["whitebox"]["asr"] == recs["main"]["asr"]
    assert np.array_equal(recs["whitebox"]["margins"], recs["main"]["margins"])
    assert np.array_equal(recs["xi_only_victim"]["margins"], recs["xi_only_attacker"]["margins"])


def test_pb_runs_and_does_not_change_state(restored, source):
    w, snap = restored
    before = _state(w)
    out = D3.probe_pb(w, snap, 1, source["metrics"])
    assert _same_state(before, _state(w))
    assert set(out["V"]) == set(D3.PB_COLUMNS)
    assert all(v is None or 0.0 <= v <= 1.0 for v in out["V"].values())
    assert out["recorded_V"] == pytest.approx(np.mean([source["rec"][e]["client_benign"] for e in w.victims]))


# ── P-c ─────────────────────────────────────────────────────────────────────
def test_pc_runs_and_does_not_change_state(restored):
    w, snap = restored
    before = _state(w)
    out = D3.probe_pc(w, snap, 1)
    assert _same_state(before, _state(w))
    assert sorted(out["edges"]) == ["0", "1"]
    for v in out["edges"].values():
        assert np.isfinite(v["kl_all"]) and v["kl_all"] >= 0 and v["n"] == 16
        assert v["n_nontarget"] == 12                          # 标签 0–3 循环，目标类 0 占 4 张
        assert set(k for k in v if k.startswith("kl_stage")) == {"kl_stage1", "kl_stage2", "kl_stage3", "kl_stage4"}


def test_pc_zero_trigger_gives_zero_shift(restored, monkeypatch):
    w, snap = restored
    att = w.cloud.eval_attacker
    monkeypatch.setattr(att, "eval_xi", lambda m, x, y=None, rng=None: np.zeros_like(x))
    monkeypatch.setattr(att, "eval_delta", lambda x: np.zeros_like(x))
    out = D3.probe_pc(w, snap, 1)
    assert all(v["kl_all"] == 0.0 for v in out["edges"].values())


# ── 一个 run 的全流程 ────────────────────────────────────────────────────────
def test_probe_run_end_to_end_is_jsonable_and_leaves_no_trace(source, restored, hardener, d3cfg):
    w, snap = restored
    h, base = hardener
    before = _state(w)
    out = D3.probe_run(w, d3cfg, {"pa": [1], "pb": [1], "pc": [1]}, {1: source["snap"]}, source["metrics"],
                       hardener=h, base_cfg=base)
    assert _same_state(before, _state(w))
    R = out["rounds"]["1"]
    assert R["reproduce"]["pass"] and {"pa", "pb", "pc"} <= set(R)
    blob = json.loads(json.dumps(S.jsonable(out)))
    assert blob["rounds"]["1"]["pa"]["arms"]["none"]["main"]["n"] >= 1


# ── 入口 main()：读清单 → 核对快照 sha → 写 JSON → 退出码 ───────────────────
@pytest.fixture()
def e2e(source, restored, tmp_path, monkeypatch, d3cfg):
    import shutil
    import yaml
    from hardening import spec as HS
    w, _ = restored
    monkeypatch.setattr(P0, "build_world", lambda path: w)
    run = "TEST__cell__s42"
    here = tmp_path / "here"
    (here / "configs").mkdir(parents=True)
    (here / "configs" / f"{run}.yaml").write_text("unused: true\n")
    shutil.copy(HERE / "p0_configs.yaml", here / "p0_configs.yaml")
    c = copy.deepcopy(d3cfg)
    for p in ("pa", "pb", "pc"):
        c[p]["runs"], c[p]["rounds"] = [run], [1]
    c["pb"]["reference_runs"] = []
    (here / "d3_configs.yaml").write_text(yaml.safe_dump(c, allow_unicode=True))
    snap = Path(source["snap"])

    def call(sha_ok=True, bump=0.0, extra=()):
        m = copy.deepcopy(source["metrics"])
        m["per_edge_rounds"]["1"] = [{**d, "client_benign": d["client_benign"] + bump} for d in m["per_edge_rounds"]["1"]]
        m["dumps"] = {"snapshots": [{"round": 1, "path": snap.name, "sha": HS.sha12(snap) if sha_ok else "000000000000"}]}
        res = here / "results/P2/TEST"
        res.mkdir(parents=True, exist_ok=True)
        (res / f"{run}.metrics.json").write_text(json.dumps(m))
        out = tmp_path / "out.json"
        rc = D3.main(["--run", run, "--here", str(here), "--dumps-root", str(snap.parent), "--out", str(out), *extra])
        return rc, json.loads(out.read_text()), c
    return call


def test_main_writes_a_complete_result(e2e):
    rc, out, c = e2e()
    assert rc == 0
    assert out["schema"] == S.SCHEMA and out["run_id"] == "TEST__cell__s42" and out["seed"] == 42
    assert out["config_sha"] == S.content_sha(c) and out["plan"] == {"pa": [1], "pb": [1], "pc": [1]}
    R = out["rounds"]["1"]
    assert R["reproduce"]["pass"] and {"pa", "pb", "pc"} <= set(R)


def test_main_refuses_a_snapshot_whose_sha_differs(e2e):
    with pytest.raises(RuntimeError, match="sha"):
        e2e(sha_ok=False)


def test_main_reproduce_failure_exits_3_but_still_writes_everything(e2e):
    rc, out, _ = e2e(bump=0.5)
    R = out["rounds"]["1"]
    assert rc == 3 and not R["reproduce"]["pass"] and "pa" in R
