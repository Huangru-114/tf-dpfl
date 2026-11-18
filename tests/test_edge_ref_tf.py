"""
tests/test_edge_ref_tf.py  —  方向 2（D-131）参照打分的 TF 侧（真跑 cloud / edge / client）

硬要求：**记录不得改变训练**（同 S5 / S6 / update_ck）
  · edge_ref（observe）开 vs 不设防：逐轮 [Checksum] 相同、已有评估行逐字相同；
  · 反向锚点：去掉 random 围栏，开它就改变训练（草稿模型第一次创建推进 Python random，F-078）；
  · 一次打分前后，传入的权重 / 上传、edge 模型、Python random、numpy 全局 RNG 逐位不变。
接线：防御经 EdgeServerBase.__init__ 注入 edge 与统计量索引（F-107：测试走生产路径）。
内容：
  · TF 取出的「BN 层输入的逐样本矩」= 手写的整批训练模式前向（逐层卷积 + 用整批统计量归一化）；
  · 上传的统计量被人为挪开 → 它的 KL 变大；上传 = 参照本身 → KL ≈ 0；
  · [EdgeRefScore] / [EdgeRefNull] 的条数、有效轮、cid 归属；与三层个性化（edge_shared_blocks: 1）同开能跑，阶段名对得上。
"""

import random
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                          # noqa: E402
from test_eval_grid_tf import _prod_seeding, OLD_TAGS        # noqa: E402
from aggregation.client_update import ClientUpdate           # noqa: E402
from defense.edge_ref import EdgeRefDefense                  # noqa: E402
from server import eval_grid as EG                           # noqa: E402
from utils.kvline import collect_kv, parse_list              # noqa: E402

FED = {"edge_schedule": "interleaved"}
R, G, N = 4, 2, 2
EDGE_REF = {"name": "edge_ref", "edge_ref": {"mode": "observe", "n_fake": 3, "fake_n": 12}}


def _prepare(edges):
    for e in edges:
        r = np.random.default_rng(100 + int(e.edge_id))
        e.clean_x = r.normal(size=(40, TEI.IMG, TEI.IMG, 3)).astype(np.float32)
        e.clean_y = (np.arange(40) % TEI.NCLS).astype(np.int64)


def _run(capsys, monkeypatch, on):
    orig = TEI._cfg

    defense = on if isinstance(on, dict) else (EDGE_REF if on else None)   # on：False / True（observe）/ 一个 defense 块

    def cfg_with_defense(aligned):
        c = orig(aligned)
        if defense:
            c["defense"] = {"name": "edge_ref", "edge_ref": dict(defense["edge_ref"])}
        return c
    monkeypatch.setattr(TEI, "_cfg", cfg_with_defense)
    ev = {"eval_interval": EG.full_interval(G, R), "eval_grid": G}
    random.seed(42)
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=ev,
                                            federation=dict(FED, edge_rounds=R, n_rounds=N))
    _prod_seeding(monkeypatch)
    _prepare(edges)
    cfg["backdoor"]["eval_interval"] = ev["eval_interval"]
    cloud.bd_eval_interval = ev["eval_interval"]
    capsys.readouterr()
    cloud.run()
    monkeypatch.setattr(TEI, "_cfg", orig)
    return capsys.readouterr().out, cloud, clients, edges


def _checksums(out):
    return [ln for ln in out.splitlines() if ln.startswith("[Checksum]")]


def _old_by_round(out):
    chunks, cur = {}, None
    for ln in out.splitlines():
        m = re.match(r"\s*\[Round\s+(\d+)\]", ln)
        if m:
            cur = int(m.group(1))
            chunks[cur] = []
            continue
        if cur is not None and (ln.startswith(OLD_TAGS) or ln.strip().startswith(OLD_TAGS)):
            chunks[cur].append(re.sub(r"time=[\d.]+s", "time=?", ln))
    return chunks


# ══════════════════════════════════════════════════════════════════════════
# 记录不改变训练
# ══════════════════════════════════════════════════════════════════════════
def test_observe_on_off_same_training_and_same_numbers(capsys, monkeypatch):
    off, *_ = _run(capsys, monkeypatch, False)
    on, *_ = _run(capsys, monkeypatch, True)
    assert _checksums(off) and _checksums(off) == _checksums(on)
    assert _old_by_round(off) == _old_by_round(on)
    assert collect_kv(on.splitlines(), "[EdgeRefScore]") and not collect_kv(off.splitlines(), "[EdgeRefScore]")


def test_without_the_fence_edge_ref_changes_training(capsys, monkeypatch):
    """反向锚点：去掉 random 复原 → 草稿模型第一次创建推进 Python random → legacy 管线的洗牌变。"""
    off, *_ = _run(capsys, monkeypatch, False)
    monkeypatch.setattr(random, "setstate", lambda st: None)
    on, *_ = _run(capsys, monkeypatch, True)
    assert _checksums(off) != _checksums(on)


# ══════════════════════════════════════════════════════════════════════════
# 接线与记录的内容
# ══════════════════════════════════════════════════════════════════════════
def test_injected_through_the_production_path_and_line_counts(capsys, monkeypatch):
    out, cloud, clients, edges = _run(capsys, monkeypatch, True)
    for e in edges:
        assert isinstance(e.defense, EdgeRefDefense) and e.defense.edge is e and e.defense.stat_idx
    scores = collect_kv(out.splitlines(), "[EdgeRefScore]")
    nulls = collect_kv(out.splitlines(), "[EdgeRefNull]")
    # 每个 (云轮, edge, edge 轮) 一行 null；有效轮 = (g − 1)·R + er
    assert sorted({(d["round"], d["edge_id"], d["edge_round"]) for d in nulls}) == [
        (g, e, er) for g in range(1, N + 1) for e in range(2) for er in range(1, R + 1)]
    assert all(d["effective_round"] == (d["round"] - 1) * R + d["edge_round"] for d in scores + nulls)
    assert all(len(parse_list(d["kl_all"])) == 3 for d in nulls)          # n_fake = 3
    by_edge = {int(e.edge_id): {int(c.client_id) for c in e.clients} for e in edges}
    assert all(d["cid"] in by_edge[d["edge_id"]] for d in scores)
    assert all(d["kl_all"] is not None and d["kl_all"] >= 0 and d["kl_stem"] == d["kl_all"] for d in scores)
    assert all(d["kl_stage4"] is None for d in scores)                       # 夹具的模型只有一个（stem）BN 层
    dec = [ln for ln in out.splitlines() if "[Decision]" in ln and "edge_ref" in ln]
    assert dec and all("rejected=[]" in ln for ln in dec)                   # observe：全部接纳
    assert len(collect_kv(out.splitlines(), "[TimingEdgeRef]")) == len(nulls)


def _setup_edge(monkeypatch):
    orig = TEI._cfg
    monkeypatch.setattr(TEI, "_cfg", lambda aligned: {**orig(aligned), "defense": dict(EDGE_REF)})
    cfg, cloud, clients, edges = TEI._setup(True, federation=dict(FED, edge_rounds=R, n_rounds=N))
    _prod_seeding(monkeypatch)
    _prepare(edges)
    for e in edges:
        e.current_round = (1, 1)
    return cfg, cloud, clients, edges


def _h(ws):
    import hashlib
    return hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(w)).tobytes() for w in ws)).hexdigest()


def test_scoring_leaves_everything_untouched_and_shifted_stats_score_higher(monkeypatch, capsys):
    cfg, cloud, clients, edges = _setup_edge(monkeypatch)
    e = edges[0]
    d = e.defense
    w0 = e.model.get_weights()
    # 「上传」= edge 权重，统计量换成干净集上的参照 → KL ≈ 0；再把均值挪开 → KL 变大
    d._build()
    mom = d._moments(w0, e.clean_x)
    ref_mu = mom[0][0].mean(0)
    ref_var = np.maximum(mom[0][1].mean(0) - ref_mu ** 2, 0.0)
    L = d._layers[0]
    same = [np.array(w, copy=True) for w in w0]
    same[L["mean_idx"]], same[L["var_idx"]] = ref_mu.astype(np.float32), ref_var.astype(np.float32)
    moved = [np.array(w, copy=True) for w in same]
    moved[L["mean_idx"]] = moved[L["mean_idx"]] + 3.0 * np.sqrt(ref_var).astype(np.float32)
    ups = [ClientUpdate(same, 24, 0.0, 0.0, client_id=1), ClientUpdate(moved, 24, 0.0, 0.0, client_id=2)]
    h_up, h_edge = [_h(u[0]) for u in ups], _h(e.model.get_weights())
    py, npst = random.getstate(), np.random.get_state()[1].copy()
    rows = d.score(ups, w0)
    assert [_h(u[0]) for u in ups] == h_up and _h(e.model.get_weights()) == h_edge
    assert random.getstate() == py and np.array_equal(np.random.get_state()[1], npst)
    by = {r["cid"]: r for r in rows}
    assert by[1]["kl_all"] < 1e-4                          # float32 舍入之内
    assert by[2]["kl_all"] == pytest.approx(4.5, rel=0.02)  # 只挪均值 3σ → KL = 3² / 2
    assert by[2]["mix_res"] > by[1]["mix_res"]


def test_moments_equal_a_hand_written_whole_batch_training_forward():
    """三个 BN 层的小网络：TF 取出的第 2、3 层 BN 输入的矩 = 手写「卷积 → 用整批统计量归一化 → ReLU → 卷积」。"""
    import test_three_tier_personalization as T3
    tf.keras.utils.set_random_seed(3)
    model = T3._model()
    d = EdgeRefDefense({"n_fake": 2, "fake_n": 4})

    class _E:
        pass
    edge = _E()
    edge.model, edge.edge_id = model, 0
    d.set_edge(edge)
    d._build()
    x = np.random.default_rng(0).normal(size=(10, T3.IMG, T3.IMG, 3)).astype(np.float32)
    w = model.get_weights()
    mom = d._moments(w, x)
    assert [L["stage"] for L in d._layers] == ["stem", "stage3", "stage4"]
    h = tf.convert_to_tensor(x)
    want = []
    for name in ("stem", "stage3", "stage4"):
        conv = model.get_layer(f"{name}_conv" if name == "stem" else f"{name}_conv1")
        bn = model.get_layer(f"{name}_bn" if name == "stem" else f"{name}_bn1")
        z = tf.nn.conv2d(h, conv.kernel, 1, "SAME")
        want.append(((tf.reduce_mean(z, [1, 2])).numpy(), tf.reduce_mean(tf.square(z), [1, 2]).numpy()))
        m, v = tf.nn.moments(z, [0, 1, 2])
        h = tf.nn.relu(tf.nn.batch_normalization(z, m, v, bn.beta, bn.gamma, bn.epsilon))
    for (a1, a2), (b1, b2) in zip(mom, want):
        assert np.allclose(a1, b1, atol=1e-4) and np.allclose(a2, b2, atol=1e-4)
    # 训练模式前向只动草稿模型自己的统计量：原模型不变
    assert all(np.array_equal(a, b) for a, b in zip(model.get_weights(), w))


def test_runs_together_with_three_tier_personalization(capsys):
    """edge_shared_blocks: 1 + edge_ref 同开：跑满 3 个云轮，阶段名对得上（夹具模型有 stem / stage3 / stage4）。

    夹具 `_setup` 不接受 defense 参数 → 建好之后用生产路径的工厂（create_defense）换上防御，并照 EdgeServerBase.__init__ 注入。"""
    import test_three_tier_personalization as T3
    from defense import create_defense
    from models.cnn import get_bn_stat_indices
    cloud, clients, edges = T3._setup(1)
    for e in edges:
        e.defense = create_defense(dict(e.config, defense=dict(EDGE_REF)), layer="edge")
        e.defense.set_stat_indices(get_bn_stat_indices(e.model), label=e.edge_id)
        e.defense.set_edge(e)
        r = np.random.default_rng(int(e.edge_id))
        e.clean_x = r.normal(size=(20, T3.IMG, T3.IMG, 3)).astype(np.float32)
        e.clean_y = (np.arange(20) % T3.NCLS).astype(np.int64)
    capsys.readouterr()
    cloud.run()
    out = capsys.readouterr().out
    scores = collect_kv(out.splitlines(), "[EdgeRefScore]")
    assert scores and all(d["kl_stage4"] is not None and d["kl_stage3"] is not None and d["kl_stage1"] is None
                          for d in scores)
    assert all(np.isfinite(d["mix_res"]) for d in scores)



# ══════════════════════════════════════════════════════════════════════════
# 第二段：过滤（N-014 / D-132）
# ══════════════════════════════════════════════════════════════════════════
def _filt(threshold, score="kl_all"):
    return {"name": "edge_ref", "edge_ref": {"mode": "filter", "n_fake": 3, "fake_n": 12,
                                             "score": score, "threshold": threshold}}


def test_filter_that_rejects_nobody_trains_exactly_like_observe(capsys, monkeypatch):
    """τ 远大于任何分数 → 没有人被剔 → 与 observe（= 不设防）逐轮 checksum 相同；admitted 列全是 1。"""
    obs, *_ = _run(capsys, monkeypatch, True)
    flt, *_ = _run(capsys, monkeypatch, _filt(1e9))
    assert _checksums(obs) and _checksums(obs) == _checksums(flt)
    rows = collect_kv(flt.splitlines(), "[EdgeRefScore]")
    assert rows and all(d["admitted"] == 1 for d in rows)


def test_filter_that_rejects_everybody_freezes_the_edge_models(capsys, monkeypatch):
    """τ = 0 → 每个上传都被剔（KL > 0）→ 每个边缘轮聚合都返回下发权重 → 训练与 observe 不同；[Decision] 每轮全剔。"""
    obs, *_ = _run(capsys, monkeypatch, True)
    flt, cloud, clients, edges = _run(capsys, monkeypatch, _filt(0.0))
    rows = collect_kv(flt.splitlines(), "[EdgeRefScore]")
    assert rows and all(d["admitted"] == 0 for d in rows)
    dec = [ln for ln in flt.splitlines() if "[Decision]" in ln and "edge_ref" in ln]
    assert dec and all(" admitted 0/" in ln for ln in dec)
    assert _checksums(obs) != _checksums(flt)


def test_filter_aggregates_exactly_the_admitted_uploads(monkeypatch, capsys):
    """两个上传：统计量 = 参照（KL ≈ 0）与挪开 3σ（KL ≈ 4.5）。τ = 1 → 只聚合前者，逐位等于 fedavg_aggregate([前者])；
    τ = 0 → 两个都剔 → 返回下发权重的副本（不是同一个对象）。"""
    from aggregation.fedavg import aggregate as fedavg_aggregate
    cfg, cloud, clients, edges = _setup_edge(monkeypatch)
    e = edges[0]
    d = e.defense
    w0 = e.model.get_weights()
    d._build()
    mom = d._moments(w0, e.clean_x)
    mu = mom[0][0].mean(0)
    var = np.maximum(mom[0][1].mean(0) - mu ** 2, 0.0)
    L = d._layers[0]
    same = [np.array(w, copy=True) for w in w0]
    same[L["mean_idx"]], same[L["var_idx"]] = mu.astype(np.float32), var.astype(np.float32)
    moved = [np.array(w, copy=True) for w in same]
    moved[L["mean_idx"]] = moved[L["mean_idx"]] + 3.0 * np.sqrt(var).astype(np.float32)
    ups = [ClientUpdate(same, 24, 0.0, 0.0, client_id=1), ClientUpdate(moved, 30, 0.0, 0.0, client_id=2)]
    d.p.update(mode="filter", score="kl_all", threshold=1.0)
    agg = d.aggregate(ups, w0)
    want = fedavg_aggregate([ups[0]])
    assert d.last_admitted == [0] and all(np.array_equal(a, b) for a, b in zip(agg, want))
    d.record_decision(ups)
    assert d.last_rejected_ids == [2]
    d.p.update(threshold=0.0)
    agg0 = d.aggregate(ups, w0)
    assert d.last_admitted == [] and all(np.array_equal(a, b) for a, b in zip(agg0, w0))
    assert all(a is not b for a, b in zip(agg0, w0))
