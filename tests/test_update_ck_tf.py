"""
tests/test_update_ck_tf.py  —  S6b：在线 c_k 的 TF 侧（真跑 cloud / edge / client；D-087）

小 HFL（2 edge × 3 FedRep 客户端，2 个 Bad-PFL 恶意端；夹具同 test_s6_tf），R=4、网格 G=2、评分间隔 2：

硬要求（同 S5 / S6a）：**记录不得改变训练**
  · update_ck 开 vs 关：逐轮 [Checksum] 相同、已有评估行逐字相同；
  · 反向锚点：去掉 random 围栏，开关就改变训练（专用评分模型第一次创建推进 Python random，F-078）；
  · 一次评分前后，传入的权重 / 上传、Python random、numpy 全局 RNG 逐位不变。
内容：
  · θ_i = edge_w 上换入上传的**可训练权重**，BN 统计量与 head 位置保持 edge_w 的；
  · 上传 == edge_w（Δ = 0）→ c_i == c_before（构造得出的精确值）；
  · [CkBefore] / [CkScore] 的条数、c 的长度、恶意标记、cid 归属 edge；评分点只落在 eff % every == 0；
  · `ck_reached_batched` 与逐类的 `ck_reached`：ε = 0 逐位相同，ε > 0 的 c_k 均值一致。
"""

import random
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                          # noqa: E402
from test_eval_grid_tf import _prod_seeding, OLD_TAGS        # noqa: E402
from aggregation.client_update import ClientUpdate           # noqa: E402
from analysis import ck_snapshot as CK                       # noqa: E402
from analysis import functional_score as FS                  # noqa: E402
from models.cnn import get_bn_stat_indices                   # noqa: E402
from server import eval_grid as EG                           # noqa: E402
from utils.kvline import collect_kv, parse_list              # noqa: E402

CKCFG = {"update_ck": True, "update_ck_every": 2, "update_ck_n": 8, "update_ck_steps": 2}
FED = {"edge_schedule": "interleaved"}
R, G, N = 4, 2, 2


def _prepare(cloud, edges):
    """给每个 edge 挂上干净集（main.py 在开关开时做的事）。"""
    for e in edges:
        r = np.random.default_rng(100 + int(e.edge_id))
        e.clean_x = r.normal(size=(40, TEI.IMG, TEI.IMG, 3)).astype(np.float32)
        e.clean_y = (np.arange(40) % TEI.NCLS).astype(np.int64)


def _run(capsys, monkeypatch, ck):
    ev = {"eval_interval": EG.full_interval(G, R), "eval_grid": G, **(CKCFG if ck else {})}
    random.seed(42)
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=ev,
                                            federation=dict(FED, edge_rounds=R, n_rounds=N))
    _prod_seeding(monkeypatch)
    _prepare(cloud, edges)
    cfg["backdoor"]["eval_interval"] = ev["eval_interval"]
    cloud.bd_eval_interval = ev["eval_interval"]
    capsys.readouterr()
    cloud.run()
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
def test_ck_on_off_same_training_and_same_numbers(capsys, monkeypatch):
    off, *_ = _run(capsys, monkeypatch, False)
    on, cloud, *_ = _run(capsys, monkeypatch, True)
    assert _checksums(off) and _checksums(off) == _checksums(on)
    assert _old_by_round(off) == _old_by_round(on)
    assert collect_kv(on.splitlines(), "[CkScore]") and not collect_kv(off.splitlines(), "[CkScore]")


def test_without_the_fence_ck_changes_training(capsys, monkeypatch):
    """反向锚点：去掉 random 复原 → 专用评分模型第一次创建推进 Python random → legacy 管线的洗牌变。"""
    off, *_ = _run(capsys, monkeypatch, False)
    monkeypatch.setattr(random, "setstate", lambda st: None)
    on, *_ = _run(capsys, monkeypatch, True)
    assert _checksums(off) != _checksums(on)


# ══════════════════════════════════════════════════════════════════════════
# 记录的内容
# ══════════════════════════════════════════════════════════════════════════
def test_lines_counts_flags_and_scoring_points(capsys, monkeypatch):
    out, cloud, clients, edges = _run(capsys, monkeypatch, True)
    before = collect_kv(out.splitlines(), "[CkBefore]")
    scores = collect_kv(out.splitlines(), "[CkScore]")
    # 评分点只在 eff % 2 == 0：R=4 → 每个云轮的 edge 轮 2、4
    assert {(d["edge_round"] % 2) for d in before} == {0}
    assert sorted({(d["round"], d["edge_round"]) for d in before}) == [
        (g, er) for g in range(1, N + 1) for er in (2, 4)]
    assert all(d["effective_round"] == (d["round"] - 1) * R + d["edge_round"] for d in before)
    assert all(len(parse_list(d["c"])) == TEI.NCLS for d in before + scores)
    assert all(d["n_proto"] == 40 and d["n_attack"] == 8 for d in before)
    by_edge = {int(e.edge_id): {int(c.client_id) for c in e.clients} for e in edges}
    for d in scores:
        assert d["cid"] in by_edge[d["edge_id"]]
        assert d["mal"] == int(d["cid"] in TEI.MAL)
        assert 0.0 <= d["ncm_acc"] <= 1.0
    # 每个评分点的更新条数 = 该 edge 该轮的上传数（[CkBefore] 一条对应 ≥ 1 条 [CkScore]）
    keys_b = {(d["round"], d["edge_id"], d["edge_round"]) for d in before}
    keys_s = {(d["round"], d["edge_id"], d["edge_round"]) for d in scores}
    assert keys_b == keys_s
    assert len(collect_kv(out.splitlines(), "[TimingCk]")) == len(before)


def _cloud_with_ck(monkeypatch, **extra):
    ev = {"eval_grid": G, "eval_interval": EG.full_interval(G, R), **CKCFG, **extra}
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=ev,
                                            federation=dict(FED, edge_rounds=R, n_rounds=N))
    _prod_seeding(monkeypatch)
    _prepare(cloud, edges)
    for e in edges:                                    # 先训出 head、让 edge 权重不是初值
        e.run_edge_round(1, 1)
    return cfg, cloud, clients, edges


def _h(ws):
    import hashlib
    return hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(w)).tobytes() for w in ws)).hexdigest()


def test_zero_update_scores_exactly_like_the_before_model(monkeypatch, capsys):
    cfg, cloud, clients, edges = _cloud_with_ck(monkeypatch)
    e = edges[0]
    ew = e.model.get_weights()
    capsys.readouterr()
    ups = [ClientUpdate([np.array(w, copy=True) for w in ew], 5, 0.1, 0.0, client_id=int(e.clients[0].client_id))]
    cloud._ck.score(e, 1, 2, ups, ew, e._base_w_idx)
    out = capsys.readouterr().out
    b = collect_kv(out.splitlines(), "[CkBefore]")[0]
    s = collect_kv(out.splitlines(), "[CkScore]")[0]
    assert parse_list(b["c"]) == parse_list(s["c"])             # Δ = 0 → c_i == c_before
    assert b["ncm_acc"] == s["ncm_acc"]


def test_theta_i_takes_trainable_weights_but_keeps_edge_stats_and_head(monkeypatch):
    cfg, cloud, clients, edges = _cloud_with_ck(monkeypatch)
    e = edges[0]
    ew = e.model.get_weights()
    stat = set(get_bn_stat_indices(e.model))
    assert stat                                                  # 夹具里有 BN
    up_w = [np.asarray(w) + 1.0 for w in ew]                     # 每个坐标都不同于 edge_w
    seen = []
    orig = type(cloud._ck)._measure

    def spy(self, weights, *a, **k):
        seen.append([np.array(w, copy=True) for w in weights])
        return orig(self, weights, *a, **k)
    monkeypatch.setattr(type(cloud._ck), "_measure", spy)
    cloud._ck.score(e, 1, 2, [ClientUpdate(up_w, 5, 0.1, 0.0, client_id=0)], ew, e._base_w_idx)
    before, theta_i = seen
    for j in range(len(ew)):
        if j in set(e._base_w_idx) and j not in stat:
            np.testing.assert_array_equal(theta_i[j], up_w[j])   # 可训练权重 ← 上传
        else:
            np.testing.assert_array_equal(theta_i[j], ew[j])     # 统计量 / head ← edge_w
    for a, b in zip(before, ew):
        np.testing.assert_array_equal(a, b)


def test_a_scoring_call_mutates_nothing_and_touches_no_global_rng(monkeypatch):
    cfg, cloud, clients, edges = _cloud_with_ck(monkeypatch)
    e = edges[0]
    ew = e.model.get_weights()
    ups = [ClientUpdate([np.asarray(w) + 0.1 for w in ew], 5, 0.1, 0.0, client_id=0)]
    up_h, ew_h = _h(ups[0][0]), _h(ew)
    model_h = (_h(e.model.get_weights()), _h(cloud.global_model.get_weights()),
               [_h(c.model.get_weights()) for c in clients])
    st_py, st_np = random.getstate(), np.random.get_state()[1].copy()
    cloud._ck.score(e, 1, 2, ups, ew, e._base_w_idx)
    assert _h(ups[0][0]) == up_h and _h(ew) == ew_h
    assert (_h(e.model.get_weights()), _h(cloud.global_model.get_weights()),
            [_h(c.model.get_weights()) for c in clients]) == model_h
    assert random.getstate() == st_py
    np.testing.assert_array_equal(np.random.get_state()[1], st_np)


def test_non_scoring_rounds_do_nothing_and_missing_clean_set_is_loud(monkeypatch, capsys):
    cfg, cloud, clients, edges = _cloud_with_ck(monkeypatch)
    e = edges[0]
    ew = e.model.get_weights()
    capsys.readouterr()
    cloud._ck.score(e, 1, 1, [ClientUpdate(ew, 5, 0.1, 0.0, client_id=0)], ew, e._base_w_idx)   # eff=1
    assert capsys.readouterr().out == ""
    e.clean_x = None
    with pytest.raises(ValueError, match="干净集"):
        cloud._ck.score(e, 1, 2, [ClientUpdate(ew, 5, 0.1, 0.0, client_id=0)], ew, e._base_w_idx)


# ══════════════════════════════════════════════════════════════════════════
# 批量 PGD 与逐类 PGD 的一致性（analysis/ck_snapshot.py）
# ══════════════════════════════════════════════════════════════════════════
def _tiny():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((8, 8, 3))
    x = tf.keras.layers.Conv2D(6, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(5, activation="softmax")(x))


def test_batched_pgd_matches_per_class_pgd():
    m = _tiny()
    feat = CK.feature_model(m)
    r = np.random.default_rng(0)
    xp = r.normal(size=(40, 8, 8, 3)).astype(np.float32)
    yp = (np.arange(40) % 5).astype(np.int64)
    xa = r.normal(size=(12, 8, 8, 3)).astype(np.float32)
    protos, _ = FS.class_prototypes(CK.extract(feat, xp), yp, 5)
    lo, hi = np.full(3, -50.0, np.float32), np.full(3, 50.0, np.float32)
    z = np.zeros(3, np.float32)
    np.testing.assert_array_equal(CK.ck_reached_batched(feat, protos, xa, z, 3, lo, hi, 5),
                                  CK.ck_reached(feat, protos, xa, z, 3, lo, hi, 5))
    e = np.full(3, 1.0, np.float32)
    a = CK.ck_reached_batched(feat, protos, xa, e, 4, lo, hi, 5)
    b = CK.ck_reached(feat, protos, xa, e, 4, lo, hi, 5)
    assert abs(float(a.mean()) - float(b.mean())) <= 0.05               # 批组成不同，数值上可有个别翻转
    # 分块（max_batch 小）与不分块相同
    c = CK.ck_reached_batched(feat, protos, xa, e, 4, lo, hi, 5, max_batch=12)
    np.testing.assert_array_equal(a, c)
