"""
tests/test_static_poison_tf.py  —  静态触发器逐批投毒（`malicious_strategy: static`，D-141）在真客户端 / 小 HFL 上的接线（需要 TF）

纯算术、配置、登记、解析在 test_static_poison.py；这里验证：
  · 类组合：恶意端 = StaticPoison → HierFedRep（mixin 组合，不是替换，陷阱 #1）；良性端就是方法类本身；
  · 每批恰好掩码行被换成 trigger(增强后的 x) + 目标标签，掩码 = poison_mask(self.rng, n, ρ, bernoulli) 逐位相同；
  · 攻击时间窗外：批原样返回、不消耗投毒随机数、不打 [Poison]；没注入触发器就抛错；
  · ρ = 0 的 static 恶意端与同编号的良性端上传逐位相同（mixin 除了投毒不碰训练）；
  · DBA：10 个攻击者按编号轮流分到官方 4 块（3/3/2/2），每块只改自己的 6 个像素；评估触发器 = 24 个像素的并集；
  · 评估：没有固定攻击者（select_eval_attacker → None），评估触发器 = 静态触发器；
  · 小 HFL（交错调度 + 统一网格 + 云聚合后评估点，per_epoch 取数）跑 3 个云轮：没有被吞掉的客户端异常、
    [Poison] 计数 > 0、云聚合后评估点有攻击成功率；同配置跑两次 [Checksum] 逐位相同。
"""

import random

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import sys                                                    # noqa: E402
from pathlib import Path                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

import collect_metrics as CM                                  # noqa: E402
import test_eval_integration as TEI                           # noqa: E402
from attack.backdoor import make_dba_local_triggers           # noqa: E402
from attack.poison_mask import poison_mask                    # noqa: E402
from attack.triggers import build_trigger, trigger_space, make_badnet_trigger   # noqa: E402
from client.client_static import StaticPoisonMixin            # noqa: E402
from client.compose import compose_client_class               # noqa: E402
from client.hier_fedrep import HierFedRepClient               # noqa: E402
from data.partition import merge_test_datasets                # noqa: E402
from main import build_eval_trigger, resolve_client_classes, select_eval_attacker   # noqa: E402
from models.model_utils import clone_model                    # noqa: E402
from server import eval_grid as EG                            # noqa: E402
from server.backdoor_server import BackdoorCloudServer        # noqa: E402
from server.hier_fedrep import HierFedRepEdgeServer           # noqa: E402
from test_eval_grid_tf import _prod_seeding                   # noqa: E402

IMG, NCLS, MAL, TARGET = TEI.IMG, TEI.NCLS, {0, 3}, 0
OFFICIAL_DBA = [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
                [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
                [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
                [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]]
Mal = compose_client_class(HierFedRepClient, None, StaticPoisonMixin)


def _cfg(**bd):
    c = TEI._cfg(True)
    c["data"].update(batch_pipeline="per_epoch", augment=True)
    c["backdoor"].update({"malicious_strategy": "static", "trigger": "badnet", "poison_ratio": 0.2,
                          "badnet_size": 3, "badnet_value": 1.0, "target_label": TARGET, **bd})
    for k in ("badpfl_xi", "badpfl_bn_mode", "badpfl_generator", "eval_xi_model", "badpfl_gen_steps"):
        c["backdoor"].pop(k, None)
    return c


def _data(cid, n=24):
    r = np.random.default_rng(cid)
    return (r.normal(size=(n, IMG, IMG, 3)).astype(np.float32), (np.arange(n) % NCLS).astype(np.int64))


def _client(cfg, cid=0, cls=Mal, model=None, malicious=True, trigger=True):
    x, y = _data(cid)
    m = model if model is not None else clone_model(TEI._model())
    c = cls(client_id=cid, dataset=tf.data.Dataset.from_tensor_slices((x, y)).batch(8),
            model=m, config=cfg, n_samples=len(x))
    c.is_malicious = malicious
    c.set_train_source(x, y, np.arange(len(x)))
    tx, ty = _data(100 + cid, 12)
    c.set_test_dataset(tf.data.Dataset.from_tensor_slices((tx, ty)).batch(8))
    if trigger and hasattr(c, "set_static_trigger"):
        c.set_static_trigger(build_trigger(cfg["backdoor"], img_size=IMG, config=cfg))
    return c


def _copy_rng(rng):
    r = np.random.default_rng()
    r.bit_generator.state = rng.bit_generator.state
    return r


# ══════════════════════════════════════════════════════════════════════════
# 客户端
# ══════════════════════════════════════════════════════════════════════════
def test_classes_are_composed_not_replaced():
    cfg = _cfg()
    cfg["training"]["drift_correction"] = "hier_fedrep"
    ben, mal = resolve_client_classes(cfg)
    assert ben is HierFedRepClient
    mro = mal.__mro__
    assert mro.index(StaticPoisonMixin) < mro.index(HierFedRepClient)


def test_each_batch_poisons_exactly_the_mask_rows_after_augmentation():
    cfg = _cfg()
    c = _client(cfg)
    c.on_round_start(1)
    assert c._attack_active
    trig = make_badnet_trigger(*trigger_space(cfg), size=3, value=1.0)
    total = 0
    for x, y in c.epoch_batches():                          # 已增强的批（与训练同一个取数函数）
        x_np, y_np = np.asarray(x), np.asarray(y)
        want = poison_mask(_copy_rng(c.rng), len(x_np), 0.2, "bernoulli")
        xb, yb = c.on_batch(x, y)
        xb, yb = np.asarray(xb), np.asarray(yb)
        assert np.array_equal(xb[want], trig(x_np[want]))
        assert np.all(yb[want] == TARGET)
        assert np.array_equal(xb[~want], x_np[~want]) and np.array_equal(yb[~want], y_np[~want])
        assert yb.dtype == y_np.dtype
        total += int(want.sum())
    assert c._static_n_poisoned == total and c._static_n_seen == 24


def test_outside_the_window_nothing_is_touched_and_no_rng_is_consumed(capsys):
    cfg = _cfg(attack_start_round=5, attack_stop_round=9)
    c = _client(cfg)
    for r, on in ((4, False), (5, True), (9, False)):
        c.on_round_start(r)
        assert c._attack_active is on
        if on:
            continue
        x, y = next(iter(c.epoch_batches()))
        st = repr(c.rng.bit_generator.state)
        xb, yb = c.on_batch(x, y)
        assert np.array_equal(np.asarray(xb), np.asarray(x)) and np.array_equal(np.asarray(yb), np.asarray(y))
        assert repr(c.rng.bit_generator.state) == st
        capsys.readouterr()
        assert c.on_upload([np.zeros(1)], r)[0].shape == (1,)
        assert "[Poison]" not in capsys.readouterr().out


def test_missing_trigger_raises_instead_of_training_clean():
    c = _client(_cfg(), trigger=False)
    with pytest.raises(RuntimeError, match="set_static_trigger"):
        c.on_round_start(1)


def test_rho0_static_attacker_uploads_exactly_what_the_benign_client_uploads(capsys):
    cfg = _cfg(poison_ratio=0.0)
    g = TEI._model()
    a = _client(cfg, cid=0, cls=Mal, model=clone_model(g))
    b = _client(cfg, cid=0, cls=HierFedRepClient, model=clone_model(g), malicious=False)
    w0 = g.get_weights()
    for c in (a, b):
        c.set_weights(w0, w0)
    ua, *_ = a.local_train(1)
    ub, *_ = b.local_train(1)
    assert all(np.array_equal(u, v) for u, v in zip(ua, ub))
    out = capsys.readouterr().out
    assert "[Poison] Round 1 | client=0 | n_poisoned=0" in out


def test_dba_local_triggers_and_the_global_union():
    cfg = _cfg(trigger="dba", dba_patterns=OFFICIAL_DBA)
    mal_ids = [6, 7, 9, 10, 15, 16, 17, 19, 20, 24]
    local = make_dba_local_triggers(cfg["backdoor"], mal_ids, config=cfg)
    x = np.zeros((1, 32, 32, 3), np.float32)
    touched = {}
    for rank, cid in enumerate(sorted(mal_ids)):
        diff = np.argwhere(np.any(local[cid](x)[0] != x[0], axis=-1))
        touched[cid] = {tuple(p) for p in diff}
        assert touched[cid] == {tuple(p) for p in OFFICIAL_DBA[rank % 4]}, cid
    counts = [sum(touched[c] == {tuple(p) for p in pat} for c in mal_ids) for pat in OFFICIAL_DBA]
    assert counts == [3, 3, 2, 2]
    glob = build_trigger(cfg["backdoor"], img_size=32, config=cfg)
    union = {tuple(p) for p in np.argwhere(np.any(glob(x)[0] != x[0], axis=-1))}
    assert union == {tuple(p) for pat in OFFICIAL_DBA for p in pat} and len(union) == 24


def test_eval_has_no_fixed_attacker_and_uses_the_static_trigger():
    cfg = _cfg()
    clients = [_client(cfg, cid=i, cls=Mal if i in MAL else HierFedRepClient, malicious=i in MAL)
               for i in range(4)]
    assert select_eval_attacker(clients, MAL, 42) is None
    static = build_trigger(cfg["backdoor"], img_size=IMG, config=cfg)
    fn = build_eval_trigger(cfg["backdoor"], static, clients, cfg)
    x, _ = _data(9, 5)
    assert np.array_equal(fn(None, x), static(x)) and np.array_equal(fn(clients[1].model, x, None), static(x))


# ══════════════════════════════════════════════════════════════════════════
# 小 HFL：交错调度 + 网格 + 云聚合后评估点
# ══════════════════════════════════════════════════════════════════════════
R, G, N = 2, 2, 3


def _setup(trigger="badnet"):
    cfg = _cfg(trigger=trigger, dba_patterns=[[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
    cfg["federation"].update(edge_schedule="interleaved", edge_rounds=R, n_rounds=N)
    iv = EG.full_interval(G, R)
    cfg["evaluation"].update(eval_interval=iv, eval_grid=G, post_agg_eval=True)
    cfg["backdoor"]["eval_interval"] = iv
    g = TEI._model()
    dba = make_dba_local_triggers(cfg["backdoor"], MAL, config=cfg) if trigger == "dba" else {}
    static = build_trigger(cfg["backdoor"], img_size=IMG, config=cfg)
    clients = []
    for cid in range(6):
        c = _client(cfg, cid=cid, cls=Mal if cid in MAL else HierFedRepClient,
                    model=clone_model(g), malicious=cid in MAL, trigger=False)
        if cid in MAL:
            c.set_static_trigger(dba.get(cid, static))
        c.assigned_edge = cid // 3
        clients.append(c)
    edges = [HierFedRepEdgeServer(e, clients[3 * e:3 * e + 3], clone_model(g), cfg) for e in range(2)]
    for e in edges:
        e.set_test_dataset(merge_test_datasets(e.clients, 8))
    cloud = BackdoorCloudServer(
        global_model=g, edge_servers=edges, test_dataset=merge_test_datasets(edges, 8), config=cfg,
        bd_cfg=cfg["backdoor"], x_test=None, y_test=None,
        trigger_fn=build_eval_trigger(cfg["backdoor"], static, clients, cfg),
        malicious_ids=MAL, eval_attacker=select_eval_attacker(clients, MAL, cfg["seed"]))
    return cfg, cloud


def _run(capsys, monkeypatch, trigger="badnet"):
    random.seed(42)
    cfg, cloud = _setup(trigger)
    _prod_seeding(monkeypatch)
    capsys.readouterr()
    cloud.run()
    return capsys.readouterr().out, cloud


@pytest.mark.parametrize("trigger", ["badnet", "dba"])
def test_tiny_federation_runs_poisons_and_has_post_aggregation_points(capsys, monkeypatch, trigger):
    out, cloud = _run(capsys, monkeypatch, trigger)
    assert cloud.eval_attacker is None
    m = CM.collect(out)
    assert m["client_failures"] == [] and m["errors"] == []
    pc = {r["client"]: r for r in m["poison_counts"]}
    assert set(pc) <= MAL and pc and all(r["n_poisoned"] > 0 for r in pc.values())
    post = m["post_agg_rounds"]
    assert [p["round"] for p in post] == [2, 3]
    assert all(p["local_benign_asr"] is not None for p in post)
    assert m["light_rounds"] == [] or all(r["local_benign_asr"] is not None for r in m["light_rounds"])
    assert m["frozen_rounds"] == []                     # 冻结触发器列关着


def test_same_config_twice_gives_bitwise_identical_checksums(capsys, monkeypatch):
    a, _ = _run(capsys, monkeypatch)
    b, _ = _run(capsys, monkeypatch)
    ca = [ln for ln in a.splitlines() if ln.startswith("[Checksum]")]
    cb = [ln for ln in b.splitlines() if ln.startswith("[Checksum]")]
    assert ca and ca == cb


# ══════════════════════════════════════════════════════════════════════════
# main.build_clients 的真接线（S3 划分 + 集中布点，同 test_designed_partition F 的夹具）
# ══════════════════════════════════════════════════════════════════════════
def test_build_clients_injects_per_client_triggers_keeps_clean_data_and_forces_serial(capsys):
    import main as M
    from data import designed_partition as D
    labels = np.repeat(np.arange(D.K), D.SUPPLY_PER_CLASS)
    images = np.zeros((len(labels), 4, 4, 3), dtype=np.float32)
    pats = [[[0, 0]], [[0, 3]], [[3, 0]], [[3, 3]]]                  # 4×4 图里放得下的 4 块「局部触发器」
    cfg = {"seed": 42,
           "data": {"dataset": "cifar10", "per_client_test_ratio": 0.25, "batch_size": 32, "img_size": 4,
                    "num_classes": 10, "batch_pipeline": "per_epoch"},
           "federation": {"partition": "designed", "n_clients": 100, "n_edges": 4, "n_workers": 4,
                          "design": {"condition": "C1", "strength": 0.25},
                          "edge_assignment": "block", "client_fraction": 0.1},
           "training": {"drift_correction": "hierfedavg", "learning_rate": 0.1, "lr_decay": 1.0,
                        "local_epochs": 1},
           "backdoor": {"enabled": True, "malicious_strategy": "static", "trigger": "dba",
                        "dba_patterns": pats, "target_label": 0, "poison_ratio": 0.2,
                        "malicious_placement": "by_edge", "malicious_per_edge": [10, 0, 0, 0]}}
    model = tf.keras.Sequential([tf.keras.Input((4, 4, 3)), tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(10, activation="softmax")])
    clients, assignments, _ = M.build_clients(images, labels, model, cfg, s3_out={})
    out = capsys.readouterr().out
    assert cfg["federation"]["n_workers"] == 1                        # 与 Bad-PFL 一样强制串行
    assert "poisoned dataset built" not in out                        # 不走旧的静态投毒数据集
    mal = sorted(c.client_id for c in clients if c.is_malicious)
    assert len(mal) == 10 and all(assignments[i] == 0 for i in mal)
    x = np.zeros((1, 4, 4, 3), np.float32)
    for rank, cid in enumerate(mal):
        c = clients[cid]
        assert isinstance(c, StaticPoisonMixin) and c._static_trigger is not None
        hit = {tuple(p) for p in np.argwhere(np.any(c._static_trigger(x)[0] != x[0], axis=-1))}
        assert hit == {tuple(pats[rank % 4][0])}, cid                 # DBA：本端的局部触发器
    assert not any(isinstance(c, StaticPoisonMixin) for c in clients if not c.is_malicious)
