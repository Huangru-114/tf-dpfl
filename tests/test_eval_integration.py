"""
tests/test_eval_integration.py  —  A4：评估口径的集成断言（AUDIT A02 / A06 / A28）

用一个很小的真 HFL（2 edge × 3 个 FedRep 客户端，其中 2 个 Bad-PFL 恶意端）跑一轮
BackdoorCloudServer.run_round，断言：

  A28 / D-033（硬规则）：每个客户端的主 pm_acc 与主 ASR 在**同一个**模型上测，
      且它 = compose_pm(所在 edge 的当前权重, 该端私有部分)；陈旧列用 client.model。
  A02 / D-015：主 ASR 的 ξ 与被评估的模型无关（换受害者模型结果不变）；选中者 = rng 解析值。
  A06：four_way 下打出 [ASR4]；模板全关时这些副列一条都不打（旧口径逐字不变）。

需要 TF。这也是一条接线测试：新评估路径在一轮完整的 run_round 里跑通。
"""

import hashlib

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import attack.backdoor_eval as BE                          # noqa: E402
from client.client_badpfl import BadPFLMixin                # noqa: E402
from client.client_base import FLClientBase                 # noqa: E402
from client.compose import compose_client_class             # noqa: E402
from client.hier_fedrep import HierFedRepClient             # noqa: E402
from data.partition import merge_test_datasets              # noqa: E402
from main import select_eval_attacker                       # noqa: E402
from models.autoencoder import build_generator              # noqa: E402
from models.model_utils import clone_model                  # noqa: E402
from server.backdoor_server import BackdoorCloudServer      # noqa: E402
from server.hier_fedrep import HierFedRepEdgeServer         # noqa: E402
from utils.pm import compose_pm                             # noqa: E402

IMG, NCLS, MAL = 16, 10, {0, 3}

P2_EVAL = {
    "training": {"fedrep_bn_stats": "private", "fedrep_poison_phases": "body"},
    "evaluation": {"pm_model": "fresh", "asr_columns": "four_way"},
    "backdoor": {"badpfl_xi": "pgd", "badpfl_bn_mode": "official",
                 "badpfl_generator": "official", "eval_xi_model": "fixed_attacker",
                 "poison_sampling": "bernoulli"},
}


def _cfg(aligned: bool):
    c = {
        "seed": 42,
        "data": {"dataset": "cifar10", "img_size": IMG, "batch_size": 8, "num_classes": NCLS},
        "federation": {"n_clients": 6, "n_edges": 2, "client_fraction": 0.5,
                       "edge_rounds": 1, "n_rounds": 3, "n_workers": 1},
        "training": {"drift_correction": "hier_fedrep", "learning_rate": 0.1,
                     "lr_decay": 1.0, "local_epochs": 1, "plocal_epochs": 1,
                     "head_lr_rep": 0.05},
        "evaluation": {"eval_interval": 1},
        "backdoor": {"enabled": True, "malicious_strategy": "badpfl", "target_label": 0,
                     "poison_ratio": 0.5, "eval_interval": 1, "badpfl_gen_steps": 1,
                     "asr_max_samples": 0, "feature_eval": False, "forgetting_epochs": 0},
        "defense": {"name": "none"},
    }
    if aligned:
        for sec, kv in P2_EVAL.items():
            c[sec].update(kv)
    return c


def _model():
    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input((IMG, IMG, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(NCLS, activation="softmax")(x))


def _setup(aligned: bool):
    cfg = _cfg(aligned)
    g = _model()
    Mal = compose_client_class(HierFedRepClient, None, BadPFLMixin)
    shared = build_generator(cfg)
    opt = tf.keras.optimizers.Adam(0.01)
    clients = []
    for cid in range(6):
        r = np.random.default_rng(cid)
        x = r.normal(size=(24, IMG, IMG, 3)).astype(np.float32)
        y = (np.arange(24) % NCLS).astype(np.int64)
        tx = r.normal(size=(12, IMG, IMG, 3)).astype(np.float32)
        ty = (np.arange(12) % NCLS).astype(np.int64)
        cls = Mal if cid in MAL else HierFedRepClient
        c = cls(client_id=cid, dataset=tf.data.Dataset.from_tensor_slices((x, y)).batch(8),
                model=clone_model(g), config=cfg, n_samples=24)
        c.is_malicious = cid in MAL
        c.set_train_source(x, y, np.arange(24))
        c.set_test_dataset(tf.data.Dataset.from_tensor_slices((tx, ty)).batch(8))
        c.assigned_edge = cid // 3
        if cid in MAL:
            c.set_shared_generator(shared, opt)
        clients.append(c)
    edges = [HierFedRepEdgeServer(e, clients[3 * e:3 * e + 3], clone_model(g), cfg)
             for e in range(2)]
    for e in edges:
        e.set_test_dataset(merge_test_datasets(e.clients, 8))
    gt = merge_test_datasets(edges, 8)
    att = select_eval_attacker(clients, MAL, cfg["seed"])
    mal0 = [c for c in clients if c.client_id in MAL][0]
    cloud = BackdoorCloudServer(
        global_model=g, edge_servers=edges, test_dataset=gt, config=cfg,
        bd_cfg=cfg["backdoor"], x_test=None, y_test=None,
        trigger_fn=lambda model, x, y=None: mal0.eval_trigger(model, x, y),
        malicious_ids=MAL, eval_attacker=att)
    return cfg, cloud, clients, edges


def _h(model):
    return hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes()
                                   for w in model.get_weights())).hexdigest()


def test_selected_attacker_is_the_rng_value_and_order_free():
    cfg, cloud, clients, _ = _setup(True)
    mal = sorted(MAL)
    want = mal[int(np.random.default_rng([42, 0xA02]).integers(len(mal)))]
    assert cloud.eval_attacker.client_id == want
    assert select_eval_attacker(list(reversed(clients)), MAL, 42).client_id == want


def test_main_pm_acc_and_main_asr_use_the_same_composed_model(monkeypatch, capsys):
    cfg, cloud, clients, edges = _setup(True)
    by_ds = {id(c.test_dataset): c.client_id for c in clients}
    acc_h, asr_h, acc_stale_is_model = {}, {}, {}

    orig_eval = FLClientBase.evaluate_on

    def rec_eval(self, fallback_dataset=None, model=None, verbose=True):
        if verbose:
            acc_h[self.client_id] = _h(model)
        else:
            acc_stale_is_model[self.client_id] = model is self.model
        return orig_eval(self, fallback_dataset, model, verbose)
    monkeypatch.setattr(FLClientBase, "evaluate_on", rec_eval)

    orig_asr = BE.compute_asr_four_way

    def rec_asr(model, ds, *a, **k):
        if id(ds) in by_ds:
            asr_h[by_ds[id(ds)]] = _h(model)
        return orig_asr(model, ds, *a, **k)
    monkeypatch.setattr(BE, "compute_asr_four_way", rec_asr)

    cloud.run_round(1)
    out = capsys.readouterr().out

    for c in clients:
        e = cloud.edge_of(c)
        idx, val = c.private_state()
        want = compose_pm(e.model.get_weights(), val, idx)
        want_h = hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes()
                                         for w in want)).hexdigest()
        assert acc_h[c.client_id] == asr_h[c.client_id] == want_h, c.client_id
        assert acc_stale_is_model[c.client_id]              # 陈旧列 = client.model
    for tag in ("[ASR4]", "[ASRwb]", "[StaleASR]", "[Stale]"):
        assert tag in out, f"没有打出 {tag}"


def test_main_xi_does_not_depend_on_the_victim_model():
    cfg, cloud, clients, edges = _setup(True)
    cloud.run_round(1)
    cloud.begin_pm_eval()
    x = np.random.default_rng(9).normal(size=(12, IMG, IMG, 3)).astype(np.float32)
    y = (np.arange(12) % NCLS).astype(np.int64)
    a = cloud._attacker_trigger(1, 0, "fresh")(clients[1].model, x, y)
    b = cloud._attacker_trigger(1, 0, "fresh")(edges[1].model, x, y)
    np.testing.assert_array_equal(a, b)
    wb1 = clients[0].eval_trigger(clients[1].model, x, y, rng=np.random.default_rng(1))
    wb2 = clients[0].eval_trigger(edges[1].model, x, y, rng=np.random.default_rng(1))
    assert not np.array_equal(wb1, wb2)                     # 反向锚点：白盒 ξ 随受害者变


def test_legacy_config_prints_no_side_columns_and_uses_client_model(monkeypatch, capsys):
    cfg, cloud, clients, edges = _setup(False)
    seen = {}
    orig_eval = FLClientBase.evaluate_on

    def rec_eval(self, fallback_dataset=None, model=None, verbose=True):
        seen[self.client_id] = model is self.model
        return orig_eval(self, fallback_dataset, model, verbose)
    monkeypatch.setattr(FLClientBase, "evaluate_on", rec_eval)
    cloud.run_round(1)
    out = capsys.readouterr().out
    assert all(seen.values()) and len(seen) == 6
    for tag in ("[ASR4]", "[ASRwb]", "[StaleASR]", "[Stale]"):
        assert tag not in out
