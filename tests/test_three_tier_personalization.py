"""
tests/test_three_tier_personalization.py  —  S8：3-E 三层个性化（需要 TF）

纯 python 的规则在 tests/test_tier_split.py。这里守 TF 那一半：

  1. 真 ResNet-10（resnet10 / resnet10_torch）的三分组：k=1 恰好是 stage4 的 15 个张量
     （两个 3×3 conv、shortcut conv、三个 BN 的 γ/β/统计量），k=2 再加 stage3 的 15 个；
     k=0 时前四个键与改动前逐字相同；
  2. 真跑 CloudServer + HierFedRepEdgeServer + HierFedRepClient（2 edge × 2 端）：
     - 首次广播：各 edge 的 edge 段 = 全局初值（同一起点）；
     - cloud 聚合：cloud 段 = edge 的样本加权均值，edge 段 = 聚合前的值（全局那一段永远是初值，D-057 / Q4）；
     - 之后的广播：edge 段保留本 edge 自己的，其余照常被覆盖；client 收到的是本 edge 的 edge 段；
     - fresh-PM（D-033）：edge 段来自所在 edge；
     - 反向锚点：k=0 与不写这个键的 [Checksum] 逐轮相同；k=1 与 k=0 不同（测试有分辨力）。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from aggregation.fedavg import aggregate                  # noqa: E402
from client.hier_fedrep import HierFedRepClient            # noqa: E402
from data.partition import merge_test_datasets             # noqa: E402
from models.cnn import (build_resnet10, build_resnet10_torch,  # noqa: E402
                        get_base_head_indices, get_bn_stat_indices,
                        weight_owner_names)
from models.model_utils import clone_model                 # noqa: E402
from server.hier_fedrep import HierFedRepEdgeServer        # noqa: E402
from server.server import CloudServer                      # noqa: E402
from utils.kvline import parse_kv                          # noqa: E402

NCLS = 10


# ══════════════════════════════════════════════════════════════════════════
# 1. 真 ResNet-10 的三分组
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module", params=["resnet10", "resnet10_torch"])
def resnet(request):
    build = {"resnet10": build_resnet10, "resnet10_torch": build_resnet10_torch}[request.param]
    return build(input_shape=(16, 16, 3), num_classes=NCLS)


def _owned(model, prefixes):
    return {i for i, o in enumerate(weight_owner_names(model))
            if o is not None and o.startswith(prefixes)}


def test_k0_leaves_the_fedrep_split_untouched(resnet):
    old = get_base_head_indices(resnet, NCLS)
    new = get_base_head_indices(resnet, NCLS, 0)
    assert new == old
    assert new["edge_weight_indices"] == [] and new["cloud_weight_indices"] == new["base_weight_indices"]


@pytest.mark.parametrize("k,stages,n", [(1, ("stage4_",), 15), (2, ("stage3_", "stage4_"), 30)])
def test_edge_segment_is_exactly_the_last_k_blocks(resnet, k, stages, n):
    s = get_base_head_indices(resnet, NCLS, k)
    edge, cloud, base, head = (set(s["edge_weight_indices"]), set(s["cloud_weight_indices"]),
                               set(s["base_weight_indices"]), set(s["head_weight_indices"]))
    assert edge == _owned(resnet, stages) and len(edge) == n
    assert edge <= base and not (edge & head)
    assert edge | cloud == base and not (edge & cloud)
    # 块内带权重的层全在里面：两个 3×3 conv、shortcut conv、三个 BN（γ/β + 2 个统计量）
    owners = weight_owner_names(resnet)
    for st in stages:
        assert {owners[i] for i in edge if owners[i].startswith(st)} == {
            f"{st}conv1", f"{st}bn1", f"{st}conv2", f"{st}bn2", f"{st}sc_conv", f"{st}sc_bn"}
    assert len(edge & set(get_bn_stat_indices(resnet))) == 6 * k


def test_arch_without_stage_names_raises():
    m = tf.keras.Sequential([tf.keras.layers.Input((4,)), tf.keras.layers.Dense(NCLS)])
    assert get_base_head_indices(m, NCLS)["edge_weight_indices"] == []
    with pytest.raises(ValueError, match="stage4_"):
        get_base_head_indices(m, NCLS, 1)


# ══════════════════════════════════════════════════════════════════════════
# 2. 真跑一轮：2 edge × 2 个 FedRep 端
# ══════════════════════════════════════════════════════════════════════════
IMG = 8


def _model():
    """层名沿用 ResNet-10 的 stage 前缀（按名字切分），但只有几百个参数，CPU 上秒级。"""
    L = tf.keras.layers
    inp = tf.keras.Input((IMG, IMG, 3))
    x = inp
    for name in ("stem", "stage3", "stage4"):
        conv = f"{name}_conv" if name == "stem" else f"{name}_conv1"
        bn = f"{name}_bn" if name == "stem" else f"{name}_bn1"
        x = L.Conv2D(4, 3, padding="same", use_bias=False, name=conv)(x)
        x = L.BatchNormalization(momentum=0.9, epsilon=1e-5, name=bn)(x)
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, L.Dense(NCLS, activation="softmax", name="head")(x))


def _setup(k):
    """k=None → 配置里不写这个键（改动前的配置长这样）。"""
    tf.keras.utils.set_random_seed(0)
    cfg = {
        "seed": 42,
        "data": {"dataset": "cifar10", "img_size": IMG, "batch_size": 8, "num_classes": NCLS},
        "federation": {"n_clients": 4, "n_edges": 2, "client_fraction": 1.0,
                       "edge_rounds": 2, "n_rounds": 3, "n_workers": 1},
        "training": {"drift_correction": "hier_fedrep", "learning_rate": 0.1,
                     "lr_decay": 1.0, "local_epochs": 1, "plocal_epochs": 1,
                     "head_lr_rep": 0.05, "fedrep_bn_stats": "private"},
        "evaluation": {"eval_interval": 1, "pm_model": "fresh"},
        "backdoor": {"enabled": False},
        "defense": {"name": "none"},
    }
    if k is not None:
        cfg["federation"]["edge_shared_blocks"] = k
    g = _model()
    clients = []
    for cid in range(4):
        r = np.random.default_rng(cid)
        # 两个 edge 的数据分布不同（均值差 2）→ edge 段会各自走开
        x = (r.normal(size=(16, IMG, IMG, 3)) + 2.0 * (cid // 2)).astype(np.float32)
        y = ((np.arange(16) + 3 * (cid // 2)) % NCLS).astype(np.int64)
        c = HierFedRepClient(client_id=cid,
                             dataset=tf.data.Dataset.from_tensor_slices((x, y)).batch(8),
                             model=clone_model(g), config=cfg, n_samples=16)
        c.set_test_dataset(tf.data.Dataset.from_tensor_slices((x[:8], y[:8])).batch(8))
        c.assigned_edge = cid // 2
        clients.append(c)
    edges = [HierFedRepEdgeServer(e, clients[2 * e:2 * e + 2], clone_model(g), cfg)
             for e in range(2)]
    for e in edges:
        e.set_test_dataset(merge_test_datasets(e.clients, 8))
    cloud = CloudServer(g, edges, merge_test_datasets(edges, 8), cfg)
    return cloud, clients, edges


def _eq(a, b, idx):
    return all(np.array_equal(a[i], b[i]) for i in idx)


def _differ(a, b, idx):
    return any(not np.array_equal(a[i], b[i]) for i in idx)


def test_settings6_is_printed_where_the_indices_are_computed(capsys):
    cloud, _, _ = _setup(1)
    d = [parse_kv(ln, "[设定6]") for ln in capsys.readouterr().out.splitlines()]
    d = [x for x in d if x is not None]
    assert len(d) == 1
    assert d[0]["edge_shared_blocks"] == 1 and d[0]["edge_shared_prefixes"] == "stage4"
    assert d[0]["n_edge_tensors"] == len(cloud._edge_seg_idx) == 5      # conv + BN 的 4 项


def test_first_broadcast_gives_every_edge_the_same_initial_segment():
    cloud, _, edges = _setup(1)
    for e in edges:                                  # 先把 edge 弄乱：首次接收必须整体覆盖
        e.model.set_weights([w + 1.0 + e.edge_id for w in e.model.get_weights()])
    cloud.broadcast_to_edges()
    gw = cloud.global_model.get_weights()
    for e in edges:
        assert _eq(e.model.get_weights(), gw, range(len(gw)))


def test_cloud_does_not_aggregate_the_edge_segment_and_broadcast_keeps_it():
    cloud, clients, edges = _setup(1)
    idx = cloud._edge_seg_idx
    s = get_base_head_indices(cloud.global_model, NCLS, 1)
    cloud_idx, head_idx = s["cloud_weight_indices"], s["head_weight_indices"]
    init = [w.copy() for w in cloud.global_model.get_weights()]

    cloud.run_round(1)
    gw = cloud.global_model.get_weights()
    ew = [e.model.get_weights() for e in edges]          # = 各 edge 上传的权重（还没被广播覆盖）
    # 全局的 edge 段 = 初值；cloud 段 = edge 的样本加权均值
    assert _eq(gw, init, idx)
    want = aggregate([(w, e.n_samples, 0.0, 0.0) for w, e in zip(ew, edges)])
    assert _eq(gw, want, cloud_idx)
    # edge 段真的在训练、而且两个 edge 各走各的（否则下面的「保留」断言没有分辨力）
    assert _differ(ew[0], init, idx) and _differ(ew[0], ew[1], idx)

    cloud.broadcast_to_edges()
    after = [e.model.get_weights() for e in edges]
    for e, before, now in zip(edges, ew, after):
        assert _eq(now, before, idx), f"edge{e.edge_id} 的 edge 段被 cloud 广播覆盖了"
        assert _eq(now, gw, cloud_idx + head_idx)
        # client 收到的是本 edge 的 edge 段（私有统计量除外，A27）
        e.broadcast_to_clients(e.clients, global_weights=e._global_weights_ref)
        for c in e.clients:
            priv = set(c.private_state()[0])
            keep = [i for i in idx if i not in priv]
            assert keep and _eq(c.model.get_weights(), now, keep)

    cloud.run_round(2)
    assert _eq(cloud.global_model.get_weights(), init, idx)       # 永远是初值


def test_fresh_pm_takes_the_segment_of_the_clients_own_edge():
    cloud, clients, edges = _setup(1)
    idx = cloud._edge_seg_idx
    cloud.run_round(1)
    ew = {e.edge_id: e.model.get_weights() for e in edges}
    assert _differ(ew[0], ew[1], idx)
    cloud.begin_pm_eval()
    for c in clients:
        priv = set(c.private_state()[0])
        keep = [i for i in idx if i not in priv]
        pm = cloud.main_pm(c).get_weights()
        assert _eq(pm, ew[c.assigned_edge], keep)
        assert _differ(pm, ew[1 - c.assigned_edge], keep)


def _checksums(k, capsys):
    capsys.readouterr()
    cloud, _, _ = _setup(k)
    for r in (1, 2):
        cloud.run_round(r)
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[Checksum]")]


def test_k0_is_byte_identical_to_not_writing_the_key(capsys):
    """反向锚点：k=0 与改动前的配置（不写这个键）逐轮 checksum 相同；k=1 不同。"""
    absent, zero, one = (_checksums(k, capsys) for k in (None, 0, 1))
    assert len(absent) == 2 and absent == zero
    assert one != zero


def test_k0_still_aggregates_every_body_index_at_the_cloud():
    """k=0 下全局的 stage4 照常被聚合（与 k=1 的「保持初值」相反）。"""
    cloud, _, edges = _setup(0)
    assert cloud._edge_seg_idx == []
    stage4 = get_base_head_indices(cloud.global_model, NCLS, 1)["edge_weight_indices"]
    init = [w.copy() for w in cloud.global_model.get_weights()]
    cloud.run_round(1)
    assert _differ(cloud.global_model.get_weights(), init, stage4)
