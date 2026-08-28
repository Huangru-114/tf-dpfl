"""
tests/test_probe_layermap.py  —  get_weights() 下标 ↦ (层, 张量种类) 的映射（需 TF）。

映射错了不会崩，只会让 Figure 2/3/4 的每一根柱子贴错标签 —— 这是最难事后发现的
一类错误，所以断言精确到「哪个下标是哪个种类」。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from models.cnn import build_model, get_base_head_indices     # noqa: E402
from probe.layermap import weight_index_map, layer_groups     # noqa: E402


def _bn_model(num_classes=3, img=16):
    L = tf.keras.layers
    inp = L.Input((img, img, 3))
    x = L.Conv2D(4, 3, padding="same", name="c1")(inp)
    x = L.BatchNormalization(name="bn1")(x)
    x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, L.Dense(num_classes, activation="softmax", name="head")(x))


def test_entries_align_one_to_one_with_get_weights():
    m = _bn_model()
    imap = weight_index_map(m)
    ws = m.get_weights()
    assert len(imap["entries"]) == len(ws)
    for e, w in zip(imap["entries"], ws):
        assert e["shape"] == tuple(w.shape)
        assert e["size"] == int(w.size)


def test_batchnorm_four_tensors_are_classified_exactly():
    m = _bn_model()
    kinds = [e["kind"] for e in weight_index_map(m)["entries"]]
    assert kinds == ["kernel", "bias", "bn_gamma", "bn_beta",
                     "bn_moving_mean", "bn_moving_var", "kernel", "bias"]


def test_bn_state_indices_are_exactly_the_two_buffers():
    """
    moving_mean / moving_variance 是 model state / buffer，**不在** trainable_variables
    里。它们是后续 Metric J 的观察对象，必须能被单独取出来。
    """
    m = _bn_model()
    imap = weight_index_map(m)
    assert imap["bn_state_indices"] == [4, 5]
    for i in imap["bn_state_indices"]:
        assert imap["entries"][i]["trainable"] is False


def test_trainable_flag_matches_keras():
    m = _bn_model()
    imap = weight_index_map(m)
    n_trainable = sum(1 for e in imap["entries"] if e["trainable"])
    assert n_trainable == len(m.trainable_variables)


def test_layer_groups_partition_all_indices_without_overlap():
    m = _bn_model()
    imap = weight_index_map(m)
    g = layer_groups(imap)
    seen = [i for idxs in g.values() for i in idxs]
    assert sorted(seen) == list(range(len(m.get_weights())))
    assert len(seen) == len(set(seen))


def test_layer_groups_restricted_to_backbone_drops_the_head_layer():
    """陷阱 #9：只在 backbone 索引上分组时，head 层必须整个消失。"""
    m = _bn_model()
    imap = weight_index_map(m)
    base = get_base_head_indices(m, 3)["base_weight_indices"]
    g = layer_groups(imap, base)
    assert "head" not in g
    assert set(g) == {"c1", "bn1"}


def test_consistent_with_get_base_head_indices():
    m = _bn_model()
    imap = weight_index_map(m)
    split = get_base_head_indices(m, 3)
    head_idx = set(split["head_weight_indices"])
    # head 的两个下标必须恰好落在同一个层里，且那个层只有它们
    head_layers = {imap["entries"][i]["layer"] for i in head_idx}
    assert len(head_layers) == 1
    only = head_layers.pop()
    assert set(imap["by_layer"][only]) == head_idx


def test_resnet10_bn_tensors_are_found():
    """真实模型（Bad-PFL 论文对齐用的 resnet10，带 BN）上的端到端核对。"""
    m = build_model(input_shape=(32, 32, 3), num_classes=10, arch="resnet10")
    imap = weight_index_map(m)
    assert len(imap["bn_state_indices"]) > 0
    assert len(imap["bn_state_indices"]) % 2 == 0        # mean/var 成对出现
    n_bn_layers = sum(1 for e in imap["entries"] if e["kind"] == "bn_moving_mean")
    assert n_bn_layers * 2 == len(imap["bn_state_indices"])
    # backbone 分组必须能覆盖到 BN 的 state 张量（它们不在 trainable 里，
    # 但确实是 get_weights() 的一部分、也确实随训练变化）
    base = get_base_head_indices(m, 10)["base_weight_indices"]
    assert set(imap["bn_state_indices"]) <= set(base)


def test_resnet10_by_kind_covers_every_index():
    m = build_model(input_shape=(32, 32, 3), num_classes=10, arch="resnet10")
    imap = weight_index_map(m)
    seen = [i for idxs in imap["by_kind"].values() for i in idxs]
    assert sorted(seen) == list(range(len(m.get_weights())))
    assert "other" not in imap["by_kind"], \
        f"有张量未被分类: {[imap['entries'][i] for i in imap['by_kind'].get('other', [])]}"


def test_no_bn_model_has_empty_bn_state():
    """fedavg_cnn 没有任何归一化层 —— BN 相关的指标在它上面必须是空集，不是报错。"""
    m = build_model(input_shape=(32, 32, 3), num_classes=10, arch="fedavg_cnn")
    assert weight_index_map(m)["bn_state_indices"] == []
