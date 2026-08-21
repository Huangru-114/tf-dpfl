"""
tests/test_resnet10_fedrep_split.py  —  ResNet10 与 FedRep backbone/head 切分兼容性

exp004 把骨干从 fedavg_cnn 换成 resnet10（对齐 Bad-PFL 论文，BN 版）。fedrep 靠
get_base_head_indices 从 get_weights() 末尾扫描出 head=最后一个 Dense(num_classes)。
本测试守卫：resnet10 上这个切分仍正确（否则 fedrep 会静默切错、聚合错误的层）。

需要 TF，本地自动 skip，在集群跑。
"""

import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from models.cnn import build_resnet10, get_base_head_indices   # noqa: E402


NCLS = 10


def test_resnet10_ends_with_dense_num_classes():
    m = build_resnet10(input_shape=(32, 32, 3), num_classes=NCLS)
    last = m.layers[-1]
    assert isinstance(last, tf.keras.layers.Dense)
    assert last.units == NCLS
    # softmax 末层（from_logits=False，与全仓库一致）
    assert last.activation is tf.keras.activations.softmax


def test_get_base_head_indices_on_resnet10():
    """head 恰为最后 Dense(num_classes) 的 kernel+bias；backbone 为其余；映射非空。"""
    m = build_resnet10(input_shape=(32, 32, 3), num_classes=NCLS)
    split = get_base_head_indices(m, NCLS)

    w = m.get_weights()
    hwi = split["head_weight_indices"]
    bwi = split["base_weight_indices"]

    # head 恰 2 个权重（kernel + bias），且是 get_weights 的最后两个
    assert len(hwi) == 2, f"head 应为 2 个权重，得到 {len(hwi)}"
    assert set(hwi) == {len(w) - 2, len(w) - 1}, "head 不是末尾两个权重（切分错位）"

    # head kernel 的输出维 == num_classes，bias 长度 == num_classes
    shapes = sorted((w[i].ndim, tuple(w[i].shape)) for i in hwi)
    assert shapes[0] == (1, (NCLS,)), f"head bias 形状不对: {shapes[0]}"
    assert shapes[1][0] == 2 and shapes[1][1][-1] == NCLS, f"head kernel 形状不对: {shapes[1]}"

    # backbone = 其余全部；head/base 不相交且并集为全集
    assert set(bwi).isdisjoint(hwi)
    assert set(bwi) | set(hwi) == set(range(len(w)))

    # trainable 映射非空（BN 的 gamma/beta 在 backbone trainable 里，moving_* 不在）
    assert len(split["head_trainable_indices"]) == 2, "head trainable 应为 kernel+bias 两个"
    assert len(split["base_trainable_indices"]) > 0, "backbone trainable 不应为空"
