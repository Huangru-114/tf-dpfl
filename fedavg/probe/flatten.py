"""
probe/flatten.py  —  权重列表 ↔ 展平向量 + 分组下标（**纯 numpy，不 import tf**）。

与 `defense/base_defense.flatten_weights` 的区别：那个只管整体展平给鲁棒聚合用；
这里还要**保住每个坐标属于哪一层**，因为 Exp 0.3 的 Metric B/C/D 都要 layer-wise，
而陷阱 #9 要求 backbone 与私有 head 全程分开、绝不混进同一个展平向量。

分组表示 = `dict[str, np.ndarray[int]]`，值是**进入展平向量后的下标**。
用显式下标数组而不是 slice：keras 的 `get_weights()` 里同一层的张量通常相邻，
但这不是语言保证的，一旦不相邻，slice 版本会静默串层。
"""
import numpy as np


def flatten(weights, indices=None) -> np.ndarray:
    """把 weights[indices] 依序展平拼接成 float64 一维向量（indices=None → 全部）。"""
    idx = range(len(weights)) if indices is None else indices
    parts = [np.asarray(weights[i], dtype=np.float64).ravel() for i in idx]
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def flat_offsets(weights, indices=None) -> dict:
    """
    返回 {tensor_index: (start, stop)} —— 每个权重张量在 `flatten(weights, indices)`
    产出的向量里占据的区间。分组下标由此拼出来。
    """
    idx = list(range(len(weights))) if indices is None else list(indices)
    out, pos = {}, 0
    for i in idx:
        n = int(np.asarray(weights[i]).size)
        out[i] = (pos, pos + n)
        pos += n
    return out


def build_groups(weights, tensor_groups: dict, indices=None) -> dict:
    """
    把「层名 → 该层占用的**张量下标**列表」翻译成「层名 → 展平向量里的坐标下标数组」。

    Args:
        weights:       权重列表（只用来读 shape）
        tensor_groups: {group_name: [tensor_index, ...]}，由 layermap.weight_index_map 产出
        indices:       参与展平的张量下标（None=全部）。**不在 indices 里的张量会被跳过**
                       —— 这正是「只在 backbone 上算 layer-wise 指标」的实现方式。
    Returns:
        {group_name: np.ndarray[int]}；某组的张量全部不在 indices 里时，该组不出现。
    """
    off = flat_offsets(weights, indices)
    out = {}
    for name, tidx in tensor_groups.items():
        spans = [np.arange(*off[i]) for i in tidx if i in off]
        if not spans:
            continue
        out[name] = np.concatenate(spans)
    return out


def delta(a, b, indices=None) -> np.ndarray:
    """展平后的 a − b（两者必须同形）。"""
    return flatten(a, indices) - flatten(b, indices)
