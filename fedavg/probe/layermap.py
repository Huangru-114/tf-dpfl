"""
probe/layermap.py  —  `get_weights()` 下标 ↦ (层名, 张量种类) 的映射。

需要一个真实的 keras model 对象（因此要 TF），但**不做任何前向/训练**——
纯结构解析。Exp 0.3 的 layer-wise 指标（Metric B/C/D）与后续的 BN state 会话
（Metric J）都建在它上面。

为什么不复用 `models/cnn.get_base_head_indices`：那个只回答「哪两个下标是 head」，
是二分的；这里要的是**逐层分组**与「这个张量是 kernel / bias / BN 的哪一个」。
两者必须一致，守卫见 tests/test_probe_layermap.py。

Keras 约定（`get_base_head_indices` 已验证并依赖）：
    model.weights 与 model.get_weights() 索引一一对齐。
本模块额外依赖：把 model.layers 按序展开各自的 .weights，拼出来的序列与
model.weights 同序。**这一条不是语言保证**，所以下面显式核对，对不上就报错退出，
而不是产出一份悄悄串层的映射（串层的后果是 Figure 2/3/4 每一根柱子都贴错标签）。
"""

# BN 的四个张量：前两个 trainable（gamma/beta），后两个是 model state / buffer。
# 后两个正是 Metric J 的观察对象，且**不在** model.trainable_variables 里。
_BN_KINDS = ("bn_gamma", "bn_beta", "bn_moving_mean", "bn_moving_var")

_NAME_TO_KIND = {
    "kernel": "kernel", "bias": "bias",
    "gamma": "bn_gamma", "beta": "bn_beta",
    "moving_mean": "bn_moving_mean", "moving_variance": "bn_moving_var",
    "depthwise_kernel": "kernel", "pointwise_kernel": "kernel",
    "embeddings": "kernel",
}


def _var_short_name(v) -> str:
    """取变量的短名（'kernel' / 'moving_mean' / …），兼容 Keras 2 的 'dense/kernel:0'。"""
    name = getattr(v, "name", "") or ""
    name = name.split(":")[0]          # 去掉 Keras 2 的 ':0'
    return name.split("/")[-1]


def _classify(layer, pos: int, var) -> str:
    """判定张量种类：先按变量短名，认不出再按 BN 的固定次序兜底。"""
    kind = _NAME_TO_KIND.get(_var_short_name(var))
    if kind is not None:
        return kind
    if "BatchNormalization" in type(layer).__name__ and pos < len(_BN_KINDS):
        return _BN_KINDS[pos]
    return "other"


def weight_index_map(model) -> dict:
    """
    解析模型结构。

    Returns:
        {
          "entries":  [{"index", "layer", "layer_type", "kind", "shape", "size",
                        "trainable"}, ...]  # 与 get_weights() 同序、等长
          "by_layer": {layer_name: [get_weights 下标, ...]}   # 保序
          "by_kind":  {kind: [get_weights 下标, ...]}
          "bn_state_indices": [...]   # moving_mean / moving_variance 的下标（Metric J）
        }
    Raises:
        RuntimeError: 逐层展开的变量序列与 model.weights 对不上（说明本模块的
                      结构假设在这个模型上不成立）——宁可拒绝启动，不产出错标签。
    """
    model_weights = list(model.weights)
    flat, entries = [], []

    for layer in model.layers:
        for pos, var in enumerate(getattr(layer, "weights", []) or []):
            flat.append(var)
            entries.append({
                "layer": layer.name,
                "layer_type": type(layer).__name__,
                "kind": _classify(layer, pos, var),
            })

    if len(flat) != len(model_weights):
        raise RuntimeError(
            f"逐层展开得到 {len(flat)} 个变量，model.weights 有 {len(model_weights)} 个。"
            f"本模块假设两者同序等长（嵌套子模型 / 共享层会破坏它）。"
            f"不产出可能串层的映射。")
    for i, (a, b) in enumerate(zip(flat, model_weights)):
        if a is not b:
            raise RuntimeError(
                f"下标 {i} 处变量身份不一致：逐层展开得到 {getattr(a, 'name', a)!r}，"
                f"model.weights 是 {getattr(b, 'name', b)!r}。不产出可能串层的映射。")

    tv_ids = {id(v) for v in model.trainable_variables}
    by_layer, by_kind, bn_state = {}, {}, []
    for i, (e, var) in enumerate(zip(entries, model_weights)):
        e["index"] = i
        e["shape"] = tuple(int(s) for s in var.shape)
        e["size"] = int(var.shape.num_elements()) if hasattr(var.shape, "num_elements") \
            else int(_prod(e["shape"]))
        e["trainable"] = id(var) in tv_ids
        by_layer.setdefault(e["layer"], []).append(i)
        by_kind.setdefault(e["kind"], []).append(i)
        if e["kind"] in ("bn_moving_mean", "bn_moving_var"):
            bn_state.append(i)

    return {"entries": entries, "by_layer": by_layer,
            "by_kind": by_kind, "bn_state_indices": bn_state}


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def layer_groups(imap: dict, indices=None) -> dict:
    """
    {层名: [该层落在 indices 里的 get_weights 下标]}。

    indices 给 backbone 索引时，只含 head 的层会整个消失 —— 这正是陷阱 #9 要的
    「送去度量的索引子集」语义：私有 head 逐客户端 warm-start、发散最快，
    混进同一个展平向量会主导任何距离/余弦。
    """
    keep = None if indices is None else set(int(i) for i in indices)
    out = {}
    for name, idxs in imap["by_layer"].items():
        sel = [i for i in idxs if keep is None or i in keep]
        if sel:
            out[name] = sel
    return out
