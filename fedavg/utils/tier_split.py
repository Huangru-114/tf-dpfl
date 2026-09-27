"""
utils/tier_split.py  —  3-E 三层个性化的权重切分（S8，DECISIONS D-057）

get_weights() 的索引分三组（互不相交，并起来 = 全部）：

  head   客户端私有（FedRep 原有，models.cnn.get_base_head_indices）
  edge   edge 内共享、**不上云**：ResNet-10 的最后 k 个残差块
         （块内全部带权重的层：两个 3×3 conv、shortcut conv、三个 BN 的 γ/β 与 moving 统计量）
  cloud  其余 body：edge 内聚合后再上云聚合

k = `federation.edge_shared_blocks` ∈ {0, 1, 2}（原文 §8 的 (a)/(b)/(c)）：
  0 = FedRep 基线，edge 段为空 → 与改动前逐字节相同；1 = stage4；2 = stage3 + stage4。

k > 0 时的数据流（只有打 ★ 的两处改了代码）：
  client  训 body（cloud 段 + edge 段）→ 上传给 edge          不改（body = 非 head 全部）
  edge    聚合 body 索引（含 edge 段）                       不改
  cloud   只聚合 cloud 段；全局模型里的 edge 段保持原值（初值）★ CloudServer.aggregate_edges（D-057 / Q4）
  广播    edge 段不被 cloud 覆盖（首次接收除外，各 edge 同一初值）★ HierFedRepEdgeServer.set_weights
  评估    fresh-PM = [edge 当前权重（自带本 edge 的 edge 段）, 私有 head, 私有统计量]   不改（D-033 自然扩展）

纯 python，不 import TF —— 规则本地秒级可测（tests/test_tier_split.py）。
"""

from __future__ import annotations

EDGE_SHARED_KEY = "federation.edge_shared_blocks"
EDGE_SHARED_CHOICES = (0, 1, 2)
N_STAGES = 4                      # ResNet-10：stage1..stage4，每个 stage 恰好 1 个 BasicBlock
# 块按层名前缀认（`stage{i}_…`，models/cnn.py 的两个 ResNet-10 同名）；别的 arch 没有这套名字
TIER_ARCHS = frozenset({"resnet10", "resnet10_torch"})
# 只有 FedRep 的 edge server 实现了「广播不覆盖 edge 段」；别的方法下这个键会静默不生效
TIER_METHODS = frozenset({"hier_fedrep"})


def edge_shared_blocks(config: dict) -> int:
    """配置里的 k；不写 = 0。非法取值直接报错（bool 也算非法：Python 里 True == 1）。"""
    fed = (config or {}).get("federation") or {}
    v = fed.get("edge_shared_blocks", None)
    if v is None:
        return 0
    if isinstance(v, bool) or not isinstance(v, int) or v not in EDGE_SHARED_CHOICES:
        raise ValueError(f"{EDGE_SHARED_KEY}={v!r} 不是合法取值 {list(EDGE_SHARED_CHOICES)}"
                         f"（0 = FedRep 基线 / 1 = 末 1 块 edge 内共享 / 2 = 末 2 块）")
    return v


def edge_shared_prefixes(n_blocks: int, n_stages: int = N_STAGES) -> tuple:
    """最后 n_blocks 个残差块的层名前缀，如 1 → ('stage4_',)；0 → ()。"""
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or not 0 <= n_blocks <= n_stages:
        raise ValueError(f"n_blocks={n_blocks!r} 超出 [0, {n_stages}]")
    return tuple(f"stage{s}_" for s in range(n_stages - n_blocks + 1, n_stages + 1))


def indices_owned_by(owners: list, prefixes) -> list:
    """owners[i] = get_weights() 第 i 项所属层的名字；返回名字以任一前缀开头的索引（升序）。"""
    prefixes = tuple(prefixes)
    if not prefixes:
        return []
    return [i for i, name in enumerate(owners)
            if name is not None and str(name).startswith(prefixes)]


def keep_segment(new_w: list, old_w: list, idx: list) -> list:
    """new_w 的 idx 位置换回 old_w 的值，其余照 new_w。返回新列表，两个输入都不改。

    cloud 聚合（edge 段不聚合）与 edge 接收广播（edge 段不被覆盖）是同一个操作，
    也就是 fresh-PM 的组装（utils/pm.compose_pm）—— 形状核对一并继承。
    """
    from utils.pm import compose_pm
    return compose_pm(new_w, [old_w[int(i)] for i in idx], idx)


def describe(n_blocks: int, edge_idx: list, weights: list) -> dict:
    """`[设定6]` 行的字段。在 CloudServer **真正算出索引的地方**打印 → 证明接线生效，
    而不只是回显配置（陷阱 #7 / #18 的教训：配置写了 ≠ 代码走了）。
    n_*_params 按 get_weights() 的元素数算（含 BN 的 moving 统计量）。"""
    edge_set = {int(i) for i in edge_idx}
    n_edge = sum(int(getattr(weights[i], "size", 0)) for i in edge_set)
    n_all = sum(int(getattr(w, "size", 0)) for w in weights)
    return {
        "edge_shared_blocks": int(n_blocks),
        "edge_shared_prefixes": ",".join(p.rstrip("_") for p in edge_shared_prefixes(n_blocks))
                                or "none",
        "n_edge_tensors": len(edge_set),
        "n_edge_params": n_edge,
        "n_params": n_all,
    }
