"""
tests/test_cloud_aggregate_default.py  —  留接口不等于改行为

本会话把 `robust_mean` 从 EdgeServerBase 抽成 RobustAggregationMixin，让 CloudServer
也能继承（为将来的 cloud 层防御留通道），并把 `CloudServer._aggregate_global` 的
`return aggregate(edge_updates)` 换成 `return self.aggregate_edges(...)`。

**这条测试守的就是「换了入口但没换行为」**：`defense.layers` 缺省 ["edge"] 时
cloud 层拿到的 defense 是 None，`robust_mean` 必须回退到
`aggregation.fedavg.aggregate`，且**逐元素相等**（不是「差不多」）。

纯 numpy，本地可跑。
"""

import numpy as np
import pytest

from aggregation.fedavg import aggregate as fedavg_aggregate
from defense import configured_layers, create_defense
from server.robust_aggregation import RobustAggregationMixin


class _Dummy(RobustAggregationMixin):
    """最小宿主：只提供 mixin 需要的 _default_ref_weights。"""

    def __init__(self, config, layer, ref):
        self._ref = ref
        self._init_defense(config, layer=layer)

    def _default_ref_weights(self):
        return self._ref


def _updates(rng, n_clients=4, shapes=((3, 2), (2,))):
    out = []
    for i in range(n_clients):
        w = [rng.normal(size=s).astype(np.float32) for s in shapes]
        out.append((w, 10 * (i + 1), 0.5, 0.1))
    return out


def _ref(shapes=((3, 2), (2,))):
    return [np.zeros(s, dtype=np.float32) for s in shapes]


# ══════════════════════════════════════════════════════════════════════════
# 1. 默认配置下 cloud 层不设防
# ══════════════════════════════════════════════════════════════════════════
def test_default_layers_is_edge_only():
    """缺省 defense.layers 必须是 ["edge"] —— 这是「行为不变」的前提。"""
    assert configured_layers({"defense": {"name": "flame"}}) == ("edge",)
    assert configured_layers({}) == ("edge",)


def test_cloud_defense_is_none_by_default():
    """即使配了 flame，只要 layers 没写 cloud，cloud 层就必须拿到 None。"""
    cfg = {"defense": {"name": "flame"}, "federation": {"n_edges": 2},
           "backdoor": {"n_malicious": 2}}
    assert create_defense(cfg, layer="cloud") is None
    assert create_defense(cfg, layer="edge") is not None


def test_defense_none_means_no_defense_anywhere():
    cfg = {"defense": {"name": "none"}}
    assert create_defense(cfg, layer="edge") is None
    assert create_defense(cfg, layer="cloud") is None


# ══════════════════════════════════════════════════════════════════════════
# 2. 不设防时 robust_mean ≡ fedavg_aggregate（逐元素）
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("layer", ["edge", "cloud"])
def test_robust_mean_without_defense_equals_fedavg_exactly(layer):
    rng = np.random.default_rng(0)
    ups = _updates(rng)
    host = _Dummy({"defense": {"name": "none"}}, layer, _ref())

    got = host.robust_mean(ups)
    want = fedavg_aggregate(ups)

    assert len(got) == len(want)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_cloud_path_equals_old_aggregate_call():
    """
    改动前 cloud 层是 `return aggregate(edge_updates)`。
    改动后是 `self.aggregate_edges(...)` → `robust_mean(...)` → 同一个 aggregate。
    这里断言默认配置下两条路径逐元素相等。
    """
    rng = np.random.default_rng(7)
    edge_updates = _updates(rng, n_clients=3)
    prev_global = _ref()

    host = _Dummy({"defense": {"name": "flame"}, "federation": {"n_edges": 2},
                   "backdoor": {"n_malicious": 2}}, "cloud", prev_global)
    # flame 配了，但 layers 缺省只有 edge → cloud 仍然不设防
    assert host.defense is None

    got = host.robust_mean(edge_updates, prev_global)
    want = fedavg_aggregate(edge_updates)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


# ══════════════════════════════════════════════════════════════════════════
# 3. 显式打开 cloud 层时通道真的通
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name,coord", [("flame", False), ("multi_krum", False),
                                        ("dnc", False), ("median", True),
                                        ("trimmed_mean", True)])
def test_every_defense_emits_the_uniform_decision_line(capsys, name, coord):
    """
    `admitted_count` 是 CLAUDE.md 要求 metrics.json 回传的 4 个字段之一，
    但各防御原本各打各的日志（flame「admitted 7/10」、multi_krum「selected N clients」、
    dnc「keep N clients」、trimmed_mean/median 干脆不打）→ 回程解析器
    **5 个防御里只有 1 个读得出来**，defense 轴 4/5 是瞎的。

    统一判决行由 robust_mean 这个唯一收口处发出，所以每个防御都必须有。
    对应的解析在 tests/test_collect_metrics.py。
    """
    cfg = {"defense": {"name": name, "layers": ["edge"]},
           "federation": {"n_edges": 1}, "backdoor": {"n_malicious": 1}}
    rng = np.random.default_rng(11)
    ups = _updates(rng, n_clients=6)
    host = _Dummy(cfg, "edge", _ref())
    assert host.defense is not None

    host.robust_mean(ups)
    out = capsys.readouterr().out

    assert "[Decision]" in out, f"{name} 没有发出统一判决行 —— 回程读不到 admitted_count"
    assert f"| {name} |" in out
    if coord:
        assert "coordinate-wise" in out, (
            f"{name} 是坐标类防御，必须显式标注「无客户端级判决」，"
            f"否则会被记成 admitted=0（= 全部剔除）")
    else:
        assert "admitted " in out and "rejected=" in out


def test_cloud_layer_can_be_enabled_explicitly():
    """
    留接口要留得真：写了 layers: [edge, cloud] 就必须在 cloud 层拿到防御实例，
    且聚合结果不再等于朴素 FedAvg（否则这个接口是假的）。
    """
    cfg = {"defense": {"name": "median", "layers": ["edge", "cloud"]},
           "federation": {"n_edges": 2}, "backdoor": {"n_malicious": 2}}
    assert create_defense(cfg, layer="cloud") is not None

    rng = np.random.default_rng(3)
    ups = _updates(rng, n_clients=5)
    host = _Dummy(cfg, "cloud", _ref())

    got = host.robust_mean(ups)
    plain = fedavg_aggregate(ups)
    # median 与样本加权均值在随机数据上不可能逐元素相等
    assert any(not np.array_equal(g, p) for g, p in zip(got, plain))
