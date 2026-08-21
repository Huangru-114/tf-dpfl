"""
tests/test_broadcast_coverage.py  —  下行广播必须走唯一通道

**这条测试存在的理由**（真实存在过的结构问题）：
`EdgeServerBase.broadcast_to_clients()` 看着像下行的窄腰，实际上 **6 个 edge server
里只有 1 个在用**（edge_server_fedavg），其余 5 个内联
`for client in selected: client.set_weights(...)` 把它绕过去了。

后果：任何需要「服务器下发额外载荷给客户端」的**主动防御**（裁剪上界、挑战样本、
参考统计量），写在 broadcast_to_clients 里会在 5/6 的 PFL 方法下**静默送不到**，
而日志照样打印 `[Defense] enabled`。这与 test_defense_coverage.py 守的是同一种病
（防御轴静默失效），只是方向相反 —— 那条守上行聚合，这条守下行广播。

上行窄腰（robust_mean）当初是 6/6 且有守卫，下行没有守卫，所以烂到 1/6 都没人发现。

与 test_defense_coverage.py 同一套写法：AST 静态检查，不需要 TF、毫秒级，
新增方法时自动被覆盖。
"""

import ast
import re
from pathlib import Path

import pytest

from config_validate import METHOD_SUPPORTS_DEFENSE

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"
SERVER_DIR = FEDAVG / "server"
DEFAULT_EDGE_CLASS = "PFedMeEdgeServer"
# 下行通道定义在基类里，基类自己当然要出现 client.set_weights
BASE_FILE = "edge_server_base.py"


def _index_server_classes():
    """类名 → (基类名列表, 源码, 文件名)。"""
    index = {}
    for path in sorted(SERVER_DIR.glob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                bases += [b.attr for b in node.bases if isinstance(b, ast.Attribute)]
                index[node.name] = (bases, ast.get_source_segment(src, node) or "",
                                    path.name)
    return index


def _dispatch_map():
    src = (FEDAVG / "main.py").read_text()
    body = src.split("edge_cls = {", 1)
    assert len(body) == 2, "main.py 里找不到 edge_cls 字典 —— 分发结构变了，请更新本测试"
    body = body[1].split("}", 1)[0]
    return dict(re.findall(r'"([\w]+)"\s*:\s*(\w+)', body))


SERVER_CLASSES = _index_server_classes()
DISPATCH = _dispatch_map()


def _edge_class_for(method: str) -> str:
    return DISPATCH.get(method, DEFAULT_EDGE_CLASS)


def _mro_sources(cls_name: str):
    out, seen, stack = [], set(), [cls_name]
    while stack:
        name = stack.pop()
        if name in seen or name not in SERVER_CLASSES:
            continue
        seen.add(name)
        bases, src, fname = SERVER_CLASSES[name]
        out.append((name, src, fname))
        stack.extend(bases)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1. 每个方法的下行都必须经过 broadcast_to_clients
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("method", sorted(METHOD_SUPPORTS_DEFENSE))
def test_method_broadcasts_through_single_channel(method):
    cls_name = _edge_class_for(method)
    chain = _mro_sources(cls_name)
    assert chain, f"找不到 {method} 对应的 edge server 类 {cls_name}"

    calls_broadcast = any("self.broadcast_to_clients(" in src for _, src, _ in chain)
    assert calls_broadcast, (
        f"方法 {method} → {cls_name}：基类链 {[n for n, _, _ in chain]} 上没有任何一层"
        f"调用 self.broadcast_to_clients()。\n"
        f"  它的下行绕开了唯一通道 → 主动防御的下行载荷对该方法**静默送不到**"
        f"（日志仍会打印 [Defense] enabled）。")


# ══════════════════════════════════════════════════════════════════════════
# 2. 更强的形式：edge server 自己那一层不许内联 client.set_weights
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("method", sorted(METHOD_SUPPORTS_DEFENSE))
def test_edge_server_does_not_inline_set_weights(method):
    """
    5 个 edge server 当初就是内联 `client.set_weights(...)` 绕过去的。
    只有基类（定义通道的那一处）允许出现。
    """
    cls_name = _edge_class_for(method)
    bases, own_src, fname = SERVER_CLASSES.get(cls_name, ([], "", ""))
    if fname == BASE_FILE:
        pytest.skip("基类本身就是下行通道的定义处")

    assert not re.search(r"client\.set_weights\(", own_src), (
        f"{cls_name}（{fname}）里内联了 client.set_weights(...) —— "
        f"请改调 self.broadcast_to_clients(selected, global_weights=..., "
        f"round_idx=...)，否则主动防御的下行载荷对 {method} 送不到。")


def test_broadcast_channel_carries_control():
    """
    通道本身必须真的下发 control —— 否则「大家都走通道」也是空的。
    """
    _, base_src, _ = SERVER_CLASSES["EdgeServerBase"]
    assert "def broadcast_to_clients(" in base_src
    assert "client.set_control(" in base_src, (
        "EdgeServerBase.broadcast_to_clients 没有调用 client.set_control(...) —— "
        "下行载荷通道是空的，主动防御拿不到任何东西。")
