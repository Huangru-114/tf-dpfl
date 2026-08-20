"""
tests/test_defense_coverage.py  —  防御轴必须覆盖每一个 PFL 方法

**这条测试存在的理由**（真实发生过的 bug）：
`EdgeServerBase.robust_mean()` 是防御的统一入口，但每个 PFL 方法的 edge server
各写各的 `run()`。曾经有 3/7 的方法在 run 里**手写加权平均**、绕开 robust_mean，
于是 `defense=flame` 对它们完全无效 —— 而 `create_defense` 在基类 __init__ 里
无条件调用，日志照样打印 `[Defense] enabled: flame`。
按 config 的取值算，那是 7 attack × 3 method × 5 defense = 105 个静默失效的格子。

「聚合信息流是否统一」是**结构性**属性，所以用 AST 静态检查：不需要 TF、毫秒级，
且新增方法时会自动被覆盖。行为层面的确认由 L2 smoke 负责。
"""

import ast
import re
from pathlib import Path

import pytest

from config_validate import METHOD_SUPPORTS_DEFENSE

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"
SERVER_DIR = FEDAVG / "server"
BASE_CLASS = "EdgeServerBase"
# _select_method_classes 里 .get(method, PFedMeEdgeServer) 的默认分支
DEFAULT_EDGE_CLASS = "PFedMeEdgeServer"


# ── 把 server/ 下所有类索引出来：类名 → (基类名列表, 源码) ──────────────────
def _index_server_classes():
    index = {}
    for path in sorted(SERVER_DIR.glob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                bases += [b.attr for b in node.bases if isinstance(b, ast.Attribute)]
                index[node.name] = (bases, ast.get_source_segment(src, node) or "")
    return index


def _dispatch_map():
    """从 main.py 的 _select_method_classes 里抽出 method → edge server 类名。"""
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
    """沿基类链收集源码，直到 EdgeServerBase（含）。"""
    out, seen, stack = [], set(), [cls_name]
    while stack:
        name = stack.pop()
        if name in seen or name not in SERVER_CLASSES:
            continue
        seen.add(name)
        bases, src = SERVER_CLASSES[name]
        out.append((name, src))
        stack.extend(bases)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1. 台账与实际分发必须一致
# ══════════════════════════════════════════════════════════════════════════
def test_registry_matches_main_dispatch():
    """
    config_validate.METHOD_SUPPORTS_DEFENSE 是「防御覆盖」的单一事实来源。
    它必须与 main.py 的实际分发对得上，否则校验器放行的方法可能根本不存在。
    """
    dispatched = set(DISPATCH)
    registered = set(METHOD_SUPPORTS_DEFENSE)
    # 分发表里的每个 key 都必须登记
    missing = dispatched - registered
    assert not missing, (
        f"main.py 分发了 {sorted(missing)}，但 METHOD_SUPPORTS_DEFENSE 没登记 —— "
        f"校验器会把它们当成未知方法拒绝。")
    # 登记表里未出现在分发表的，只能是走默认分支的 pfedme 系
    extra = registered - dispatched
    assert extra <= {"pfedme", "hierpfedme"}, (
        f"{sorted(extra)} 登记了但 main.py 不分发，会静默落到 "
        f"{DEFAULT_EDGE_CLASS} 默认分支。")


def test_removed_method_is_really_gone():
    """hier_perfedavg 已移除：分发表、server/ 目录里都不该再有。"""
    assert "hier_perfedavg" not in DISPATCH
    assert not (SERVER_DIR / "hier_perfedavg.py").exists()
    assert "HierPerFedAvgEdgeServer" not in SERVER_CLASSES


# ══════════════════════════════════════════════════════════════════════════
# 2. 声称支持防御的方法，聚合必须真的经过 robust_mean
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("method", sorted(METHOD_SUPPORTS_DEFENSE))
def test_method_routes_aggregation_through_robust_mean(method):
    """
    这就是当初那 105 个静默失效格子的守卫。
    某方法的 edge server（或其基类链上任一层）必须调用 self.robust_mean(...)。
    """
    if not METHOD_SUPPORTS_DEFENSE[method]:
        pytest.skip(f"{method} 已显式标记为不支持防御")

    cls_name = _edge_class_for(method)
    chain = _mro_sources(cls_name)
    assert chain, f"找不到 {method} 对应的 edge server 类 {cls_name}"

    calls_robust = any("self.robust_mean(" in src for _, src in chain)
    assert calls_robust, (
        f"方法 {method} → {cls_name}：基类链 "
        f"{[n for n, _ in chain]} 上没有任何一层调用 self.robust_mean()。\n"
        f"  它的 edge 聚合绕开了防御入口 → defense 轴对该方法**静默失效**"
        f"（日志仍会打印 [Defense] enabled）。\n"
        f"  要么把聚合改走 robust_mean，要么在 METHOD_SUPPORTS_DEFENSE 里标 False。")


@pytest.mark.parametrize("method", sorted(METHOD_SUPPORTS_DEFENSE))
def test_method_does_not_handroll_weighted_mean(method):
    """
    更强的形式：edge server 自己那一层不该出现手写加权平均的特征写法。
    `hier_ditto_rep` 当初就是 `sum(upd[0][j] * (upd[1] / total_n) ...)` 绕过去的。
    """
    cls_name = _edge_class_for(method)
    own_src = SERVER_CLASSES.get(cls_name, ([], ""))[1]

    handrolled = [
        r"upd\[0\]\[\w+\]\s*\*\s*\(?upd\[1\]",     # sum(upd[0][j] * upd[1] / total_n)
        r"w_sum\s*/\s*total_n",
    ]
    for pat in handrolled:
        assert not re.search(pat, own_src), (
            f"{cls_name} 里出现手写加权平均（匹配 {pat!r}）—— "
            f"请改调 self.robust_mean()，否则防御轴对 {method} 失效。")
