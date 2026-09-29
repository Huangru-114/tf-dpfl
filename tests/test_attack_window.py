"""
tests/test_attack_window.py  —  攻击时间窗（Neurotoxin 式持久性协议）

## 这条协议要回答什么

攻击者在第 T 个 **global(cloud) round** 及之后停止投毒，之后只观察后门衰减。
层级结构天然提出的新问题：**edge 模型会不会成为后门的「蓄水池」，
让它在攻击者走后活得更久？** 这与「终值上层级组不比 flat 低」不冲突 ——
强度和持久性是两件事。

衰减曲线**不需要新指标**：`rounds[]` 已经是逐轮三层 ASR 的时间序列，
T 之后的那一段就是衰减段。要写的只有「让攻击者停下来」这一件事。

## 设计的两条要害（这两条如果错了，整批持久性数据都白跑）

1. **闸门是 `_attack_active`，不是 `is_malicious`。**
   `is_malicious` 同时是**身份标记**：`malicious_participation_by_client` 的统计、
   逐 edge 的 same/diff-edge 分组、`build_eval_trigger` 挑哪个生成器，全依赖它。
   把它翻成 False 会让「攻击者退出」在 metrics.json 里**看起来像「这一格压根
   没有恶意端」** —— 而那正是「攻击完全失效」的样子，两者数值上无法区分。

2. **`vanilla` 用不了时间窗。** vanilla 的投毒是在 `build_clients` 阶段把恶意端的
   **数据集静态改掉**，钩子拦不住。配上 `attack_stop_round` 会静默无效，
   而日志与正常 run 一模一样 → `config_validate` 必须直接拒绝。

本模块纯 stdlib + AST，不 import TF，本地秒级。
"""

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEDAVG = ROOT / "fedavg"
sys.path.insert(0, str(FEDAVG))


# ══════════════════════════════════════════════════════════════════════════
# 1. attacking() 的判定 —— 不 import TF，直接拿类的函数对象在假对象上跑
# ══════════════════════════════════════════════════════════════════════════

def _client_base_fn(name):
    """
    从源码里取出 `FLClientBase.<name>` 的函数体并编译。

    不 `from client.client_base import FLClientBase` —— 那会 import TF，
    本地跑不了。这里只要纯 Python 的判定函数；它们调用的 `in_window` /
    `generator_on` 来自不 import TF 的 attack/attack_window.py（S4），注入进命名空间。
    """
    import attack.attack_window as AW
    src = (FEDAVG / "client" / "client_base.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert fn is not None, (
        f"FLClientBase 里找不到 {name}() —— 攻击时间窗的判定没了，"
        "attack_stop_round / attack_start_round 会静默无效。")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"in_window": AW.in_window, "generator_on": AW.generator_on}
    exec(compile(ast.fix_missing_locations(mod), f"<{name}>", "exec"), ns)
    return ns[name]


class _FakeClient:
    def __init__(self, is_malicious, stop, start=None, schedule="window"):
        self.is_malicious = is_malicious
        self._attack_stop_round = stop
        self._attack_start_round = start
        self._generator_schedule = schedule


@pytest.fixture(scope="module")
def attacking():
    """惰性取 —— 函数不存在时只让这几条红，不要在收集期把整个模块带下线。"""
    return _client_base_fn("attacking")


@pytest.fixture(scope="module")
def generating():
    return _client_base_fn("generating")


@pytest.mark.parametrize("round_idx,expected", [
    (0, True), (39, True), (40, False), (41, False), (999, False),
])
def test_malicious_client_stops_at_T(attacking, round_idx, expected):
    """T 是**左闭右开**的边界：第 T 轮本身已经不投毒了。"""
    assert attacking(_FakeClient(True, 40), round_idx) is expected


def test_stop_none_means_never_stop(attacking):
    """默认 None → 与加这条之前的行为逐字节一致（全长都在投毒）。"""
    c = _FakeClient(True, None)
    assert all(attacking(c, r) for r in (0, 1, 40, 400, 10_000))


def test_benign_client_never_attacks(attacking):
    """反向锚点：良性端在任何轮、任何 T 下都不投毒。"""
    for stop in (None, 0, 40):
        c = _FakeClient(False, stop)
        assert not any(attacking(c, r) for r in (0, 39, 40, 400))


# ── S4：起点 + 生成器语义（D-078 / D-079）───────────────────────────────────
@pytest.mark.parametrize("round_idx,expected", [
    (1, False), (4, False), (5, True), (8, True), (9, False), (19, False),
])
def test_window_is_start_inclusive_stop_exclusive(attacking, round_idx, expected):
    """G5 的 t20 格：attack_start_round=5、attack_stop_round=9 → 第 5–8 个 cloud 轮投毒。"""
    assert attacking(_FakeClient(True, 9, start=5), round_idx) is expected


def test_start_only_attacks_until_the_end(attacking):
    c = _FakeClient(True, None, start=10)
    assert [attacking(c, r) for r in (9, 10, 11, 500)] == [False, True, True, True]


def test_start_none_is_byte_identical_to_the_stop_only_rule(attacking):
    """默认 start=None 必须与 S4 之前的判定（`stop is None or r < stop`）逐轮相同。"""
    for stop in (None, 1, 31, 151):
        c = _FakeClient(True, stop)
        old = [stop is None or r < stop for r in range(0, 400)]
        assert [attacking(c, r) for r in range(0, 400)] == old


@pytest.mark.parametrize("start,stop", [(None, None), (None, 31), (5, 9), (29, 33), (10, None)])
def test_window_schedule_generates_exactly_when_attacking(attacking, generating, start, stop):
    """A 臂（window，缺省）：生成器闸门 ≡ 投毒闸门 → G8 与以前的 run 行为不变。"""
    c = _FakeClient(True, stop, start=start, schedule="window")
    assert [generating(c, r) for r in range(0, 60)] == [attacking(c, r) for r in range(0, 60)]


def test_always_schedule_trains_the_generator_outside_the_window(attacking, generating):
    """B 臂（always）：窗口外只是不投毒，生成器每一轮都训（= ρ=0 影子攻击者）。"""
    c = _FakeClient(True, 9, start=5, schedule="always")
    assert all(generating(c, r) for r in range(1, 20))
    assert [attacking(c, r) for r in (4, 5, 8, 9)] == [False, True, True, False]


def test_benign_client_never_generates(generating):
    for sched in ("window", "always"):
        assert not any(generating(_FakeClient(False, 9, start=5, schedule=sched), r)
                       for r in range(0, 20))


def test_attack_window_module_rejects_unknown_schedules():
    import attack.attack_window as AW
    with pytest.raises(ValueError):
        AW.generator_on(1, None, None, "sometimes")
    assert AW.SCHEDULES == ("window", "always") and AW.DEFAULT_SCHEDULE == "window"


# ══════════════════════════════════════════════════════════════════════════
# 2. 三个攻击 mixin 的闸门都换过来了（AST，防止新增攻击时漏掉）
# ══════════════════════════════════════════════════════════════════════════

# 钩子里读 `self.is_malicious` 作为**投毒闸门**的写法必须绝迹 ——
# 漏掉任何一个，那个攻击的时间窗就静默失效（数据照跑，衰减段是假的）。
GATED_HOOKS = ("on_round_start", "on_batch", "on_extra_loss", "on_upload")
ATTACK_MIXIN_FILES = ("client_badpfl.py", "client_neurotoxin.py", "client_cerp.py")


def _body_source(fn: ast.FunctionDef) -> str:
    """函数体源码，**不含 docstring** —— 文档里提到 `_attack_active` 是常事，
    拿它当「代码读了这个标志」会得到假绿，顺序断言更会被 docstring 骗到。"""
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return "\n".join(ast.unparse(n) for n in body)


def _hook_sources(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: _body_source(n) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in GATED_HOOKS}


# Bad-PFL 的 on_round_start 只训生成器，闸门是生成器闸门 `_gen_active`（S4；window 语义下
# ≡ `_attack_active`）。其余钩子（投毒本身）一律读 `_attack_active`。
GEN_GATED = {("client_badpfl.py", "on_round_start")}


def _gate_of(fname, name):
    return "self._gen_active" if (fname, name) in GEN_GATED else "self._attack_active"


@pytest.mark.parametrize("fname", ATTACK_MIXIN_FILES)
def test_hooks_gate_on_attack_active_not_is_malicious(fname):
    hooks = _hook_sources(FEDAVG / "client" / fname)
    assert hooks, f"{fname} 里一个受闸门管的钩子都没找到 —— 文件结构变了？"
    for name, src in hooks.items():
        assert "self.is_malicious" not in src, (
            f"{fname}::{name} 仍然拿 self.is_malicious 当投毒闸门。\n"
            f"  时间窗只作用于 self._attack_active（基类 on_round_start 每轮刷新），\n"
            f"  用 is_malicious 会让 attack_stop_round 对这个攻击**静默失效**。")
        gate = _gate_of(fname, name)
        assert gate in src, f"{fname}::{name} 没有读 {gate} —— 闸门去哪了？"


def test_badpfl_poisoning_still_gates_on_the_attack_window():
    """B 臂的要害：生成器全程训，但**投毒**（on_batch）仍只在窗口内。"""
    hooks = _hook_sources(FEDAVG / "client" / "client_badpfl.py")
    assert "self._attack_active" in hooks["on_batch"]
    assert "self._gen_active" not in hooks["on_batch"]


@pytest.mark.parametrize("fname", ATTACK_MIXIN_FILES)
def test_hooks_call_super_before_reading_the_flag(fname):
    """
    `_attack_active` 由**基类**的 on_round_start 刷新，所以 mixin 必须先调 super()。
    先读后调 = 读到上一轮的值 → 退出轮整体偏移一轮，而且完全看不出来。
    """
    src = _hook_sources(FEDAVG / "client" / fname).get("on_round_start")
    if src is None:
        pytest.skip(f"{fname} 没有 on_round_start")
    i_super = src.find("super().on_round_start")
    i_flag  = src.find(_gate_of(fname, "on_round_start").replace("self.", ""))
    assert i_super != -1, f"{fname}::on_round_start 没调 super()"
    if i_flag != -1:
        assert i_super < i_flag, (
            f"{fname}::on_round_start 在调 super() 之前就读了 _attack_active")


def test_base_on_round_start_refreshes_the_flag():
    """基类那两句是整条链的源头，不能被优化掉。"""
    src = _hook_sources(FEDAVG / "client" / "client_base.py")["on_round_start"]
    assert "_attack_active" in src and "attacking" in src, src
    assert "_gen_active" in src and "generating" in src, src


def test_is_malicious_is_not_mutated_anywhere():
    """
    **要害 1 的守卫**：谁都不许赋值 `self.is_malicious`（除了 client_base 的初始化
    兜底与 main.py 构造后的赋值）。翻它 = 把「攻击者退出」伪装成「没有恶意端」。
    """
    from test_main_names_are_bound import DEAD_FILES     # 单一事实来源

    offenders = []
    for path in sorted((FEDAVG / "client").rglob("*.py")):
        if path.name == "client_base.py":
            continue
        if str(path.relative_to(FEDAVG)) in DEAD_FILES:  # 语法都不过的死文件
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == "is_malicious"
                        and isinstance(t.value, ast.Name) and t.value.id == "self"):
                    offenders.append(f"{path.relative_to(FEDAVG)}:{node.lineno}")
    assert not offenders, (
        "有人给 self.is_malicious 赋值：" + ", ".join(offenders) +
        "\n  它是身份标记，参与度统计 / 逐 edge 分组 / build_eval_trigger 都依赖它。"
        "\n  要让攻击停下来请用 attack_stop_round → _attack_active。")


# ══════════════════════════════════════════════════════════════════════════
# 3. config_validate 拦住三种静默失效
# ══════════════════════════════════════════════════════════════════════════

def _cfg(**bd_over):
    return {
        "seed": 42,
        "training": {"drift_correction": "hier_fedrep",
                     "local_epochs": 1, "plocal_epochs": 1},
        "defense": {"name": "none"},
        "federation": {"n_clients": 100, "n_edges": 2, "edge_rounds": 5,
                       "n_rounds": 80, "client_fraction": 0.1,
                       "edge_assignment": "block"},
        "data": {"num_classes": 10},
        "model": {"arch": "resnet10"},
        "evaluation": {"eval_interval": 2},
        "backdoor": {"enabled": True, "malicious_strategy": "badpfl",
                     "trigger": "badnet", "target_label": 0, "poison_ratio": 0.2,
                     "n_malicious": 10, "eval_interval": 2,
                     "malicious_placement": "by_edge",
                     "malicious_per_edge": [5, 5], **bd_over},
    }


def _validate(cfg):
    pytest.importorskip("numpy")     # config_validate 惰性 import defense 包
    import io
    from contextlib import redirect_stdout
    from config_validate import ConfigError, validate_config
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            validate_config(cfg)
        except ConfigError as e:
            return None, str(e)
    return buf.getvalue(), None


def test_stop_round_beyond_n_rounds_is_rejected():
    """
    T ≥ n_rounds = 攻击者到跑完都没退出 → **静默等于「从不停止」**，
    而 metrics.json 会写着有退出轮。这是最难发现的一种：数据看起来齐全。
    """
    _, err = _validate(_cfg(attack_stop_round=80))
    assert err and "n_rounds" in err


def test_stop_round_zero_is_rejected():
    """T ≤ 0 是「一轮都不投毒」，那是无攻击对照，不该用退出轮伪装。"""
    _, err = _validate(_cfg(attack_stop_round=0))
    assert err and "无攻击对照" in err


def test_vanilla_with_stop_round_is_rejected():
    """**要害 2 的守卫**：vanilla 静态投毒数据集，钩子拦不住。"""
    _, err = _validate(_cfg(malicious_strategy="vanilla", attack_stop_round=40))
    assert err and "vanilla" in err


def test_valid_stop_round_passes():
    """反向锚点：合法的 T 必须放行，否则上面三条可能只是「什么都拒绝」。"""
    out, err = _validate(_cfg(attack_stop_round=40))
    assert err is None, err
    assert "attack_stop_round=40" in out


def test_no_stop_round_passes_unchanged():
    """不写这个键的配置（= 现存全部格子）行为不变。"""
    out, err = _validate(_cfg())
    assert err is None, err
    assert "attack_stop_round=n/a" in out


# ── S4：起点与生成器语义的校验（D-078 / D-079）───────────────────────────────
def test_start_round_at_or_after_stop_is_rejected():
    """空窗口 = 一轮都不投毒，而 metrics.json 会写着有窗口。"""
    _, err = _validate(_cfg(attack_start_round=40, attack_stop_round=40))
    assert err and "空的" in err


def test_start_round_below_one_is_rejected():
    _, err = _validate(_cfg(attack_start_round=0, attack_stop_round=40))
    assert err and "从 1 起" in err


def test_start_round_beyond_n_rounds_is_rejected():
    _, err = _validate(_cfg(attack_start_round=80))
    assert err and "n_rounds" in err


def test_vanilla_with_start_round_is_rejected():
    _, err = _validate(_cfg(malicious_strategy="vanilla", attack_start_round=5,
                            attack_stop_round=9))
    assert err and "vanilla" in err


def test_unknown_generator_schedule_is_rejected():
    _, err = _validate(_cfg(generator_schedule="sometimes"))
    assert err and "generator_schedule" in err


def test_always_schedule_needs_badpfl():
    _, err = _validate(_cfg(malicious_strategy="neurotoxin", generator_schedule="always",
                            attack_start_round=5, attack_stop_round=9))
    assert err and "badpfl" in err


def test_valid_window_passes_and_self_describes():
    """反向锚点：合法窗口放行，[设定7] 打出起点与有效的生成器语义。"""
    out, err = _validate(_cfg(attack_start_round=5, attack_stop_round=9,
                              generator_schedule="always"))
    assert err is None, err
    assert "[设定7] attack_start_round=5 | generator_schedule=always" in out
    assert "attack_stop_round=9" in out


def test_default_config_self_describes_the_default_schedule():
    """不写这两个键 → 起点 n/a（从第一轮起）、语义 window（缺省）。"""
    out, err = _validate(_cfg())
    assert err is None, err
    assert "[设定7] attack_start_round=n/a | generator_schedule=window" in out

