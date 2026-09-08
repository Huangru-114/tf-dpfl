"""
L1：静态「未定义名」检查 —— 防的是「删改时漏掉一处赋值」这一整类。

## 为什么需要这条

2026-09-08 的 smoke run 在集群上炸在：

    File "fedavg/main.py", line 695, in run_experiment
      clients, global_model, config,
    NameError: name 'clients' is not defined

根因是 `a723a22`（方案 B 回退）把

    clients, baked_assignments, edge_fine_classes = build_clients(x_train, y_train, ...)

整段替换成了两行 `np.concatenate`，**忘了把 build_clients 的调用加回来** ——
于是 `build_clients` 成了全仓库没有调用者的死函数，`x_all/y_all` 算完没人用，
`clients` 从未绑定。

**当时 L1 全绿。** `tests/test_no_test_leakage.py::test_merge_is_kept_because_it_is_not_the_bug`
只断言 `run_experiment` 里还有 `np.concatenate([x_train, x_test])` —— 这句话在那份
炸掉的代码上照样成立。而 `fedavg/` 绝大多数模块 import TF，本地根本跑不起来，
所以这个 NameError 只能等 GPU 排到、跑到第 695 行才暴露。

## 这条测试查什么

对每个函数做一次**保守的**作用域分析：函数体里每一个被读取（`Name(Load)`）的名字，
必须在「函数内某处被绑定」∪「模块级名字」∪「builtins」里。

保守 = 只报**必然**的错，宁可漏不可误报：
  · 嵌套函数/推导式的绑定一律并进外层（over-approximate）——闭包与推导式作用域
    的细节不是这条测试的目标；
  · `global` / `nonlocal` 声明的名字视为已绑定；
  · 一个名字只要在函数里被赋值过**一次**（哪怕在 if 分支里）就算绑定 ——
    真正的「赋值前使用」不在射程内，那类错误交给运行时。
即便这么弱，它也足以抓住上面那次删漏：`clients` 在 `run_experiment` 里
**一次都没被绑定过**。
"""

import ast
import builtins
import pathlib

import pytest

FEDAVG = pathlib.Path(__file__).resolve().parents[1] / "fedavg"

BUILTINS = set(dir(builtins))


def _module_level_names(tree: ast.Module) -> set:
    """模块级可见的名字：import / 顶层赋值 / def / class。"""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names |= _bound_by_target(t)
        elif isinstance(node, (ast.If, ast.Try)):
            # 条件 import（try: import x / except ImportError: x = None）也算
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        names |= _bound_by_target(t)
    return names


def _bound_by_target(target: ast.AST) -> set:
    """一个赋值目标绑定了哪些名字（含 tuple 解包、星号）。"""
    return {n.id for n in ast.walk(target)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}


def _bound_in_function(fn: ast.AST) -> set:
    """
    函数子树里所有被绑定的名字（保守地包含嵌套作用域）。
    参数 / 赋值 / for / with-as / except-as / 推导式 / walrus /
    import / 嵌套 def & class / global & nonlocal 声明。
    """
    names = set()

    def add_args(a: ast.arguments):
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
            names.add(arg.arg)
        if a.vararg:
            names.add(a.vararg.arg)
        if a.kwarg:
            names.add(a.kwarg.arg)

    add_args(fn.args)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node is not fn:
                names.add(node.name)
                add_args(node.args)
        elif isinstance(node, ast.Lambda):
            add_args(node.args)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _outermost_functions(tree: ast.Module):
    """
    只取**最外层**的函数（模块级函数、以及模块级 class 的方法）。

    嵌套函数不单独分析：闭包读外层局部变量是完全正常的写法，单独分析会把
    每一个闭包都报成未定义。它们的函数体已经被外层的子树遍历覆盖到了 ——
    绑定和读取都在同一个 over-approximate 的集合里比对。
    """
    out = []

    def visit(node, inside_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not inside_function:
                    out.append(child)
                visit(child, True)
            else:
                visit(child, inside_function)

    visit(tree, False)
    return out


def _undefined_names(source: str):
    """返回 [(函数名, 未定义的名字, 行号), ...]。"""
    tree = ast.parse(source)
    module_names = _module_level_names(tree) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__", "__builtins__",
    }
    problems = []
    for fn in _outermost_functions(tree):
        bound = _bound_in_function(fn) | module_names | BUILTINS
        seen = set()
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id not in bound and node.id not in seen):
                seen.add(node.id)
                problems.append((fn.name, node.id, node.lineno))
    return problems


PY_FILES = sorted(FEDAVG.rglob("*.py"))


def test_the_checker_actually_catches_an_undefined_name():
    """
    反向锚点。没有这条，「零个问题」既可能是代码干净，也可能是检查器坏了。
    用的就是 2026-09-08 那次删漏的最小复现。
    """
    broken = (
        "import numpy as np\n"
        "def build_clients(x, y): return [], None, None\n"
        "def run_experiment(x_train, y_train, x_test, y_test):\n"
        "    x_all = np.concatenate([x_train, x_test])\n"
        "    return build_edge_servers(clients, x_all)\n"
    )
    found = _undefined_names(broken)
    assert ("run_experiment", "clients", 5) in found, \
        f"检查器没抓住 clients 未定义，实际抓到：{found}"


def test_the_checker_does_not_cry_wolf():
    """反向锚点之二：正常写法不能被误报（闭包 / 推导式 / except-as / walrus）。"""
    fine = (
        "import numpy as np\n"
        "TOP = 1\n"
        "def f(a, *args, k=None, **kw):\n"
        "    xs = [i * a for i in range(10)]\n"
        "    def g():\n"
        "        return xs, TOP, np, k, kw, args\n"
        "    try:\n"
        "        (n := len(xs))\n"
        "    except ValueError as e:\n"
        "        print(e, n)\n"
        "    with open('x') as fh:\n"
        "        return fh, g()\n"
    )
    assert _undefined_names(fine) == []


# ── 已知死文件 ──────────────────────────────────────────────────────────────
# 这三个文件本来就带着未定义名（甚至语法都不过），但**仓库里没有任何 import
# 指向它们** —— 由下面的 test_dead_files_are_still_dead 每次实测，不是靠这句注释。
# 它们不在本次改动的射程内（一个会话一个模块），先如实记账。
# 一旦有人开始 import 其中任何一个，那条守卫立刻变红，逼着先把文件修好。
DEAD_FILES = {
    "client/Client_BadPFL.py":
        "语法都过不了（IndentationError）；在用的是 client/client_badpfl.py",
    "data/partition_pfedme.py":
        "用了 make_client_dataset 但没 import（它在 data/partition.py）；"
        "main.py 走的是 data.partition",
    "data/clustering_pfedme.py":
        "用了 _print_assignment 但没 import（它在 data/clustering.py）；"
        "main.py 走的是 data.clustering",
}


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(FEDAVG)))
def test_no_undefined_names(path):
    rel = str(path.relative_to(FEDAVG))
    if rel in DEAD_FILES:
        pytest.skip(f"已知死文件：{DEAD_FILES[rel]}")
    try:
        tree_src = path.read_text(encoding="utf-8")
        problems = _undefined_names(tree_src)
    except SyntaxError as e:
        pytest.fail(f"{rel} 语法都过不了：{e}")
    assert not problems, "\n".join(
        f"{rel}:{line} 函数 {fn}() 读取了从未绑定的名字 {name!r}"
        for fn, name, line in problems
    )


def test_dead_files_are_still_dead():
    """
    白名单不能变成活代码的挡箭牌：DEAD_FILES 里的每个模块都必须仍然没有任何
    import 指向它。谁要开始用它，先把里面的未定义名修掉。
    """
    sources = [p.read_text(encoding="utf-8")
               for p in [*FEDAVG.rglob("*.py"),
                         *(FEDAVG.parent / "tests").rglob("*.py"),
                         *(FEDAVG.parent / "harness").rglob("*.py")]]
    for rel in DEAD_FILES:
        mod = pathlib.PurePosixPath(rel).stem
        pkg = pathlib.PurePosixPath(rel).parent.name
        importers = [
            s for s in sources
            if f"import {mod}" in s or f"from {pkg}.{mod} " in s or f"{pkg}.{mod} import" in s
        ]
        # 允许测试文件在注释/字符串里提它（例如说明为什么跳过它）
        real = [s for s in importers
                if any(ln.strip().startswith(("import ", "from "))
                       and mod in ln for ln in s.splitlines())]
        assert not real, (
            f"{rel} 已经被 import 了，但它带着未定义名 —— "
            f"要么修好它并从 DEAD_FILES 里删掉，要么别 import 它")


def test_the_scan_is_not_empty():
    """再一条反向锚点：别哪天 rglob 什么都没扫到，却报「全部通过」。"""
    assert len(PY_FILES) > 20, f"只扫到 {len(PY_FILES)} 个 py 文件，路径大概错了"
    assert any(p.name == "main.py" for p in PY_FILES), "main.py 没被扫到"
    scanned = {str(p.relative_to(FEDAVG)) for p in PY_FILES}
    missing = set(DEAD_FILES) - scanned
    assert not missing, f"DEAD_FILES 里有已经不存在的条目，该清掉：{missing}"
