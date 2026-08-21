"""
tests/test_config_cli.py  —  `--config` 必须真的生效

**这条测试存在的理由**（真实发生过、且瞒过了整个 L2 验收层的 bug）：
`main.load_config(path)` 旧实现是

    with open(path) as f: config = yaml.safe_load(f)   # ← 先读文件
    parser.add_argument("--config", default=path)
    args, _ = parser.parse_known_args()                 # ← 后解析
    # args.config 之后**再也没被使用过**

于是 `--config` 被静默忽略。`run_smoke.sh` 传
`--config ../experiments/smoke-base.yaml` 完全无效 ——
每次「3 分钟 smoke（10 client / 5 round / cifar10）」实际跑的都是
`fedavg/config/config.yaml`（cifar100 / 100 client / 40 round），
日志里没有任何异常，L2 这一整级验收等于不存在。

用 AST 静态检查（不需要 TF、毫秒级），断言两件事：
  1. `--config` 解析出来之后**被用到了**；
  2. 读文件发生在 parse_known_args() **之后**。
"""

import ast
from pathlib import Path

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"
MAIN = FEDAVG / "main.py"


def _load_config_fn():
    tree = ast.parse(MAIN.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_config":
            return node
    raise AssertionError("main.py 里找不到 load_config —— 结构变了，请更新本测试")


def test_config_arg_is_actually_used():
    """解析出的 args.config 必须被用到，否则 --config 是个装饰品。"""
    fn = _load_config_fn()
    uses = [n for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and n.attr == "config"
            and isinstance(n.value, ast.Name) and n.value.id == "args"]
    assert uses, (
        "load_config 解析了 --config 但从未使用 args.config —— "
        "传进来的配置文件被静默忽略，实际跑的是函数默认的 config/config.yaml。")


def test_file_is_opened_after_arg_parsing():
    """
    open() 必须在 parse_known_args() 之后。顺序反了就等于用默认路径读文件，
    args.config 再怎么用也来不及。
    """
    fn = _load_config_fn()
    parse_line = open_line = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "parse_known_args"):
            parse_line = n.lineno
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open":
            open_line = n.lineno if open_line is None else min(open_line, n.lineno)

    assert parse_line is not None, "load_config 里找不到 parse_known_args()"
    assert open_line is not None, "load_config 里找不到 open()"
    assert parse_line < open_line, (
        f"load_config 在第 {open_line} 行就 open() 了，而 parse_known_args() 在第 "
        f"{parse_line} 行 —— 文件先于参数被读取，--config 必然失效。")


def test_smoke_script_passes_config():
    """run_smoke.sh 依赖 --config 生效，这里顺带钉住它确实在传。"""
    src = (FEDAVG.parent / "run_smoke.sh").read_text()
    assert "--config" in src, "run_smoke.sh 不再传 --config —— L2 会跑成默认全量配置"
    assert "smoke-base.yaml" in src
