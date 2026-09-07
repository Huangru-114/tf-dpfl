"""
tests/test_seeding_completeness.py  —  set_seed 必须覆盖所有被实际使用的全局 RNG

**这条测试存在的理由**（真实的、瞒过了两个 seed 的 Experiment 3 的 bug）：
`main.set_seed` 只播了两个 RNG ——

    np.random.seed(seed)
    tf.random.set_seed(seed)

而它自己的 docstring 写着「随机性来自**三处**」。漏掉的第三处是 Python 内置
`random`：五个方法客户端的本地训练循环每个 epoch 都用它打乱 batch 顺序
（`hier_fedrep.py` / `client_pfedme.py` / `hier_ditto.py` / `hier_ditto_rep.py` /
`hier_pfedme_rep.py` 里的 `random.shuffle(eb)`）。

后果：**Rep / Ditto / pFedMe 全家在固定 seed 下都不可复现** —— 也就是
Experiment 3 的每一个格子。这会伪装成「种子方差很大」：exp3 的 3C 轴上
seed42 ≈ 0.80、seed43 ≈ 0.58，落差里有多少是真的种子方差、多少是这个
未播种的洗牌，在修好之前无法分离。

用 AST 静态检查（不需要 TF、毫秒级）。第二条测试是**通用守卫**：
不写死那五个文件名，而是扫描 client/ 与 server/ 下所有模块 ——
只要有人用了 `random.*`，`set_seed` 就必须播 `random`。
以后新增方法时不会重蹈覆辙。
"""

import ast
from pathlib import Path

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"
MAIN = FEDAVG / "main.py"

# 会消耗 Python 内置 random 全局状态的函数
_RANDOM_FUNCS = {"shuffle", "sample", "choice", "random", "randint",
                 "randrange", "uniform", "gauss", "normalvariate"}


def _set_seed_fn():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_seed":
            return node
    raise AssertionError("main.py 里找不到 set_seed —— 结构变了，请更新本测试")


def _seeded_rngs():
    """返回 set_seed 里实际播种的 RNG 名字集合，如 {'random', 'np.random', 'tf.random'}。"""
    seeded = set()
    for node in ast.walk(_set_seed_fn()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("seed", "set_seed"):
            continue
        seeded.add(ast.unparse(node.func.value))
    return seeded


def _modules_using_python_random():
    """扫描 client/ 与 server/，返回用到内置 random.* 的模块名列表。"""
    hits = []
    for sub in ("client", "server"):
        for path in sorted((FEDAVG / sub).glob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue          # 仓库里有已知的死文件 Client_BadPFL.py，跳过
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "random"
                        and node.func.attr in _RANDOM_FUNCS):
                    hits.append(f"{sub}/{path.name}:{node.lineno}")
    return hits


def test_set_seed_seeds_all_three_rngs():
    """numpy / tensorflow / Python random —— 一个都不能少。"""
    seeded = _seeded_rngs()
    for rng in ("random", "np.random", "tf.random"):
        assert rng in seeded, (
            f"set_seed 没有播种 {rng}；实际播了 {sorted(seeded)}。"
            " 漏播任何一个都会让固定 seed 的重跑对不上。")


def test_any_use_of_python_random_is_covered_by_set_seed():
    """
    通用守卫：只要 client/ 或 server/ 里有人用内置 random，set_seed 就必须播它。
    这样新增方法时不需要回来改这条测试。
    """
    users = _modules_using_python_random()
    if not users:
        return                     # 没人用就没有约束
    assert "random" in _seeded_rngs(), (
        "以下位置使用了未播种的 Python 内置 random：\n  "
        + "\n  ".join(users)
        + "\n而 main.set_seed 没有调用 random.seed() —— 这些位置的随机性不受 seed 控制。")


def test_the_known_shuffle_sites_are_still_detected():
    """
    冒烟：上面那条通用守卫真的能看见东西。
    如果哪天这些 random.shuffle 被改成了 seeded RNG，这条会失败 ——
    那时把它删掉即可（守卫本身仍然有效）。
    """
    users = _modules_using_python_random()
    assert users, ("扫描器一处 random.* 都没找到 —— 要么实现改了（好事，删掉本测试），"
                   "要么扫描器坏了（坏事，去修 _modules_using_python_random）")
