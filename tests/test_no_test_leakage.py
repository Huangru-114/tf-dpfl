"""
tests/test_no_test_leakage.py  —  官方 test split 绝不能进客户端训练集

**这条测试存在的理由**（真实的、存活了约一年的 bug）：
`main.run_experiment` 曾经这样构造客户端数据池 ——

    # 合并全量数据后分区：确保 per-client train/test 同分布（PFLlib 标准做法）
    # x_test/y_test 保留用于 edge/GM 评估，不受影响。      ← 这句断言是错的
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    clients, ... = build_clients(x_all, y_all, ...)

而同一份 `x_test` 又被交给 `BackdoorCloudServer(x_test=x_test, ...)` 去算
global / edge / local 六个 ASR 与 global_acc。

泄漏是两步的必然结果，与 seed / alpha 无关：
  1. `noniid_partition` 里 `np.split(class_indices, cut_points)` 是一个**划分** ——
     传进去的每一个 index 都会落到某个 client 上，没有任何 index 被排除；
  2. `split_client_train_test` 再把每个 client 分片的 (1 − test_ratio) 划进训练集。
  ⇒ 官方 test set 的 (1 − test_ratio) = **75%**（exp3 的 test_ratio=0.25）被训练过。

误导性在于 `build_clients(..., x_test_np=x_test, y_test_np=y_test)` 这两个参数：
docstring 说「存在时注入」，但函数体里**从未引用过**，看起来像做了隔离，其实没有。

下面两条测试合起来构成完整证明：
  A. 分区是**穷尽**的（需要 TF，集群上跑）——「传进去多少就用掉多少」；
  B. 调用方只传 **x_train**（纯 AST，本地毫秒级）——「传进去的里面没有 test」。
"""

import ast
from pathlib import Path

import pytest

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"
MAIN = FEDAVG / "main.py"


# ─────────────────────────── A. 分区是穷尽的（需要 TF） ───────────────────────────

def test_partition_is_exhaustive_so_nothing_passed_in_is_ever_held_out():
    """
    机制证明：喂给分区的每一个 index 最终都会成为某个客户端的数据。
    所以「传进去但指望它不被用到」是不可能的 —— 隔离只能发生在调用方。
    """
    pytest.importorskip("tensorflow")
    import numpy as np
    import sys
    sys.path.insert(0, str(FEDAVG))
    from data.partition import noniid_partition, split_client_train_test

    N, N_CLIENTS, N_CLASSES = 2000, 10, 10
    rng = np.random.default_rng(0)
    images = rng.random((N, 8, 8, 3), dtype=np.float32)
    labels = np.repeat(np.arange(N_CLASSES), N // N_CLASSES)
    cfg = {"data": {"batch_size": 16, "shuffle_buffer": 100, "img_size": 8}}

    _, client_indices = noniid_partition(images, labels, N_CLIENTS, cfg, alpha=0.5)

    covered = set()
    for idx in client_indices:
        covered |= set(np.asarray(idx).tolist())
    assert covered == set(range(N)), (
        f"分区没有覆盖全部 index（覆盖 {len(covered)}/{N}）—— 若这条失败，"
        "说明分区语义变了，本文件的推理需要重做")

    _, train_indices, _ = split_client_train_test(
        images, labels, client_indices, cfg, test_ratio=0.25)
    train_pool = set()
    for idx in train_indices:
        train_pool |= set(np.asarray(idx).tolist())
    frac = len(train_pool) / N
    assert 0.70 < frac < 0.80, (
        f"喂进去的 index 有 {frac:.1%} 进了训练集（预期 ≈1−test_ratio=75%）")


# ─────────────────── B. 调用方只把 train split 喂进去（纯 AST） ───────────────────

def _run_experiment_fn():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_experiment":
            return node
    raise AssertionError("main.py 里找不到 run_experiment —— 结构变了，请更新本测试")


def _build_clients_call():
    for node in ast.walk(_run_experiment_fn()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "build_clients"):
            return node
    raise AssertionError("run_experiment 里找不到 build_clients(...) 调用")


def test_build_clients_receives_the_train_split_only():
    """前两个位置实参必须字面上是 x_train / y_train。"""
    call = _build_clients_call()
    got = [ast.unparse(a) for a in call.args[:2]]
    assert got == ["x_train", "y_train"], (
        f"build_clients 的前两个实参是 {got}，应为 ['x_train', 'y_train']。"
        " 任何把 x_test 混进客户端数据池的写法都会让 ASR/MTA 失去意义。")


def test_test_split_is_never_merged_into_the_client_pool():
    """
    run_experiment 里不允许出现把 x_test / y_test 并进另一个数组的拼接。
    这正是被移除的那两行 `np.concatenate([x_train, x_test])`。
    """
    offenders = []
    for node in ast.walk(_run_experiment_fn()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("concatenate", "vstack", "hstack", "append"):
            continue
        src = ast.unparse(node)
        if "x_test" in src or "y_test" in src:
            offenders.append(f"line {node.lineno}: {src}")
    assert not offenders, (
        "run_experiment 里把官方 test split 拼进了别的数组：\n  "
        + "\n  ".join(offenders)
        + "\n它随后会被 build_clients 分区 → 75% 进客户端训练集，"
          "而 BackdoorCloudServer 仍用同一份 x_test 算 ASR/global_acc。")


def test_build_clients_has_no_unused_test_set_parameters():
    """
    `x_test_np` / `y_test_np` 曾是「看起来像隔离、实际从未被引用」的参数。
    留着它们迟早会有人再次以为隔离已经做过了。
    """
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_clients":
            names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            for dead in ("x_test_np", "y_test_np"):
                assert dead not in names, (
                    f"build_clients 又出现了参数 `{dead}`。若真要用它做隔离，"
                    "请在函数体里实际引用并补一条断言；否则不要留这个误导性签名。")
            return
    raise AssertionError("main.py 里找不到 build_clients")
