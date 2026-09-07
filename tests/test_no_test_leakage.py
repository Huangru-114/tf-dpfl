"""ASR 探针必须是**留出分片**，不能是原始 x_test。

# 先说清楚什么不是 bug

把 train+test **合并后再逐客户端分区**是 PFLlib 的标准做法，本仓库照做，
**这不是泄漏**。关键性质是：

  1. `noniid_partition` 是一个**划分** —— `np.split` 让 x_all 的每个 index
     恰好归属一个客户端；
  2. `split_client_train_test` 再在**每个客户端自己的分片内部**切 train/test。

⇒ 「所有客户端留出分片的并集」与「所有训练数据的并集」**全局不相交**。
   那才是真正的留出集，而且它保住了全部 60k 数据。

# 真正错过的地方

评估侧曾经把**原始 `x_test` 数组**又当成一份独立探针，拿去算六个 ASR
（`main.py` → `BackdoorCloudServer(x_test=x_test)` → `_backdoor_eval`）。
一旦合并，官方 test split 里的图已经分给客户端、其中 1−`per_client_test_ratio`
进了训练集，再拿它当探针就是**在训练过的图上测攻击成功率**。

顺带的第二个问题：那样一来 ASR 测在类别均匀的官方测试集上，而 `pm_acc` 测在
非 IID 的 per-client 分片上 —— 两个数字根本不在同一个样本 population 上。

现在三层 ASR 一律测在留出分片上：
  global ← `merge_test_datasets(所有 edge)`｜edge ← `edge.get_test_dataset()`
  ｜client ← `client.test_dataset`（与它自己的 `pm_acc` 同一个集合）
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FEDAVG = ROOT / "fedavg"


# ═══════════════ A. 划分性质：训练与留出全局不相交（需要 TF） ═══════════════

def test_train_and_heldout_pools_are_globally_disjoint():
    """
    **这是 B 方案成立的全部依据。**

    不是「客户端 i 的 test 与它自己的 train 不相交」（那是显然的），
    而是「所有 test 分片的并集」与「所有 train 分片的并集」不相交 ——
    因为每个 index 只归属一个客户端。若这条不成立，用留出分片当 ASR 探针
    就和用 x_test 一样有问题。
    """
    pytest.importorskip("tensorflow")
    import numpy as np
    sys.path.insert(0, str(FEDAVG))
    from data.partition import noniid_partition, split_client_train_test

    N, N_CLIENTS, N_CLASSES = 2000, 10, 10
    rng = np.random.default_rng(0)
    images = rng.random((N, 8, 8, 3), dtype=np.float32)
    labels = np.repeat(np.arange(N_CLASSES), N // N_CLASSES)
    cfg = {"data": {"batch_size": 16, "shuffle_buffer": 100, "img_size": 8}}

    _, client_indices = noniid_partition(images, labels, N_CLIENTS, cfg, alpha=0.5)

    # 1) 分区是划分：每个 index 恰好出现一次
    flat = [int(i) for idx in client_indices for i in np.asarray(idx)]
    assert len(flat) == len(set(flat)) == N, (
        f"分区不是划分：{len(flat)} 个 index，去重后 {len(set(flat))}，应为 {N}")

    # 2) 训练并集与留出并集不相交
    _, train_idx, _ = split_client_train_test(
        images, labels, client_indices, cfg, test_ratio=0.25)
    train_pool = {int(i) for idx in train_idx for i in np.asarray(idx)}
    heldout_pool = set(range(N)) - train_pool
    assert not (train_pool & heldout_pool), "训练池与留出池相交了"
    frac = len(heldout_pool) / N
    assert 0.20 < frac < 0.30, f"留出池占 {frac:.1%}，应 ≈ 25%"


# ═══════════════ B. 评估侧：探针来自留出分片（纯 AST，本地可跑） ═══════════════

def _fn(path: Path, name: str):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} 里找不到 {name}")


def test_hierarchical_asr_is_called_with_the_heldout_dataset():
    """`_backdoor_eval` 传给 ASR 的必须是留出集，不是 x_test 派生的数组。"""
    fn = _fn(FEDAVG / "server" / "backdoor_server.py", "_backdoor_eval")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "evaluate_hierarchical_asr"):
            args = [ast.unparse(a) for a in node.args]
            assert "self.test_dataset" in args, (
                f"实参是 {args}，其中应当有 self.test_dataset（留出集的并集）")
            joined = " ".join(args)
            assert "x_test" not in joined and "y_test" not in joined, (
                f"实参里出现了 x_test/y_test：{args}。合并分区之后官方 test split "
                "已经分给客户端了，不能再拿它当 ASR 探针。")
            return
    raise AssertionError("_backdoor_eval 里找不到 evaluate_hierarchical_asr(...) 调用")


def test_no_asr_subsampling_over_the_official_test_array():
    """
    旧路径会缓存一份 `_asr_subsample_idx` 对 x_test 做随机子采样。
    现在改成在留出集上「取前 N 个合格样本」（dataset 是 shuffle=False 建的，
    顺序固定，确定性），那份缓存索引不该再存在。
    """
    src = (FEDAVG / "server" / "backdoor_server.py").read_text(encoding="utf-8")
    assert "_asr_subsample_idx" not in src, (
        "还留着对官方 x_test 的随机子采样缓存 —— 说明探针没换干净")


def test_compute_asr_returns_none_when_no_eligible_samples():
    """
    非 IID 下客户端的留出分片**可能一个非目标类样本都没有** → ASR 无定义。
    返回 0.0 会被读成「攻击完全失败」，那是个强结论。
    B 方案把探针换成 per-client 分片之后，这种情况从罕见变成常态。
    """
    src = (FEDAVG / "attack" / "backdoor_eval.py").read_text(encoding="utf-8")
    for name in ("compute_asr", "compute_asr_on_dataset"):
        fn = _fn(FEDAVG / "attack" / "backdoor_eval.py", name)
        body = ast.unparse(fn)
        assert "return 0.0" not in body, (
            f"{name} 里还有 `return 0.0` —— 无合格样本时必须返回 None")
    assert "return None" in src


def test_per_client_asr_uses_that_client_own_split():
    """
    每个客户端的 ASR 探针必须是它**自己**的 test_dataset ——
    这样 ASR 与 pm_acc 才在同一个 population 上。
    """
    fn = _fn(FEDAVG / "attack" / "backdoor_eval.py", "evaluate_hierarchical_asr")
    # 找客户端循环里那次 compute_asr_on_dataset，断言第二个实参就是 `ds`，
    # 而 `ds` 来自 c.test_dataset。用 AST 而不是子串匹配，改了缩进也不会假绿。
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "compute_asr_on_dataset"]
    assert calls, "找不到 compute_asr_on_dataset 调用"
    probes = {ast.unparse(c.args[1]) for c in calls if len(c.args) > 1}
    assert "ds" in probes, f"客户端 ASR 的探针实参是 {probes}，应含 ds"
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "ds" for t in n.targets)]
    assert assigns and "c.test_dataset" in ast.unparse(assigns[0]), \
        "ds 不是从 c.test_dataset 来的 —— 客户端探针不是它自己的分片"


def test_edge_asr_uses_that_edge_own_split():
    fn = _fn(FEDAVG / "attack" / "backdoor_eval.py", "evaluate_hierarchical_asr")
    src = ast.unparse(fn)
    assert "edge.get_test_dataset()" in src, "edge ASR 没有用该 edge 的留出集"


def test_merge_is_kept_because_it_is_not_the_bug():
    """
    合并 train+test 再分区是 PFLlib 口径，**不是** bug，不要再被删掉。
    （这条测试存在是因为我们曾经误删过一次。）
    """
    fn = _fn(FEDAVG / "main.py", "run_experiment")
    merged = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "concatenate"
              and {"x_train", "x_test"} <= set(ast.unparse(n).replace("[", " ")
                                               .replace("]", " ").replace(",", " ").split())]
    assert merged, "run_experiment 里没有合并 x_train/x_test —— PFLlib 口径被丢掉了"
