"""
tests/test_epoch_pipeline.py  —  A4：取数管线（AUDIT A25 / D-026，开关 data.batch_pipeline）

旧 FedRep 客户端构造时把 tf.data 缓存成 batch 列表：之后每个 epoch 只打乱 batch 的
**顺序**，batch 的**组成**与被丢的尾批永远不变 → seed42 下 3.2% 的训练样本从未参与训练
（FINDINGS F-025）。per_epoch 管线每 epoch 重洗、drop_last 且尾批轮换、重新增强。

纯 numpy + AST，不 import TF。
"""

import ast
from pathlib import Path

import numpy as np

from data.epoch_pipeline import augment_batch, epoch_index_batches

FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"


def _legacy_cached(n, bs, rng):
    """旧 FedRep：构造时分批一次（组成固定），之后每 epoch 只打乱批的顺序、丢非满批。"""
    perm = rng.permutation(n)
    batches = [perm[i:i + bs] for i in range(0, n, bs)]

    def epoch():
        return [batches[i] for i in rng.permutation(len(batches)) if len(batches[i]) == bs]
    return epoch


def test_each_epoch_covers_n_minus_remainder_distinct_samples():
    r = np.random.default_rng(0)
    for n in (133, 401, 909, 64, 31):
        for _ in range(5):
            bs = epoch_index_batches(n, 32, r)
            flat = np.concatenate(bs) if bs else np.array([], int)
            assert all(len(b) == 32 for b in bs)
            assert len(flat) == n - n % 32 and len(set(flat.tolist())) == len(flat)


def test_consecutive_epochs_differ_in_composition_and_dropped_samples():
    r = np.random.default_rng(1)
    n = 133
    e1, e2 = epoch_index_batches(n, 32, r), epoch_index_batches(n, 32, r)
    comp1 = {frozenset(b.tolist()) for b in e1}
    comp2 = {frozenset(b.tolist()) for b in e2}
    assert comp1 != comp2
    drop1 = set(range(n)) - set(np.concatenate(e1).tolist())
    drop2 = set(range(n)) - set(np.concatenate(e2).tolist())
    assert drop1 != drop2


def test_dropped_samples_rotate_so_every_sample_is_eventually_used():
    """尾批轮换：多个 epoch 后，每个样本都至少被训练过一次（旧缓存下有 n mod 32 个永远不用）。"""
    r = np.random.default_rng(2)
    n, used = 133, set()
    for _ in range(40):
        used |= set(np.concatenate(epoch_index_batches(n, 32, r)).tolist())
    assert used == set(range(n))


def test_legacy_cache_freezes_composition_reverse_anchor():
    """反向锚点：旧缓存口径下，连续两个 epoch 的批组成与被丢样本完全相同。"""
    r = np.random.default_rng(3)
    epoch = _legacy_cached(133, 32, r)
    e1, e2 = epoch(), epoch()
    assert {frozenset(b.tolist()) for b in e1} == {frozenset(b.tolist()) for b in e2}
    used = set(np.concatenate(e1).tolist())
    assert len(set(range(133)) - used) == 133 % 32


def test_augment_is_a_flip_plus_zero_padded_shift():
    r = np.random.default_rng(4)
    x = r.normal(size=(64, 8, 8, 3)).astype(np.float32) + 5.0     # 全正，0 只可能来自补边
    y = augment_batch(x, np.random.default_rng(5))
    assert y.shape == x.shape and y.dtype == np.float32
    for i in range(64):
        found = False
        for src in (x[i], x[i, :, ::-1, :]):
            p = np.pad(src, ((4, 4), (4, 4), (0, 0)))
            for oy in range(9):
                for ox in range(9):
                    if np.array_equal(p[oy:oy + 8, ox:ox + 8], y[i]):
                        found = True
        assert found, f"样本 {i} 不是「翻转 + 补零平移」的结果"


def test_augment_flip_rate_and_offsets_are_uniform():
    r = np.random.default_rng(6)
    x = np.zeros((4000, 4, 4, 1), np.float32)
    x[:, :, 0, :] = 1.0                                          # 左列标记，用来看翻转
    rng = np.random.default_rng(7)
    y = augment_batch(x, rng, pad=0)                             # pad=0：只剩翻转
    flipped = (y[:, :, -1, 0] == 1.0).all(axis=1)
    assert abs(flipped.mean() - 0.5) < 0.03


def test_augment_uses_only_the_given_rng():
    x = np.ones((4, 8, 8, 3), np.float32)
    np.random.seed(0)
    a = augment_batch(x, np.random.default_rng(9))
    np.random.seed(1)
    b = augment_batch(x, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


def _calls_epoch_batches(path: Path, func: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr == "epoch_batches":
                    return True
    return False


def test_the_three_fetch_sites_share_one_function():
    """AST：FedAvg 本地训练、FedRep 两个阶段、Bad-PFL 生成器训练都经 epoch_batches 取数。"""
    assert _calls_epoch_batches(FEDAVG / "client/client_fedavg.py", "local_train")
    assert _calls_epoch_batches(FEDAVG / "client/hier_fedrep.py", "_epoch")
    assert _calls_epoch_batches(FEDAVG / "client/client_badpfl.py", "_atk_gen_batches")
    src = (FEDAVG / "client/hier_fedrep.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    phases = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for ph in ("_head_phase", "_body_phase"):
        calls = {s.func.attr for s in ast.walk(phases[ph])
                 if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)}
        assert "_epoch" in calls, f"{ph} 没有经 _epoch 取数"
