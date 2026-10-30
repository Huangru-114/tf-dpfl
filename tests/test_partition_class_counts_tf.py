"""
`harness/partition_class_counts.py`（F-110）的复刻划分 == `data/partition.py` 的真函数（同一条全局 np.random 序列）。
真函数要建 tf.data → 需要 TF；分析函数是纯 python，另有本地测试（下半部分，不需要 TF）。
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))
import partition_class_counts as P                     # noqa: E402

CFG = ROOT / "experiments/defense/edge-native/configs/CCSF__off__s42.yaml"


def test_replica_equals_the_real_partition_functions():
    pytest.importorskip("tensorflow")
    from data.partition import noniid_partition, split_client_train_test
    cfg = copy.deepcopy(yaml.safe_load(CFG.read_text(encoding="utf-8")))
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 10, size=3000)
    images = np.zeros((len(labels), 4, 4, 3), np.float32)
    cfg["data"]["img_size"] = 4
    np.random.seed(42)
    _, idx = noniid_partition(images, labels, 20, cfg, alpha=0.5)
    _, train, _ = split_client_train_test(images, labels, idx, cfg, test_ratio=0.25)
    rtrain, rtest = P.replicate_partition(labels, 20, 0.5, 0.25, 42)
    assert len(train) == len(rtrain) == 20
    for a, b, t, full in zip(train, rtrain, rtest, idx):
        assert np.array_equal(a, b)                                    # 训练索引逐个相同（含顺序）
        assert sorted(np.concatenate([b, t]).tolist()) == sorted(np.asarray(full).tolist())


def test_analysis_groups_and_spearman_on_synthetic_counts():
    counts = {"train_counts": [[0, 5], [3, 5], [20, 5], [60, 5]], "target_label": 0}
    metrics = {"client_final": {"client_id": [0, 1, 2, 3], "asr": [0.0, 0.2, 0.6, 0.9],
                                "malicious": [False, False, False, False]}}
    r = P.analyse(counts, metrics, 0)
    assert r["n_benign"] == 4
    assert r["spearman_asr_vs_n_target"] == pytest.approx(1.0)
    assert r["by_n_target"]["0"] == {"n": 1, "mean_asr": 0.0, "n_asr_lt_0.1": 1}
    assert r["by_n_target"][">=50"]["mean_asr"] == 0.9
    metrics["client_final"]["malicious"][3] = True                     # 攻击端不计
    assert P.analyse(counts, metrics, 0)["n_benign"] == 3


def test_spearman_handles_ties_and_constant_input():
    assert P.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
    assert P.spearman([1, 1, 1], [1, 2, 3]) is None
