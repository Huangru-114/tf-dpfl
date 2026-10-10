"""
harness/ck_probe_classes.py  —  在线 c_k（S6b，`server/update_ck.py`）实际用到的攻击样本的类构成（FINDINGS F-095）

    python3 harness/ck_probe_classes.py experiments/attack/hfl-mechanism/configs/G1R5__C1-collocated-R5__s42.yaml ...

`update_ck.py:87` 取 edge 干净集的前 `update_ck_n` 张当攻击样本；干净集由 `designed_partition` 的 `take()` 逐类拼接 → 按类排序。
这里用每类 6 000 张的合成标签（CIFAR-10 合并池的类供给）跑**真实的** `designed_partition`：
干净集各类的张数只取决于类供给、设计参数与 seed，与真实标签在数组里的位置无关 → 前 n 张的类构成与 run 里的相同。
纯 numpy，不 import TF。
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))
from data.designed_partition import designed_partition
labels = np.repeat(np.arange(10), 6000)          # CIFAR-10 合并池：每类 6000（与真实标签的位置无关，只影响索引）
for cfgp in sys.argv[1:]:
    cfg = yaml.safe_load(open(cfgp))
    out = designed_partition(labels, cfg)
    n = int(cfg['evaluation'].get('update_ck_n', 64))
    print('==', cfgp.split('/')[-1], 'n =', n)
    for e, idx in enumerate(out['clean_indices']):
        y = labels[np.asarray(idx)]
        first = y[:n]
        comp = np.bincount(first, minlength=10)
        elig = [int((first != k).sum()) for k in range(10)]
        print(f'  edge{e}: first-{n} class counts {comp.tolist()}  | eligible for k=0 (y_t): {elig[0]}  | clean counts {np.bincount(y, minlength=10).tolist()}')
