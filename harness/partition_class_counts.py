#!/usr/bin/env python3
"""
每个客户端本地各类别的样本数（旧 `noniid` 划分，Dir α；阶段三 F-110：验证「本地没有目标类 → 个性化模型对后门免疫」）。

为什么要它：noniid 划分不打 `[Partition]` 行，metrics.json 里没有逐端的类别分布；F-110 只能用 yt_clean（干净样本被判成目标类的比例）
当代理。这里在集群上按 main.py 的同一条随机序列重算一遍划分，只输出计数（100 × 10 × 2 个整数），不回传任何数组。

复刻（与 `data/partition.py` 逐行对应，`tests/test_partition_class_counts_tf.py` 在合成标签上与真函数对拍）：
  np.random.seed(seed) → noniid_partition：逐类 shuffle + Dirichlet(α·1) 切分 → split_client_train_test：逐端 permutation，
  前 max(1, int(n·test_ratio)) 个为测试。main.run_experiment 在这之间不消耗全局 np.random（读数据、建模型都走 TF）——
  **这一句由下面的锚点核对，不靠相信**：逐端测试集大小必须等于给定 metrics.json 的 client_final.n，不等就退出码 3。

用法（集群上，要 TF 读 CIFAR-10；见 experiments/defense/edge-native/partition_counts.sbatch）：
  python3 harness/partition_class_counts.py --config <cfg.yaml> --metrics <run.metrics.json> --out <analysis/xxx.json>
本地（已有计数 json 时）只做相关分析：--counts <json> --metrics <run.metrics.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def replicate_partition(labels: np.ndarray, n_clients: int, alpha: float, test_ratio: float, seed: int):
    """返回 (train_idx 列表, test_idx 列表)，与 noniid_partition + split_client_train_test 同一随机序列。"""
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(n_clients)]
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha=np.ones(n_clients) * alpha)
        cut_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for cid, split in enumerate(np.split(class_indices, cut_points)):
            client_indices[cid].extend(split.tolist())
    train, test = [], []
    for indices in client_indices:
        indices = np.array(indices)
        n = len(indices)
        n_test = max(1, int(n * test_ratio))
        perm = np.random.permutation(n)
        test.append(indices[perm[:n_test]])
        train.append(indices[perm[n_test:]])
    return train, test


def class_counts(idx_list, labels, num_classes):
    return [np.bincount(labels[i], minlength=num_classes).astype(int).tolist() for i in idx_list]


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):                       # 并列取平均秩
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = math.fsum(ra) / n, math.fsum(rb) / n
    num = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(math.fsum((x - ma) ** 2 for x in ra) * math.fsum((y - mb) ** 2 for y in rb))
    return None if den == 0 else num / den


def analyse(counts: dict, metrics: dict, target: int) -> dict:
    """逐端 ASR（client_final，良性端）对本地目标类训练样本数 / 占比的相关与分组。"""
    cf = metrics["client_final"]
    rows = []
    for cid, asr, mal in zip(cf["client_id"], cf["asr"], cf["malicious"]):
        if mal or asr is None:
            continue
        tr = counts["train_counts"][int(cid)]
        rows.append({"client_id": int(cid), "asr": asr, "n_target": tr[target], "n_train": sum(tr),
                     "share_target": tr[target] / max(1, sum(tr))})
    A = [r["asr"] for r in rows]
    groups = {}
    for name, lo, hi in (("0", 0, 1), ("1-9", 1, 10), ("10-49", 10, 50), (">=50", 50, 10 ** 9)):
        g = [r["asr"] for r in rows if lo <= r["n_target"] < hi]
        groups[name] = {"n": len(g), "mean_asr": None if not g else round(math.fsum(g) / len(g), 4),
                        "n_asr_lt_0.1": sum(a < 0.1 for a in g)}
    return {"target_label": target, "n_benign": len(rows),
            "spearman_asr_vs_n_target": spearman(A, [r["n_target"] for r in rows]),
            "spearman_asr_vs_share_target": spearman(A, [r["share_target"] for r in rows]),
            "spearman_asr_vs_n_train": spearman(A, [r["n_train"] for r in rows]),
            "by_n_target": groups, "clients": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="noniid 划分的逐端类别计数 + 与逐端 ASR 的相关（F-110）")
    ap.add_argument("--config")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--counts", help="已有的计数 json（只做分析）")
    ap.add_argument("--target", type=int, help="目标类（缺省读配置的 backdoor.target_label）")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    metrics = json.loads(Path(a.metrics).read_text(encoding="utf-8"))
    if a.counts:
        counts = json.loads(Path(a.counts).read_text(encoding="utf-8"))
    else:
        import yaml
        sys.path.insert(0, str(ROOT / "fedavg"))
        from data.dataset import load_cifar10                     # 与 main.run_experiment 同一个读法
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        assert cfg["federation"].get("partition", "noniid") == "noniid", "只复刻旧 noniid 划分"
        _, _, _, y_train, _, y_test = load_cifar10(cfg)
        labels = np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0).reshape(-1).astype(int)
        train, test = replicate_partition(labels, int(cfg["federation"]["n_clients"]),
                                          float(cfg["federation"].get("alpha", 0.5)),
                                          float(cfg["data"].get("per_client_test_ratio", 0.2)), int(cfg.get("seed", 42)))
        nc = int(cfg["data"]["num_classes"])
        counts = {"config": a.config, "seed": int(cfg.get("seed", 42)),
                  "target_label": int((cfg.get("backdoor") or {}).get("target_label", 0)),
                  "train_counts": class_counts(train, labels, nc), "test_counts": class_counts(test, labels, nc)}
        # 锚点：逐端测试集大小 = metrics.json 的 client_final.n（同一个划分的唯一外部证据）
        cf = metrics["client_final"]
        want = {int(c): int(n) for c, n in zip(cf["client_id"], cf["n"])}
        got = {i: sum(t) for i, t in enumerate(counts["test_counts"])}
        bad = [i for i in want if want[i] != got.get(i)]
        counts["anchor"] = {"n_clients_checked": len(want), "n_mismatch": len(bad), "first_mismatch": bad[:5]}
        if bad:
            print(json.dumps(counts["anchor"]))
            print("[partition_counts] 测试集大小与 metrics.json 对不上 → 没复现 main.py 的随机序列，结果不可用")
            if a.out:
                Path(a.out).write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")
            return 3
    target = a.target if a.target is not None else int(counts.get("target_label", 0))
    res = {"counts": counts, "analysis": analyse(counts, metrics, target)}
    summary = {k: v for k, v in res["analysis"].items() if k != "clients"}
    print(json.dumps({"anchor": counts.get("anchor"), **summary}, ensure_ascii=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
