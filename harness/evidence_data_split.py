"""数据划分的确凿证据：官方 test split 到底有没有进客户端训练集。

# 这个脚本证明什么

Experiment 3 的全部 ASR / MTA 绝对值此前都建立在一个错误的前提上。
`fedavg/main.py` 在 commit `53a076d` **之前**是这么构造客户端数据池的
（旧文件的 664-667 行，可用 `git show 53a076d~1:fedavg/main.py` 复核）：

    # 合并全量数据后分区：确保 per-client train/test 同分布（PFLlib 标准做法）
    # x_test/y_test 保留用于 edge/GM 评估，不受影响。      ← 这句断言是错的
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    clients, ... = build_clients(x_all, y_all, ...)

而同一份 `x_test` 又被交给 `BackdoorCloudServer(x_test=x_test, ...)`
（`main.py:706` → `server/backdoor_server.py:121-155`）去算 global / edge / local
六个 ASR 与 `global_acc`。

**本脚本调用的是仓库里真实的 `noniid_partition` 与 `split_client_train_test`，
不是复刻件。** 它把两条路径都跑一遍，数出官方 test set 有多少张落进了客户端的
**训练**集：

  · legacy 路径：喂 `concat(x_train, x_test)`（复现旧行为，逐字对照上面的引用）
  · fixed  路径：只喂 `x_train`（当前代码的行为）

预期：legacy ≈ 75%（= 1 − `per_client_test_ratio`），fixed 恰好 0。

# 为什么 legacy 路径要在脚本里重建

因为修复已经把那两行删掉了。直接 checkout 旧的 `main.py` 会**同时**带回
未播种的 `random`、填 0 的空组指标、`int()` 截断三个已修问题，比较就不是单变量的。
在脚本里只重建「合并」这一件事，其余全部走当前代码，才是干净的单变量对照。

# 怎么跑（集群，容器内，**不需要 GPU**）

    apptainer exec <sif> python3 harness/evidence_data_split.py \\
        --config experiments/attack/hfl-propagation/3c_R5.yaml

退出码 0 = fixed 路径零泄漏且 legacy 路径确实泄漏（两者都符合预期）；1 = 有问题。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))

import numpy as np                                     # noqa: E402
import yaml                                            # noqa: E402

from data.dataset import load_cifar10, load_cifar100   # noqa: E402
from data.partition import (noniid_partition,          # noqa: E402
                            split_client_train_test)


def _train_pool(images, labels, n_clients, cfg, alpha, test_ratio):
    """跑真实的分区 + train/test 切分，返回所有客户端**训练**索引的并集。"""
    _, client_indices = noniid_partition(images, labels, n_clients, cfg, alpha=alpha)
    _, train_indices, _ = split_client_train_test(
        images, labels, client_indices, cfg, test_ratio=test_ratio)
    pool = set()
    for idx in train_indices:
        pool |= set(np.asarray(idx).tolist())
    return pool, client_indices


def _current_main_has_no_merge() -> bool:
    """断言当前 main.py 里确实没有把 x_test 并进池子的写法。

    否则本脚本的 "fixed" 一列描述的就不是当前代码。
    """
    tree = ast.parse((ROOT / "fedavg" / "main.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_experiment":
            for n in ast.walk(node):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "concatenate"):
                    if "x_test" in ast.unparse(n) or "y_test" in ast.unparse(n):
                        return False
            return True
    raise SystemExit("main.py 里找不到 run_experiment")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "experiments/attack/hfl-propagation/3c_R5.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    n_clients = int(cfg["federation"]["n_clients"])
    alpha = float(cfg["federation"].get("alpha", 0.5))
    test_ratio = float(cfg["data"].get("per_client_test_ratio", 0.2))
    dataset = str(cfg["data"]["dataset"]).lower()

    loader = load_cifar100 if dataset == "cifar100" else load_cifar10
    _, _, x_train, y_train, x_test, y_test = loader(cfg)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    n_train, n_test = len(y_train), len(y_test)

    print(f"\nconfig            : {args.config.name}")
    print(f"dataset           : {dataset}  train={n_train}  官方 test={n_test}")
    print(f"n_clients / alpha : {n_clients} / {alpha}")
    print(f"per_client_test_ratio : {test_ratio}  → 每个客户端 {1-test_ratio:.0%} 的分片进训练集")
    print(f"seed              : {seed}")

    # 官方 test split 在合并池 x_all 里的 index 就是 [n_train, n_train+n_test)
    official_test = set(range(n_train, n_train + n_test))

    # ── legacy：复现 commit 53a076d 之前的合并（**只重建这一件事**）────────
    np.random.seed(seed)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    legacy_pool, legacy_idx = _train_pool(x_all, y_all, n_clients, cfg, alpha, test_ratio)
    covered = set()
    for idx in legacy_idx:
        covered |= set(np.asarray(idx).tolist())
    legacy_leak = official_test & legacy_pool

    # ── fixed：当前代码的行为（只喂 x_train）──────────────────────────────
    np.random.seed(seed)
    fixed_pool, _ = _train_pool(x_train, y_train, n_clients, cfg, alpha, test_ratio)
    # fixed 路径的索引空间只有 [0, n_train)，与官方 test 天然不相交；
    # 这里显式检查越界，等价于"官方 test 一张都没进来"。
    fixed_leak = {i for i in fixed_pool if i >= n_train}

    print("\n" + "=" * 74)
    print(f"{'':<34}{'legacy（修复前）':>19}{'fixed（当前）':>19}")
    print("-" * 74)
    print(f"{'分区覆盖的 index 数':<32}{len(covered):>16}/{n_train+n_test:<8}"
          f"{len(fixed_pool):>12}/{n_train}")
    print(f"{'官方 test 进【客户端训练集】':<28}{len(legacy_leak):>16}/{n_test:<8}"
          f"{len(fixed_leak):>12}/{n_test}")
    print(f"{'  占官方 test 的比例':<32}{len(legacy_leak)/n_test:>19.1%}"
          f"{len(fixed_leak)/n_test:>19.1%}")

    # ASR 探针：backdoor_server.py:121-131 用 default_rng(seed) 抽 asr_max_samples
    n_probe = int(cfg["backdoor"].get("asr_max_samples", 0) or 0)
    if n_probe and n_probe < n_test:
        pick = np.random.default_rng(seed).choice(n_test, n_probe, replace=False)
        probe = {int(i) + n_train for i in pick}
        print(f"{'ASR 探针中被训练过的':<30}{len(probe & legacy_pool):>16}/{n_probe:<8}"
              f"{len(probe & fixed_leak):>12}/{n_probe}")
        print(f"{'  占探针的比例':<32}{len(probe & legacy_pool)/n_probe:>19.1%}"
              f"{len(probe & fixed_leak)/n_probe:>19.1%}")
    print("=" * 74)

    ok = True
    print("\n判据：")
    c1 = len(covered) == n_train + n_test
    print(f"  {'✅' if c1 else '❌'}  分区是穷尽的：喂进去的 {n_train+n_test} 个 index "
          f"全部归属某个客户端（覆盖 {len(covered)}）")
    ok &= c1
    c2 = len(legacy_leak) / n_test > 0.5
    print(f"  {'✅' if c2 else '❌'}  legacy 路径确实泄漏：{len(legacy_leak)}/{n_test} "
          f"= {len(legacy_leak)/n_test:.1%}（应 ≈ {1-test_ratio:.0%}）")
    ok &= c2
    c3 = len(fixed_leak) == 0
    print(f"  {'✅' if c3 else '❌'}  fixed 路径零泄漏：{len(fixed_leak)}/{n_test}")
    ok &= c3
    c4 = _current_main_has_no_merge()
    print(f"  {'✅' if c4 else '❌'}  当前 main.py 的 run_experiment 里没有把 "
          f"x_test 并进池子的 concatenate")
    ok &= c4

    if not ok:
        print("\n❌ 有判据不成立 —— 见上。")
        return 1
    print(f"\n✅ 全部成立。修复前官方 test 的 {len(legacy_leak)/n_test:.1%} 被客户端训练过，"
          f"修复后为 0。\n"
          f"   注意：这证明的是**泄漏的存在与规模**，不是它把 ASR/MTA 抬高了多少 ——\n"
          f"   后者要用同一 config 跑 legacy/fixed 两次训练来测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
