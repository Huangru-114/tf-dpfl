"""数据划分与 ASR 探针的证据。

# 先说清楚什么**不是** bug

把 train+test 合并后再逐客户端分区，是 PFLlib 的标准做法，本仓库照做。
**合并本身不造成任何泄漏**，因为：

  1. `noniid_partition` 是一个**划分** —— `np.split` 让 x_all 的每个 index
     恰好归属一个客户端；
  2. `split_client_train_test` 再在**每个客户端自己的分片内部**切 train/test。

⇒ 「所有客户端留出分片的并集」与「所有训练数据的并集」**全局不相交**。
   这是真正的留出集，而且保住了全部 60k 数据和 PFLlib 口径。

# 真正错过的地方（评估侧，不是划分侧）

曾经把**原始 `x_test` 数组**又当成一份独立探针去算六个 ASR。合并之后官方
test split 里的图已经分给客户端了，其中 1−`per_client_test_ratio` 进了训练集，
再拿它当探针就是在训练过的图上测攻击成功率。

本脚本用仓库里**真实的** `noniid_partition` / `split_client_train_test`
把这几件事量出来：

  · 分区是划分吗（每个 index 恰好一次）
  · 训练池 ∩ 留出池 是不是空（B 方案成立的依据）
  · 旧探针（官方 x_test）有多少张落进训练池 —— 这是它不能当探针的原因
  · 其中多少落进**恶意端**的训练分片 —— 那部分是被真的投毒训过的，
    模型对它们是**记忆**而非泛化，ASR 会被直接抬高
  · 新探针（留出池）与训练池的交集 —— 必须是 0

# 怎么跑（集群，容器内，**不需要 GPU**）

    bash run_evidence.sh                                   # 默认 3c_R5.yaml
    bash run_evidence.sh experiments/attack/hfl-propagation/4edge_collocated.yaml

退出码 0 = 全部判据成立。
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

from data.clustering import block_assignment            # noqa: E402
from data.dataset import load_cifar10, load_cifar100    # noqa: E402
from data.partition import (noniid_partition,           # noqa: E402
                            split_client_train_test)
from attack.backdoor import resolve_malicious_ids       # noqa: E402


def _pools(images, labels, n_clients, cfg, alpha, test_ratio):
    """跑真实的分区 + 切分，返回 (每客户端训练索引, 训练池, 留出池, 全部索引)。"""
    _, client_indices = noniid_partition(images, labels, n_clients, cfg, alpha=alpha)
    _, train_indices, _ = split_client_train_test(
        images, labels, client_indices, cfg, test_ratio=test_ratio)
    all_idx = set()
    for idx in client_indices:
        all_idx |= set(np.asarray(idx).tolist())
    train_pool = set()
    per_client_train = []
    for idx in train_indices:
        cur = set(np.asarray(idx).tolist())
        per_client_train.append(cur)
        train_pool |= cur
    return per_client_train, train_pool, all_idx - train_pool, all_idx


def _asr_probe_is_heldout() -> bool:
    """AST：`_backdoor_eval` 传给 ASR 的是留出集，不是 x_test 派生的数组。"""
    src = (ROOT / "fedavg" / "server" / "backdoor_server.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_backdoor_eval":
            for n in ast.walk(node):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "evaluate_hierarchical_asr"):
                    args = " ".join(ast.unparse(a) for a in n.args)
                    return ("self.test_dataset" in args
                            and "x_test" not in args and "y_test" not in args)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "experiments/attack/hfl-propagation/3c_R5.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    n_clients = int(cfg["federation"]["n_clients"])
    n_edges = int(cfg["federation"]["n_edges"])
    alpha = float(cfg["federation"].get("alpha", 0.5))
    test_ratio = float(cfg["data"].get("per_client_test_ratio", 0.2))
    dataset = str(cfg["data"]["dataset"]).lower()

    loader = load_cifar100 if dataset == "cifar100" else load_cifar10
    _, _, x_train, y_train, x_test, y_test = loader(cfg)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    n_train, n_test = len(y_train), len(y_test)

    print(f"\nconfig  : {args.config.name}   dataset={dataset}")
    print(f"          train={n_train}  官方 test={n_test}  n_clients={n_clients}  "
          f"n_edges={n_edges}  alpha={alpha}  per_client_test_ratio={test_ratio}")

    # PFLlib 口径：合并后分区（当前代码的行为）
    np.random.seed(seed)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    per_client_train, train_pool, heldout_pool, all_idx = _pools(
        x_all, y_all, n_clients, cfg, alpha, test_ratio)

    official_test = set(range(n_train, n_train + n_test))   # x_all 里官方 test 的位置

    # 恶意端是谁（复刻 main.py 的接线）
    assign = (block_assignment(list(range(n_clients)), n_edges)
              if str(cfg["federation"].get("edge_assignment", "random")).lower() == "block"
              else None)
    mal_ids = set(int(i) for i in
                  resolve_malicious_ids(cfg["backdoor"], n_clients,
                                        assignments=assign, seed=seed))
    mal_train = set()
    for cid in mal_ids:
        mal_train |= per_client_train[cid]

    print("\n" + "=" * 76)
    print("【划分侧】合并分区本身有没有问题")
    print("-" * 76)
    print(f"  分区覆盖的 index          : {len(all_idx)} / {n_train + n_test}")
    print(f"  训练池                    : {len(train_pool)}")
    print(f"  留出池（= 所有留出分片）  : {len(heldout_pool)}")
    print(f"  训练池 ∩ 留出池           : {len(train_pool & heldout_pool)}   ← 必须是 0")

    old_leak = official_test & train_pool
    old_leak_mal = official_test & mal_train
    print("\n【旧探针】官方 x_test 当 ASR 探针会怎样")
    print("-" * 76)
    print(f"  官方 test 落进训练池      : {len(old_leak)} / {n_test} "
          f"= {len(old_leak)/n_test:.1%}")
    print(f"    其中落进**恶意端**训练分片: {len(old_leak_mal)} / {n_test} "
          f"= {len(old_leak_mal)/n_test:.1%}")
    print(f"      → 这部分被真的加过触发器、翻过标签，模型是**记忆**了它们；")
    print(f"        其余 {len(old_leak)-len(old_leak_mal)} 张是良性端按正确标签训的，")
    print(f"        方向相反（更难被翻）。净效果需要跑对照实验才知道。")

    print("\n【新探针】留出池当 ASR 探针")
    print("-" * 76)
    print(f"  留出池 ∩ 训练池           : {len(heldout_pool & train_pool)}   ← 必须是 0")
    print(f"  留出池里来自官方 test 的  : {len(heldout_pool & official_test)} "
          f"（无所谓：它们没被任何客户端训练过）")
    print("=" * 76)

    ok = True
    print("\n判据：")
    for text, cond in [
        (f"分区是划分：{n_train+n_test} 个 index 全部恰好归属一个客户端",
         len(all_idx) == n_train + n_test),
        ("训练池与留出池全局不相交（B 方案成立的依据）",
         not (train_pool & heldout_pool)),
        (f"旧探针确有重叠：官方 test 的 {len(old_leak)/n_test:.0%} 被训练过"
         f"（应 ≈ {1-test_ratio:.0%}）", len(old_leak) / n_test > 0.5),
        ("恶意端记忆的那部分可量化（> 0 说明确有记忆成分）", len(old_leak_mal) > 0),
        ("当前代码的 ASR 探针是留出集，不含 x_test", _asr_probe_is_heldout()),
    ]:
        print(f"  {'✅' if cond else '❌'}  {text}")
        ok &= bool(cond)

    if not ok:
        print("\n❌ 有判据不成立 —— 见上。")
        return 1
    print("\n✅ 全部成立。结论：**合并分区没问题**（PFLlib 口径，训练/留出全局不相交）；"
          f"\n   问题只在评估侧曾用官方 x_test 当探针，而它有 {len(old_leak)/n_test:.0%} "
          f"被训练过、{len(old_leak_mal)/n_test:.1%} 被投毒训练过。现已改为留出集。"
          "\n   ⚠️ 这证明的是**重叠的存在与规模**，不是它把 ASR 抬高了多少 ——"
          "\n      净方向需要同一 config 跑新旧两套探针各一次才能定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
