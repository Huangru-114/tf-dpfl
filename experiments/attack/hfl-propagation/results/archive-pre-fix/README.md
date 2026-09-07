# 归档：修复前的 Experiment 3 结果（⛔ 勿引用数字）

这 29 个 `metrics.json` 是 2026-08 那一轮 Experiment 3 的产物。它们**已经作废**，
移到这里有两个目的：留档，以及**让重跑不会跳过它们**。

## 为什么必须移走

`run_exp3.sh` 的断点续跑是按文件判定的：

```bash
# run_exp3.sh:12,46
# 断点续跑：results/<exp_id>.metrics.json 里 exit_code==0 即算完成、跳过。
[ -s "$1" ] && grep -q '"exit_code": *0' "$1"
```

这 29 个文件全部带 `"exit_code": 0`。留在 `results/` 里，下一次
`bash run_exp3.sh` 会**静默跳过每一个格子**，一个 GPU 作业都不提交，
而输出看起来一切正常（`已完成=29`）。这类失败最难发现。

## 为什么作废

1. **ASR 探针用了官方 `x_test`**。注意：**合并 train+test 再分区不是问题**
   （PFLlib 口径，且分区是划分 → 留出分片与训练数据全局不相交）。问题是评估侧
   又拿原始 `x_test` 当独立探针，而它的图已经分给客户端、约 1−test_ratio
   进了训练集。现已改为三层都测留出分片。影响幅度未知（恶意端那部分是记忆、
   抬高 ASR；良性端那部分压低 ASR），跑 `bash run_evidence.sh` 可量化。
2. **`target_label` 由 9(truck) 改为 0(airplane)**，与 Bad-PFL 库统一。
3. **固定 seed 下不可复现**：`set_seed` 漏播 Python `random`，而五个方法客户端
   每 epoch 都用它打乱 batch。这批数据的「seed42 vs seed43」落差里有多少是真的
   种子方差，无法分离。
4. **无定义指标被填 0**：`diff_edge_asr` / `same_edge_asr` 的 `0.000` 有一部分
   是「无定义」而非「ASR 为零」，现已改报 `null`。
5. **参与端配额**：`int()` 截断使 4-edge 的格子每轮少训 20% 的客户端。
6. **`eval_interval`** 改为按有效轮对齐（原来一律 5 个 cloud round，
   于是 flat 有 80 个评估点而 R40 只有 2 个）。

## 还能用它们做什么

只有一件事：**证明报告里的 Figure 12 是用过期数据画的**。
`3c_R4/R5/R10/R20/R40` 的 seed43 在 commit `0f7a716`（2026-08-25）就已入库，
而 Figure 12 的 R≥4 是单点无带 —— 反推它画的是 seed42 单值
（0.813/0.798/0.811/0.828），R=2 那点才是 2-seed 均值 0.698。
对这批归档数据跑一次 `plot_exp3.py`，R≥4 就会出现 min–max 带，
差异即为证据。**这是零机时的，做完即可回答导师关于图的质疑。**
