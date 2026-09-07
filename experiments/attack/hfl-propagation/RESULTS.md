# Experiment 3（HFL 结构与后门传播）—— 中间结果快照

> # ⛔ 本文记录的全部数字已作废，勿引用
>
> 下面这份快照写于 2026-08-26，其中的结论建立在**三个此后被证伪或修掉的前提**上。
> 保留原文是为了留档「当时是怎么读的」，**不要**再从中取任何数字。
>
> **1. 测试集泄漏（最严重）**
> `main.py` 曾把 CIFAR-10 官方 10k test split 并进分给客户端的数据池，
> 而同一份 `x_test` 又用来算 ASR 与 `global_acc`。分区是穷尽的、每个 client 再按
> `1−per_client_test_ratio` 划进训练集 → **官方 test 的 75.5% 被训练过**，
> ASR 探针 2000 张里 1509 张被训练过。`global_asr / edge_asr / local_benign_asr /
> same_edge / diff_edge / local_malicious` 六个 ASR 与 `global_acc` 全部受影响。
> 抬高的是**绝对值**；组间相对趋势多半仍成立，但任何绝对数字都不能外发。
>
> **2. 固定 seed 下不可复现**
> `set_seed` 漏播了 Python 内置 `random`，而 `hier_fedrep` 等五个方法客户端每个
> epoch 都用它打乱 batch。所以下表「seed42 / seed43」的落差里，有多少是真的种子
> 方差、多少是不受控的非确定性，**当时无法分离**。
>
> **3. 无定义指标被填 0**
> `diff_edge_asr = 0.000`（distributed 布点下没有干净 edge）与
> `same_edge_asr = 0.000`（`10edge_collocated` 的 E0 没有良性端）都不是「ASR 为零」，
> 是**无定义**。现已改报 `null`。
>
> 另有两处**本文对仓库状态的描述本身就是错的**（写于数据入库的次日）：
> 「3C seed43 因 GPU 分配失败，R4/R5/R10/R20 只有 seed42」和「3c_R40 两 seed 均失败」
> —— 这两批数据其实在 2026-08-25 的 commit `0f7a716` 就已入库且 `exit_code: 0`。
> 报告里 Figure 12「R≥4 无 min–max 带」正是用这批过期数据画的。
>
> 此外 `target_label` 已由 9(truck) 统一为 0(airplane) 以对齐 Bad-PFL 库，
> `eval_interval` 改为按有效轮对齐，参与端配额改为与拓扑无关 —— 三者都会改变数值。
>
> **重跑之后本文整体替换。**

---


> 状态：**2 seed（42/43）中间结果 + flat 两层基线**。图在 `results/figures/`（`plot_exp3.py` 生成）。
> 数据尚有缺口（见文末），结论按「已可下 / 待更多 seed」分级。

## 设定（跨所有格恒定）

100 client · 10 恶意（全局 10%）· cifar10 · hier_fedrep · resnet10(BN) · badpfl ·
`shared_generator=true` · `forced_participation=false` · `poison_ratio=0.2` ·
有效预算 `edge_rounds × n_rounds = 400`。唯一自变量 = 恶意端空间分布（3A/3B）或聚合频率（3C）。
**flat 基线** = `exp007`（`n_edges=1` + `edge_rounds=1` × 400，退化成普通 FedAvg，无 edge 层），
复制为 `results/flat_baseline_seed42.metrics.json`。

## 主发现 1（架构效应，最强、已可下）

**引入 edge 层显著抑制并拖慢后门向个性化模型的传播。**
`fig_timeseries_topology.png`（benign 本地 ASR vs 有效轮 = cloud_round × edge_rounds）：
flat 基线（黑虚线）几乎全程压在所有层级组之上、爬升更快（flat 在 ~75 有效轮已 ~0.79，
层级组同有效轮低得多），最终 flat benign ASR ~0.84。而 **MTA（PM acc）各组几乎一致（~0.75–0.77）**
——即层级结构主要削的是后门传播，不是主任务。这与用户此前「拉平成两层 benign ASR 明显更高」
的观察一致，且现在是在**同有效预算**下、逐轮轨迹上看到的。

## 主发现 2（3C 漂移，机制，已可下）

`fig_3c_frequency.png` 下panel：**参数漂移（abs `‖Δ‖` + rel）与表示漂移（repr mean/median）
随 `edge_rounds` 单调上升，R=20 陡增。** 即 edge 在两次 cloud 聚合之间训得越久，越偏离共识
（backbone 层面）。上panel 的三层 ASR 随 R_edge 大致平、略升。
→ 量化了「聚合越稀疏 → edge 漂移越大」的机制通道。**（drift 的 TF 特征抽取路径已在真实 run 上验证。）**

## 观察 3（拓扑内部差异，**待更多 seed**）

- `10edge_collocated` 轨迹最低：E0 是 100% 恶意，但其 backbone 贡献在 cloud 被 9 个干净 edge 稀释。
- 其余拓扑最终 benign ASR 都落在 **0.74–0.85**，2 seed 下 min–max 带很宽、彼此大量重叠。

| 拓扑 | final benign ASR (seed42 / seed43) | final PM |
|---|---|---|
| 2edge_collocated | 0.729 / 0.757 | ~0.76 |
| 2edge_distributed | 0.871 / 0.648 | ~0.76 |
| 4edge_collocated | 0.838 / 0.866 | ~0.77 |
| 4edge_distributed | 0.859 / 0.716 | ~0.77 |
| 4edge_mixed | 0.858 / 0.708 | ~0.77 |
| 10edge_collocated | 0.709 / 0.824 | ~0.77 |
| 10edge_distributed | 0.897 / 0.672 | ~0.77 |
| 10edge_mixed | 0.790 / 0.857 | ~0.77 |
| **flat (2-layer)** | **0.840** (seed42) | 0.76 |

**判定：拓扑（collocated/distributed/mixed、edge 数）之间的差异目前落在种子噪声内，尚不可分。**
需要 seed44（收窄带）才能对 3A（空间分布）/3B（密度）下结论。architecture（flat vs 层级）的差异
则已足够大、可下。

## 数据缺口 / 质量

- ~~**3C seed43 因 GPU 分配失败**：R4/R5/R10/R20 目前只有 seed42~~
  ❌ **这句是错的**。seed43 的 R4/R5/R10/R20/R40 全部在 commit `0f7a716`（2026-08-25）
  就已入库，`exit_code: 0`。本文写于 8-26，比数据还晚一天却没有更新。
  **报告里的 Figure 12 就是用这份过期认知画的**：R=2 那一点是 2-seed 均值 0.698，
  R≥4 是 seed42 单值 —— 图上「从 0.70 跳到 0.81 然后走平」纯粹是种子可得性的假象。
- ~~**3c_R40 两 seed 均失败**~~ ❌ 同样是错的，两个 seed 都跑完了。
  R40 被排除出 Figure 12 的真实原因是它当时只有 **2 个评估点**
  （`eval_interval` 数 cloud round，而 R40 只有 10 个 cloud round）。
- **flat 仅 seed42** ✅ 这一条属实。而它在 Figure 10 里被当作黑虚线基准，
  去比有 min–max 带的层级组 —— n=1 对 n=2。
- 拓扑 8 组各 2 seed（带偏宽）。
- **下一步**：补 seed44 + 重跑上述失败格（`run_exp3.sh` 断点续跑，命令见会话记录 / README）。
  补完 `python3 plot_exp3.py` 五张图自动刷新。

## 已在真实数据上验证的基础设施

per-edge 指标导出、`by_edge` 确定性布点、forced-participation 补位修复、
**drift 仪表（含 TF 特征抽取路径）**、时间序列「有效轮」对齐、flat 基线纳入 —— 均跑通。

## 图清单（`results/figures/`）

`fig_timeseries_topology.png`（主发现1）· `fig_timeseries_3c.png` · `fig_3c_frequency.png`（主发现2）·
`fig_topology_summary.png` · `fig_per_edge_propagation.png`。
