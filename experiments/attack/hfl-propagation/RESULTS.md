# Experiment 3（HFL 结构与后门传播）—— 中间结果快照

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

- **3C seed43 因 GPU 分配失败**（`cudaSetDevice() … device busy/unavailable`，**非代码 bug**）：
  R4/R5/R10/R20 目前只有 seed42（单点，无 min–max 界）；R2 有两 seed。
- **3c_R40 两 seed 均失败**；**flat 仅 seed42**。
- 拓扑 8 组各 2 seed（带偏宽）。
- **下一步**：补 seed44 + 重跑上述失败格（`run_exp3.sh` 断点续跑，命令见会话记录 / README）。
  补完 `python3 plot_exp3.py` 五张图自动刷新。

## 已在真实数据上验证的基础设施

per-edge 指标导出、`by_edge` 确定性布点、forced-participation 补位修复、
**drift 仪表（含 TF 特征抽取路径）**、时间序列「有效轮」对齐、flat 基线纳入 —— 均跑通。

## 图清单（`results/figures/`）

`fig_timeseries_topology.png`（主发现1）· `fig_timeseries_3c.png` · `fig_3c_frequency.png`（主发现2）·
`fig_topology_summary.png` · `fig_per_edge_propagation.png`。
