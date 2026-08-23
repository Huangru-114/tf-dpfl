# Experiment 3 enabling —— 新增的接线（本会话，供 3A/3B/3C 用）

分支 `claude/exp3-hfl-enabling-*`。三件 enabling 基础，都带 L1。**不含实验判据**——
开跑前另写 `current-focus.md`（一个会话一个模块）。

## ① 逐 edge 指标导出

`evaluate_hierarchical_asr` 现返回 `per_edge`（+ `edge_ids`）；`backdoor_server` 每轮每 edge
打一条可解析行：
```
[Backdoor] Round R | edge0 | edge_asr=.. | client_benign=.. | client_malicious=.. | n_benign=.. | n_malicious=.. | has_malicious=True
```
`harness/collect_metrics.py` 解析进 `metrics.json` 的 `per_edge_rounds`（{round: [per-edge…]}）
与 `per_edge_final`（末轮快照）。**判 amplification/dilution/cross-edge cancellation 用这个，
不要用被均值抹平的 `edge_asr_mean`。**

判据对应关系（逐 edge 比较）：
- Local amplification：`per_edge[e].edge_asr > per_edge[e].client_benign`
- Hierarchical amplification：`final.global_asr > mean(per_edge[*].edge_asr)`
- Dilution：`per_edge[e].edge_asr < per_edge[e].client_malicious`
- Cross-edge cancellation：每个 edge 的 `edge_asr` 都高，但 `global_asr < mean(edge_asr)`

## ② 确定性按 edge 布点

新 `edge_assignment: block`（确定性连续分块，client id 段 → edge，可从 id 直观预期）+
新 `malicious_placement: by_edge`，读 `backdoor.malicious_per_edge`（按 edge id 索引的个数列表）。
config_validate 早筛长度/容量。

三种拓扑（40 client / 4 edge，配 `edge_assignment: block`）：
```yaml
federation: { n_clients: 40, n_edges: 4, edge_assignment: block }
backdoor:
  malicious_placement: by_edge
  # A. Collocated   —— 全挤在 E0
  malicious_per_edge: [4, 0, 0, 0]
  # B. Distributed  —— 每 edge 1 个
  # malicious_per_edge: [1, 1, 1, 1]
  # C. Mixed        —— 部分高部分低
  # malicious_per_edge: [3, 1, 0, 0]
```
> 全局恶意比例相同（都是 4/40）但**局部密度**不同——正是 3A/3B 的自变量。

## ③ forced_participation 补位不再饿死良性

`install_forced_participation` 的包装器改为 `selected = base良性 + must_in恶意`：恶意端按 Q
调度强制在场，**base 抽中的良性一个不砍**。之前「只砍良性」在恶意密集 edge 会把良性砍光
（exp006 的 pm_acc 崩溃根因）。3B 若开 forced_participation 控密度，不会再复发。

## 提醒（沿用上个会话结论）

- Exp 3 全程 `badpfl_shared_generator: true`（eval trigger 取 mal[0]，共享 generator 才对全体
  有代表性）。
- 等概率抽样有参与度方差，**每格 ≥3 seed，报 mean±std**，seed 作矩阵一轴。
- `diff_edge` 在 distributed/mixed 下恒 0（每 edge 都含恶意）——用 ① 的逐 edge 面板，别用它。
