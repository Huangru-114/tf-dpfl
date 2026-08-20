# current-focus：FLAME

## 现在要回答的唯一问题

把 FLAME 从「majority-cosine 近似」改写成文献方案（HDBSCAN 余弦聚类 + 中位数范数裁剪
+ 无权平均 + 高斯噪声）后，`tests/test_flame.py` 的五条不变量是否全过。

## 判定「完成」的客观标准

- **L1**：`pytest tests/test_flame.py -v` → 5 passed（当前 **3 failed / 2 passed**）
- **L2**：`experiments/defense/flame/expNNN.metrics.json`
  - 无攻击时 `admitted_count_mean == total`（防御不误伤）
  - badnet 攻击下 `final.local_benign_asr` 明显低于 `defense=none` 的对照

## 参考实现

- 文献：Nguyen et al., *FLAME: Taming Backdoors in Federated Learning*, USENIX Security 2022
- 源仓库：**待确认**（clone 前先核实 URL，不要凭印象写）
- 依赖：`sklearn.cluster.HDBSCAN`（sklearn ≥ 1.3；集群已有 1.8，无需第三方 `hdbscan` 包）

## 语义 diff 表

| 论文步骤 | 论文定义 | 本仓库现实现 (`defense/flame.py`) | 差异 | 怎么验证 |
|---|---|---|---|---|
| 聚类过滤 | 两两**余弦距离**跑 HDBSCAN，`min_cluster_size=m//2+1`, `min_samples=1`, `allow_single_cluster=True`，取最大簇 | 余弦相似度取 off-diagonal **中位数**为阈值，保留邻居数 ≥ m/2 者 | **根本不同**。更新彼此接近时中位数阈值退化成抛硬币 | `test_all_benign_admits_everyone` / `test_rejects_opposite_direction_attackers` |
| 范数裁剪 | `S_t = median(‖e_i‖₂)`，`γ_i = min(1, S_t/‖e_i‖)` | 同左（median 只在 admitted 上取） | median 取值范围不同（admitted vs 全体） | `test_clipping_bounds_aggregate_norm` ✅ 已过 |
| 聚合 | admitted 上的**无权平均** `1/L·Σ` | `np.average(..., weights=n_samples)` | **不同**。按样本数加权 → 谎报样本数成为免费放大通道 | `test_aggregation_is_unweighted_over_admitted` |
| 加噪 | `N(0, (λ·S_t)²)` 加到聚合结果 | 同左，λ=`noise_scale`=0.001 | 一致 | `test_zero_noise_is_deterministic` ✅ 已过 |

## 已排除的可能

- ~~缺 HDBSCAN 依赖所以只能用近似~~ —— 已排除：集群 site-packages 里有 scikit-learn 1.8.0，
  `sklearn.cluster.HDBSCAN` 自 1.3 起内置。原注释里的「无依赖」理由不再成立。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 按文献写 5 条 L1 不变量，跑当前实现 | `3 failed, 2 passed`；全良性 20 个更新只接纳 11 个（`admitted 11/20, cos_thresh=0.997`） | 确认现实现在**无攻击时也会误伤近半良性客户端**，非文献方案 |
