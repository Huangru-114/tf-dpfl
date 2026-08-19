# current-focus：Neurotoxin

## 现在要回答的唯一问题

`neurotoxin_mask_ratio` 到底是「**保留**的坐标比例」（官方语义）还是「**屏蔽**的坐标比例」
（本仓库现实现）？——这一个符号决定了恶意更新被限制在 5% 还是 95% 的坐标上。

> 这必须先于任何调参回答。在它确定之前，所有关于 ASR 的观察都没有意义。

## 判定「完成」的客观标准

- **L1**：`pytest tests/test_neurotoxin_mask.py -v` → 3 passed（集群上跑，本地无 TF 会 skip）
- **L2**：`experiments/attack/neurotoxin/expNNN.metrics.json`
  - `n_malicious_participations == n_rounds`（连续参与真的生效了）
  - `final.local_benign_asr` 随轮次单调上升（持久性攻击的特征），而不是原地不动

## 参考实现

- 文献：Zhang et al., *Neurotoxin: Durable Backdoors in Federated Learning*, ICML 2022
- 源仓库：`github.com/jhcknzzm/Federated-Learning-Backdoor`（**clone 前核实**）
- 关键函数：`grad_mask_cv` / `apply_grad_mask`
- clone 的 commit：**待填**

## 语义 diff 表（待用官方实现逐格证实/证伪）

| 步骤 | 官方实现（待核实） | 本仓库现实现 (`client/client_neurotoxin.py`) | 差异 | 怎么验证 |
|---|---|---|---|---|
| benign 梯度来源 | 干净数据上的真实梯度 | 同左（`_benign_grads`，在收到的 edge 模型上算） | 一致 | — |
| mask 选取 | 取 \|g\| **最小**的 `ratio` 比例坐标，mask=1（**保留**） | `(np.abs(a) < thr)`，thr = 全局 top-`mask_ratio` 分位 → **保留 1−ratio** | **疑似反向** | `test_mask_keeps_smallest_benign_gradient_coords` |
| 阈值粒度 | **逐层** | 全局（所有层拼平后取一个阈值） | 不同 | `test_threshold_is_per_layer` |
| 投影 | 逐步 mask 梯度 | 对整轮累计 update 做一次 mask | 一轮内 mask 固定时数学等价（现注释的说法，**未验证**） | `test_masked_coords_have_exactly_zero_update` |
| 参与频率 | 每轮在场 | `main.py` 对 neurotoxin 设 Q=1 | 一致 | L2 `n_malicious_participations` |
| 放大 | 部分设定下有 model-replacement 缩放 | 无 | 不同 | L2 ASR 趋势 |

## 已排除的可能

- ~~benign 梯度用 edge 权重差代理导致 mask 失效~~ —— 已在上一轮修掉（commit `841d5be`），
  现在用真实干净数据梯度。但 ASR 仍不起来，说明问题不在这里。

## 已知的接线阻塞（陷阱 #1，需先修）

`NeurotoxinClient` 继承 `FedAvgClient` 且在 `main.py:395` **替换**掉 PFL 方法类。
`drift_correction != hierfedavg` 时，恶意客户端跑的是另一套算法。
**在正交化修好之前，L2 的 ASR 数字不可解释。** 证据：`tests/test_attack_method_orthogonality.py`

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 按文献语义写 3 条 L1 不变量 | 本地无 TF，skip；待集群跑 | 待集群 L1 结果 |
