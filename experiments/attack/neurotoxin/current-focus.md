# current-focus：Neurotoxin

> **前置**：本方法依赖「陷阱 #1 正交化」先完成。若 `tests/test_attack_method_orthogonality.py`
> 还是失败/skip 状态，先停下来告诉我 —— 在那之前 L2 的 ASR 数字不可解释。

## 现在要回答的唯一问题

官方 `grad_mask_cv` 里的 `ratio` 到底是「**保留**的坐标比例」还是「**屏蔽**的坐标比例」？

这一个符号决定恶意更新被限制在 5% 还是 95% 的坐标上，也就决定 Neurotoxin 是不是
退化成了普通 BadNet。**在 clone 官方实现证实它之前，不要调任何超参、不要看 ASR。**

## 判定「完成」的客观标准

| 级别 | 判据 | 现状 |
|---|---|---|
| L1 | `pytest tests/test_neurotoxin_mask.py -v` → 3 passed（需 TF，集群跑） | 本地 skip，未验证 |
| L2-a | `n_malicious_participations == n_rounds`（连续参与真的生效） | 未跑 |
| L2-b | `final.local_benign_asr` **随轮次单调上升**，而非原地不动 | 未跑 |
| L2-c | 同一配置换 `drift_correction=hierpfedme` 后 ASR 量级不崩（正交化真的生效） | 未跑 |

L2 跑法：
```bash
bash run_smoke.sh attack neurotoxin neurotoxin none exp001
```

> **注意 L2-b 是「趋势」不是「阈值」**。Neurotoxin 的卖点是**持久性**不是绝对 ASR，
> 所以判据是曲线形状。若只在最后一轮看一个数字，等于没验证这个方法。

## 参考实现

- 文献：Zhang et al., *Neurotoxin: Durable Backdoors in Federated Learning*, ICML 2022
- 源仓库：`github.com/jhcknzzm/Federated-Learning-Backdoor`（**clone 前先核实 URL**）
- 关键函数：`grad_mask_cv` / `apply_grad_mask`
- clone 的 commit：**待填**

## 语义 diff 表（逐格用官方代码证实/证伪，不靠印象）

| 步骤 | 官方实现（待核实） | 本仓库现实现 `client/client_neurotoxin.py` | 差异 | 怎么验证 |
|---|---|---|---|---|
| benign 梯度来源 | 干净数据上的真实梯度 | `_benign_grads`，在收到的 edge 模型上算 | 一致 | — |
| mask 选取 | 取 \|g\| **最小**的 `ratio` 比例，mask=1（**保留**） | `(np.abs(a) < thr)`，thr = 全局 top-`mask_ratio` 分位 → 保留 1−ratio | **疑似反向** | `test_mask_keeps_smallest_benign_gradient_coords` |
| 阈值粒度 | **逐层** | 全局（所有层拼平后一个阈值） | 不同 | `test_threshold_is_per_layer` |
| 投影时机 | 逐步 mask 梯度 | 对整轮累计 update 做一次 mask | 一轮内 mask 固定时**据称**等价，未验证 | `test_masked_coords_have_exactly_zero_update` |
| 上传对象 | — | `self.model.set_weights(w_proj)` 后取 weights | **PFL 下会污染个性化模型**，应只变换返回的 upload | 正交化后复查 |
| 参与频率 | 每轮在场 | `main.py` 对 neurotoxin 设 Q=1 | 一致 | L2-a |
| 放大 | 部分设定有 model-replacement 缩放 | 无 | 不同 | L2-b 趋势 |

## 已排除的可能

- ~~benign 梯度用 edge 权重差代理导致 mask 失效~~ —— 已在 commit `841d5be` 修掉，
  现在用真实干净数据梯度。ASR 仍不起来，说明问题不在这里。
- ~~随机性未播种导致不可复现~~ —— 已在 commit `d2717cd` 修掉。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 按文献语义写 3 条 L1 不变量 | 本地无 TF，skip | 待集群 L1 结果 |
