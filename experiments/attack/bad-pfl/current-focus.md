# current-focus：Bad-PFL

## 现在要回答的唯一问题

生成器 G 只在恶意客户端**自己的**模型上训练，评估却要迁移到良性客户端的**个性化**模型
（`main.py:build_eval_trigger` 取 `mal[0]` 的 generator）。
这个迁移在本仓库的设定下到底成不成立？——若不成立，ASR≈0 是**预期行为**而不是 bug，
需要改的是攻击的建模方式（官方靠触发器绑定目标类的自然特征），不是调参。

## 判定「完成」的客观标准

- **L1**：`pytest tests/test_badpfl_trigger.py -v` → 5 passed（集群上跑）
- **L2**：分两级读 `expNNN.metrics.json`
  1. `final.local_malicious_asr` > 0.9 —— 攻击在**恶意客户端自己的模型**上先得成立。
     这一条不过，谈迁移毫无意义。
  2. `final.local_benign_asr` —— 迁移是否成立的真实读数。

> 先看第 1 条。现在的失败很可能停在这一步，而我们一直在盯第 2 条。

## 参考实现

- 文献：*Bad-PFL: Exploiting Backdoor Attacks against Personalized Federated Learning*, ICLR 2025
- 源仓库：`github.com/fmy266/Bad-PFL`（**clone 前核实**）
- 关键文件：`fba.py` / `generator.py`
- clone 的 commit：**待填**

## 语义 diff 表（待用官方实现逐格证实/证伪）

| 步骤 | 官方 | 本仓库现实现 (`client/client_badpfl.py`) | 差异 | 怎么验证 |
|---|---|---|---|---|
| 触发器 | `T(x) = x + ξ + δ` | 同左 | 一致 | — |
| ξ（破坏噪声） | FGSM，σ=4/255（[0,1] 空间） | `sigma_norm·sign(∇_x CE)` | 一致；但换算用的 STD 见下 | `test_fgsm_noise_respects_sigma_budget` |
| δ（生成器） | Autoencoder，ε=4/255，输出有界 | `generator(x)·eps_norm` | 待核实输出层是否 tanh | `test_generator_delta_respects_eps_budget` |
| 归一化换算 | — | 硬用 `attack/triggers.CIFAR10_STD` | **不同**：`data/dataset.py:121` 对 cifar100 用 `CIFAR100_STD`，两者差 ~8% | `test_eps_conversion_uses_dataset_matching_std` |
| 生成器训练目标 | 让加噪样本被判为 target | 同左（model 冻结，30 步） | 一致 | L2 第 1 条判据 |
| 生成器是否上传 | 本地训练、不上传聚合 | 同左 | 一致 | — |
| 投毒样本选择 | — | 全局 `np.random.shuffle` | **不可复现**（陷阱 #2） | `test_poison_selection_is_reproducible` |
| 训练路径 | — | 手写 eager 循环，绕开父类 `_train_step` | **不同**：丢失 lr 衰减/近端项；且逼得 `n_workers=1` | `tests/test_attack_method_orthogonality.py` |

## 已知的接线阻塞（陷阱 #1，需先修）

同 Neurotoxin：`BadPFLClient` 继承 `FedAvgClient` 并替换掉 PFL 方法类。
**在正交化修好之前，L2 的 ASR 数字不可解释。**

## 已排除的可能

- （待填）

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 按文献写 5 条 L1 不变量；确认 cifar100 归一化常数不一致（代码阅读） | `data/dataset.py:121` 用 CIFAR100_STD，`client_badpfl.py:24` 导入 CIFAR10_STD | 归一化偏差属实但幅度小（~8%），不足以解释 ASR≈0；主嫌疑仍是迁移性 + 接线 |
