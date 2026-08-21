# current-focus：Bad-PFL

> **前置**：本方法依赖「陷阱 #1 正交化」先完成。若 `tests/test_attack_method_orthogonality.py`
> 还是失败/skip 状态，先停下来告诉我。

## 现在要回答的唯一问题

**攻击在恶意客户端**自己的**模型上到底成不成立？**

不是「ASR 为什么是 0」。现在一直在盯良性客户端的 ASR，但那是**第二级**问题。
生成器只在恶意客户端自己的模型上训练，如果连它自己都骗不动，讨论迁移到良性
个性化模型毫无意义。**先把第一级钉死。**

## 判定「完成」的客观标准（分两级，必须按顺序）

| 级别 | 判据 | 含义 | 现状 |
|---|---|---|---|
| L1 | `pytest tests/test_badpfl_trigger.py -v` → 5 passed（需 TF，集群跑） | 扰动预算 / 投毒比例 / 归一化 / 可复现 | 本地 skip |
| **L2-一级** | `final.local_malicious_asr > 0.9` | **攻击本身学会了** | 未跑 |
| L2-二级 | `final.local_benign_asr` | 迁移性是否成立 | 未跑 |

L2 跑法：
```bash
sbatch run_smoke.sh attack bad-pfl badpfl none exp001
```

### 关于「修好了」的定义 —— 开工前必须先认这一条

- **一级不过** → 是 bug，继续修。
- **一级过、二级 ≈ 0** → 这**不是 bug，是科学结论**：在本仓库的设定下生成器不可迁移。
  这时候要么按论文的做法改建模（官方靠触发器绑定**目标类的自然特征**，而不是靠
  一个学出来的生成器硬迁移），要么把它记录成 negative result 写进 notes。
  **不要通过调超参去把二级"调"上来** —— 那是在拟合噪声。

> 没有这条约定，这个会话会无限期地追 ASR。

## 参考实现

- 文献：*Bad-PFL: Exploiting Backdoor Attacks against Personalized Federated Learning*, ICLR 2025
- 源仓库：`github.com/fmy266/Bad-PFL`（**clone 前先核实 URL**）
- 关键文件：`fba.py` / `generator.py`
- clone 的 commit：**待填**

## 语义 diff 表（逐格用官方代码证实/证伪）

| 步骤 | 官方 | 本仓库现实现 `client/client_badpfl.py` | 差异 | 怎么验证 |
|---|---|---|---|---|
| 触发器 | `T(x) = x + ξ + δ` | 同左 | 一致 | — |
| ξ（破坏噪声） | FGSM，σ=4/255（[0,1] 空间） | `sigma_norm·sign(∇_x CE)` | 一致 | `test_fgsm_noise_respects_sigma_budget` |
| δ（生成器） | Autoencoder，ε=4/255，输出有界 | `generator(x)·eps_norm` | 待核实输出层是否 tanh | `test_generator_delta_respects_eps_budget` |
| 归一化换算 | — | 硬用 `attack/triggers.CIFAR10_STD` | **不同**：`data/dataset.py:121` 对 cifar100 用 `CIFAR100_STD`，差 ~8% | `test_eps_conversion_uses_dataset_matching_std` |
| 生成器训练目标 | 让加噪样本被判为 target | 同左（model 冻结，30 步） | 一致 | **L2-一级** |
| 生成器是否上传 | 本地训练、不上传聚合 | 同左 | 一致 | — |
| 触发器与目标类的关系 | 绑定目标类的**自然特征**（可迁移性的来源） | 无此机制 | **可能是关键差异** | L2-二级 |
| 训练路径 | — | 手写 eager 循环，绕开父类 `_train_step` | 丢 lr 衰减/近端项；逼得 `n_workers=1` | 正交化的 `on_batch` 钩子应替掉它 |
| 评估侧触发器 | 每个恶意端用自己的 generator | `build_eval_trigger` 统一取 `mal[0]` | 多恶意端时不对 | L2-二级 |

## 已排除的可能

- ~~随机性未播种导致不可复现~~ —— 已在 commit `d2717cd` 修掉（`_poison_batch` 改用 `self.rng`）。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 写 5 条 L1 不变量；确认 cifar100 归一化常数不一致 | `data/dataset.py:121` 用 CIFAR100_STD，`client_badpfl.py:24` 导入 CIFAR10_STD | 归一化偏差属实但幅度小（~8%），**不足以解释 ASR≈0**；主嫌疑是迁移性 + 接线 |
