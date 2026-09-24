# 对齐审计 —— tf-dpfl 与官方 Bad-PFL / FedRep / 原文献

> **门槛：本表所有行关闭之前，不跑任何 P2 run。**（DECISIONS.md D-006）
> 关闭 = 状态为 `done`（已对齐并有 L1 测试）或 `deviate`（有意偏离，理由已写明、你签过字）。
> `harness/status.py` 读本表的「状态」列：只要还有一行不是 `done` / `deviate`，
> 所有 `requires: [audit]` 的运行组都显示 **blocked**。
>
> **状态取值**（反引号里的词是机器读取的）：
> `open` 未决定 · `align` 已决定对齐、待实现 · `deviate` 有意偏离（理由见「决定」列）· `done` 已对齐且有测试

## 参考源（逐字读过的）

| 源 | 位置 | 读取方式 | 版本标识 |
|---|---|---|---|
| Bad-PFL 官方 | `github.com/fmy266/Bad-PFL`，main 分支 | `raw.githubusercontent.com`，2026-09-24 | `fba.py` etag `d65deddbd62a…` |
| 读过的文件 | `fba.py` `client.py` `server.py` `main.py` `fl_process.py` `utils.py` `generator.py` `pfl.py` | — | — |
| **没读的** | `resnet.py`(152 行) `trigger.py` `event_emitter.py` | — | 见 A17/A18 |
| FedRep 官方 | `lgcollins/FedRep`：raw 路径 master/main 都返回 **404** | — | 见 A19 |
| Bad-PFL 论文 | ICLR 2025 —— 本环境访问不了 arxiv / openreview | — | 见 A20 |

TF 侧行号基于 `df015b2`。

---

## 一、已找到的差异（2026-09-24 初查，A01–A16）

| ID | 项 | 官方（出处） | tf-dpfl（出处） | 状态 | 决定 / 理由 | 怎么验证 |
|---|---|---|---|---|---|---|
| A01 | ξ 的构造 | `fba.py:6-22` `pgd_attack`：随机起点 `uniform_(-ε,ε)`(:8) → clamp[0,1](:9) → 一步 `α·sign(grad)`(:17) → η clamp ±ε(:18) → clamp[0,1](:19) | `client_badpfl.py:85-91`：`σ·sign(∇ₓCE(model(x),y))`，无随机起点、无任何 clamp | `open` | 建议对齐。TF 在标准化空间，[0,1] 要换成逐通道 `[(0−μ)/σ, (1−μ)/σ]` | L1：固定输入下与手算的 numpy 参考实现逐元素相等；输出落在合法像素范围 |
| A02 | 评估时 ξ 用哪个模型 | `fba.py:64` `eval_func=partial(..., poison_ratio=1., client=client)`；`:53` 用 `client.local_model` → **攻击者**的模型（循环覆盖，留下的是**最后一个**恶意端） | `main.py:170-172` → `client_badpfl.py:164-172` `eval_trigger(model,…)` 在**被评估模型**上白盒算 ξ | `align` | D-004：主 ASR 对齐官方；白盒 ξ 另记一列作上界。「用哪个恶意端」在 A1 定 | L1：主 ASR 路径里 ξ 与受害者模型无关（换受害者模型，ξ 不变） |
| A03 | 投毒量 | `fba.py:48` 逐样本伯努利 `rand() <= ρ` | `client_badpfl.py:146` `k = int(round(n·ρ))`，每 batch 恰好 k 个 | `open` | 建议对齐（用客户端自己的 seeded rng） | L1：大样本下投毒比例 → ρ；ρ=0 时 0 个 |
| A04 | 生成器训练数据 | `fba.py:36` `client.fetch_data()`；`client.py:124-125` `PoisonClient.fetch_data` 返回 `poison_func(...)` → **按 ρ 已投毒**、标签为 target | `main.py:495-512` 动态投毒保留干净 ds；`client_badpfl.py:100` 用 `self.dataset` 干净批次 | `open` | 建议对齐。这同时复现 ρ=1 时 PGD 推离 target、生成器推向 target 的抵消（FINDINGS F-013） | L1：生成器训练批次中被投毒的比例 = ρ |
| A05 | BN 训练/推理模式 | 生成器**从不** `.eval()`（一直用 batch 统计）；训练期投毒 PGD 在 `local_update` 的 `train()` 模式下（`client.py:35`，`fba.py:53`）；生成器训练期间模型在 `eval()`（`fba.py:32`） | 生成器与 FGSM 都 `training=False`（`client_badpfl.py:89,95`） | `open` | 逐项对齐或写理由 | L1：各调用点的 training 标志与官方一致 |
| A06 | ASR 定义 | `main.py:131` + `utils.py:29-48`：**不过滤**，目标类样本也计入 | `backdoor_eval.py:54`、`:71-97` `_collect_eligible` 只数非目标类 | `open` | 建议两列都报，主指标待定（METRICS.md 记过两者约差 10 个点） | L1：构造含目标类的探针，两个定义各得解析值 |
| A07 | 聚合权重 | `server.py:4-10` `agg_avg`：**不加权**平均 | edge：`aggregation/fedavg.py:17-27` 按参与客户端样本数加权；cloud：`server.py:179-195` → 同一函数，按 `edge.n_samples`（**全部成员**样本数，不只是参与者）加权 | `open` | 待定。flat 下与官方不同；HFL 下 cloud 权重的定义本身要决定 | L1：flat 配置下与不加权平均逐元素相等（若对齐） |
| A08 | 学习率 | `main.py:57` `SGD(lr=0.1)`，**无调度器** | `client_base.py:117-126` `lr0·0.992^cloud_round` | `open` | **重新决定**（D-005 已作废）。建议常数（对齐官方，flat/HFL 混淆一并消失） | L1：flat 与 R_edge=5 在同一有效轮上 lr 相同 |
| A09 | 本地训练量 | `main.py:32` + `fl_process.py:29-30`：每轮 **15 步** × batch 32 | `training.local_epochs: 5`（完整 epoch） | `open` | 待定。标定：ep1 下 ASR 停在 0.66、ep5 下趋向 1.0 —— **影响最大的一条** | L1：每轮 SGD 步数 = 配置值 |
| A10 | 预处理 / 增强 | `main.py:58` 只有 `ToTensor()`：不标准化、不增强 | `dataset.py:168` 标准化；`partition.py:38-40` 翻转+裁剪；但 `hier_fedrep.py:64` `self._batch_list = list(self.dataset)` 只缓存一次 → **增强只采一次就冻结** | `open` | 建议对齐（去掉增强），至少不要半开半关 | L1：两个 epoch 的同一 batch 是否相同与配置一致 |
| A11 | 数据划分 | `main.py:60-83` + `utils.py:51-85`：train/test **各自** Dirichlet(0.5)、共享类先验、每客户端**等大小** | 合并 60k（`main.py:687`）→ 逐类 Dirichlet、客户端**不等大** → 客户端内切 25% test | `open` | HFL 需要新划分（S3）→ 部分 `deviate`；客户端等大小可对齐 | L1：各客户端样本数 |
| A12 | PFL 方法 | 上游 `pfl.py:3-24` **只有 FedBN**（BN 层不聚合不下发），没有 FedRep | FedRep：head = 最后一层 Dense，BN 进聚合，`head_lr_rep=0.005`，`plocal_epochs=1` / `local_epochs=5` | `open` | 先找参考实现（A19）。`hier_fedrep.py:21-22` docstring 写「head 多步、body 少步」，与配置 1<5 **相反** | 待 A19 |
| A13 | 数值精度 | `client.py:59` `autocast` 混合精度 | fp32 | `open` | 建议 `deviate`（数值更稳），写理由 | — |
| A14 | 生成器结构 | `generator.py:6-40`：Conv k4 s2 **p1** + BN（带 bias） | `models/autoencoder.py:31-43`：`padding="same"`、`use_bias=False` | `open` | 陷阱 #6：padding 在 stride>1 时不一定等价 | L1：同权重下与 torch 参考输出数值等价（需要一次集群或离线核对） |
| A15 | 确定性 | `utils.py:8-13` `cudnn.deterministic=True` | 实测同 seed 重跑第 1 轮即分叉（FINDINGS F-002） | `open` | 试 `tf.config.experimental.enable_op_determinism()` | 集群：同 seed 两次 run 逐轮相等 |
| A16 | 死配置 | 无 label smoothing | `training.label_smoothing: 0.1` 在配置里，代码不读 | `open` | 建议从配置删除，避免误导 | grep 守卫 |

## 二、还没查的（每一项都是 `open`，查完才能关）

| ID | 项 | 需要什么 | 状态 |
|---|---|---|---|
| A17 | ResNet-10 逐层对比（`resnet.py` 152 行 vs `models/resnet.py`） | 直接读 raw | `open` |
| A18 | `trigger.py`、`event_emitter.py` 是否影响攻击/评估流程 | 直接读 raw | `open` |
| A19 | FedRep 参考实现：你 fork 里加的 `--pfl fedrep` 分支，或 Collins 官方代码的正确地址 | **需要你提供** | `open` |
| A20 | 论文正文超参（轮数、本地步数、FedRep 设置、ASR 定义） | **需要你提供 PDF** | `open` |
| A21 | HFL 的形式化（Liu et al. HierFAVG：edge/cloud 聚合权重、κ₁/κ₂） | 文献 | `open` |

## 三、有意偏离登记（HFL 特有，官方没有对应物）

这些都是设计选择，**要你签字**才算关闭。签字前一律 `open`。

| ID | 项 | 现状（出处） | 理由（草案） | 代价 | 状态 |
|---|---|---|---|---|---|
| D01 | edge 层 | `server/server.py:208-230`，edge 顺序跑 `edge_rounds` 轮 | 研究对象本身 | — | `open` |
| D02 | 参与配额 | `server/participation.py:44-52`：全局预算 `B=max(n_edges, round(N·frac))`，整数配额按轮轮转，每 edge ≥1 | 不同拓扑下：每有效轮训练的客户端数相同；恶意端期望参与次数相同（collocated 4edge：2.5×10/25=1.0；distributed 10edge：1×1/10×10=1.0）；不空转。n_edges=1 时退化成官方「100 抽 10」 | **10 edge 时每 edge 每轮只有 1 个客户端** → edge 内聚合退化成复制；n_edges>B 时 `max` 悄悄抬高预算 | `open` |
| D03 | 按 edge 布点 | `malicious_placement: by_edge` | 3-B/3-C 需要控制「哪个 edge 有攻击者」 | — | `open` |
| D04 | 探针 = 客户端留出分片 | 陷阱 #11；`backdoor_eval.py:188-343` | 与 pm_acc 同一 population；PFLlib 口径 | 与官方「官方 test 独立划分」不同（A11） | `open` |
| D05 | 逐 edge 指标 | `per_edge_*` | 3-B/3-C 的判据是逐 edge 的 | — | `open` |
| D06 | 自适应停轮 | `server/stopping.py` | 跨拓扑在相同完成状态下读数 | 各格长度不同；判据参数是在 P1 口径下标定的，A4 后要复核 | `open` |

## 四、流程

1. **A1**（攻击：A01–A06、A14）→ **A2**（训练协议：A07–A11、A13、A15、A16）→ **A3**（FedRep / ResNet / 论文：A12、A17–A21）→ D01–D06 签字。
2. 每个审计会话按 CLAUDE.md 的格式出语义 diff 表：论文公式 | 官方实现 | 本仓库实现 | 差异 | 怎么验证。你**逐行**拍板，结论写回本表的「状态」和「决定」两列，并在 DECISIONS.md 记一条。
3. **A4** 按拍板改代码，每个 `align` 配 L1 测试 → 状态改 `done`；口径版本升 P2（`fedavg/utils/provenance.py`）；2 个 smoke run 复核标定。
4. 本表全部关闭 → `status` 解除 blocked → 开跑 P2。
