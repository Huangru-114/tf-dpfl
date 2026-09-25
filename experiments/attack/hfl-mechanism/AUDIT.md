# 对齐审计 —— tf-dpfl 与官方 Bad-PFL / FedRep / 原文献

> **门槛：本表所有行关闭之前，不跑任何 P2 run。**（DECISIONS.md D-006）
> 关闭 = 状态为 `done`（已对齐并有 L1 测试）或 `deviate`（有意偏离，理由已写明、你签过字）。
> `harness/status.py` 读本表的「状态」列：只要还有一行不是 `done` / `deviate`，
> 所有 `requires: [audit]` 的运行组都显示 **blocked**。
>
> **状态取值**（反引号里的词是机器读取的）：
> `open` 未决定 · `align` 已决定对齐、待实现 · `deviate` 有意偏离（理由见「决定」列）· `done` 已对齐且有测试
> （第二节「还没查的」里，`done` = 已经查完、结论已转进对应的 A 行）

## 参考源（逐字读过的）

| 源 | 位置 | 读取方式 | 版本标识 |
|---|---|---|---|
| Bad-PFL 官方 | `github.com/fmy266/Bad-PFL`，main 分支 | `raw.githubusercontent.com`，2026-09-24 | `fba.py` etag `d65deddbd62a…` |
| 读过的文件 | `fba.py` `client.py` `server.py` `main.py` `fl_process.py` `utils.py` `generator.py` `pfl.py` | — | — |
| 重读（A2 会话，2026-09-25） | 11 个文件按 raw 重新下载（含 `resnet.py` `trigger.py` `event_emitter.py`）；`fba.py` `client.py` `main.py` 的 etag 与 A1 相同。另读了 `fl_process.py` / `server.py` / `utils.py` / `pfl.py` 的训练与划分部分 | raw | 同上 |
| 重读（A1 会话，2026-09-25） | 同上 8 个文件按 raw 重新下载逐行核对；`fba.py` etag 仍为 `d65deddbd62a…`，未变。**`client.py` 的 `PMClient` / `PMPoisonClient` 从未被 import**（`main.py:6` 只 import `BasicClient, PoisonClient`）→ 官方实际执行的只有 FedBN 单模型（F-022） | raw | `client.py` etag `69e2b095c234…`、`main.py` `93855051ac20…` |
| **没读的** | `resnet.py`(152 行) `trigger.py` `event_emitter.py` | — | 见 A17/A18 |
| FedRep 原作者 | `github.com/LittleStory233/FedRep`（README 写明为 Collins 等人 ICML'21 官方代码；`lgcollins/FedRep` 的 raw 路径 404）；读过 `models/Update.py`、`utils/options.py`、`main_fedrep.py` | raw，2026-09-25 | — |
| Bad-PFL 论文 | ICLR 2025 正文 + 附录（28 页，用户 2026-09-25 提供的 PDF；**未入库**，10.5 MB 超过单文件红线）。引用写成「论文 p.N」。A2 会话用户重新上传，用 pypdf 抽取 p.4、p.7、p.8、p.13、p.14、p.18 核对 | 用户上传 | — |
| PFLlib（旁证） | `github.com/TsingZ0/PFLlib`：`system/flcore/clients/clientrep.py`、`servers/serverrep.py`、`serverbase.py`、`main.py` | raw，2026-09-25 | — |

TF 侧行号：A01–A06、A14、A24 基于 `317fbb5`（A1 会话重核；`fedavg/main.py` 相对 `df015b2` 在前 40 行内多了 2 行、`load_config` 里多了 12 行，其余 TF 文件未变）；A07–A13、A15、A16、A22、A23、A25 中 A2 会话新写的行号基于 `8fc613b`；其余行基于 `df015b2`。

---

## 一、已找到的差异（2026-09-24 初查 A01–A16；2026-09-25 读论文后加 A22–A23；同日 A1 会话拍板 A01–A06、A14，并加 A24；同日 A2 会话拍板 A07–A13、A15、A16、A22、A23，并加 A25）

| ID | 项 | 官方（出处） | tf-dpfl（出处） | 状态 | 决定 / 理由 | 怎么验证 |
|---|---|---|---|---|---|---|
| A01 | ξ 的构造 | `fba.py:6-22` `pgd_attack`：随机起点 `uniform_(-ε,ε)`(:8) → clamp[0,1](:9) → 在**起点处**求梯度(:14-16) → 一步 `α·sign(grad)`(:17) → η clamp ±ε(:18) → clamp[0,1](:19)；`:55` 加 δ 之后**不再** clamp；ε=α=4/255。论文 Eq.6（p.6）只写 `σ·sign(∇ₓL(F(x),y))`：在 x 处、无随机起点、无 clamp —— **论文与代码不一致** | `client_badpfl.py:85-91`：在 x 处 `σ·sign(∇ₓCE(model(x),y))`，无随机起点、无任何 clamp | `align` | **D-014：对齐官方代码**（论文 Eq.6 不采纳）：U(−σ,σ) 起点 → clamp 合法像素范围 → 在起点处求梯度 → +σ·sign → 投影 ±σ → clamp；+δ 之后不 clamp。生成器训练 / 训练期投毒 / 评估三处都用它。TF 在标准化空间：[0,1] 换成逐通道 `[(0−μ)/s, (1−μ)/s]`、σ 换成 σ/s（sign 在正仿射变换下不变）。量级：官方 ξ 只有约 50% 像素满幅、均值 0.75σ，TF 现为处处 σ（F-019） | L1：① 线性 softmax 小模型（输入梯度可解析）+ 注入固定 u → 与 numpy 参考实现逐元素相等；② x+ξ 落在合法范围；③ \|ξ\| ≤ σ_norm；④ 反向锚点：内点 \|ξ\|=σ 的比例 ≈ 0.5（现实现 = 1.0 → 改前失败） |
| A02 | 评估时 ξ 用哪个模型 | `fba.py:60-64` 循环里每个恶意端都覆盖一次 `eval_func` → 留下的是 `main.py:95` `shuffle(clients)` 之后**最后一个**恶意端（≈ 开跑时随机选一个、全程固定）；`:53` 用它的 `client.local_model`，即它最后一次训练后的陈旧模型 | `main.py:164-177` → `client_badpfl.py:164-172` `eval_trigger(model,…)` 在**被评估模型**上白盒算 ξ | `align` | D-004：主 ASR 对齐官方，白盒 ξ 另记一列作上界。**D-015**：setup 时用 `default_rng([seed, …])` 从恶意端里固定选**一个**、全程不变，它的 id 与所在 edge 写进 `[设定]` 行；用它最后一次训练后的模型（与官方一样陈旧） | L1：换受害者模型，主 ASR 路径的 ξ 不变；选中者 = rng 的解析结果；`[设定]` 行能被 `collect_metrics` 解析 |
| A03 | 投毒量 | `fba.py:48` 逐样本伯努利 `rand() <= ρ`；`:49-50` 一个都没抽中就原样返回。论文 Eq.4（p.3）是 α 加权损失（期望意义） | `client_badpfl.py:146` `k = int(round(n·ρ))`，每 batch 恰好 k 个 | `align` | **D-016：对齐伯努利**，用客户端自己的 seeded rng。现实现在 batch 32 下有取整偏差：ρ=0.2 → 0.1875（−6.3%）、ρ=0.05 → +25%、ρ=0.02 → +56%（F-020） | L1：10⁴ 个 batch 的投毒比例均值落在 ρ±3σ_binom；每 batch 计数方差 > 0（反向锚点：现实现 ρ=0.2 时恒为 6）；ρ=0 → 输入逐字节不变；ρ=1 → 全部投毒 |
| A04 | 生成器训练数据 | `fba.py:36` `clean_data, clean_label = client.fetch_data()`，但 `client.py:124-125` `PoisonClient.fetch_data` 覆写成 `poison_func(...)` → 实际拿到**按 ρ 已投毒**、标签为 target 的批次；且 `:54` 的内层 δ 没 detach，梯度穿过两次生成器。论文 Eq.7（p.6）：E_{(x,y)∼D_i}，**干净** x、真实 y —— **论文与代码不一致** | `main.py:472,510-525` 动态投毒保留干净 ds；`client_badpfl.py:100` 用 `self.dataset` 干净批次 | `deviate` | **D-017：对齐论文 Eq.7（干净数据），偏离官方代码**（= 现状，A4 不改代码）。理由：论文写得明确；官方变量名 `clean_data` 表明意图是干净数据，却被 `PoisonClient.fetch_data` 覆写，疑似 bug。代价：F-013 所述「ρ=1 时 PGD 与生成器相互抵消」不会复现，大 ρ 下与官方行为不同 | L1（守住现状）：生成器训练批次中被投毒的比例 = 0、标签 = 真实标签 |
| A05 | BN 训练/推理模式 | 6 个调用点：#1 训 G 时 G 前向 → train（G **从不** `.eval()`）；#2 训 G 时 F 前向 → `eval()`（`fba.py:32`）；#3 训练期投毒 G 生成 δ → train（本 batch 32 张的统计量）；#4 训练期投毒 F 算 ξ → train（`client.py:35` 先 `.train()` 再 `fetch_data`）；#5 评估时 G 生成 δ → train（测试 batch 32，`drop_last`）；#6 评估时攻击者 F 算 ξ → **取决于客户端列表顺序**（攻击者自己被评估前停在 train，`utils.py:33` 之后是 eval） | 除 #1 `training=True`（`client_badpfl.py:109`）外全部 `training=False`（`:89,95`）；评估时整个探针（≤ `asr_max_samples`）一次喂进生成器 | `align` | **D-018**：#3 #4 #5 对齐官方（G 在投毒 / 评估时用 batch 统计，评估按 32 张分块，尾块怎么处理在 A4 定——官方是 `drop_last` 丢弃；训练期求 ξ 时 F 用训练模式）；#1 #2 已一致；**#6 偏离官方，用推理模式**，理由：官方此处是顺序依赖的偶然行为（还会在评估中改写攻击者的 running stats）。现状问题：G 按 batch 统计优化、按 moving stats 使用，Keras momentum 0.99 下首次 30 步后仍残留 74% 初值（F-021，对 ASR 的数值影响无证据） | L1：用记录器包住 model 与 generator，断言 6 个调用点的 `training` 标志与本行逐项相同；评估时生成器每次调用的 batch ≤ 32 |
| A06 | ASR 定义 | `main.py:127-134` + `utils.py:29-48`：**不过滤**，目标类样本也计入；对**全部**客户端（含 10 个恶意端）的 `local_model` 逐客户端 ASR 等权求均值，只在训练结束时评一次。论文 p.7：「ASR over triggered samples for clients' personalized models on their test sets」 | `backdoor_eval.py:43-68`、`:71-97` `_collect_eligible` 只数非目标类；良性端与恶意端分开报（`:274-283`），逐客户端等权平均 | `align` | **D-019**：四列都报（过滤 / 不过滤 × 全体 / 仅良性，均为逐客户端等权平均）；**主指标 = 过滤 × 仅良性**（= 现 `local_benign_asr`）；「不过滤 × 全体」作为与论文可比的列。理由：3-B 刻意改变各 edge 的 y_t 占比（C2 约 30%、C3 < 1%），不过滤的口径会被占比机械抬高；恶意端自身 ASR 在 ρ=1 时是常数预测 | L1：构造探针（t 个目标类 + m 个其他类）：恒预测 target → 四列都为 1；恒预测真标签 → 不过滤 = t/(t+m)、过滤 = 0；加入恶意端，验证「全体」与「仅良性」的解析均值 |
| A07 | 聚合权重 | `server.py:4-10` `agg_avg`：**不加权**平均；论文 p.8「the FedAvg aggregator, where each client's contribution is treated equally during aggregation」。官方客户端等大小（`main.py:65-66`：每端 500 / 100 张），所以等权 ≡ 样本加权 | edge：`aggregation/fedavg.py:17-27` 按参与客户端样本数加权；cloud：`server.py:179-195` → 同一函数，按 `edge.n_samples`（`edge_server_base.py:36`，**全部成员**样本数，不只是参与者）加权。seed42 下客户端 133–909 张，权重是均值的 0.30–2.02 倍（FINDINGS F-028） | `deviate` | **D-027**：保留样本加权（HierFAVG 的标准形式）。与官方的差别只来自客户端不等大小；S3 的新划分要求客户端等大小（D-027），届时两者数值上等价 | 守卫（S3 实现）：P2 划分下全体客户端 `n_samples` 相等 |
| A08 | 学习率 | `main.py:57` `SGD(lr=0.1)`，**无调度器**；论文 p.7「SGD with a learning rate of 0.1」，未提衰减 | `client_base.py:117-126` `lr0·0.992^cloud_round`；FedRep head `hier_fedrep.py:75-85` `head_lr_rep=0.005`，同样按云轮衰减。两者来自 `9d303262` / `27d7d0dc`（修 FedRep 故障，FINDINGS F-030） | `open` | **D-023**（取代 D-010）：保留调过参的 0.992 与 head 0.005，但改为**按有效轮衰减**：`0.1·0.992^t_eff`、head `0.005·0.992^t_eff`，t_eff = 截至本 edge 轮已完成的有效轮数（R_edge=1 时恰为现在的云轮号）→ flat 逐字节不变，F-003 的 flat/HFL 混淆消失。偏离官方的常数 lr。做成配置开关（默认 = 现行为）；**D-029 可行性实验通过后改 `deviate`** | L1：同一有效轮上 flat 与 R_edge ∈ {5, 40} 的 base / head lr 相同；flat 下新旧开关给出的 lr 序列逐元素相同；集群：D-029 |
| A09 | 本地训练量 | `main.py:32` + `fl_process.py:29-30`：每轮 **15 步** × batch 32（每端 500 张、`drop_last` → 恰 1 epoch）；论文 p.7「batch size of 32 for 15 steps (roughly one epoch)」、1000 轮；p.18「when the training round reaches 500, both the models and the attack methods converge」 | `training.local_epochs: 5`（完整 epoch，≈68 步） | `deviate` | **D-024**：维持 ep5。**总本地训练量已与论文相当**：论文每端 1000 × 10% × 15 步 ≈ 100 epoch（500 轮收敛 ≈ 50 epoch）；P1 每端 150–300 有效轮 × 10% × 5 epoch ≈ 75–150 epoch（FINDINGS F-029）。差别在 τ（两次聚合之间的本地步数 ≈68 vs 15）与通信轮数，不在训练总量。ep5 是 Stage B 按墙钟标定选出的（`experiments/calibration/RESULTS.md` §二） | — |
| A10 | 预处理 / 增强 | `main.py:58` 只有 `ToTensor()`：[0,1]、不标准化、不增强；论文未提 | `dataset.py:168` 逐通道标准化；`partition.py:38-40` 翻转 + 裁剪（每 epoch 是否重新采样见 A25） | `deviate` | **D-025**：主协议保留标准化 + 增强。「官方预处理」（无标准化、无增强）作**对比实验组 G7**（PLAN §4），检验用户的假设「官方设定降低了攻击难度」。前提：`client_badpfl.py:53-56` 的 ε/σ 换算与 `attack/triggers.py` 必须跟随标准化开关，否则 ξ 放大 3.8–4.1×（FINDINGS F-027） | L1（G7 的前提）：关标准化时模型输入 == `x/255`、`_atk_eps_norm == 4/255`（反向锚点：现在 ≈ 0.0635） |
| A11 | 数据划分 | `main.py:60-84` + `utils.py:51-85`：每客户端类先验 Dir(0.5)，train / test **各自**按同一先验抽，每客户端**等大小**（500 / 100）；numpy 未播种（F-024） | 合并 60k（`main.py:701`）→ 逐类 Dir(0.5·1₁₀₀)（`partition.py:147-183`）、客户端**不等大**（seed42：133–909）→ 客户端内切 25% test | `deviate` | **D-027**：不照抄官方划分 —— 标签偏斜同量级（5 个 seed：客户端平均标签熵 1.75–1.77 vs 1.65–1.68 nats；论文 p.14 的 60.88 分辨不出两种方案，F-031），为对齐而改没有收益。**真正的问题是大小不等**：名义 10% 的恶意端，在 P1 各布点下的数据占比为 0.092–0.142（F-028），而布点正是自变量 → **S3 的新划分硬要求客户端等大小** | S3 的 L1：每客户端样本数相等；恶意端数据占比 = n_malicious / n_clients |
| A12 | FedRep 本地训练 | 上游代码只有 FedBN（`pfl.py:3-24`）。**论文 p.13 附录 A**：「For the PFL methods, we use the same training configuration as that of the local models to train personalized models」→ head 与 body **各** SGD lr=0.1、batch 32、15 步（p.7） | `client/hier_fedrep.py`：先 head `plocal_epochs=1`（lr `head_lr_rep=0.005`）后 body `local_epochs=5`（lr 0.1·0.992^r）；head = 最后一层 Dense，BN 进聚合 | `deviate` | **D-024（取代 D-012）**：维持 P1 现状 —— head `plocal_epochs: 1`、`head_lr_rep: 0.005`，body `local_epochs: 5`，lr 按 D-023。理由：总本地训练量已与论文相当（F-029）；head 的独立低 lr 是 `27d7d0dc` 针对 head 过拟合调出来的（F-030）。偏离论文附录 A「the same training configuration」。**原决定**：D-012：对齐 Bad-PFL 论文：head、body 各 15 步，lr 同为 0.1，batch 32，去掉单独的 head lr。原作者（4:1、momentum 0.5、wd 1e-4、lr 0.01）与 PFLlib（1:1、lr 0.005、batch 10）只作旁证，见 FINDINGS F-016。「head = 哪几层、BN 算不算 body」论文没写，在 A3 定 | —（维持现状，无需改代码） |
| A13 | 数值精度 | `client.py:59-73` `autocast` 只包训练的前向 + loss；每客户端一个 `GradScaler`（初始 scale 2¹⁶，溢出即跳过该步）；PGD 与生成器训练不在 autocast 内（fp32） | fp32 | `deviate` | **D-028**：保持 fp32。TF 的 `mixed_float16` 是全局 Keras policy，会同时改到末层 softmax（本仓库 `from_logits=False`）、BN、生成器与 PGD 的输入梯度；照官方的作用域复刻要逐模型设 policy，收益没有证据。代价：不复现 GradScaler 可能跳过的前几步（机制推断，无数值证据） | grep 守卫：`fedavg/` 中无 `mixed_float16` |
| A14 | 生成器结构 | `generator.py:6-40`：Conv / ConvT 均为 k4 s2 **p1**，全部带 bias；末层 ConvT → Tanh，**无 BN**；torch 默认 BN eps 1e-5、kaiming_uniform(a=√5) 初始化；输入是 [0,1] 图。论文 p.14 表 5 末层写「ConvTranspose2d + BatchNorm2d + Tanh」—— **论文与代码不一致** | `models/autoencoder.py:31-43`：`padding="same"`、前 7 层 `use_bias=False`、末层带 bias 无 BN、Keras BN eps 1e-3、glorot_uniform；输入是标准化图 | `align` | **D-020：以官方代码为准**：保留 `same`（偶数输入、k4 s2 时 SAME 的 pad_total=(out−1)·2+4−in=2 → 前后各 1，与 p1 等价，用 L1 证明）；加 bias；BN eps 1e-5；init 用 `VarianceScaling(1/3, "fan_in", "uniform")`（= torch 默认；ConvT 的 fan_in 两边都按输出通道数算）；生成器输入先反标准化到 [0,1]（零填充的语义才一致）；末层不加 BN（表 5 不采纳） | L1：同权重下 SAME ≡ `ZeroPadding2D(1)`+VALID、ConvT SAME ≡ VALID+`Cropping2D(1)`，逐元素相等；输出 32×32×3 ∈ [−1,1]；参数量 = torch 解析值；BN eps = 1e-5；生成器实际输入 ∈ [0,1] |
| A15 | 确定性 | `utils.py:8-13` 只播 torch + `cudnn.deterministic=True`；**numpy 与 python `random` 没播** → 划分（`main.py:67`、`utils.py:65`）与客户端顺序（`main.py:95`）每次运行都不同，官方自身不可复现（FINDINGS F-024） | 三处随机源都播了（`main.py:325-339`）；GPU 算子未设确定性；tf.data 并行 `map` 里有带状态的随机增强（`partition.py:40`）。实测同 seed 重跑第 1 轮即分叉（F-002） | `align` | **D-028**：`tf.config.experimental.enable_op_determinism()`（为内部效度，比官方更严）。F-002 的分叉来自 GPU kernel 还是增强，**没有证据**。若容器 TF 版本有算子不支持（抛 `UnimplementedError`）或慢得不可接受，回审计重议 | L1：AST —— `run_experiment` 在建模型之前调用它；集群：加一条 `[Checksum]` 自描述行（每轮全局权重 sha256 前 12 位），同 seed 两次 run 前 5 轮哈希相同 |
| A16 | 死配置 | 无 label smoothing | `training.label_smoothing: 0.1` 在配置里，代码不读（A2 会话 grep 复核：`fedavg/` 中 0 处） | `align` | **D-028**：从 P2 配置删除；冻结的旧配置不动（D-003） | L1：P2 配置不含 `label_smoothing`；`fedavg/` 中读取它的地方 = 0（将来谁实现了它，这条先红，旧配置里的 0.1 不会悄悄生效） |
| A22 | 目标标签 | 论文 p.7「The target label y_t is randomly generated」；官方代码 `main.py:36` 默认 `--ba_target_label 0` | 所有配置 `backdoor.target_label: 0` | `align` | **D-028**（用户拍板）：固定 0，= 官方代码、偏离论文。理由：3-B 的 C1–C4 围绕 y_t 的占比设计；P1 的噪声估计都在 y_t=0；与 Bad-PFL 库可同框。代价：结论只针对 class 0（airplane） | L1：P2 基配置 `target_label == 0` |
| A23 | 训练轮数 | 论文 p.7：1000 轮；p.18：约 500 轮时模型与攻击都已收敛；官方代码 `main.py:23` 默认 `--total_round 300` | 标定后：有效轮 floor 150 / cap 300 | `deviate` | **D-028**：保留 floor 150 / cap 300。按 A09 的换算，floor 150 ≈ 每端 75 epoch，已超过论文收敛点（500 轮 ≈ 50 epoch）。A08 改成按有效轮衰减后 HFL 格子的 lr 变低，收敛是否受影响由 D-029 的 `stop_reason` 与 pm_acc 判据复核 | 集群：D-029 |
| A24 | FedRep 哪个阶段投毒 | 官方实际执行的只有 FedBN 单模型（`main.py:6,89-94` 只用 `BasicClient` / `PoisonClient`），**没有对应物**；`client.py:86-137` 的 `PMClient` / `PMPoisonClient`（两步都投毒）是死代码，从未被 import（FINDINGS F-022）。论文 Alg.1（p.5）：第 11 行恶意端按 Eq.4 训 F(·;θg)；第 15 行「Train the personalized model with the predefined PFL method」在 `end` 之后，良性端与恶意端共用，未提投毒 | `client/hier_fedrep.py:139-150`：head 阶段（`:141`）与 body 阶段（`:148`）都经过 `on_batch` → 都投毒（`client_badpfl.py:25-27` 的「决策 B」） | `align` | **D-021：按论文 Alg.1 字面，只在 body 阶段投毒**，head 阶段用干净数据。只改 `hier_fedrep`（其他方法的决策 B 不在实验 3 范围）。影响 3.2：head 不再直接看到投毒样本（FINDINGS N-003）。训练顺序（Alg.1 先 local 后 personalized；TF 先 head 后 body）在 A3 定 | L1：FedRep 恶意端一次 `local_train` 中，head 阶段 0 个 batch 被投毒，body 阶段每个 batch 都经过投毒分支 |
| A25 | 每 epoch 的取数 | `main.py:78-80` `SubsetRandomSampler` + `drop_last=True`，`client.py:48-57` 持久迭代器：**每 epoch 重洗**，丢掉的尾批每 epoch 不同 | `hier_fedrep.py:65` 构造时 `list(self.dataset)` 缓存一次 → 增强冻结（F-014）、**batch 组成冻结**，`:173-177` 只打乱 batch 顺序并永远丢同一个尾批 → seed42 下 1453/45037 = 3.2% 的训练样本从未参与训练，单端最高 19.5%（F-025）。FedAvg 客户端（`client_fedavg.py:44`）每 epoch 重读 tf.data（重洗、重增强、含尾批）→ G4 混入数据流差异（F-026）；生成器（`client_badpfl.py:100`）另读一遍 tf.data、循环取批 | `align` | **D-026**：修复 —— 统一为官方语义：每 epoch 重洗 + 重增强、`drop_last` 且尾批每 epoch 轮换；FedAvg、FedRep、生成器走同一个取数函数。做成配置开关（默认 = 现行为），并入 D-029 可行性实验 | L1：同一客户端连续两个 epoch 的 batch 组成不同、被丢的样本不同（反向锚点：现缓存下相同）；每 epoch 覆盖 n − (n mod 32) 个不重复样本；AST：三处取数调用同一函数；集群：D-029 |

## 二、还没查的（每一项都是 `open`，查完才能关）

| ID | 项 | 需要什么 / 查到了什么 | 状态 | 去向 |
|---|---|---|---|---|
| A17 | ResNet-10 逐层对比（`resnet.py` 152 行 vs `models/resnet.py`） | 直接读 raw | `open` | — |
| A18 | `trigger.py`、`event_emitter.py` 是否影响攻击/评估流程 | 直接读 raw | `open` | — |
| A19 | FedRep 参考实现 | 已定：**以 Bad-PFL 论文附录 A 为准**（D-012）；原作者与 PFLlib 已读，差异记在 F-016 | `done` | 参考源问题已结，实现差异转到 A12 |
| A20 | 论文正文超参 | A2 会话按原文复核（p.7 §4.1、p.8、p.13–14 附录 A、p.18）：100 客户端、**1000 轮**（p.18：约 500 轮收敛）、10 个恶意端、每轮 10%、Dirichlet 0.5、SGD lr 0.1、batch 32、15 步、投毒率 0.2、ε=σ=4/255、生成器 Adam 0.01 × 30 步、**目标标签随机生成**、FedAvg 聚合时各客户端等权（p.8）；MultiKrum f=1 选 5 个。p.7 注明 FL 设定「Following existing studies (Zhuang et al., 2024)」—— 训练超参是继承来的，不是攻击的一部分（D-022）。去向：100 客户端 / 10 恶意 / 每轮 10% / batch 32 / 投毒率数值 / 生成器 Adam 0.01×30 步 = 一致；Dirichlet → A11；1000 轮 → A09、A23；lr → A08；15 步 → A09；等权 → A07；目标标签 → A22；ε=σ（配置 0.01569，与 4/255 差 0.02%）→ A01、A10；投毒取整 → A03；ResNet-10 → A17；MultiKrum → 不在实验 3 范围 | `done` | 已逐项分派到各 A 行（A2 会话，F-031） |
| A21 | HFL 的形式化（Liu et al. HierFAVG：edge/cloud 聚合权重、κ₁/κ₂） | 文献 | `open` | — |

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

> **术语**：A1–A4 是**会话名**（PLAN §5），A01–A25 是本表的**行号**，两者无关。A4 = 按拍板改代码的实现会话（与行 A04「生成器训练数据」撞名，没有改名，因为 `harness/registry.py` 的报错文案和两条测试引用了「A4」）。

1. **A1**（攻击：A01–A06、A14、A24 —— 2026-09-25 已拍板）→ **A2**（训练协议：A07–A11、A13、A15、A16、A20、A22、A23、A25 —— 2026-09-25 已拍板，原则见 DECISIONS D-022；A08 待 D-029 可行性实验）→ **A3**（FedRep / ResNet / HFL：A12 的实现细节、A17、A18、A21；另加 **FedRep 训练顺序**：论文 Alg.1 先 local 后 personalized，TF 先 head 后 body，A24 只定了「哪个阶段投毒」）→ D01–D06 签字。
2. 每个审计会话按 CLAUDE.md 的格式出语义 diff 表：论文公式 | 官方实现 | 本仓库实现 | 差异 | 怎么验证。你**逐行**拍板，结论写回本表的「状态」和「决定」两列，并在 DECISIONS.md 记一条。
3. **A4** 按拍板改代码，每个 `align` 配 L1 测试 → 状态改 `done`；跑 D-029 可行性实验（A08、A25），通过后 A08 改 `deviate`；口径版本升 P2（`fedavg/utils/provenance.py`）；2 个 smoke run 复核标定。
4. 本表全部关闭 → `status` 解除 blocked → 开跑 P2。
