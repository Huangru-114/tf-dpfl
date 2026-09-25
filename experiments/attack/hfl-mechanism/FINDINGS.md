# 结论台账 —— Experiment 3（改版）

> 每条结论 = 声明 + 证据（能重跑的命令或 文件:行）+ 状态。
> 状态：`confirmed`（证据是代码或数据本身）· `provisional`（只有 P1 / 单 seed，当假设用）· `retracted`（撤回，写原因）。
> **数字只从脚本产物或复核命令里来，不凭记忆写**（陷阱 #14）。
> P2 数据回来后，结论性的条目由 `harness/verdicts.py` 的输出支撑，并注明 run 集合与 n_seeds。

## 2026-09-24（S1，评审时核实）

### F-001 `confirmed` —— 旧方案的 6 个 `def_*` 格没有开防御

- `experiments/attack/hfl-propagation/exp3_cell.sbatch:46` 写死 `--defense none`，而 `fedavg/main.py:225-230` 让 CLI 覆盖 yaml 的 `defense.name`（yaml 里写的是 `median` / `multi_krum`）。
- 6 个 `results/def_*_seed42.metrics.json` 全部是 `run.defense="none"`、`admitted=[]`。
- 修复：D-007 推迟；陷阱 #19。S1 起 `[Provenance]` 行会记录 `cli_overrides`，`harness/status.py` 会把这类格子标成 `mismatch`。

### F-002 `confirmed` —— 同配置、同 seed 重跑，第 1 轮就分叉

F-001 的 6 个格子等于 3 组同配置、同 seed 的重复（恶意端 id 一致）：

| 组 | 第 1 轮 gm_acc | 末 10 点 benign ASR |
|---|---|---|
| flat_baseline | 0.1144 / 0.1116 / 0.1122 | 0.755 / 0.696 / 0.685 |
| 2edge_distributed | 0.2973 / 0.3285 / 0.3331 | 0.738 / 0.723 / 0.756 |
| 4edge_distributed | 0.2167 / 0.2201 / 0.2115 | 0.972 / 0.963 / 0.910 |

按**实际因素**分组后（`harness/runs_table.py`），还多出两组之前没人注意到的同配置同 seed 重复：

- `3c_R5` 与 `2edge_distributed` 的因素完全相同 → 2edge_distributed 实际有 **4** 份：末 10 点 benign ASR 为 0.738 / 0.724 / 0.723 / 0.756；T_0.5 为 46.8 / 46.9 / 43.6 / 48.2 有效轮。
- `stoptest_10edge` 是 `10edge_distributed` 的第 2 份：末 10 点 benign ASR 为 0.729 / 0.723。

复核：`python3 harness/runs_table.py experiments/attack/hfl-propagation/results --out <dir> --legacy-protocol P1` → `runs.csv` 的 `replicate` / `n_replicates` 列。

影响：按 seed 配对只锁得住划分和布点，锁不住训练轨迹。AUDIT A15。

### F-003 `confirmed` —— 学习率按云轮衰减，flat 与 HFL 的 LR 日程不同

- `fedavg/client/client_base.py:117-126`：`lr = 0.1·0.992^cloud_round`。
- 同样 200 有效轮，末端 lr：flat（R=1）0.020 · R=5 0.073 · R=40 0.096。
- 官方是常数 lr（AUDIT A08）。

### F-004 `confirmed` —— 评估时 ξ 在受害者模型上白盒计算，官方在攻击者模型上

- TF：`main.py:170-172` → `client_badpfl.py:164-172`。
- 官方：`fba.py:53,64`。
- 已决定对齐（D-004）。
- `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」这句与**两边都不符**：官方是循环里最后一个恶意端，TF 是受害者模型。该文件与 Bad-PFL 库双份同步（`test_metrics_doc.py`），留给两库一起改。

### F-005 `confirmed` —— 旧方案的 edge 与数据无关

- 25 个 yaml 全是 `partition: "noniid"` + `edge_assignment: "block"`。
- `inter_edge` / `intra_edge` 只在 `partition: hierarchical` 分支被读（`main.py:373-385`），所以在这些配置下是死配置。

### F-006 `confirmed` —— 旧 rho0 格不是 Bad-PFL 的下限

- 没有恶意端时，`build_eval_trigger` 回退到静态 BadNet 触发器（`main.py:171`），0.037 测的是 BadNet。
- 真正的下限用 ρ=0 影子攻击者（PLAN §1.1）。

### F-007 `confirmed` —— FedRep 下 edge / global 模型的 head 是初始化 head；PM 是陈旧的

- `server/hier_fedrep.py:59-65` 只把 backbone 索引写回 edge 模型，head 保持下发值；cloud 平均的是各 edge 未变的 head。
- `client.model` 只在被抽中时更新（`client/hier_fedrep.py:91-109`），10% 参与率下大多数良性端是旧 body。
- 影响：edge_asr / global_asr 的含义要重新定义；3-C 主曲线用 fresh-PM ASR。

### F-008 `provisional` —— P1 中 4edge_distributed / 4edge_mixed 的 ASR 明显高于 2edge 和 10edge

- 末 10 点 benign ASR 约 0.97 / 0.94，高于 2edge 的 0.74 和 10edge 的 0.73。
- 在 3 次同 seed 重复里都出现（0.972 / 0.963 / 0.910），所以不是训练噪声。
- 候选解释是 seed42 的划分 × 布点，**没有证据**。只作假设，由 P2 检验。
- 2026-09-25 补充（F-028）：4edge 三格的恶意端**数据占比**（0.127–0.142）高于 2edge / 10edge / flat（0.092–0.114），是一个候选协变量；但 4edge_collocated 占比同为 0.128，benign ASR 却只有 0.715 → 1 个 seed 分不开，仍是假设。

### F-009 `provisional` —— P1 中 flat 并不比 HFL 快

- T_0.5(benign)：flat 57 / 2edge_dist 47 / 4edge_dist 23 有效轮（`exp3_analysis.csv`）。
- 单 seed，且有 F-003 的 LR 混淆 → 只作假设；正是 3-A 要回答的问题。

### F-010 `confirmed` —— 旧分析口径的两个问题

- `harness/analyze_exp3.py` 的终值取**单个末轮**，而标定规定用末 10 点均值（`experiments/calibration/RESULTS.md` §4.2）。
- HHI 回归把 ρ / def / 3C / stoptest 等格子也混了进去（n=25，R²≈0.02）。
- S1 的 `runs_table.py` 改用末 10 点均值，并按实际因素分组。

### F-011 `confirmed` —— 旧 metrics.json 没有溯源信息

- 没有 git commit、config hash、run_id，也没有时间戳；`run_name` 在所有 exp3 格子里都相同。
- 一个格子的身份只靠文件名。S1 加 `[Provenance]` 行。

### F-012 `confirmed` —— FedRep 的 head / body 训练量与 docstring 相反

- `client/hier_fedrep.py:21-22` 写「plocal_epochs > local_epochs」（head 多步、body 少步，FedRep 原则），而全部 exp3 配置是 `plocal_epochs: 1` / `local_epochs: 5`。
- 影响未知 → AUDIT A12。

### F-013 `confirmed`（机制）/ 无数值证据（效果）—— 官方 ρ=1 时 PGD 与生成器相互抵消，TF 版不会

- 官方生成器用 `PoisonClient.fetch_data()` 的数据训练（`fba.py:36`，`client.py:124-125`）。这些数据已按 ρ 投毒、标签为 target，于是 PGD 推离 target、生成器损失推向 target。
- TF 版用干净数据训练生成器（`client_badpfl.py:100`），所以不会抵消。
- 这证实了 METRICS.md 的推断在官方代码路径上成立，但**抵消在数值上有多大仍无证据**。AUDIT A04。

### F-014 `confirmed` —— 数据增强只采样一次

- `client/hier_fedrep.py:64` 在构造时 `list(self.dataset)`，把增强后的 batch 缓存下来，之后每个 epoch 重复使用。
- 官方不做增强。AUDIT A10。

### F-015 `confirmed` —— `training.label_smoothing: 0.1` 是死配置

- `fedavg/client` / `server` / `models` 里没有任何地方读它。AUDIT A16。

### 设计备注（不是缺陷，但解读结果时必须带上）

- **N-001**：参与配额下，10 个 edge 时每个 edge 每轮只有 1 个客户端 → edge 内聚合退化成复制这一个模型，edge 内稀释不存在。AUDIT D02。
- **N-002**：`grid_too_coarse` 有两处含义：
  - 停轮规则（`server/stopping.py:116-117`）：评估点凑不齐 `pm_window=10` 个。3c_R40 只有 8 个点，所以一直跑到上限。
  - T_θ（`harness/analyze_exp3.py:71,314`）：少于 4 个点就不插值。

## 2026-09-25（读 Bad-PFL 论文与两份 FedRep 实现）

### F-016 `confirmed` —— 三份 FedRep「实现」彼此都不一样，本仓库又和它们都不一样

| 项 | 原作者（`LittleStory233/FedRep`） | PFLlib（`TsingZ0/PFLlib`） | Bad-PFL 论文 | 本仓库（TF，改前） |
|---|---|---|---|---|
| head : body 训练量 | 4 : 1 epoch（`options.py` `local_ep=5`、`local_rep_ep=1`；`Update.py:591`） | 1 : 1 epoch（`plocal_epochs=1`、`local_epochs=1`） | 各 15 步（p.7 + p.13） | 1 : 5 epoch |
| 优化器 | 一个 SGD，momentum 0.5，weight decay 1e-4（`Update.py:557-563`） | 两个普通 SGD（`clientrep.py:11-20`） | SGD | 两个普通 SGD |
| lr | 0.01（两者相同） | 0.005（相同） | 0.1（相同） | body 0.1，head 0.005 |
| lr 衰减 | 无（`lr_decay=1.0`） | 默认关（`learning_rate_decay=False`） | 未提 | 0.992 / 云轮 |
| batch | 10 | 10 | 32 | 32 |

用户判断「PFLlib 与原作者不一样」属实。决定（D-012）：对齐 Bad-PFL 论文。AUDIT A12 → `align`。

### F-017 `confirmed` —— 论文正文超参与本仓库的差异（AUDIT A20 / A22 / A23）

- 论文 p.7：100 客户端、**1000 轮**、10 个恶意端、每轮 10%、Dirichlet 0.5、SGD lr 0.1、batch 32、15 步（约一个 epoch）、投毒率 0.2、ε=σ=4/255、生成器 Adam 0.01 × 30 步、**目标标签随机生成**。
- 本仓库：约 200 有效轮（floor 150 / cap 300）、`local_epochs: 5`、LR 按云轮衰减、目标标签固定为 0。
- 顺带（不属于实验 3，与陷阱 #4 有关）：论文 p.13 附录 A 写 Neurotoxin 更新的是**底部 10%** 的坐标（「We choose to update the bottom 10%…」）；
  MultiKrum 取 f=1、选 5 个客户端聚合（p.14）。

## 2026-09-25（A1 审计会话：攻击部分）

### F-018 `confirmed` —— 集群上 `[Provenance]` 行拿得到 git commit

- 复核文件：`experiments/attack/hfl-propagation/smoke_prov.metrics.json`（`317fbb5`，作者机 `arrhenius1`；命令 `sbatch run_smoke.sh attack hfl-propagation badpfl none smoke_prov hier_fedavg_fedrep`）。
- `run.provenance`：`git=76c72a752f82`（= 交接提交 `76c72a7`）、`dirty=0`、`branch=claude/federated-learning-experiment-review-pt5j1b`、`host=n26`（计算节点）、`job=2984665`、`protocol=P1`、`config_sha=2be16d22e944`。
- `run.cli_overrides` 记下了两条：`backdoor.malicious_strategy` vanilla→badpfl、`training.drift_correction` hierfedavg→hier_fedrep。
- `errors=[]`、`client_failures=[]`，5 轮跑完。`exit_code=null` 是预期的：`run_smoke.sh` 不注入它，只有 `cell.sbatch` / `exp3_cell.sbatch` 注入。
- 结论：`.git` 在 `--bind <仓库上一级>` 范围内，容器里读得到。S1 挂着的「集群核对」关闭。

### F-019 `confirmed` —— 论文与官方代码有三处不一致，逐条选了一边

| AUDIT | 论文 | 官方代码 | 用户选择 |
|---|---|---|---|
| A01 ξ | p.6 Eq.6：在 x 处 FGSM σ·sign(∇ₓL) | `fba.py:6-22`：随机起点 + 在起点处求梯度 + 投影 + clamp 的单步 PGD | 代码（D-014） |
| A04 生成器数据 | p.6 Eq.7：干净 (x, y) | `fba.py:36` + `client.py:124-125`：按 ρ 已投毒的批次 | 论文（D-017） |
| A14 生成器末层 | p.14 表 5：ConvT + BN + Tanh | `generator.py:33-34`：ConvT + Tanh，无 BN | 代码（D-020） |

- A01 的量级差异（numpy 模拟 10⁶ 个像素，内点、不触发 [0,1] clamp）：官方 ξ 有 49.96% 的像素 \|ξ\|=ε，均值 \|ξ\|/ε = 0.7497；FGSM 为 100% / 1.0。
  推导：u∼U(−ε,ε)、sign=+1 时 ξ = min(u+ε, ε)，u≥0 取 ε、u<0 取 U[0,ε)。
- 没有「论文与代码冲突时统一以谁为准」的总规则，逐条决定。

### F-020 `confirmed` —— 「每 batch 恰好 round(nρ) 个」的取整偏差

batch = 32（全部 exp3 配置），`k = int(round(32·ρ))`：

| ρ | k | 实际比例 | 相对偏差 |
|---|---|---|---|
| 0.02 | 1 | 0.0312 | +56.2% |
| 0.05 | 2 | 0.0625 | +25.0% |
| 0.2 | 6 | 0.1875 | −6.3% |
| 0.25 / 0.5 / 1.0 | 8 / 16 / 32 | 与 ρ 相同 | 0 |

- 影响：P1 的全部 ρ=0.2 格子实际投毒率是 0.1875。决定改伯努利（D-016）。

### F-021 `confirmed`（机制）/ 无数值证据（效果）—— TF 生成器的 BN「按 batch 统计优化、按 moving stats 使用」

- 训练：`client_badpfl.py:109` `training=True`。使用：`:95` `training=False`，投毒和评估都走这里。
- Keras BN 默认 momentum 0.99：首次训 30 步后，moving stats 里仍有 0.99³⁰ = 74% 是初值（均值 0、方差 1）；torch 默认 momentum 0.1 下为 0.9³⁰ = 4%。
- 官方的生成器从不 `.eval()`，三处都用 batch 统计，没有这个失配。
- 失配对 ASR 的数值影响**没有证据**。已决定对齐（D-018）。

### F-022 `confirmed` —— 官方 `PMClient` / `PMPoisonClient` 是死代码

- `client.py:86-137` 定义了它们（`PMPoisonClient.fetch_data` 会让 local 与 personalized 两步都投毒）。
- 但 `main.py:6` 只 `from client import BasicClient, PoisonClient`，`main.py:89-94` 也只实例化这两个类。官方实际执行的只有 FedBN（`pfl.py`）单模型，没有单独的个性化训练阶段。
- 本会话一度把 `PMPoisonClient` 当成「官方两阶段都投毒」的证据，用户指出后撤回。任何「官方 PFL 两阶段怎么做」的说法都没有官方执行代码作证据。
- 复核：`grep -n 'PMPoisonClient\|PMClient' *.py`（官方仓库）只命中 `client.py` 的定义处。

### F-023 `confirmed` —— smoke-base 的 `spread` 布点在 `edge_assignment: random` 下没有跨 edge 分散

- smoke 的恶意端是 {0, 9}，两个都在 edge 1（`per_edge_final`：edge0 `n_malicious=0`，edge1 `n_malicious=2`）。
- 原因：非 baked 分区下，`resolve_malicious_ids` 在 random 分配**之前**被调用，`assignments=None` → 走等距 id 回退（`attack/backdoor.py:69-72`）；之后 random 分配把两个都放进了同一个 edge。
- 只影响 smoke-base（exp3 用 `by_edge` + `block`）。smoke 里的 same/diff edge 数字不代表 distributed 布点。

### 设计备注

- **N-003**：D-021 规定只在 body 阶段投毒之后，3.2 的「ρ=1 时私有 head 吸收后门」里 head 不再直接看到投毒样本。head 只在干净数据上训，body 阶段 head 冻结。
  原文 §3.2 的假设（「私有头甚至 bias 就能实现一律预测 y_t」）在这个设定下的机制要重新表述，在 S6 / G4 之前与用户确认。

## 2026-09-25（A2 审计会话：训练协议）

### F-024 `confirmed` —— 官方代码自身不可复现：只播了 torch

- 官方 `utils.py:8-13` `set_random_seed` 只调 `torch.manual_seed` / `cuda.manual_seed_all`，并设 `cudnn.deterministic=True`。
- 划分用的是未播种的 numpy：`main.py:67` `np.random.dirichlet`，`utils.py:65,71,74` `np.random.randint / uniform`。
- 客户端顺序用的是未播种的 python `random`：`main.py:12,95` `from random import shuffle` → `shuffle(clients)`，决定谁排在哪、最后一个恶意端是谁（A02）。
- 结论：官方每次运行的划分与客户端顺序都不同。AUDIT A15 原来只写了「官方 cudnn.deterministic」，只说了一半。本仓库三处都播了（`main.py:325-339`），在划分上比官方严格。

### F-025 `confirmed` —— FedRep 客户端的取数被缓存冻结：3.2% 的训练样本从未参与训练

- `client/hier_fedrep.py:65` 在构造时执行 `self._batch_list = list(self.dataset)`：tf.data 的洗牌 + 增强 + 分批只走一遍。
- `:173-177` `_shuffled_batches` 每个 epoch 只打乱 **batch 的顺序**，并丢掉 `x.shape[0] != batch_size` 的那一批 —— 永远是同一个尾批。
- 后果：增强冻结（F-014）；batch 的组成冻结；同一批 `n mod 32` 个样本从头到尾没被训练过。
- 数值（seed42，P1 划分）：1453 / 45037 = **3.2%** 的训练样本从未参与训练，单个客户端最高 19.5%。
- 复核：按 `main.py` 的 RNG 调用顺序重放划分，只需要标签。10 个 block edge 的样本和 `[5112, 4722, 4309, 4001, 5131, 3435, 4350, 4904, 4046, 5027]` 与 P1 `10edge_distributed_seed42.metrics.json` 的 `per_edge_acc_final[*].n_samples` **逐个相等**，说明重放是精确的：

```python
import numpy as np
np.random.seed(42)                        # main.set_seed；load_cifar10 / build_model 不消耗 numpy RNG
labels = np.repeat(np.arange(10), 6000)   # 合并后的 60k；shuffle 消耗的 RNG 只取决于长度
ci = [[] for _ in range(100)]
for c in range(10):                       # data/partition.py:161-170
    ix = np.where(labels == c)[0]; np.random.shuffle(ix)
    p = np.random.dirichlet(np.ones(100) * 0.5)
    for k, s in enumerate(np.split(ix, (np.cumsum(p) * len(ix)).astype(int)[:-1])): ci[k].extend(s)
n_train = []
for ind in ci:                            # data/partition.py:59-62，test_ratio=0.25
    n = len(ind); nt = max(1, int(n * 0.25)); np.random.permutation(n); n_train.append(n - nt)
n_train = np.array(n_train)               # min 133 / median 401 / max 909，CV 0.40
print((n_train % 32).sum() / n_train.sum())   # 0.0323
```

### F-026 `confirmed` —— G4「FedRep vs FedAvg」混入了数据流差异

- FedAvg 客户端（`client/client_fedavg.py:44`）每个 epoch 都 `for x, y in self.dataset` 重新读 tf.data：重新洗牌、重新增强，并且训练尾批（不 drop）。
- FedRep 客户端走 F-025 的冻结缓存。
- 所以 G4（3.2 的 FedRep ↔ FedAvg 对照）里两边的差别不只是方法。
- Bad-PFL 生成器训练（`client/client_badpfl.py:100`）又是第三种：每轮重读一遍 tf.data，按 `step % len(batches)` 循环取批。
- 修复：D-026（AUDIT A25）。

### F-027 `confirmed` —— 标准化开关与 Bad-PFL 的 ε 换算绑在一起

- `client/client_badpfl.py:53-56`：`_atk_eps_norm = ε / CIFAR10_STD`，无条件执行。
- 若数据管线关掉标准化而这里不改，[0,1] 空间里的 ξ 预算会放大 1/std 倍：三个通道分别 = 4.05 / 4.11 / 3.82。
- `attack/triggers.py:25-37` 的 BadNet 触发器也是按标准化空间算的像素值（F-006 的静态回退会用到）。
- 影响：G7（官方预处理对比，D-025）必须先让这两处跟随开关。

### F-028 `provisional` —— 客户端大小不等 → 名义 10% 的恶意端，数据占比随布点在 0.092–0.142 之间变化

- 数据：seed42 的重放大小（F-025）× P1 各格 `run.malicious_ids`。
- 恶意端数据占比（10 个恶意端 / 全部训练数据）与末 10 点 benign ASR：

| 格子 | 数据占比 | benign ASR |
|---|---|---|
| flat_baseline | 0.092 | 0.755 |
| 2edge_distributed | 0.094 | 0.738 |
| 10edge_distributed | 0.094 | 0.729 |
| 2edge_collocated | 0.106 | 0.889 |
| 10edge_mixed | 0.109 | 0.715 |
| 10edge_collocated | 0.114 | 0.829 |
| 4edge_distributed | 0.127 | 0.972 |
| 4edge_collocated | 0.128 | 0.715 |
| 4edge_mixed | 0.142 | 0.935 |

- 三个 distributed 格子里，占比最高的 4edge（0.127）正是 ASR 异常高的那格（F-008）。但 4edge_collocated 占比相同，ASR 却只有 0.715。
- 只有 1 个 seed，**不能**说占比解释了 F-008。能确定的只有一点：大小不等给布点这个自变量附带了一个没被控制的协变量。
- 样本加权聚合 + 按 epoch 训练会让大客户端的权重更大、本地步数也更多，两条机制都成立；数值影响**无证据**。
- 处理：D-027（S3 的新划分要求客户端等大小）。

### F-029 `confirmed` —— 总本地训练量已与论文相当；差别在每次聚合间的步数，不在训练总量

- 论文 p.7：1000 轮、每轮 10%、每轮 15 步（≈ 1 epoch）→ 每个客户端约 1000 × 0.1 × 15 = 1500 步 ≈ **100 epoch**。p.18：约 500 轮时模型与攻击都收敛 → ≈ 50 epoch。
- P1（ep5）：有效轮 floor 150 / cap 300 × 0.1 × 5 epoch → 每个客户端约 **75–150 epoch**。
- 差别在 τ（两次聚合之间的本地步数：约 68 vs 15）和通信轮数（约 200 vs 1000）。
- **撤回**：A2 会话第一版建议（只在对话里给出、未入库）说「A09 影响最大、对齐 15 步后预算约 ×3」。那一版只比了每轮步数，没有比总量。

### F-030 `confirmed`（来历）/ 无记录（取值）—— 0.992 与 head 0.005 的来历

- `9d303262`（2026-06-12）：把按 optimizer step 衰减改为按轮衰减 `lr0·0.992^round`。修的是 FedRep head 每轮约 25 次衰减、第约 20 轮 head lr≈0 被冻死、PM acc 卡在 60%。
- `27d7d0dc`（同日）：加 `head_lr_rep=0.005`。修的是 head 与 backbone 同 lr（当时 base lr 0.02）时，30 轮后 PM loss 回升。
- 修复的动机在 commit message 里有记录；**0.992 与 0.005 这两个值是怎么选的，仓库里没有扫参数据**。当时的 base lr 是 0.02，不是现在的 0.1。

### F-031 `confirmed` —— 论文原文核对（A2 会话，用户重新上传的 PDF）

- p.7：FL 设定「Following existing studies (Zhuang et al., 2024)」 —— 训练超参是继承来的，不属于攻击本身（D-022）。
- p.8：「the FedAvg aggregator, where each client's contribution is treated equally during aggregation」（A07）。
- p.18：「when the training round reaches 500, both the models and the attack methods converge」（A09 / A23）。
- 论文**没有**提到数据增强、标准化、LR 衰减。
- p.14：Dirichlet 0.5 下「average standard deviation for the clients is … 60.88」。官方划分函数重放 5 个 seed 得 54.4–55.6，本仓库方案（训练部分）得 51.7–54.1。两种方案同一量级，这个数分辨不出来。
- 同样 5 个 seed：客户端平均标签熵，官方 1.75–1.77 nats、本仓库 1.65–1.68；最大类占比 0.35–0.37 vs 0.37–0.39；客户端大小的 CV 是 0 vs 0.41–0.49。**差别主要在大小**（A11 → D-027）。

### F-032 `confirmed` —— 官方的事件系统与网格触发器不在执行路径上（A18 的预读，行状态留给 A3）

- `fl_process.py` 的 6 处 `fl_event_emitter.emit(...)` 在 11 个官方文件里**没有任何** `.on(...)` / 装饰器注册 → 全部是空操作。
- `main.py:11` import 了 `grid_trigger_adder`，但全仓库没有调用（`--ba our` 只走 `use_our_attack`）。
- 复核：在官方仓库执行 `grep -n "fl_event_emitter\|\.on(\|grid_trigger_adder" *.py`。
