# 结论台账 —— 阶段三（edge 原生防御）

> 每条结论 = 声明 + 证据（能重跑的命令或 文件:行）+ 状态。编号接续 `experiments/attack/hfl-mechanism/FINDINGS.md`（F-086 之后从 F-087 起；D-090）。
> 状态：`confirmed`（证据是代码、数据或原文本身）· `provisional`（单 seed / 推断，当假设用）· `retracted`（撤回，写原因）。
> **数字只从脚本产物、复核命令或原文表格里来，不凭记忆写**（陷阱 #14）。预注册的判读规则写成 `N-` 条目，冻结后才交作业。

## 2026-10-09（阶段三计划定稿会话）

### F-087 `confirmed`（代码与配置核对）—— G8 / G6 / G6D 用的旧 `noniid` 划分没有 edge 干净集 → P0 不能用 G8 存盘，改用新组 SNAP

- `experiments/attack/hfl-mechanism/base.yaml:42`：`partition: "noniid"`；G8（`registry.yaml:266-269`）、G6（`:228-234`）、G6D（`:280-286`）的 `set:` 都没有改划分。
- edge 干净集只在 S3 划分（designed / hdir / equal_random）下产生：`fedavg/main.py:506` 只在 S3 路径写 `s3_out["clean_indices"]`，`main.py:774-775` 其余情况下 `edge.clean_indices = None`。
- `fedavg/analysis/ck_snapshot.py` 的说明里写明：G8 没有 edge 干净集，F-081 的 c_k 预检拿**良性端留出分片**当干净数据 —— 只读评估没问题，
  但拿它做**加固**（训练）就是在评估探针上训练（陷阱 #11）。
- 后果：P0 的离线反事实需要「S3 划分 + edge 干净集 + 上传前 body」的快照 → `PLAN.md` §4 的 **SNAP**（C1 × {集中, 分散} × R5 × s42–44，快照第 6 / 15 / 60 轮）。
  集中布点的 SNAP 与 G1R5 只差快照开关，按 F-084 / D-073 应逐位相同（顺带复验）；SNAP 同时是全部主检验的无防御对照臂。
- G8 存盘（集群 `tfdpfl-dumps/G8__a__s4x.*`，约 0.81 GB）因此不再是阶段三的前置；删不删由用户定（D-073）。

### F-088 `confirmed`（原文核对，用户上传的 PDF）/ 一处出处待确认 —— Bad-PFL 的触发器是两个 ℓ∞ 对抗扰动之和；flat 文献里已知防御几乎都挡不住它，对抗训练没被测过；CCS 是唯一的正面证据，但由两个组件组成

数字与原文表号见 `LITERATURE.md` 第 2 部分。对计划有直接影响的六点：

1. **触发器结构**（Bad-PFL Eq. 5–7）：δ = ε·G_w(x) 定向推向 y_t、ξ = σ·sign(∇L) 非定向（FGSM），ε = σ = 4/255 → ‖δ + ξ‖∞ ≤ 8/255。
   → 对抗训练的扰动集要覆盖 8/255，且要对定向扰动鲁棒（`PLAN.md` §3.2 的 ε 网格与 AT-tgt）。
2. **δ 是主体，且有剂量反应**（表 4 / 18，FedRep）：去掉 δ → 12.82%、去掉 ξ → 79.32%；δ 的预算 0 / 1 / 2 / 3 / 4（/255）→ 12.82 / 54.92 / 79.68 / 86.79 / 97.95%。
   → 加固不必做到完全鲁棒（**推断**：这条曲线是攻击者改 ε 测的，不是防御方改鲁棒性测的）。
3. **已知防御在 flat 下几乎都失败**（FedRep，ASR）：FT-15 / 30 / 45 97.31 / 97.76 / 97.01、NAD 86.25、I-BAU 76.58、Simple-Tuning 88.82、BAERASER 91.54、MAD 90.74、
   ClipAvg 97.28、Multi-Krum 96.15、Median 77.21；Sign 20.32 但 Acc 34.49（训练崩）。检测也弱：Neural Cleanse 异常指数 2.2、STRIP 熵 0.77（干净 0.92）。
   → 干净微调降为归因对照、删 NAD 式蒸馏、删 Simple-Tuning（D-094）；I-BAU 的通用扰动遗忘失效 → 改成逐样本的定向 AT；H5 的多数基线预期失败 → s42 先筛。
4. **Bad-PFL 原文没有测对抗训练**；附录 D 认为唯一可想到的对策是「不含目标类数据地微调」。
5. **CCS**（ICASSP 2026，flat、**FedPer**、ResNet-18）：Bad-PFL 94.42% → 8.54%（MTA 79.46 → 79.76），α 0.1–1 都约 10%。组成 = 客户端对抗训练（CE + 0.01·KL + 0.01·MMD）
   + server 对 BN running 统计量做 HDBSCAN 剔除；威胁模型明写攻击者不知道防御。CCS 的 ASR 不过滤目标类（无攻击时约 9–10.5%），8.54% 即无攻击水平。
6. **CCS 的消融出处待确认**：用户记得原文做过消融、对抗训练约占 85%；上传的这一版**没有**（`pdftotext <CCS PDF> - | grep -a -i -E 'ablat|w/o|85%'` 无结果，
   实验部分只有表 1 与图 2）。而本仓库 F-085 测到 BN 统计量通道单独 AUROC 0.90、且在 FedRep 下可零代价伪造 → 本设定下聚类的份额可能比原文大（推断）
   → CCSP 三臂复核组件贡献 + AA-S（伪造统计量）检验聚类部分；拿到消融后把「ΔV(CCS-AT) ≈ 0.85 × ΔV(CCS-full)」写成预注册预测。

### F-089 `confirmed`（环境）—— 本地用 Python ≥ 3.12 跑 L1 会多出与代码无关的红；用 Python 3.11 与交接基线逐项相同

| 解释器（numpy） | L1（`bash run_l1.sh`，本地无 TF） |
|---|---|
| Python 3.11.17（numpy 2.4.6） | **1599 passed / 45 skipped / 3 xfailed → PASS**（= 交接记录的基线） |
| Python 3.12.3（numpy 2.4.6，只跑下面两个文件） | `test_designed_partition.py` 1 条红 + `test_report_figures.py` 1 条红 |
| Python 3.13.16（numpy 2.5.3，本容器默认 `python3`） | **4 failed** / 1595 passed / 45 skipped / 3 xfailed → FAIL |

- **AST 指纹**（`tests/test_designed_partition.py` 的 `test_old_partition_*` 三条）：测试对 `ast.dump()` 的输出取哈希。Python 3.12 给每个 `FunctionDef` 加了 `type_params=[]` 字段
  （函数级指纹那 1 条变红）；Python 3.13 的 `ast.dump()` 默认省略空列表与 None 字段（3 条全红）。复现：
  `for v in 3.11 3.12 3.13; do python$v -c "import ast; print(ast.dump(ast.parse('def f(a):\n    if a:\n        return 1').body[0]))"; done`。
- **第 4 位小数的舍入**（`tests/test_report_figures.py::test_g1_views_figure_reads_the_committed_verdict_and_it_reproduces`）：重算 `g1_verdict.judge_3c` 与入库的
  `analysis/g1_verdict.json` 只差 C1·R20 的两个 `d_jump`（s43 0.1706 vs 0.1707、s44 0.202 vs 0.2021），判定标签不变。同一 numpy 2.4.6 下 3.11 过、3.12 红 →
  是 Python 版本造成的：3.12 起内置 `sum()` 对浮点用补偿求和（`python3.11 -c "print(sum([0.1]*10))"` → 0.9999999999999999，3.12 → 1.0），
  `harness/g1_verdict.py:82` 用内置 `sum` 求均值，落在舍入边界上的值就翻了。
- **代码与数据都没有问题**：集群容器是 Python 3.10.12（CLAUDE.md 的实测记录）；交接记录的 1599 / 45 / 3 与本会话 Python 3.11 的结果逐项相同。
- 做法（本会话起）：本地用 3.11 的 venv 跑 L1 —— `uv venv -p python3.11 <dir>`、装 `pytest numpy pyyaml matplotlib`，再 `TFDPFL_PY=<dir>/bin/python bash run_l1.sh`
  （`cluster_env.sh` 会打印 `mode=override`）。要不要把这几条测试改成与解释器版本无关（例如 AST 指纹先做与版本无关的规范化、均值改用 `math.fsum`），由用户定；本会话没有改。

### F-090 `confirmed`（代码核对）—— CCS 官方代码：FL 主干 = FL-bench；Dir 0.5 = 逐类 Dirichlet、客户端不等大 = 本仓库旧 `noniid`；另有三处与原文 / 计划不一致

- 来源：`https://github.com/chenjian0924/Paper-Code`（用户提供；`README.md`：「Code reference for this document: https://github.com/KarhouTam/FL-bench」）。
  本地 clone 在 `reference/ccs-code/`（gitignore）；FL-bench 的划分代码 sparse clone 在 `reference/fl-bench/`（HEAD `c88d3bc`，2026-01-25 —— **不是** CCS 当时用的版本，版本未知）。
- **划分**：`CCS/generate_data.py:109-118` 在 `alpha > 0` 时调 FL-bench 的 `data/utils/schemes/dirichlet.py`：先把 train + test 合并（FL-bench `datasets.py` 的 `CIFAR10`：`torch.cat([train_data, test_data])`），
  每个类 `np.random.dirichlet(np.repeat(alpha, client_num))` 切给所有客户端，客户端不等大，`test_ratio` 默认 0.25 在客户端内切。
  与本仓库 `fedavg/data/partition.py:151-188`（`noniid_partition`）逐行同构；**唯一差别**：FL-bench 重抽直到每端 ≥ `min_samples_per_client`（默认 10），本仓库不重抽。
  → 本仓库已有的 flat 组里，**G8F 用的就是这个构造**（`registry.yaml:312` 没覆盖划分 → `base.yaml:42` 的 `noniid`）；G2 的 flat 格用 `equal_random`。之前 flat 组的划分并不统一。
- **CCS 的两个组件被同一个开关打开**：`src/server/fedavg.py:533` `defence_method == 'AT'` → `aggregate_AT`（`:723`，先 `hdbscan_detect` 剔除再聚合）；客户端 `src/client/fedavg.py:272-289` 同一条件下加 `loss_ce + AT_alpha·KL + AT_beta·MMD`。代码里**没有**只开其中一个的开关。
- **与原文 / 计划不一致、待 SA-C 语义 diff 逐条定**（只读了代码，没有跑）：
  1. 聚合权重：`aggregate_AT` 用 `package["weight"]` 归一化（FL-bench 里是样本数）→ **按样本加权**；`PLAN.md` §1.1 写的是「无权 FedAvg」。
  2. KL 系数：`config/defaults.yaml` 的 `AT_alpha: 0.1`、`AT_beta: 0.01`；原文 β = γ = 0.01。`config/cifar10.yaml` 没有这两个键，最终取值取决于配置合并方式（未核对）。
  3. HDBSCAN：`src/defence/AT/hdbscan_detect.py` 用 BN `running_mean / running_var` 拼成向量、**余弦距离**、`min_cluster_size = n//2 + 2`、`min_samples = 1`、`allow_single_cluster=True`，保留最大簇；全部是噪声时不剔除。
- **攻击代码没有公开**：`src/client/fedavg.py:21` `from src.attack.BadPFL.generator import ...`，但仓库里没有 `src/attack/` → CCS 原文的 Bad-PFL 实现无法与本仓库对拍。
- 配置里的其它数（`config/cifar10.yaml`）：`PResNet18`、SGD lr 0.1 无动量、`local_epoch: 2`（**2 个 epoch**，不是 PLAN 写的「2 步」，待原文核对）、batch 64、`join_ratio: 0.1`、1000 轮、`buffers: global`。

## 2026-10-09（D0 会话：登记 + 功效分析 + 文献核对 + 预注册草案）

### F-091 `confirmed`（数据，`python3 harness/d0_power.py` → `analysis/d0_power.json`）—— 功效分析：对照臂已饱和；配对噪声远小于 0.15，但 `no_effect` 规则对真零效应也常判不出

量的定义与 `g6_verdict` / `g1_verdict` 相同（末 10 个评估点；Δ = 参照 − 处理，按 seed 配对）。逐 seed 值见 json。

| 读数 | 数 | 含义 |
|---|---|---|
| **G1R5**（= SNAP-col 的预期，C1·集中·R5，s42–44） | V 0.9959 ± 0.0008、B0 0.9993 ± 0.0013、MTA 0.8683 ± 0.0069、受害 edge margin_p50 6.44 ± 0.95 | 主检验的对照臂**饱和**：任何没把 V 压到 0.95 以下的臂都会落到天花板规则（读 Δmargin）|
| G0-C1（floor，ρ=0） | V 0.068 ± 0.024、margin −7.60 ± 0.29 | 对照 floor；防御臂 Δexcess 的参照 |
| 不饱和配置的 seed 间 SD | G1 C1·集中·R20 V 0.575 ± 0.114；G6(a) V 0.815 ± 0.170 | **CAL / W2（V 0.3–0.8）预计 seed SD 约 0.1–0.17** → W2 上的 3 seed 判定功效低（见下）|
| 配对差 SD：真实干预（3-E，G6） | a−b：ΔV 0.524 ± **0.035**、ΔMTA 0.0064 ± 0.0006；a−c：ΔV 0.730 ± **0.189**、ΔMTA 0.0171 ± 0.0042 | 「干预 × seed」噪声的两个实测量级 |
| 配对差 SD：结构变化（G1） | 集中 R10−R20：ΔV 0.338 ± 0.080、Δmargin_v 3.86 ± **0.065**；分散 R10−R20：ΔP −0.004 ± 0.034、Δmargin 0.45 ± **0.25** | margin 的配对 SD 0.07–0.25（攻击 − floor：0.66 / 0.78）→ **2 logit ≈ 2.6–30 σ** |
| ΔMTA 的配对 SD | 0.0006–0.0048（全部配对） | 0.02 的门槛是 4–30 σ → 精度判定由真实代价决定，不受噪声左右 |
| **P0 的分母**（G1R5 受害 edge Δ_jump，第 t+1 云轮云聚合后点 − 第 t 云轮末全量点） | t=6：0.231 / 0.250 / 0.336；t=15：0.423 / 0.224 / 0.146（s42 / 43 / 44） | 6 个 col 快照**全部 ≥ 0.05 → 都计入**；全部云轮里 53% 的 jump < 0.05（后期饱和）→ 第 60 轮快照不能用来算 R_H，符合 PLAN 只用 6 / 15 |

H1 规则（均值 ≥ 0.15 且最小 seed ≥ 0.10 → `protects`；三个 |Δ| < 0.05 → `no_effect`；其余 `partial`）的操作特性（正态 MC，n = 3，20 000 次，固定种子）：

| 配对 SD σ | μ = 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| 0.035（G6 a−b） | P 0.00 / N **0.62** | P 0.00 / N 0.12 | P 0.005 | P **0.48** | P **0.99** | P 1.00 |
| 0.05 | P 0.00 / N 0.32 | P 0.00 / N 0.10 | P 0.03 | P 0.43 | P 0.91 | P 1.00 |
| 0.10 | P 0.002 / N 0.06 | P 0.02 | P 0.10 | P 0.30 | P 0.58 | P 0.93 |
| 0.189（G6 a−c） | P 0.025 / N 0.01 | P 0.06 | P 0.12 | P 0.22 | P 0.35 | P 0.63 |

（P = `protects` 概率，N = `no_effect` 概率）

读法（**只报告，门槛由用户定**，PLAN §8 第 1 条）：

1. **假阳性很低**：σ ≤ 0.10 时真零效应判成 `protects` 的概率 ≤ 0.002。0.15 / 0.10 不需要收紧。
2. **功效取决于 σ**：σ ≈ 0.035 时真效应 0.20 几乎必过、0.15 约一半；σ ≈ 0.19 时连 0.30 也只有 0.63。σ 大的臂（像 3-E 的 c 臂）在 3 seed 下多半是 `partial`。
3. **`no_effect` 很难拿到**：真零效应在 σ = 0.035 时只有 62% 判成 `no_effect`，σ = 0.05 时只有 32%，其余落进 `partial` → 按 §3.5 (a) 会触发加 seed。
   这是规则的结构问题（要求三个 seed **都**在 ±0.05 内），不是数据问题。可选的改法（用户定）：把 `no_effect` 改成「均值的绝对值 < 0.05 且最大 |Δ| < 0.10」，或保持原样、接受加 seed 的开销。
4. **饱和**：对照 V 约 0.996 → H1 多半要用天花板规则。margin 的 2 logit 有余量（配对 SD ≤ 0.78，见上表）；F-086：一次云聚合把 margin 推高 1.9–3.0 logit。**2 logit 维持**。
5. **W2**：不饱和工作点的 seed SD 约 0.1–0.17。如果配对 SD 也到这个量级，W2 上的 3 seed 判定功效偏低（σ = 0.10 时 μ = 0.20 只有 0.58）→ W2 的结论预计多为 `partial`，正好落进加 seed 规则 (a)。

局限：σ 只有两个真实干预的量级（3-E 的 b / c 臂，旧划分）；对抗训练类干预的 σ 要等 P1 / H1 才知道。正态近似、3 个点估 SD 本身误差就大（自由度 2）。

### F-092 `confirmed`（代码 + 配置核对）—— SNAP 登记好了：快照里的 edge 模型就是本轮的上传物；SNAP-col 的配置与 G1R5 只差记录 / 快照开关

- **快照的时间点**：`BackdoorCloudServer.run_round` 先跑 `super().run_round()`（broadcast → R 个 edge 轮 → 云聚合）再评估，然后 `_snapshot`（`fedavg/server/backdoor_server.py:126-135`）；
  下发在**下一轮开头**（`fedavg/server/server.py:387` `broadcast_to_edges()`）→ 快照里的 `edge{e}` = 第 t 云轮最后一次 edge 聚合 = 上传物，`global` = 第 t 次云聚合的结果 G。
  这正是 P0 需要的「各 edge 的上传前 body + G」。另有每端 `private_state()`（私有 head + 私有 BN 统计量）与评估攻击者的生成器（`:417-470`）。
  edge 干净集不在快照里：它由划分（只依赖 seed）决定，离线脚本用同一份 `designed_partition` 重算 `clean_indices` 即可。
- **配置**：`experiments/defense/edge-native/registry.yaml`（base / overlay 与 hfl-mechanism 同一条链）。materialize 后 SNAP-col 与 G1R5 同 seed 的差 =
  {`update_geometry`、`update_ck*` 四个键（去掉），`frozen_trigger`、`snapshot_rounds: "6/15/60"`（加上），`meta.*`}；SNAP-dist 与 SNAP-col 只差 `malicious_per_edge`。
  守卫 `tests/test_edge_native_registry.py`；6 个配置都过 `config_validate`（只有网格下快照按全量点计数的提示，R5 时网格 = 云轮，无影响）。
- **有效性闸的依据**：去掉的 `update_ck` / `update_geometry` 与加上的 `frozen_trigger` / 快照都已证明不改变训练（F-081 / F-084、G8 对 G6(a) 第 1–30 轮，D-073）→ SNAP-col 的 `[Checksum]` 应与 G1R5 同 seed 逐轮相同（**预期，GPU 上未验证**）。
- **提交**：`PACK=3 RUN_GROUPS=SNAP bash experiments/defense/edge-native/submit.sh`（dry-run：两个 k=3 包，`-c 12 --mem=72G`）。约 6 GPU-h，盘约 1.8 GB。

### F-093 `confirmed` / 部分 `provisional`（文献核对，2026-10-09 网页检索；arxiv / ar5iv / papers.neurips.cc 被出站代理挡，未读全文）—— PLAN 凭记忆引用的条目出处都对；有两处细节要改写或带进 SA0 的语义 diff

| 条目（PLAN 里的用法） | 核对结果 | 来源 |
|---|---|---|
| TRADES（ICML 2019，「显式权衡干净精度」，β ∈ {1, 6}） | ✅ Zhang 等，ICML 2019；损失 = CE + β·KL(clean ‖ adv)。官方 `train_trades_cifar10.py` 缺省 **ε = 0.031、10 步、步长 0.007、β = 6.0**；README 写 β 可取 [1, 10] | arxiv 1901.08573 摘要页、github yaodongyu/TRADES |
| Tsipras 2019（鲁棒性–精度权衡） | ✅ Tsipras、Santurkar、Engstrom、Turner、Mądry，「Robustness May Be at Odds with Accuracy」，ICLR 2019 | iclr.cc / arxiv 1805.12152 |
| I-BAU（「反学的是一个通用扰动」） | ✅ Zeng 等，ICLR 2022。官方 README 的极小极大式里 δ 在对样本求和的外面（`max_{‖δ‖≤C} (1/n) Σ L(f(x_i+δ), y_i)`）→ **单个、与输入无关的扰动**；摘要称 100 张干净图仍有效 | github reds-lab/I-BAU |
| NAD（ICLR 2021，删除） | ✅ Li 等，ICLR 2021；teacher = 在同一干净子集上微调的副本，按中间层注意力图蒸馏；摘要：5% 干净数据 | iclr.cc / arxiv 2101.05930 |
| FLTrust（「约 100 张根数据集」） | ✅ Cao、Fang、Liu、Gong，NDSS 2021；摘要：根数据集**少于 100 张**时，在 40–60% 恶意端的自适应攻击下精度仍与无攻击的 FedAvg 相当 | ndss-symposium.org / arxiv 2012.13995 |
| A3FL（自适应攻击者模板，AA2） | ✅ Zhang、Jia、Chen、Lin、Wu，NeurIPS 2023；触发器按「全局模型被训练去反学它」的最坏情形对抗优化 → 与 AA2 的写法一致；官方代码 hfzhang31/A3FL；摘要：对 12 种防御 | neurips.cc / PSU 页面 |
| EOT（Athalye 2018，AA1） | ✅ EOT 出自 Athalye、Engstrom、Ilyas、Kwok「Synthesizing Robust Adversarial Examples」（arxiv 2017，ICML 2018）；用于打破随机化防御的是 Athalye、Carlini、Wagner「Obfuscated Gradients…」（ICML 2018）。**PLAN 写的「EOT（Athalye 2018）」应注明是前者**；AA1 的「对防御变换取期望」用的是这个思想 | arxiv 1707.07397 / 1802.00420 |
| LP（ICLR 2024，分段感知攻击者） | ✅ Zhuang 等，「Backdoor Federated Learning by Poisoning Backdoor-Critical Layers」，ICLR 2024；只毒化后门关键层，10% 恶意端下绕过 7 种防御 | iclr.cc / arxiv 2308.04466 |
| SAU（「5% 干净数据、L∞ ≤ 0.2 的 5 步 PGD」） | ✅ Wei 等，NeurIPS 2023。BackdoorBench 的 `config/defense/sau/cifar10.yaml`：`ratio: 0.05`、`norm_type: L_inf`、**`trigger_norm: 0.2`**、`adv_steps: 5`、`adv_lr: 0.2`、`pgd_init: max`、`beta_1: 0.01`、`beta_2: 1`。<br>⚠ PLAN §3.2 的 AT-sau 写的是 ε = 8/255（≈ 0.031），比官方 0.2 小约 6 倍；0.2 是在哪个空间（像素 [0,1] 还是归一化后）**没有核对** → 带进 SA0 的语义 diff 表 | github SCLBD/BackdoorBench |
| RLR（基线） | ✅ Ozdayi、Kantarcioglu、Gel，AAAI 2021；按坐标的符号投票，票数绝对值 < θ 的坐标学习率取负 | ojs.aaai.org / arxiv 2007.03767 |
| CerP（第二攻击） | ✅ Lyu 等，「Poisoning with Cerberus」，AAAI 2023（pp. 9020–9028）；联合调触发器与投毒模型的偏差 | ojs.aaai.org |
| IBA（第二攻击） | ✅ Nguyen 等，NeurIPS 2023（mlanthology 条目） | mlanthology |
| Simple-Tuning（KDD 2023，删除） | ✅ Qin 等，KDD 2023；训练后重置并只重训线性分类器（其余冻结）；结论：部分共享的 pFL 更抗后门 | arxiv 2302.01677 |

- **没核对的**：综述里标「待核实」的 BackdoorIndicator（USENIX Sec 2024）、FTA（arXiv 2309.00127）、CCS 的出处版本；Madry 2018、FLIP、FedBAP、SHIELD、PriRoAgg、DPOT 的细节。这些不影响 D0 的登记与判定规则，等对应的代码会话（SA3 / SA4）做语义 diff 时再读原文 / 官方代码。
- 对计划的影响：（1）SA0 的语义 diff 必须列出 AT-sau 的 ε 与官方配置的差别，并说明为什么取 8/255（覆盖 Bad-PFL 触发器的上界）而不是 0.2；
  （2）AT-trades 的 β ∈ {1, 6} 与官方缺省 / README 范围一致；官方 PGD 步数 10，§3.2 的「预算 {低, 高}」要写明步数；（3）PLAN 的 EOT 引文改写成上表的两篇。

### N-008 `已确认 · 生效`（预注册，写于 SNAP / P0 / P1 / CCSF 的任何数据之前；用户 2026-10-10 确认，D-101）—— SNAP 有效性闸、P0、P1、CCSF 的判读规则

> 状态：**已确认、生效**（2026-10-10，用户：「N-008 四处按你的建议改，确认生效」→ D-101）。`d0-prereg` 已加进 `registry.yaml` 的 `available`。
> 确认时改了一处（通用闸的攻击者参与，下文标 **[D-101]**，原文删除线保留）；另外三处（P1 分散布点的 `stop`、CCSF 的植入闸、P0 的阈值）按草案原文确认；
> H1 / H3 的 `no_effect` 在 `PLAN.md` §4 H1 改（不在本条范围内，同属 D-101）。
> 确认时 SNAP 已回传（F-094）：SNAP 的闸（本条第一段）读的就是它；P0 / P1 / CCSF 的数据一个都还没有。
> 判定脚本（`harness/p0_verdict.py` / `p1_verdict.py` / `ccsf_verdict.py`）在对应代码会话（SA0 / SA1 / SA-C）写、**先 commit 再打开数据**（同 D-088）。
> 之后再改本条要在 DECISIONS 留记录。

**共用定义**（末 10 个评估点；全部按 seed 配对；Δ = 对照 − 防御臂，正 = 防御臂更低；对照 = 同 seed、同布点的 SNAP）：
`V` 受害 edge E1–E3 良性端 fresh-PM ASR（`per_edge_rounds[r][e].client_benign`，每 edge 末 10 点均值，再三 edge 平均）；`B0` 同上、edge 0；`P` 全部良性端池化（`rounds[].local_benign_asr`）；
`margin_v` 受害 edge 的 `margin_p50`（`per_edge_detail_rounds`，三 edge 平均）、`margin` 池化（`rounds[].margin_p50`）；`MTA` fresh `pm_acc`（`acc_rounds[]`），陈旧列只报告；
`jump(g)` = 第 g 云轮云聚合后评估点（`per_edge_post_agg_rounds`）− 第 g−1 云轮末全量点，受害 edge 三均值（同 `g1_verdict.sawtooth`）；`jump_half` = g = 2 … 30 的均值。
均值一律 `math.fsum`（F-089）。

**通用有效性闸**（任一不满足 → 该 run `invalid`，不判标签）：`exit_code == 0`；`client_failures == []`（陷阱 #23：被吞的异常会静默剔除端）；
~~每个恶意端每云轮都参与（`malicious_selected_rounds` 覆盖全部轮次）~~ **[D-101]** 攻击者参与过：`n_malicious_participations > 0`（同 `g1_verdict.run_reasons`；
本仓库没有强制参与，随机选端下某个云轮恰好没有攻击者是正常的，F-094）；`status.py` 判 `done`（config_sha 一致、run 块因素与登记一致）。

**SNAP**：
- `snap_valid`：上面的闸 + SNAP-collocated 的 s42–44 与 G1R5 同 seed：`harness/instrumentation_check.py <G1R5> <SNAP> --upto 60` 全部 `[Checksum]` 逐轮相同、改动前就有的数值字段相同；
  每个 run 有 3 个快照（metrics.json 的 `dumps.snapshots` 轮号 = 6 / 15 / 60），快照 npz 里 `meta_json` 的 `edge_matches_eval == true`（快照的 edge 模型 = 本轮评估用的 body）。
- 任一 col run 不满足 → `snap_invalid`：停下查原因，**不读 P0**。dist 没有参照，只过通用闸。

**P0**（SA0 实现；快照 t ∈ {6, 15}；配置网格 = PLAN §3.2，SA0 在打开 s42 快照之前把**完整的配置清单与预算**写进 git）：
- 量（集中布点）：`A(·)` = 用 run 的评估代码离线算的受害 edge fresh-PM 良性 ASR（三 edge 平均），在快照 t 的固定攻击者 / 探针顺序上；
  `J_t` = run 记录的 `jump(t+1)`；`R_H = [A(G) − A(G'_H)] / J_t`，`G'_H` = 4 个 edge **都**经 H 加固后按样本加权 FedAvg；`J_t < 0.05` 的快照不计。
  分散布点只报告 `D_H = P(G) − P(G'_H)`（池化），不进判定（PLAN §4 H3：`A-pre` 在分散布点预期 `no_effect`）。
- 闸 `V0`（每个快照）：离线 FedAvg(各 edge 权重) 与快照的 `global` 逐元素 max |差| ≤ 1e-5；`|A(G) − run 记录的 post_agg(t+1)| ≤ 0.01`；`|A(edge 模型) − run 记录的第 t 云轮全量点| ≤ 0.01` → 否则 `invalid`。
- 精度过滤（每个配置 × 快照 × seed）：fresh 池化 `MTA(G) − MTA(G'_H) ≤ 0.02` 且每个 edge ≤ 0.04。配置在某 seed 上「过滤通过」= 两个快照都通过。
- **选择**（只用 s42）：在过滤通过的 AT 配置里，按 `min(R_H@6, R_H@15)` 排序，取前 3 名冻结（并列按单次加固的前向反向次数少者优先）；
  C-ft、阻尼、oracle、cloud 侧、`n_clean` 敏感性、CCS-clu 离线检测随行（只报告）。冻结清单 commit 之后才读 s43 / s44。
- 标签：
  - `go_online`：冻结的前 3 名里有配置在 **三个 seed × 两个快照**都 `R_H ≥ 0.5` 且三个 seed 都过滤通过 → 进 SA1，取名次最高的那个；
  - `kill_pre`：s42 上**全部** AT 配置、以及冻结配置在 s43 / s44 上，两个快照的 `R_H` 都 < 0.2 → `A-pre` 止步（`A-every` 不由 P0 判死）；
  - `accuracy_bound`：s42 上没有过滤通过的配置满足 `R_H ≥ 0.5`（两个快照），但有过滤**不**通过的配置满足 → 用户定；
  - `inconclusive`：其余（P1 照做，只作探路）。
- 必报：最佳配置的 `R_H − R_H(C-ft)`（逐 seed，对抗部分的贡献）；触发器范数 ‖δ‖∞ / ‖ξ‖∞ / ‖δ+ξ‖∞（像素与模型输入两个空间）；CCS-clu 离线剔除的 TPR / FPR（逐 edge）。

**P1**（SA1 实现；s42 单 seed、筛选，不进结论；对照 = SNAP s42 同布点）：
- 闸：通用闸 + 第一次加固之前的 `[Checksum]` 与 SNAP s42 逐轮相同（具体哪一行在 SA1 定、先写进 git）+ `[Harden]` 行数 = 预期（`A-pre` 4 × 60、`A-every` 4 × 300、`A-cloud` 60）。
- 集中布点 `go_main_col`：（`ΔV ≥ 0.15` 或 `jump_half(臂) ≤ 0.5 × jump_half(对照)`）且 `ΔMTA ≤ 0.02`；
  **天花板规则**：对照与臂的 V 都 ≥ 0.95 时，`ΔV` 那一项换成 `Δmargin_v ≥ 2.0`（F-091：配对 SD ≤ 0.78）。
- 分散布点 `go_main_dist`：（`ΔP ≥ 0.10` 或 `Δmargin ≥ 2.0`）且 `ΔMTA ≤ 0.02`。
- `stop_col`：`ΔV < 0.05` 且 `jump_half` 减少 < 20% 且 `Δmargin_v < 1.0`；`stop_dist`：`ΔP < 0.05` 且 `Δmargin < 1.0`（**新增**：PLAN 只写了集中布点；1.0 logit ≈ 分散布点配对 SD 0.25 的 4 倍）。
- 其余 `marginal`：只报告，不进主检验，除非用户指定。臂按布点分别进 H1（col `go_main`）/ H3（dist `go_main`）。
- `A-every` 的预算：按 P0 最佳配置缩到「墙钟开销 ≤ +50%」，换算（每次加固的前向反向次数 / 每云轮客户端训练的前向反向次数）由 SA1 先写进 git 再交。

**CCSF**（SA-C 实现；flat，s42，off vs ccs-full；划分 = 旧 `noniid`（Dir 0.5、客户端不等大，D-098）→ 报告恶意端数据占比）：
- 闸：通用闸 + off 臂必须植入：`P(off) ≥ 0.5`，否则 `invalid_no_attack`（攻击没进去时复现无从谈起）。
- `ccs_reproduces`：`P(off) − P(ccs-full) ≥ 0.5` 且 `ΔMTA ≤ 0.02`；`ccs_not_reproduced`：`P(off) − P(ccs-full) < 0.2` → 按语义 diff 表排查实现，不读 CCSP，用户定；其余 `partial_repro`（照做 CCSP，结论里注明）。
- 另报：`local_benign_asr_unfiltered`（CCS 原文的 ASR 不过滤目标类，F-088 第 5 条）、`admitted[]` / `rejected_ids` 的剔除 TPR / FPR。

**这一条里没有证据的部分**：`R_H` 的 0.5 / 0.2（来自 F-086「只需砍掉 20–40%」的推理，加一倍余量）；精度过滤里「每个 edge ≤ 0.04」；`stop` 里 1.0 logit；`V0` 的 1e-5 与 0.01 容差（离线重算与 run 内评估是不是逐位相同，要 SA0 在 GPU 上验证）。
用户确认时知道这些没有证据，**按原值确认**（D-101）。`V0` 的两个容差若在 SA0 的 GPU 验证里被证明不可达（例如离线重算本来就不逐位相同），改它要在打开 P0 数据之前、并留 DECISIONS 记录。

## 2026-10-10（SNAP 回传）

### F-094 `confirmed`（数据）—— SNAP 6 个 run 全部有效：collocated 与 G1R5 同 seed 前 60 轮逐位相同；P0 的 6 个快照分母与预测一致

- **对账**：`python3 harness/status.py experiments/defense/edge-native/registry.yaml` → SNAP 6 个 `done ✓`（config_sha 核对过）。
  6 个 run 都 exit 0、`client_failures == []`、`run.alignment.template == "p2"`。
- **有效性闸（N-008 SNAP 段）通过**：`python3 harness/instrumentation_check.py <G1R5__C1-collocated-R5__s4x> <SNAP__collocated__s4x> --upto 60`
  三个 seed 都是「✅ 前 60 轮逐位一致：60 个 checksum、4560 个已有数值字段」；另外 `post_agg_rounds[]`（去掉计时列）与 `per_edge_post_agg_rounds` 也逐项相同。
  → 去掉 `update_geometry` / `update_ck*`、加上 `frozen_trigger` 与快照，**训练与已有评估都没变**（F-092 的预期，GPU 上首次验证）。
- **快照**：6 个 run 的 `dumps.snapshots` 轮号都是 6 / 15 / 60、`errors == []`；每 run 约 0.32 GB，6 个共约 1.93 GB（≤ 20 GB，D-073）。
  `meta_json.edge_matches_eval` 在集群上的 npz 里，**本地没核对**（SA0 读快照时第一步核对）。
- **P0 的分母**（受害 edge Δ_jump，`harness/d0_power.victim_jump`）：collocated s42 / 43 / 44 的 t=6 为 0.2309 / 0.2497 / 0.336、t=15 为 0.4226 / 0.2244 / 0.146，
  与 F-091 用 G1R5 算的**逐位相同** → 6 个快照都计入（≥ 0.05）。
- **对照臂的主量**（末 10 点；`harness/d0_power.run_quantities`）：

  | run | V | B0 | P | margin（池化） | margin_v | MTA（fresh） |
  |---|---|---|---|---|---|---|
  | collocated s42 | 0.9950 | 0.9978 | 0.9955 | 6.94 | 6.69 | 0.8619 |
  | collocated s43 | 0.9965 | 1.0000 | 0.9971 | 5.66 | 5.39 | 0.8756 |
  | collocated s44 | 0.9963 | 1.0000 | 0.9969 | 7.44 | 7.25 | 0.8673 |
  | distributed s42 | — | — | 0.9982 | 5.23 | — | 0.8674 |
  | distributed s43 | — | — | 0.9977 | 7.22 | — | 0.8675 |
  | distributed s44 | — | — | 0.9614 | 4.42 | — | 0.8720 |

  两种布点的对照都饱和（V、P ≥ 0.96）→ H1 / H3 大概率要走天花板规则（读 Δmargin）。分散布点 s44 的 P = 0.961 是唯一离 1 稍远的格。
- **开销**：两个 k=3 包墙钟 15 671 s（collocated）/ 14 227 s（distributed），共约 8.3 GPU-h（外推 6 GPU-h）；每 run 显存峰值 16.76–16.95 GiB，无 OOM。
- **N-008 草案的一处措辞问题（2026-10-10 已按此改，D-101）**：通用闸写的是「每个恶意端每云轮都参与」，但本仓库没有强制参与（`run.forced_participation = false`）。
  distributed s42 第 45 轮、collocated s44 有一个云轮恰好没有攻击者被选中（`n_malicious_participations` = 59 / 60）。按原文字这两个 run 会被误判 invalid。
  建议改成与 G1 判定（`harness/g1_verdict.py:run_reasons`）相同的「`n_malicious_participations` > 0」。**还没改**。

### F-095 `confirmed`（代码 + 可复现的计算）—— G1 / G1R5 的在线 c_k 只在飞机和汽车两个源类上测：攻击样本取的是「按类排序的干净集」的前 64 张

- **代码**：`fedavg/server/update_ck.py:87` `xa, ya = xs[:n], ys[:n]`（n = `evaluation.update_ck_n` = 64）；NCM 原型用全部 500 张，攻击样本只用前 n 张。
  干净集 `edge.clean_x / clean_y` = `x_all[clean_indices]`（`main.py:778-785`），而 `clean_indices` 由 `data/designed_partition.py` 的 `take()` **逐类拼接**（先全部第 0 类、再第 1 类……）→ 按类排序。
  `analysis/functional_score.ck_from_reached` 只统计 y ≠ k 的样本。
- **复现**：`python3 harness/ck_probe_classes.py <G1 / G1R5 的配置>`（用每类 6 000 张的合成标签跑真实的 `designed_partition`；各类张数只取决于类供给、设计参数与 seed，与标签在数组里的位置无关）。
  27 个开了 update_ck 的配置、108 个 edge：

  | 划分 | 前 64 张里的飞机（y_t） | 汽车 | 其余类 | c_{y_t} 可统计的样本 |
  |---|---|---|---|---|
  | C1（G1 C1 12 个 + G1R5 3 个，60 个 edge） | 50 | 13–14 | 0–1 | 13–14 张，全是汽车 |
  | random（G1 random 12 个，48 个 edge） | 35–64 | 0–29 | 0 | 0–29 张；**8 个 edge 为 0 → c_{y_t} 无定义** |

- **后果**：c_{y_t}（推向目标类）只在十几张汽车上统计，其余 c_k 只在飞机和汽车两个源类上统计；设计意图（S6b-PLAN：在干净集上测各类的可达性）没有实现。
  F-085 / REPORT §5.13 的「在线 c_k（s_ck）undetectable」是在这个探针上得到的 —— 它说明「这个探针上没有信号」，**不能**推广成「c_k 这类功能分数检测不出恶意更新」。
  反过来，「换成按类分层的探针就能检测」**也没有证据**。G1 没存完整的单个更新（只有草图），离线重算不了；要知道答案只能带修正后的取样重跑。
- **不影响**：3-C（锯齿）、`norm_w` / 余弦的 3-D 结论（不用 c_k）；阶段三的 SNAP（`D-base` 没开 update_ck）。
- **阶段三的处理**：SA0 的 AT-tgt 求 k\* 时用全部 500 张或按类分层的子集；加固时每个 epoch 先打乱干净集（PLAN SA0 语义 diff 的修正 2 / 3）。
  `update_ck.py` 的取样要不要改、REPORT §5.13 要不要加注，由用户定（还没改）。
