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
- [D-107] 加固发散（NaN / Inf）的配置 × 快照 = 精度过滤不通过、R_H 不计入；`kill_pre` 里算「没达到 0.2」。
- 精度过滤（每个配置 × 快照 × seed）：fresh 池化 `MTA(G) − MTA(G'_H) ≤ 0.02` 且每个 edge ≤ 0.04。配置在某 seed 上「过滤通过」= 两个快照都通过。（[D-105] 判定脚本照原文：与 J_t 无关；SAU 的归因对照是 `cft-sau-*`；网格 36 个配置。）
- **选择**（只用 s42）：在过滤通过的 AT 配置里，按 `min(R_H@6, R_H@15)` 排序，取前 3 名冻结（并列按单次加固的前向反向次数少者优先）；
  C-ft、阻尼、oracle、cloud 侧、`n_clean` 敏感性、CCS-clu 离线检测随行（只报告）。冻结清单 commit 之后才读 s43 / s44。
- 标签：
  - `go_online`：冻结的前 3 名里有配置在 **三个 seed × 两个快照**都 `R_H ≥ 0.5` 且三个 seed 都过滤通过 → 进 SA1，取名次最高的那个；
  - `kill_pre`：s42 上**全部** AT 配置、以及冻结配置在 s43 / s44 上，两个快照的 `R_H` 都 < 0.2 → `A-pre` 止步（`A-every` 不由 P0 判死）；
  - `accuracy_bound`：s42 上没有过滤通过的配置满足 `R_H ≥ 0.5`（两个快照），但有过滤**不**通过的配置满足 → 用户定；
  - `inconclusive`：其余（P1 照做，只作探路）。
- 必报：最佳配置的 `R_H − R_H(C-ft)`（逐 seed，对抗部分的贡献）；触发器范数 ‖δ‖∞ / ‖ξ‖∞ / ‖δ+ξ‖∞（像素与模型输入两个空间）；~~CCS-clu 离线剔除的 TPR / FPR（逐 edge）~~ **[D-105]** 快照里没有上传的统计量，做不了 → 报私有统计量聚类 `bn_private_clu`（只作参照），CCS-clu 由 SA-C 在线给出。

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

## 2026-10-10 … 10-13（SA0 会话：edge 侧对抗训练核心 + P0 离线驱动 + 判定脚本）

### F-096 `confirmed`（代码 + 本地测试；GPU 上**没有**数据）—— SA0 实现完成；三轮对抗式审查确认的问题都已改；V0 在 CPU 小 HFL 上逐位成立

- **实现**（`8d9a63c` → `9af2527` → 本次）：`fedavg/hardening/{spec,core}.py`（加固核心，SA1 在线复用）、`fedavg/analysis/{p0_snapshot,bn_cluster}.py`（离线反事实驱动）、
  `harness/p0_verdict.py`（选择 + N-008 判定）、`p0_configs.yaml`（36 个配置，D-103 / D-104 / D-105）、`p0.sbatch`。语义 diff 表见计划文件（会话内批准），要点进了 D-103 … D-105。
- **V0 在 CPU 上逐位**（`tests/test_p0_snapshot_tf.py`，本地 TF 2.15、2 edge × 3 客户端的小 HFL）：FedAvg(快照 edge) = 快照 global；离线云聚合后点 = run 的 `[PostAgg]`（ASR / pm_acc / margin）；
  离线第 t 轮全量点 = run 的主列（client_benign + margin 中位数，margin 非零）。**GPU 上没有证据** —— 探路作业（current-focus）先看这三项。
- **记账**（`spec.fb_*`，单位 = 一个 batch 的一次前向 + 反向）：SNAP 设定下一个云周期的客户端训练 ≈ 3 300；`pgd-e8-high` 每 edge 825 × 4 edge = 3 300（≈ 一个云周期），
  pgd / trades / tgt 的 low 约 90–105 × 4 ≈ 11–13%；SAU 的 low / high 每 edge 202.5 / 1 012.5。SAU 在本预算下只有官方约 1 000 步（100 epoch × 10 批 × 256）的 ≤ 8%（步数）/ 约 1%（样本数）。
- **第一轮审查**（只完成「官方实现忠实度」一维，4 条，全部成立、已改，`9af2527`）：
  R1 SAU 的 Shared_PGD 在 tape 里裁剪 → 贴边坐标梯度 0、`init max` 下卡在 +ε（改成投影；全贴边的输入上一步后移动的坐标 0% → 62%）；
  R2 SAU 不增强；R3 Adam(amsgrad, ε 1e-8)（与 torch 公式逐步差 ≤ 3.8e-4 个 lr；Keras 缺省 Adam 差 0.035 个 lr）；R4 SAU 参照探针在推理模式特征上拟合。
- **第二轮审查**（5 维 × 反驳验证，9 / 10 个 agent 完成；tests-adequacy 的验证撞上用量上限，那一维我逐条对过代码）：成立并已改的 ——
  - 配置 / 判定（D-105）：SAU 的 C-ft 对照不匹配（major）→ `cft-sau-*`；缺快照 / 缺配置被当成 J < 0.05 静默排除（major）→ `incomplete` / `insufficient`；
    精度过滤只看计入的快照 → 两个快照都看；V0 比 N-008 严（逐 edge、含攻击者 edge）→ 受害 edge 均值；分散布点的 V0 不过会让整个 P0 invalid → 只让该 run 的 D_H 失效；
    `grid_sha` 不核对 → 核对；必报项没进判定输出 → `aux` 原样转抄；步长约定没写进表 → 记偏差；AT-tgt 对 y = k\* 做定向下降 → 跳过。
  - 驱动：**「CCS-clu」聚的是私有统计量**（major）—— head_first 下私有统计量在 head 阶段末截取（`hier_fedrep._head_phase`），只见干净批次，CCS 聚的是上传物 → 改名 `bn_private_clu`（D-105 ④）；
    `own` / `ck` / `head_diag` 与主配置用不同随机数 → 复用主评估的加固权重、探针子种子配对（测试：给 head_init = 主配置拟合的探针时两次加固逐位相同）；
    优化器每次新建 → `jit_compile=False`（GPU 上缺省 XLA 每次重编译；CPU 上 Keras 自己关掉，测试改查源码）。
  - 测试（假绿）：扰动预算测试的探针 head 全零 → 梯度为 0、什么都不动也能过 → 改随机 head 并断言扰动真的动了；各目标的训练 loss 没钉 → 逐目标对手写组合；
    V0③ 在小世界里比的是 0 == 0 → 加 margin；判定的两快照语义、缺数据、grid_sha、分散布点 V0 的反向锚点；配置清单守卫改钉全部内容的 sha；全局 RNG 的 AST 扫描扩到全部新模块（带植入的正对照）。
  - 反向锚点实测：判定的 5 条新测试在旧 `p0_verdict.py` 上红；AT-tgt 掩码、head_diag 配对在旧 `core.py` 上红；R1–R4 的 4 条在改动前的 `core.py` 上红。
  - 判为不成立：`accuracy_bound` 下不报 C-ft 归因（N-008 只要求最佳配置的归因）。
- **第三轮审查**（判定规则 + 测试充分性，各配反驳验证，4 / 4 完成）：
  - 规则（两条，验证后都是 minor）：`judge` 不核对冻结清单 → 现在对同一份 s42 screen 重跑 `select`，名单或状态不同即 `invalid`，并核对 `--grid` 与冻结文件的 sha；
    V0 在分散布点漏掉「一边 None」→ 一律判不过（分散布点的 A 仍是逐 edge 平均、不是池化 —— 只影响只报告的 D_H，记为偏差）。
  - 测试（6 major + 3 minor，全部补上，每条都做了变异 / 反向检验）：攻击方向（无目标 / 定向 PGD 翻符号 → 红；TRADES 钉第一步 = 起点 + α·sign(∇KL)，翻符号 → 红）；
    SAU 掩码（逐行可控的 logits 造出「都没翻 / 只主模型 / 只参照 / 都翻但不同类 / 共享」五种行，去掉任一掩码或「同类」条件 → 红）与提前停；
    参照模型 = 传入的权重而不是模板；head_init 真的被用；判定的 7 个区分用例（两快照、逐 seed 过滤、未冻结配置挡住 kill、max over 快照、accuracy_bound 条件、D_H 值与符号、分散布点缺配置）；
    kill 边界用浮点上恰好 0.2 的构造；HDBSCAN 的构造参数；`main()` 端到端（screen / frozen / --only / sha 不符 / V0 不过 → 退出码 3）；集中布点的受害 edge 路径；
    J_t 在三个真实 SNAP metrics.json 上 = F-094 的值。
  - **没补的**：`build_world` 没有在本地真跑（要 100 个 ResNet-10 客户端）；它与 `main.run_experiment` 的调用顺序由 AST 测试守着，
    真正的检验是 GPU 上的 V0 —— 世界建错，V0 的 ② ③ 对不上 run 记录的值，作业停（N-008：`invalid`）。
- **私有统计量的 HDBSCAN 也会误剔良性端**（`tests/test_bn_cluster.py`）：官方参数（`allow_single_cluster`、`min_samples = 1`）在紧凑的合成良性群上也把一部分标成噪声（sklearn 1.9：7 / 20）→ P0 报 FPR，不是实现错误。
- **没有证据的**：探针 head 与受害端私有 head 的迁移缺口（语义 diff A2，`head_diag` 读数会给出）；比例步长相对官方绝对步长的效果差别；小预算 AT 在 GPU 上的墙钟（探路作业回传）。

## 2026-10-16（P0 GPU 探路回传）

### F-097 `confirmed`（数据，`analysis/p0_probe/SNAP__collocated__s42.screen.json`，探路作业 `--only pgd-e8-frozen-low --rounds 6`）—— V0 在 GPU 上成立；但加固在 15 步内发散成 NaN，原因还没有证据

- **V0（第 6 轮）通过**：离线 FedAvg 与快照 global **逐位相同**（max |差| = 0）；|A 离线 − A 记录|：云聚合后点 1.6e-5、第 t 轮全量点 4.1e-4（容差 0.01；逐 edge 最大差 3.3e-5 / 4.7e-4）。
  → 离线世界的重建、快照复原、评估复用在真实 SNAP 上是对的（F-096 只在 CPU 小 HFL 上验过）。不是逐位相同（ASR 在 1e-4 量级上有差），在容差里。
- **加固发散**：4 个 edge 的 `pgd-e8-frozen-low`（lr 0.0786 = 客户端 body 第 6 轮的 lr）第一批的 loss 12.8–20.7（ln 10 = 2.3），15 步后 loss 为 NaN；
  NaN 权重被照样评估 → 预测塌成一个类：受害 ASR 1.0、pm_acc 0.10 —— **看起来像「加固反而让后门更强」，其实是数值发散**。现在驱动遇到发散不评估、记 `diverged`，判定把它当 `incomplete`（`6c150ad` 之后的改动，见本条末）。
  探针在干净集上的精度 0.972–1.0（500 张、10 类，可分）。
- **原因的两个候选，现有数据分不开**（探路只跑了 bn=frozen 一个配置，只记了首末 loss）：
  ① bn=frozen：body 在推理模式 BN（固定的 moving 统计量、不做 batch 归一化）下用客户端的 lr 训练 —— 客户端从来不这样训；
  ② 探针 head 的 logit 尺度：对抗样本上的 CE 12–21。合成特征上的 numpy 复算（500 × 512、可分）给出的 top-2 margin 约 8 —— CE 对 logits 的梯度有界，**光凭「探针过度自信」解释不了 15 步就 NaN**，这条更弱。
  ⇒ 下一个诊断作业只加固、**不评估**（`--harden-only`），记每步 loss / body 与探针的梯度范数 / 第几步发散，比 cft / pgd × bn {frozen, train}。
- **附加读数**（第 6 轮，只报告）：触发器范数 ‖δ‖∞ = ‖ξ‖∞ = 4.0/255、‖δ+ξ‖∞ = 8.0/255（像素）→ ε = 8/255 恰好覆盖 δ+ξ（SAU 的前提成立）；
  `bn_private_clu`：E0 的 TPR 0.30、FPR 0.54，受害 edge FPR 0.38–0.42 → 私有统计量分不开恶意端（与 D-105 的预期一致：私有统计量不含投毒信号）；
  阻尼 λ = 0.5 / 0.25 的受害 ASR 0.288 / 0.236、oracle 隔离 0.355（基线 A(G) = 0.428、J_6 = 0.231 → 第 6 轮全量点约 0.197）。
- **墙钟**：一个 low 配置每 edge 加固约 5.05 s、评估约 35 s → 按前向反向次数外推，s42 screen 全部 36 个配置 × 2 个快照约 **2.6 GPU-h**（PLAN 估 3–4 h）。**没有实测高预算配置**。
- **读到了什么**：这个探路作业打开了 s42 第 6 轮的基线与附加读数（A(G)、阻尼、oracle）—— 都与配置无关，选择规则不读它们；唯一的配置读数是 NaN，没有信息。
  修发散的方案要只凭诊断作业的训练轨迹定（不读 ASR），定了之后改 `p0_configs.yaml` 是改预注册 → 留 DECISIONS。

### F-098 `confirmed`（数据，`analysis/p0_probe/SNAP__collocated__s42.screen.diag.json`，`--harden-only`，第 6 轮，不读 ASR）—— 发散的原因是「推理模式 BN + 客户端 lr」；train 模式不发散，但这个 lr 也把干净拟合破坏了

- V0 再次通过（与 F-097 同值）—— 集群跑的是 `f8cad48` 之后的代码。
- 2 × 2（lr = 客户端 body 第 6 轮的 0.0786，1 epoch = 15 步，4 个 edge）：

  |  | bn=frozen | bn=train |
  |---|---|---|
  | `cft`（干净微调，无对抗） | **4 / 4 发散，第 3 步**：loss 0.41–0.96 → 35–467 → 1.6e7–3.1e13 → NaN；body 梯度范数第 1 步 12–30 → 第 2 步 307–3 169 | 不发散；loss 第 0 步 0.30–1.09 → 第 1–5 步升到 10–18 → 第 15 步 4.3–8.6；body 最大改动 0.08–0.17 |
  | `pgd-e8` | **4 / 4 发散，第 2–3 步**（同 F-097） | 不发散；loss 12.5–17.3 → 3.5–9.2 |

- **结论**：干净微调在 frozen 下一样发散 → 不是对抗 loss / 探针尺度的问题，是**推理模式 BN（固定的 moving 统计量、没有 batch 归一化）下用客户端 lr 训练 body** 会爆（F-097 的候选 ①）。
  train 模式虽不发散，但**在干净数据上把 loss 从约 0.5 抬到 4–8**（用临时探针 head 量的）→ 这个 lr 对「几步微调 + 新拟合的探针」太大，语义 diff A6「lr = 客户端 body 的 lr」不成立。
- **没有证据的**：探针 loss 变差对应多少 fresh pm_acc 下降（受害端用的是自己的私有 head）—— D-106 的 lr 扫描直接量 pm_acc。
- 读到了什么：只有训练轨迹与 V0（与 F-097 相同的基线数），没有任何配置的 ASR。

### F-099 `confirmed`（数据，`analysis/p0_probe/SNAP__collocated__s42.screen.lrscan.json`，s42 第 6 轮，只算精度、不读 ASR）—— D-106 规则选出 train ×0.1、frozen ×0.03；但在选出的 lr 下，对抗训练本身让池化精度掉 0.12–0.17

- V0 同 F-097（集群跑的是 `6b839ab`）。G 的池化 fresh pm_acc = 0.658（逐 edge 0.668 / 0.698 / 0.575 / 0.691）。
- 池化 pm_acc 下降（G − G′，4 个 edge 都加固后 FedAvg；— = 至少一个 edge 发散）：

  | 倍数（lr） | cft-train-low | cft-train-high | pgd-e8-train-low | cft-frozen-low | cft-frozen-high | pgd-e8-frozen-low |
  |---|---|---|---|---|---|---|
  | ×1（0.0786） | 0.538 | 0.516 | 0.522 | — | — | — |
  | ×0.3 | 0.033 | 0.083 | 0.268 | — | — | — |
  | ×0.1 | **0.002** | **0.015** | 0.156 | —（2 / 4 edge 发散） | —（2 / 4 edge，第 7 / 12 步） | 0.371 |
  | ×0.03 | −0.001 | 0.003 | 0.046 | **−0.006** | **−0.000** | 0.171 |
  | ×0.01 | −0.001 | 0.000 | 0.008 | 0.001 | −0.000 | 0.083 |

  SAU（官方 Adam 1e-4，不随倍数变）：train 0.029、frozen 0.122，都不发散。
- **D-106 规则的选择**：train → **×0.1**（×0.3 时 cft-high 降 0.083）；frozen → **×0.03**（×0.1 时 cft 有 edge 发散）。→ D-107 写进 `p0_configs.yaml`。
- **在选出的 lr 下，对抗部分代价很大**：pgd-e8 train ×0.1 降 0.156、frozen ×0.03 降 0.171（池化门槛 0.02）；SAU train 0.029、frozen 0.122。
  ⇒ 预计 s42 screen 里**很可能没有对抗配置过精度过滤**，结果会落在 `accuracy_bound` / `kill_pre` / `inconclusive`（看 R_H）。这是 P0 要回答的问题本身，不是工具的问题。
  直觉解释（**没有证据**）：500 张干净图、几十步、全部 4 个 edge 都做 → 共享 body 被拉向「对 8/255 扰动鲁棒」的方向，而受害端的私有 head 没有跟着调。
- **没测的**：高预算的 frozen 对抗配置（75 步 × 10 步 PGD）在 ×0.03 下稳不稳 → 发散的处理规则 D-107 先写好。
- 读到了什么：只有精度与训练轨迹（加上 F-097 已读过的基线），没有任何配置的 ASR。

## 2026-10-28（P0 s42 screen 回传）

### F-100 `confirmed`（数据，`analysis/p0/SNAP__collocated__s42.screen.json`；选择 `analysis/p0_frozen.json`）—— 能砍掉灌入的配置都过不了精度过滤；按 N-008 的逻辑，标签已经确定是 `accuracy_bound`

- 闸：两个快照 V0 都过（FedAvg 逐位；ASR 差 ≤ 4.2e-4）；`grid_sha` = 已提交的 `p0_configs.yaml`（`9019900`）。
  基线：第 6 轮 A(G) 0.428、J 0.231、pm 0.658；第 15 轮 A(G) 0.925（接近饱和）、J 0.423、pm 0.788。
- 发散（D-107：= 精度过滤不通过）：`trades-e8-b6-frozen-low`（两个快照）、`trades-e8-b6-frozen-high`（第 6 轮）。
- **R_H 与精度代价**（池化 ΔMTA 门槛 0.02，每 edge 0.04）——砍得动的配置，代价都远超门槛：

  | 配置 | R_H @6 / @15 | 池化 ΔMTA @6 / @15 | 每 edge 最大 |
  |---|---|---|---|
  | `trades-e8-b6-train-low` | 1.81 / 0.91 | 0.535 / 0.434 | 0.641（模型坏掉：ASR 0.009、margin −87）|
  | `pgd-e4-train-low` | 0.83 / 0.88 | 0.111 / 0.089 | 0.136 |
  | `pgd-e8-frozen-high` | 0.31 / 0.78 | 0.312 / 0.413 | 0.493 |
  | **`sau-e8-train-high`**（离门槛最近）| **0.58 / 0.44** | **0.032 / 0.023** | 0.051 |
  | `sau-e8-train-low` | 0.39 / 0.19 | 0.029 / 0.015 | 0.045 |
  | `sau-e51-train-low`（唯一过滤通过）| −0.13 / 0.03 | −0.003 / 0.005 | 0.025 |

  bn=frozen 的配置多数 R_H 为负（加固后受害 ASR 反而更高），如 `sau-e51-frozen-low` −1.32 / −0.14、`pgd-e8-frozen-low` −0.53 / −0.11。
- **归因（C-ft）**：SGD 的 C-ft R_H ≈ 0（−0.39 … 0.03，干净微调不砍灌入，与 Bad-PFL 原文「FT 无效」一致）；
  `cft-sau-train-high` 0.13 / 0.08 → `sau-e8-train-high` 的对抗部分贡献约 0.45 / 0.36。
- **选择**（N-008，只在过滤通过的 AT 配置里取前 3）：只有 `sau-e51-train-low` 一个，冻结清单 = `[sau-e51-train-low]`。
- **标签已由 s42 决定**（判定脚本的逻辑，`harness/p0_verdict.py:judge`）：
  `go_online` 要求冻结配置在三个 seed 上 R_H ≥ 0.5 → s42 已是 −0.13，不可能；
  `kill_pre` 要求 s42 全部 AT 配置两个快照 R_H < 0.2 → `pgd-e4-train-low` 0.88，不可能；
  `accuracy_bound` = s42 上 R_H ≥ 0.5 的配置（`trades-e8-b6-train-low`、`pgd-e4-train-low`、`trades-e8-b6-frozen-high`）全都过不了过滤 → **成立**。
  ⇒ 冻结阶段（s43 / s44 / 分散布点）**改变不了标签**，只会补 `sau-e51-train-low` 的逐 seed 数与 D_H。N-008：`accuracy_bound` → 用户定。
- **附加读数（只报告，不进判定；零训练）**：
  - **阻尼**（每个 edge 只吃 λ 份云更新：w_e + λ(G − w_e)）：λ = 0.25 → R 0.83 / 0.77，池化 ΔMTA −0.010 / 0.021（每 edge 最大 0.030 / 0.054）；
    λ = 0.5 → R 0.61 / 0.50，池化 ΔMTA −0.022 / −0.007。**在第 6 轮，λ = 0.5 的阻尼比任何 AT 配置都更好**（R 0.61 且精度反而升）。
    注意：这是一次性反事实，阻尼在线上会拖慢正常的收敛 —— 长期代价**没有证据**。
  - oracle 隔离（G 去掉 E0）：R 0.32 / 0.90，ΔMTA 0.028 / 0.016。
  - 私有 BN 统计量聚类：E0 TPR 0.3 / 0.4、各 edge FPR 0.32–0.54 → 分不开（D-105 的预期）。
  - 触发器 ‖δ+ξ‖∞ 两个快照都是 8.0/255。

## 2026-10-28（SA-C 会话：CCS 的语义对照与实现）

### F-101 `confirmed`（代码核对 + 本地可复现的计算）—— CCS 官方代码本身跑不起来；对抗样本在标准化空间里做且带 clamp；聚类与官方 `hdbscan` 包逐例相同，且在彼此很近的良性端上也会剔掉一部分

- 来源：`chenjian0924/paper-code @ b4b8653`，`CCS/`（只读 clone，未进 git；接 F-090）。
- **① 发布的代码跑不起来，两处**：
  - `src/client/fedavg.py:281` 调 `perturb_input(..., perturb_steps=15, ...)`，`src/defence/AT/AT.py:17` 的签名**没有** `perturb_steps` → `TypeError`；
  - `hdbscan_detect.py` 取 `package.get("model_params", package["client_gradient"])`。client 包里没有 `model_params`（`client/fedavg.py:218-258`）；`client_gradient` 只含 `named_parameters()` 里名字带 `base` 的键（`:51`、`:246-251`）—— BN 的 running_mean / var 是 buffer，**不在里面** → 统计量列表为空，`torch.cat([])` 报错。
  ⇒ 产出 CCS 表 1 的不是这一版代码。本仓库照的是它的**意图**（D-109），不可能逐位对拍。
- **② 对抗样本在标准化后的输入空间里做**：训练 transform 含 `Normalize`（`server/fedavg.py:434-440`，std 用的是常见的 (0.2023, 0.1994, 0.201)），而 `l_inf_patch` 分支的贴片 U(0,1)、eps 0.5、`clamp(0,1)` 都写在这个空间里。
  按常数换算（未在真实图上实测）：标准化的 [0,1] = 像素 [μ, μ+σ] ≈ [0.48, 0.69]，eps 0.5 ≈ 26/255（本仓库 σ ≈ 0.25 → ≈ 31/255）；
  clamp 把比均值暗的、比 μ+σ 亮的像素整个夹过去 —— 对抗分支看到的是一张低对比度的图。原文没写这一层。
  TF 测试 `test_one_step_equals_the_numpy_reference[input]` 里，掩码外的像素离**原图**最多 3.5（标准化单位），离 clamp 过的原图 ≤ eps —— 就是这个效应。
- **③ DPR 退化成固定圆盘**：`l_inf_patch` 分支 36 个扇区半径全取 10 → 圆心 ∈ [10, 22) 的半径 10 的 36 边形（约占 32×32 图的 30%），里面 U(0,1) 噪声；然后对**整张图** 15 步 sign 梯度（步长 0.1）上升无目标 max-margin、投影到初值 ±0.5。
  numpy 移植 `defense/ccs_spec.dpr_mask`：像素格点数与 36 边形面积 ½·36·r²·sin 10° ≈ 312.6 相差 < 15（`tests/test_ccs_spec.py`）。
- **④ BN 模式**：`perturb_input` 先 `model.eval()`；损失里的对抗前向也在 eval 模式（`model.train()` 在它**之后**才调，`client/fedavg.py:283-285`）→ 对抗分支不更新 BN 统计量。KL = `kl_div(log_softmax(adv), softmax(clean), batchmean)` = KL(p_clean ‖ p_adv)，两边都不 detach；AT 路径另有 `clip_grad_norm_(10)`。
- **⑤ 投影网络**：模型是作者自己加了投影层的 `PResNet18`（`models.py:556-601`）：avgpool → project（512→256→512）→ classifier；FedPer 下 project 与 classifier 都私有（`client/fedper.py:10-12`）；MMD 算在 project 输出上。MMD 函数不在仓库里（F-093）。
- **⑥ 系数三套**：原文 β = γ = 0.01；`config/defaults.yaml` 的 `AT_alpha 0.1 / AT_beta 0.01`；`src/utils/constants.py:120-121` 的 0.1 / 5（合并顺序未核）。攻击端走 `attack()` 不走 `fit()`（`client/fedavg.py:139-142`）→ 非自适应。聚合按样本加权（`server/fedavg.py:727-736`）。
- **⑦ 聚类：sklearn 的 HDBSCAN 与官方的 `hdbscan` 包逐例相同**（本地实测：同一组余弦距离矩阵、官方参数，300 组合成数据标签全部一致；`tests/test_ccs_spec.py::test_labels_match_the_official_hdbscan_package`，装了 `hdbscan` 才跑）。
- **⑧ 官方参数（min_samples = 1、min_cluster_size = n // 2 + 2）在彼此很近的良性端上也会剔掉一部分**：10 个只差 1% 噪声的向量里 2–3 个被判成噪声（两个库都是）。
  → CCS 的聚类有**固有的良性剔除率**，与攻击无关；CCSF 的 `rejection.fpr` 要按这个读（不是实现的 bug）。在真实上传统计量上 FPR 有多大，**没有数据**。
- 对计划的影响：语义对照表与取舍见 D-109；对抗样本空间的取舍等离线探针（`analysis/ccs_space_probe.py`）回传后由用户定（CCSF 挂 `ccs-space`）。

## 2026-10-29（CCS 对抗样本空间探针回传）

### F-102 `confirmed`（数据，`analysis/ccs_space_probe.SNAP__collocated__s42.r15.json`；零训练，24 个良性端 × 32 张，四个 edge 都有）—— 标准化空间（照官方代码）与像素空间的对抗样本都完全骗过模型，像素空间约强 1.9 倍，同一量级；clamp 动到 61% 的像素；机时估计更正为约 6–7 GPU-h / run

- 量（端均值；变体 = 空间 × 步数；同一端的四个变体用同一组掩码与噪声）：

  | | input-15 | pixel-15 | input-0 | pixel-0 |
  |---|---|---|---|---|
  | clamp_frac（原图落在工作空间 [0,1] 外的像素比例） | 0.608 | 0 | 0.608 | 0 |
  | dpx_mean / dpx_out_mask（[0,1] 像素上的平均位移 / 掩码外） | 0.168 / 0.138 | 0.269 / 0.245 | 0.071 / 0.002 | 0.094 / 0.003 |
  | adv_acc（原图精度 0.796） | 0.000 | 0.000 | 0.452 | 0.481 |
  | margin_adv | 21.5 | 39.9 | 0.00 | −0.06 |
  | KL(p_clean ‖ p_adv)（端间范围） | 17.8（12–32） | 33.4（25–62） | 1.01 | 0.93 |
  | feat_rel / MMD | 2.58 / 0.50 | 4.92 / 0.46 | 0.58 / 0.46 | 0.58 / 0.56 |
  | 每批 32 张的墙钟（s） | 0.038 | 0.039 | — | — |

  steps = 0 的变体不经过 clamp（clamp 在步内），它的 clamp_frac 只表示「原图有多少像素会被夹」。
- 读法：
  1. 两种空间 15 步后都让 fresh-PM 全错；像素空间的 KL / margin / 特征位移约是标准化空间的 1.9 倍，掩码外位移 1.8 倍 —— **同一量级**。
  2. 标准化空间里 clamp 动到 **61%** 的像素（F-101 ② 按常数估的「约一半」偏低）：比均值暗的点被抬到均值、比 μ+σ 亮的被压低 → 对抗分支看到的是低对比度的图。input-15 掩码外平均位移 0.138，eps 本身只相当于约 0.12 像素 → 掩码外的位移大半来自 clamp。
  3. 本仓库主干的预处理与 CCS 作者同类（标准化后喂模型），只差 std 常数（0.247 vs 官方 0.2023）→ input 版的幅度差约 20%，远小于 input / pixel 之间的约 2 倍；大的差别来自 CCS 代码把 [0,1] 的常数用在标准化空间里，不是主干预处理。
  4. **探针只量对抗样本本身，不量防御效果**：空间不同会不会改变 CCSF 的 ASR，**没有证据**。
  5. MMD 在四个变体里都约 0.46–0.56，几乎不随扰动强弱变化 → rbf 带宽 {10, 15, 20, 50} 相对投影特征的距离偏小、核值饱和，MMD 项近似常数（**推断，未验证**；梯度没有量）。KL 项不小：0.01 × KL ≈ 0.18（input）/ 0.33（pixel），与 CE（约 0.6）同量级。
  6. **机时更正**：造对抗样本 0.038 s / 批。CCSF 每个 ccs run 的 body 批数 ≈ 300 轮 × 10 端 × 约 14 批 × 5 epoch ≈ 21 万 → 约 +2.2 h，加上训练步多一次前向，估 ccs-full ≈ off + 2.5–3 h ≈ **6–7 GPU-h / run**。
     SA-C 交接里写的「15–25 GPU-h / run」是没有数据的高估，作废。
- 对计划的影响：D-110（CCSF 只跑 input）。
- 顺带发现（不在 CCS 范围内，**没改**）：`fedavg/config_validate.py:387` 的 `elif defense != "none":`（「backdoor.enabled=false 但 defense.name=…」警告）挂在 `if stop_cfg:` 后面，
  而不是攻击开关的判断后面 → 凡是 `stopping: null` 又开了防御的配置都会打这条**假警告**（CCSF__ccs-full 就是：攻击明明开着）。只是警告，不改行为；修法一行，留给用户定。

## 2026-10-29（P1 回传：在线上传前加固 × 集中布点 × s42）

### F-103 `confirmed`（数据，`results/P2/P1/P1__col-pre__s42.metrics.json` vs `results/P2/SNAP/SNAP__collocated__s42.metrics.json`；判定 `analysis/p1_verdict.json`）—— 按 N-008 P1 段判 `go_main_col`：受害 edge 末 10 点良性 ASR 0.995 → 0.519（ΔV 0.476），ΔMTA 0.0086；但这是**筛选用单 seed**，灌入幅度只降约 21%，攻击者 edge E0 基本没被碰到

- **闸**（D-108 ③）：arm 第 1 云轮 `[HardenAgg] unhardened = 06889698b033` = SNAP `[Checksum] Round 1`（第一次加固之前逐位相同）；`[Harden]` 240 行 = 4 × 60，`diverged_at` 全为 None；两边 exit 0、`client_failures` / `errors` 为空。
- **判定量**（`p1_verdict.py`，臂 / 对照）：

  | 量 | 臂（sau-e8-train-high） | 对照（SNAP） | Δ（对照 − 臂） |
  |---|---|---|---|
  | V（受害 edge 良性 ASR，末 10 点） | 0.519 | 0.995 | **0.476**（门槛 0.15） |
  | margin_v（受害 edge margin p50） | 0.02 | 6.69 | 6.67 |
  | jump_half（g = 2 … 30 的 Δ_jump 均值） | 0.174 | 0.220 | 比值 0.79（门槛 ≤ 0.5，**没过**） |
  | MTA | 0.853 | 0.862 | **0.0086**（门槛 ≤ 0.02） |
  | B0（攻击者 edge 良性 ASR） | 0.950 | 0.998 | 0.048 |
  | P | 0.591 | 0.996 | 0.405 |

  天花板规则未触发（臂 V < 0.95）→ 走 ΔV 那一项；`go_main_col` 只靠 ΔV 成立，jump 那一项不成立。
- **轨迹**（受害 edge 均值，`victim_by_round` = [对照, 臂]）：前 8 云轮两边相近（0.08–0.23）；对照第 40 云轮起 ≥ 0.98 饱和；
  臂在第 31–33 云轮到峰值约 0.77–0.86，之后**不饱和、回落并大幅振荡**（第 46–60 云轮 0.24–0.65；第 23 / 50 云轮单点 0.077 / 0.235）。
  → 末 10 点 V 不是一个平台值，逐点抖动远大于 F-091 的 σ≈0.09；**单 seed 上 ΔV 的大小没有可外推的证据**。
- **逐 edge**（末 10 点良性 ASR，对照 → 臂）：E0 0.998 → 0.950、E1 0.998 → 0.488、E2 0.990 → 0.469、E3 0.997 → 0.600。
  E0 几乎不动，与 PLAN E3 的事先预测一致（上传前加固碰不到攻击者 edge 内部的良性端）。末轮 pm_acc 逐 edge 变化 −0.011 … +0.016。
- **开销**：加固 12 781 s（每次约 53 s）；`round_time` 合计 9 875 → 21 323 s（+116%），连评估的墙钟约 +78%；显存峰值不变（16.76 GiB）。
- 读法：
  1. 机理上的读法（**推断**）：jump 只降 21%、但 V 降了一半 → 收益主要不在「每次云聚合灌进来的量」，而在**周期内 edge 自清洁之后、下一次灌入之前**积累不上去（margin 从 6.7 掉到约 0）。这与 P0 的设定（只量一次灌入的削减率 R_H）不是同一个量，P0 的 `accuracy_bound` 不能直接套到在线结果上。
  2. 精度：在线 ΔMTA 0.0086 低于 P0 离线同配置的池化降 0.032 / 0.023（F-100）—— 但两者是不同的量（P0 是快照上一次性回放后的池化精度，这里是 run 末的 MTA），**不能据此说在线精度代价更小已被证实**。
  3. 下一步按 PLAN：`go_main` 的臂进 s43–44 复验（PLAN 「P1 / CCSP 判 go_main 的臂 × s43–44」）；分散布点（H3）与 A-every 仍未做。由用户定。

## 2026-10-29（D-111 批次回传：H1 col-pre s43 / s44、P1 dist-pre s42；CCSF 两格）

### F-104 `confirmed`（数据，`analysis/h1_pre_verdict.json`；PLAN §4 H1 + D-101，脚本 `harness/h1_verdict.py` 先于数据 commit 于 `451dd23`）—— A-pre 集中布点 3 seed 判 `protects_victims`；攻击者 edge 的良性端 `no_effect`；每次云聚合的灌入量基本不变

- 闸：三个配对都通过（第 1 云轮 `[HardenAgg] unhardened` = 对照 `[Checksum] Round 1`；`[Harden]` 240 行、0 发散；exit 0、`client_failures` 为空）。
- 逐 seed（对照 − 臂；V / B0 / MTA 都是第 51–60 云轮的均值）：

  | seed | V 对照 → 臂 | ΔV | ΔB0 | ΔMTA | jump 比值（臂 / 对照） | 臂的受害 margin p50 |
  |---|---|---|---|---|---|---|
  | 42 | 0.995 → 0.519 | 0.476 | 0.048 | 0.009 | 0.79 | 0.02 |
  | 43 | 0.997 → 0.475 | 0.522 | 0.002 | 0.013 | 1.01 | −0.17 |
  | 44 | 0.996 → 0.769 | 0.228 | 0.000 | 0.015 | 0.96 | 0.73 |

  ΔV 均值 0.409、最小 0.228（门槛 0.15 / 0.10）；ΔMTA 均值 0.012、最大 0.015（门槛 0.02 / 0.025）→ `protects_victims`。`treats_E0` = `no_effect`。天花板规则不适用。
- 读法：
  1. seed 间差别大（0.23 – 0.52），s44 的臂 V 0.77。3 seed 是用户定的结案数（D-112），**没有 5 seed 的 CI**。
  2. jump 比值 0.79 / 1.01 / 0.96 → 每次云聚合灌进受害 edge 的量**没有被砍小**；ASR 被压低发生在两次灌入之间（与 F-103 读法 1 同向）。这一条只是从 jump 与 V 的对比得出的**推断**，没有直接测量机制的量。
  3. 机制项（Δexcess，要 ρ=0 的 floor run）没有做 → 结论只能写「压低受害 edge 的 ASR」，不能写「清除后门」。
  4. E0 不受保护：与 PLAN E3 的事先预测一致。

### F-105 `confirmed`（数据，`analysis/p1_dist_verdict.json`；N-008 P1 分散规则）—— A-pre 分散布点判 `marginal`：ASR 降得动，但精度代价超门槛，末段振荡；**只记录、不进 H3**（D-112）

- 闸通过（同 F-104）。P（全体良性端池化 ASR）0.998 → 0.713（ΔP 0.285 ≥ 0.10）；margin p50 5.23 → 1.40（Δ 3.83 ≥ 2.0）；ΔMTA **0.030 > 0.02** → 不是 `go_main_dist`；也不满足 stop → `marginal`。
- 逐轮（对照 → 臂，全量评估点的 P）：第 1–15 云轮两边相近（第 15 轮 0.70 → 0.76，臂反而略高），第 20 轮起分开（0.87 → 0.65）；
  第 51–60 轮臂依次为 0.77 / 0.75 / 0.72 / 0.83 / 0.41 / 0.81 / 0.78 / 0.41 / 0.88 / 0.77 → 0.713 不是平台值。
- 精度：臂的 MTA 末段持续低于对照（第 60 轮 0.797 vs 0.870）；末轮逐 edge pm_acc 降 0.04–0.12（E0 0.853 → 0.815、E1 0.901 → 0.820、E2 0.836 → 0.716、E3 0.891 → 0.837）—— 集中布点的逐 edge 降幅只有 −0.011 … +0.016（F-103）。
- 与 PLAN H3 的事先预测（A-pre 在分散布点 `no_effect`）相比：ASR 降幅比预测大，但精度代价超门槛。为什么分散布点下精度代价比集中布点大一倍多，**没有证据**。

### F-106 `confirmed`（数据，`results/P2/H1/H1__col-pre__pack-k2-s43.gpu.json`、`results/P2/P1/P1__dist-pre__pack-k1-s42.gpu.json`）—— 两个加固 run 装一张卡（K=2），墙钟只比单跑多约 3%

- K=2 包 `wall_s` 27 500 s（7.6 h）；单跑（k=1）26 617 s（7.4 h）。真实显存峰值 16.96 / 16.95 GiB / run，无 OOM、无主机 OOM。
- 由此，D-111 ④ 担心的「加固 run 并行会被 GPU 计算拖长墙钟」在 K=2 上不成立。K=3 仍没有测过（SNAP 的不加固 K=3 包是 4.35 h）。
- 单跑那份 gpu.json 的 `util_mean` 是 5.3%，与 K=2 包的 66.7% 对不上；原因不明，只记录、不解释。

### F-107 `confirmed`（数据 + 代码）—— CCSF ccs-full 一启动就崩：`CCSDefense.set_stat_indices` 在生产代码里从没被调用；L1 绿是因为测试替它调了（陷阱 #18 同类）

- `results/P2/CCSF/CCSF__ccs-full__s42.metrics.json`：exit 1、0 个评估点。第 1 个 edge 轮聚合时（`hier_fedrep.run_edge_round` → `robust_mean` → `CCSDefense.aggregate`）抛
  `RuntimeError: CCSDefense：没有注入 BN 统计量索引`。
- 原因：`defense/ccs.py` 写着「main.py 建好 edge server 后注入」，但 SA-C 没有在 main.py / edge server 里加这一步；全仓库唯一的调用在 `tests/test_ccs_tf.py`（手工注入）。
- 修法（D-112）：`EdgeServerBase.__init__` 建好防御后，若防御有 `set_stat_indices` 就用 `get_bn_stat_indices(self.model)` 注入。测试里的手工注入删掉，让这条 TF 测试走生产路径；
  另加本地 AST 守卫：`defense/ccs.py` 以外的 `fedavg/` 代码里必须有这一调用。
- CCSF off 正常（exit 0、60 个评估点、无 client_failures），不重跑。ccs-full 修好后先 30 分钟探路、再整跑（用户 2026-10-29）。

## 2026-10-29（CCSF ccs-full 修好接线后的两次运行：探路 job 4618401、整跑 job 4621764）

### F-108 `confirmed`（数据 + 本地实测）—— 主机内存：24G 不够，48G 下峰值约 31 GiB；大头是「每个客户端一整套模型和计算图」，CCS 端每个约是普通端的 1.8 倍

- 第一次 30 分钟探路（job 4617499，24G）：约 26 分钟被 cgroup OOM 杀掉（exit 137，`oom_kill event`）。
  48G 下：探路 1 h 的 MaxRSS 32 639 808 K（31.1 GiB），整跑 31 001 152 K（29.6 GiB），都没有 OOM。CCSF off 在 24G 下跑完。
- 本地实测（tensorflow-cpu 2.15，真实 `resnet10_torch_proj`、32×32、batch 32、steps 15；每多建一个客户端、跑两步训练后主机 RSS 的增量）：
  普通 FedRep 端中位 **118 MiB / 个**，CCS 良性端中位 **211 MiB / 个**。模型参数本身只有约 20 MB，其余是每个客户端各自的 Keras 模型对象、优化器、编译好的 `@tf.function` 图；
  CCS 多出的部分来自对抗样本的 15 步 PGD：在图里展开成 15 份前向 + 反向，每个良性端各编译一份。外推：90 个良性端多约 8 GiB，与「16.7 GiB 基线 + 8 GiB > 24G」对得上（GPU / XLA 下主机侧的量**没有直接测**）。
- 根本原因是仿真的写法：`main.py:575` 给 100 个客户端各 `clone_model` 一份完整模型，而训练是串行的（`n_workers=1`）。根治 = 所有客户端共用一个训练模型、只存私有部分 —— 是一次大改，单开会话。
- 顺带：作业脚本没有 `PYTHONUNBUFFERED`，stdout 经管道给 `tee` 是块缓冲，进程被 SIGKILL 时缓冲区里的行全丢（第一次探路的日志里一行训练输出都没有）。临时办法：`sbatch --export=ALL,PYTHONUNBUFFERED=1`。

### F-109 `confirmed`（数据 + 官方代码核对）—— ccs-full 的主任务在第 10 轮起就垮了，第 54 轮良性端 NaN 后崩溃；官方代码核实：PResNet 投影层在分类路径上、FedPer 每步联合训练，与我们 FedRep 的分阶段训练是不同的优化问题

- **运行**：整跑（job 4621764，git `71b1255`，48G）在第 54 轮崩溃：良性端 25 的 loss 为 `nan` → 上传的 BN 统计量含 NaN → `HDBSCAN` 报 `np.nan values found in precomputed-dense`（exit 1）。metrics.json 留在 results/（status = failed）→ 2026-10-30 与 off 一起移到 `legacy/CCSF-fedrep/`（CCSF 改成 FedPer 后，新 run 会写同一路径，D-114 ⑥）。
- **崩之前主任务已经垮了**（off 对 ccs-full，同 seed）：

  | 轮 | off EM / PM | ccs-full EM / PM | off 良性 ASR | ccs-full 良性 ASR |
  |---|---|---|---|---|
  | 10 | 0.468 / 0.497 | 0.389 / 0.313 | 0.046 | 0.070 |
  | 20 | 0.587 / 0.623 | 0.539 / 0.358 | 0.187 | 0.072 |
  | 30 | 0.668 / 0.674 | 0.586 / 0.360 | 0.227 | 0.053 |
  | 40 | 0.687 / 0.741 | 0.619 / 0.346 | 0.346 | 0.183 |
  | 50 | 0.735 / 0.763 | 0.609 / 0.341 | 0.607 | 0.122 |

  PM 从第 10 轮起停在 0.34–0.37；ccs-full 的低 ASR **不能读作防御有效**（主任务学不上去，后门也学不进去）。原文 CCS 的 MTA 是 79.76%。
- **PM < EM**：edge 模型的 head 与投影层从不更新（`server/hier_fedrep.py` 只聚合 body，一直是初始化值）。ccs-full 里用这组没训过的 head 得 0.61，用客户端训过的私有 head 只有 0.34；off 里是 0.73 < 0.76。→ 坏的是「私有部分与共享 body 对不上」，不是 body 本身。
- **聚类**：每轮剔 3/10（`n // 2 + 2 = 7` 的上限），与 F-101 ⑧ 的固有良性剔除一致；会减慢训练，解释不了 PM 停在 0.34。
- **官方代码核实**（2026-10-29 重读 `chenjian0924/paper-code @ b4b8653`，`CCS/`）：
  1. `src/utils/models.py`：`PResNet(DecoupledModel)` = `base`（ResNet-18 主干到 `AvgPool2d(4)` + `Flatten`）→ `project`（`Linear(512, 256)` → ReLU → `Linear(256, 512)`）→ `classifier`；`PResNet18` 登记在 `MODELS`；`config/defaults.yaml` 的 `name: PResNet18`。
     原文 Eq. 5–7 提到「个性化投影网络 p_i」，但**没写它的结构、也没写它在分类路径上** —— 只能从代码看出来。
  2. `src/client/fedper.py`：state_dict 里名字含 `classifier` / `project` 的都私有；**不重写 `fit()`** → 沿用 `FedAvgClient`，所有参数每一步一起训。
  3. `config/defaults.yaml`：`local_epoch: 2`、`batch_size: 64`、`lr: 0.1`、`momentum: 0`。原文正文写「两步本地训练」，与代码的 2 个 epoch 对不上。
     **更正**：会话里一度说「原文每轮 2 步、对抗训练量差 27 倍」—— 按代码是 2 个 epoch，我们 body 5 epoch、batch 32：数据遍数约 2.5 倍、更新步数约 5 倍。
- **由此的语义差异**（D-109 ⑥ 只写了「FedRep vs FedPer」，没写到这一层）：原文里投影层与分类头每一步都和 body 一起、以 lr 0.1 被 CE + KL + MMD 训练；
  我们的 FedRep 里它们只在 head 阶段用干净 CE 训（每轮 1 epoch、lr 0.005 衰减），**KL / MMD 的梯度永远到不了投影层**，MMD 只能靠改 body 去迁就一个几乎不动的投影层。
  这是主任务崩坏的首要嫌疑，**没有消融证据**；NaN 的来源也还不知道。下一步：加 FedPer 方法后本地消融（D-113）。

## 2026-10-30（CCSA / CCSF 的 FedPer 版回传 + 为什么 Bad-PFL 在 FedPer 下没传到良性端）

### F-110 `confirmed`（数据：`results/P2/CCSA/*`、`results/P2/CCSF/*`、`legacy/CCSF-fedrep/*`；代码核对：CCS 官方仓库、`Huangru-114/CCS-reproduce`、`Huangru-114/Bad-PFL` 的非 main 分支）—— 主任务崩坏来自框架（FedRep）；FedPer + CCS 官方超参数下，攻击只植入了共享模型、没传到大多数良性端，复现闸判 `invalid_no_attack`；首要嫌疑是每轮本地训练量太小

- **有效性**：4 个 run 都是 exit 0、无 client_failures、攻击者都参与了、`[设定13] fedper=true | bn_stats=shared`。机时：CCSA off 0.35 h / ccs 1.0 h，CCSF off 1.82 h / ccs 2.98 h；显存 16.8–17.1 GiB / run，无 OOM。
- **消融**（第 30 轮 PM 精度，off → ccs）：FedRep 0.67 → 0.36（−0.31）；FedPer + 我们的超参数 0.75 → 0.71（−0.04）；FedPer + 官方超参数 0.63 → 0.58（−0.05）。
  → F-109 的主任务崩坏来自 FedRep 框架（KL / MMD 梯度到不了私有投影层），不是超参数，也不是实现 bug。FedPer 的 run 都没有 NaN。
- **CCSF 判定 `invalid_no_attack`**（`analysis/ccsf_verdict.json`）：P(off) = 0.387（不过滤 0.412）< 0.5；ccs 0.084。另报：ΔMTA 0.21（0.833 → 0.622，第 30 轮时只差 0.05，代价随轮数累积，原因没有证据）；剔除 TPR 0.56 / FPR 0.25。
- **攻击植入了共享模型，没传到良性端**（FedPer + 官方超参数，off）：第 150 轮起 GM ASR 0.994–0.997、攻击者自身 1.000，良性端只有 0.41（> 0.5 的良性端占 42%）。
  第 300 轮逐端：90 个良性端中 28 个 < 0.1、13 个 > 0.9，中位数 0.32；逐端 ASR 与 yt_clean（干净样本被判为目标类的比例）Spearman 0.71，与测试集大小 −0.46；yt_clean < 0.5% 的 39 个端平均 0.16。
  同一组客户端在 FedRep 下（legacy off）平均 0.86，良性端中位数 1.0。
- **CCS 官方代码里没有可用的 Bad-PFL**（`chenjian0924/paper-code @ b4b8653`）：`attack()` 里 `badpfl` 落到 else 分支，**训练时不投毒**；评估用的 `trigger_gen` 随机初始化、`gen_optimizer` 全仓库无人调用；`src/attack/` 目录不存在。
  评估口径本身与我们相同（只测良性端、各自的个性化模型、本地训练前），但它的「Bad-PFL」评估 = 被测模型上 1 步 PGD + 未训练生成器的噪声 + 全部标签设为 0。
- **用户补全的 `Huangru-114/CCS-reproduce`**（`578b3a6`，照官方 `fba.py` 补了生成器训练与 0.2 投毒）：能植入 FedPer、不能植入 FedRep —— 与我们正好相反。与我们的差别（读代码，没有数值证据）：
  ξ / δ 的 4/255 直接加在**标准化后**的输入上、`pgd_attack` 在标准化空间 clamp 到 [0,1]（约 61% 的像素被夹，F-102）；评估 ξ 在被测模型上求（白盒）；评估 δ 用被测端自己的、从没训练过的生成器；ASR 不排除目标类样本。
- **官方 Bad-PFL 只实现了 FedBN**（`Huangru-114/Bad-PFL` main 的 `main.py`）；在它上面加 FedRep 的尝试（分支 `claude/federated-learning-experiment-review-pt5j1b`，HANDOFF §5c）ASR 只有 0.066–0.111、已暂停，原因未定。
  → **FedPer / FedRep 都没有官方代码的基准**；Bad-PFL 原文的 FedRep 约 98% 也没有可运行的出处。
- **剂量反应**（同一分支 HANDOFF §5d，官方 FedBN、150 轮、唯一自变量 = 本地步数）：15 / 45 / 75 步 → ASR（过滤）0.22 / 0.80 / 0.88。
  CCSF 的官方超参数（2 epoch、batch 64）约 12 步 / 轮，落在「攻击起不来」的区间；第 30 轮 FedPer + 我们的超参数（约 55 步）与 FedRep 几乎同步（良性端 0.21 vs 0.23、GM 0.49 vs 0.51），FedPer + 官方超参数的 GM 只有 0.05。
  → **首要嫌疑：每轮本地训练量太小**（推断）。检验：CCSL（D-115）。

### F-111 `confirmed`（数据：`results/P2/CCSL/CCSL__fedper-ours-off__s42.metrics.json`）—— 训练量假设被否定：FedPer 下加大每轮训练量后，良性端攻击成功率仍只有 0.41

- **有效性**：exit 0、无 client_failures、攻击者参与 197 次；前 30 轮 `[Checksum]` 与 CCSA 无防御格子 30 / 30 逐位相同；墙钟 2.85 h、显存 17.0 GiB，无 OOM。
- **结果**（末 10 点 / 末轮；对照：FedPer + CCS 官方训练量的 CCSF off、FedRep 的 legacy off）：

  | | FedPer + 我们的训练量 | FedPer + 官方训练量 | FedRep |
  |---|---|---|---|
  | 良性端攻击成功率（过滤目标类） | 0.410 | 0.387 | 0.926 |
  | 共享模型攻击成功率（第 300 轮） | 0.993 | 0.997 | 1.000 |
  | 攻击者自身 | 1.000 | 1.000 | 0.902 |
  | 良性端中位数 / > 0.5 的比例 | 0.38 / 41% | 0.32 / 42% | 1.00 / 93% |
  | 良性端 < 0.1 的端数 | 34 / 90 | 28 / 90 | 6 / 90 |
  | 主任务精度 | 0.852 | 0.833 | 0.904 |

- 训练量大了以后，后门植入共享模型快得多（第 30 轮 0.49，官方训练量是 0.05），但良性端第 100–150 轮到顶（0.53–0.54），之后回落到 0.41–0.49。
- 逐端分化不变：与「干净样本被判成目标类的比例」的等级相关 0.68；这一比例 < 0.5% 的 40 个端平均 0.18。
- → FedPer 下传不到约 6 成良性端的主因是框架本身（私有投影网络与分类头每步以 lr 0.1 训练），不是训练量（D-115 的检验结果为否定）。
  机制仍是推断：本地几乎没有目标类的端，私有层学会几乎不输出目标类。

### F-112 `confirmed`（数据：`results/P2/CCSR30/CCSR30__fedrep-projboth-ccs__s42.metrics.json`；对照 legacy/CCSF-fedrep、CCSA）—— 投影网络两段都训练没有救回主任务（假设否定）；崩坏只出现在「当前共享主干 + 客户端保存的私有部分」上，指向客户端私有的 BatchNorm 统计量与新主干失配

- **有效性**：exit 0、无 client_failures、攻击者参与 17 次；`run.alignment.fedrep_proj_phase = both`；30 次 `[CCS]` 每次 `n_valid = 10`、无 `nonfinite`、无 `[CCSStop]`；墙钟 0.96 h、显存 17.0 GiB。
- **新鲜个性化精度**（主列）：第 10 / 20 / 30 轮 0.32 / 0.35 / **0.29**；旧 FedRep + CCS 0.31 / 0.36 / 0.36；FedRep 不加防御 0.50 / 0.62 / 0.67；FedPer + 我们的训练量 + CCS 0.52 / 0.66 / 0.71。
  → D-116 的「CCS 梯度到不了投影网络」不是主因。
- **三种精度对照**（第 25 轮；共享 = 当前主干 + 初始分类头 + 聚合统计量；新鲜个性化 = 当前主干 + 客户端保存的分类头 / 投影网络 / BatchNorm 统计量；陈旧个性化 = 客户端上次参加时收到的主干 + 同一套私有部分）：

  | | 共享 | 新鲜个性化 | 陈旧个性化 |
  |---|---|---|---|
  | FedRep 不加防御 | 0.64 | 0.67 | 0.66 |
  | FedRep + CCS（旧写法 / 投影两段都训） | 0.57 / 0.55 | **0.35 / 0.32** | 0.55 / 0.54 |
  | FedPer 不加防御 | 0.64 | 0.73 | 0.67 |
  | FedPer + CCS | 0.56 | 0.69 | 0.63 |

- 读法：
  1. 陈旧与新鲜用同一套私有部分、只差主干版本：陈旧没坏、新鲜坏了 → 私有部分本身没坏，是和当前主干拼在一起时对不上。
  2. FedPer 的私有部分同样旧（客户端约每 10 轮参加一次），新鲜却好 → 剩下的区别是 BatchNorm 统计量：FedPer 随主干聚合（评估用跟着当前主干的统计量），FedRep 按对齐模板私有（评估用客户端上次参加时在旧主干上算的统计量）。
  3. **推断，没有直接证据**：CCS 的对抗训练让主干的激活分布每轮变化很大，私有的旧统计量与新主干失配。检验：CCSB30（D-117）。

### F-113 `confirmed`（数据：`results/P2/CCSB30/CCSB30__fedrep-bnshared-ccs__s42.metrics.json`）—— BatchNorm 统计量改为共享后，FedRep + CCS 的主任务不再崩坏：主因确认是客户端私有统计量与新主干失配；攻击成功率在 30 轮时还读不出结论

- **有效性**：exit 0、无 client_failures、攻击者参与 17 次；`run.alignment.fedrep_bn_stats = shared`、`template = mixed`（口径 P2-ablation）；30 次 `[CCS]` 每次 `n_valid = 10`，无 NaN；墙钟 0.97 h、显存 16.8 GiB。
- **主任务**（新鲜个性化精度，第 10 / 20 / 30 轮）：统计量共享 0.53 / 0.65 / **0.70**；统计量私有（旧）0.31 / 0.36 / 0.36；FedRep 不加防御（统计量私有）0.50 / 0.62 / 0.67；FedPer + CCS 0.52 / 0.66 / 0.71。
  只改了这一个开关 → F-109 / F-112 的主任务崩坏来自私有统计量失配（不是投影网络，也不是框架的两段式训练本身）。
- **攻击成功率**（第 10 / 20 / 30 轮；共享模型 / 良性客户端 / 攻击者自己）：
  - 统计量共享 + CCS：0.08 / 0.20 / 0.17；0.09 / 0.13 / 0.12；0.08 / 0.17 / 0.15。
  - FedRep 不加防御（统计量**私有**）：0.05 / 0.17 / 0.51；0.05 / 0.19 / 0.23；0.04 / 0.25 / 0.26。
- 读法：
  1. 30 轮时连不加防御的那格都还没植入完（FedRep 不加防御第 100 轮才 0.76），降幅现在读不出来。
  2. 手上的不加防御对照是统计量私有的；统计量改成共享以后后门本身还能不能植入 FedRep，**没有数据** → 先跑 CCSBOFF（D-118）。
  3. 攻击者自己与共享模型的攻击成功率也低（0.15 / 0.17）：CCS 可能连后门进入共享主干都压住了。单次 30 轮观察，没有证据。
