# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-28**（S3 会话结束时）。**S3 已完成**（D-061 … D-068）→ **G3 可交**（24 run）；**FLR 已登记、待交**（D-061）。
> G6 仍剩 3 个 s44 待交（D-060）；G2 与 S5 仍暂缓（D-056；S5 预案 D-055）。**下一会话由用户定**（候选：S4，它解锁 G0 / G5）。

## 几套编号（容易混，先看这里）

| 写法 | 是什么 | 在哪 |
|---|---|---|
| **A1–A4** | 审计会话的名字：A1 攻击、A2 训练协议、A3 FedRep / ResNet / HFL、**A4 = 按拍板改代码的实现会话** | PLAN §5 |
| **S1–S8** | 功能会话的名字：S3 新划分、S4 影子攻击者、S5 逐 edge 轮评估、S6 更新日志、S8 三层个性化…… | PLAN §5 |
| **A01–A29** | `AUDIT.md` 的「对齐差异」行号 | AUDIT 第一、二节 |
| **D01–D06** | `AUDIT.md` 的「有意偏离登记」行号 | AUDIT 第三节 |
| **D-001 … D-059** | `DECISIONS.md` 的决策日志（带连字符、三位数），**与登记行 D01–D06 是两套东西** | DECISIONS |
| **F-001 … F-053 / N-001 … N-006** | `FINDINGS.md` 的证据条目 / 设计备注 | FINDINGS |
| **P0 / P1 / P2** | 数据批次的口径版本；只有 P2 进结论 | PLAN §0 |

## 本会话做了什么（2026-09-28，S3：讨论 → FLR 登记 → 新划分实现）

**问题**：「S3 这个组要干什么、验证什么、和最后的防御设计有什么关系；划分怎么构造」—— 已回答、定稿并实现；另按用户要求**先做 floor 验证**。

两个提交：`6c15264`（FLR 登记 + 设计定稿）→ 其后一个提交（S3 代码）。

**S3 的代码**（第二个提交）：

| 改动 | 内容 |
|---|---|
| `fedavg/data/designed_partition.py`（新，不 import TF） | designed（C1–C4，机构式表 r = 0.25，D-067）/ hdir（社区口径，D-063）/ equal_random（D-065）；「名义 → IPF 投影 → 最大余数取整 → 无放回」；先切干净集（按 p_e，D-064）；每端 375 / 125；只用 `default_rng([seed, 0x533])` |
| `fedavg/main.py` | `build_clients` 插入 `elif partition in S3_PARTITIONS` 分支；旧 train/test 切分包进 `if s3 is None:`（**旧路径逐字节不变**，AST 指纹守着）；打 `[Partition]` + 4 条 `[PartitionEdge]`；干净集经 `s3_out` 挂到 `edge.clean_indices`（S3 不使用） |
| `fedavg/config_validate.py` §4f | S3 配置错误全部拒绝（4 edge、condition、r ∈ (0, 0.3]、target_label = 0、n × test_ratio 整数、每类供给） |
| harness | `collect_metrics` **schema 6**：`run.data`（含 `per_edge`）+ `run.partition` / `partition_condition` / `partition_alpha_edge` / `partition_n`；`registry.EXPECT_KEYS` 只挂 `federation.design.*`（挂 `federation.partition` 会让 G6 / G7 变 mismatch）；`runs_table` / `figures` 因素键；`harness/partition_preview.py`（F0 数据，F-058） |
| 登记表 | `available: [S8, S3]`；G3 = C1 / C2 / C3 / C4 / hdir-a{0.1, 0.3, 1, 10}（24 run，已 materialize）；G0 / G1 / G2 / G5 写入划分 set；`base.yaml` 不动 |

- 自描述行叫 `[Partition]` 不叫 `[Data]`：`data/dataset.py` 早就在打 `[Data] …` 行，同名会被误解析（`test_old_logs_without_partition_lines_give_none` 守着）。
- 离线预览（F-058）：C1 H_inter 0.375 / C3 最紧的一类 5543 / 6000；hdir 四档实测 0.17 / 0.47 / 0.67 / 0.86 **区间不重叠**；投影几乎不压平（≤ 0.013）；random 0.134 ≈ hdir α_e=10。
- L1（本地无 TF，venv）：1139 → **1185 passed / 38 skipped / 3 xfailed**（+45 条 `test_designed_partition.py`，+1 条是 `test_main_names_are_bound` 多扫了新文件；+1 skip = 需要 TF 的 `build_clients` 集成测试，集群上跑）。
  反向锚点：旧 noniid 分支 / `split_client_train_test` / `if s3 is None:` 里的切分语句各改一个字符 → 对应守卫变红。

**设计与 FLR**（第一个提交）：

| 决定 | 内容 |
|---|---|
| D-061 / D-068 | **FLR**：G6(a) × ρ=0 × 3 seed，与 G6(a) 按 seed 配对；预注册判定 `harness/flr_verdict.py`（全部 ≤ 0.05 → 可忽略；任一 ≥ 0.10 或 E0 差 ≥ 0.05 → 不可忽略；阈值用户已确认，D-068）→ 定 G0 规模 |
| D-067 | 比例表数值：E0 均衡；E1 automobile + truck、E2 cat + dog、E3 deer + horse，特长 r = 0.25、其余 0.025；airplane / bird / ship / frog 各 0.10 |
| D-062 | C1–C4 用**机构式**比例表（C1–C4 之间只改 y_t 列；bird / ship 均匀）→ 3-B 判定改**差中差**，G3 补 C1 格 |
| D-063 | α 按**社区口径**（每类参数 = α）；原文 Dir(α·p) 差 10 倍（F-056） |
| D-064 | 干净集 500 / edge，按本 edge 分布 p_e，与客户端不相交 |
| D-065 | P2 的 random = **等大小版**（G0-random / G1-random / G2 / G5）；写进组 `set:`，**不改 `base.yaml`** |
| D-066 | G0 = 4 edge 集中 [10,0,0,0]、R5；规模等 FLR |

- 供给核算（F-057）：n = 580 时 C2–C4 超供给，n = 560 无余量，**n 默认 500**（375 / 125）；层级 Dirichlet 名义 p_e 90–100% 超供给 → 必须投影。
- 入库：`registry.yaml` 的 FLR 组 + `configs/FLR__g6a__s4{2,3,4}.yaml`（与 G6(a) 只差 `poison_ratio` 与 meta，`test_flr_verdict.py` 守着）、
  `harness/flr_verdict.py` + 14 条 L1；`test_registry` / `test_status` 的组计数（157 → 160、非 blocked 组 + FLR）。G6 / G7 的 config_sha 不变。
- L1（本地无 TF，venv；本会话实测）：改动前 1123 passed / 39 skipped / 3 xfailed → 改动后 **1139 passed / 37 skipped / 3 xfailed**（+14 条 FLR；+2 条是装了 matplotlib 后不再 skip 的出图测试）。
  反向锚点：把配对守卫改成与 G6(b) 比 → 3 条全红。
- 完整计划（含逐格运行表）：会话 plan 文件的内容已拆进 PLAN §3 / §4 / §5 与本文件。

## 历史：S8（2026-09-27）

**问题**：「实现 3-E 三层个性化（cloud 只聚合全局共享段、edge 内共享一段中间层、客户端私有 head），让 G6 能跑。」—— 已回答。

| 改动 | 内容 | 决定 |
|---|---|---|
| `fedavg/utils/tier_split.py`（新，不 import TF） | 配置键 `federation.edge_shared_blocks ∈ {0,1,2}` 的读取 / 校验；块 → 层名前缀（`stage4_` / `stage3_`+`stage4_`）；按层名选索引；段复原 `keep_segment`（= `utils/pm.compose_pm`）；`[设定6]` 字段 | D-057 |
| `models/cnn.py` | `get_base_head_indices(model, n, edge_shared_blocks=0)` 追加 `edge_weight_indices`（⊆ base）/ `cloud_weight_indices`（= base − edge）；前四个键逐字不变；`weight_owner_names` 按变量身份认层 | D-057 |
| `server/hier_fedrep.py` | `HierFedRepEdgeServer.set_weights`：cloud 广播不覆盖 edge 段（首次接收除外） | D-057 |
| `server/server.py` | `aggregate_edges`：edge 段不聚合、全局那一段保持初值（Q4）；`__init__` 打 `[设定6]` | D-057 |
| `config_validate.py` §4e | 拒绝：非法取值（含 bool）、方法 ≠ hier_fedrep、arch ∉ ResNet-10、cloud 层防御；n_edges=1 警告 | D-057 |
| harness | `collect_metrics`：`[设定6]` → `run.edge_shared_blocks` / `run.tier_split`（**schema 5**）；`runs_table.FACTOR_KEYS` + `FACTOR_DEFAULTS`（0 记 None，老文件与 (a) 同格）；`figures.FACTOR_COLUMNS`；`registry.EXPECT_KEYS`（status 核对 k） | D-057 |
| `registry.yaml` + `configs/` | `available: [S8]`；G6：4 edge 集中 [10,0,0,0]、R5、n_rounds 60、`stopping: null`、三臂 `edge_shared_blocks` 0/1/2；**已 materialize**（G6 9 个 + G7 6 个重新生成 → G7 = stale，预期） | D-058 |
| PLAN §3 3-E 行 | 判定改用原始 benign ASR（受害 edge E1–E3，按 seed 配对），不等 S4 | D-059 |

- **L1**（本会话实测，同一环境改动前 → 后）：本地无 TF 1074 → **1115 passed** / 36 → 37 skipped / 3 xfailed（PASS）；
  TF 2.15.1 CPU venv 1199 → **1253 passed** / 23 skipped / 3 xfailed / **2 failed** —— 红灯前后都只有陷阱 #4 那 2 条。
- **反向锚点**：只把接线（edge `set_weights` / cloud `aggregate_edges` / `config_validate` §4e）换回改动前 → 新测试 14 条红；
  只去掉 edge 覆写 → 恰好 `edge0 的 edge 段被 cloud 广播覆盖了` 那条红；只去掉 cloud 复原 → cloud 聚合那条红。
  k=0 与不写这个键的 `[Checksum]` 逐轮相同，k=1 不同（`test_k0_is_byte_identical_to_not_writing_the_key`）。
- **L2 替身**（本地 CPU、随机数据，N-005 的做法；只证明接线，数字无意义）：G6 (b)/(a) 缩到 20 端 / 4 edge / R2 / 2 云轮，见下「L2 替身」。
- **真正的 L2 = 集群交 G6**（用户）：命令见下。

## 下一步

**⓪ 用户（集群，登录节点即可）：先跑一次 S3 的 TF 测试**（本地没有 TF，`build_clients` 的 S3 分支只在这里第一次真跑）

```bash
git pull
bash run_l1.sh designed_partition        # 期望：46 passed（本地是 45 passed + 1 skipped）；红了先别交 G3
```

**① 用户（集群）：交 G6 剩下的 3 个 s44 + FLR 的探路包 + G3 的探路包**

```bash
git pull                                                                 # 本分支（含 D-061）
bash experiments/attack/hfl-mechanism/submit.sh --status                 # G6：done=6、todo=3；FLR：todo=3；G7：stale=6
PACK=3 RUN_GROUPS="G6 FLR" bash experiments/attack/hfl-mechanism/submit.sh --dry-run
#   → would sbatch -c 8  pack  FLR__g6a__pack-k2-s42  FLR__g6a__s42 FLR__g6a__s43      （新格子先 K=2 探路，D-052）
#     would sbatch -c 12 pack  G6__mix-a+b+c__pack-k3-s44  G6__a__s44 G6__b__s44 G6__c__s44   （D-060）
PACK=3 RUN_GROUPS="G6 FLR" bash experiments/attack/hfl-mechanism/submit.sh
# 探路包回来后再跑一次同一条命令 → FLR__g6a__s44 按显存定 K 交出
python3 harness/flr_verdict.py --json experiments/attack/hfl-mechanism/analysis/flr_verdict.json
```

- FLR 与 G6(a) 同配置同显存（约 17 GiB / run，F-055），按 D-052 仍先探路。
- FLR 回传后核对：`run.poison_ratio == 0`、`client_failures == []`、`run.malicious_ids` 与同 seed 的 G6(a) 相同（判定脚本会报 `same_malicious_ids`）。
- 判定阈值 0.05 / 0.10 已确认（D-068）；判定为 `user_decides` 时由用户定 G0 规模。

**G3（3-B）的探路包**（S3 的真正 L2）：

```bash
PACK=3 RUN_GROUPS=G3 bash experiments/attack/hfl-mechanism/submit.sh --dry-run
#   → 8 个格子各一个 K=2 探路包（16 run），其余 8 个 held
PACK=3 RUN_GROUPS=G3 bash experiments/attack/hfl-mechanism/submit.sh
```

- 回传后先核对每个 metrics.json：`schema_version == 6`、`run.partition` / `run.partition_condition` 与格子一致、
  `run.data.client_size_min == client_size_max == 500`、`run.data.per_edge[].yt_share` 与比例表一致（C3 的 E3 ≈ 0.005）、
  `run.data.malicious_data_share == 0.1`、`client_failures == []`、`python3 harness/status.py …` 无 mismatch。
- **没有证据的**（F-058）：每端 500 张（总 52k）下的干净精度与停轮标定是否仍合适 —— 探路包的 pm_acc 与 `stop_reason` 回答。
- 3-B 的判定（差中差，D-062）要 excess ASR → 等 FLR 的判定；FLR 判 `negligible` 就用原始 ASR（D-061）。判定代码属 S7。

**G6 的 s44 的注意事项**（沿用）：

- 这是第一次 **K=3 的满长包**：回传后先看 `G6__mix-a+b+c__pack-k3-s44.gpu.json` 的 `n_oom` / `n_mem_warnings` / `run_peak_mib`。
  OOM 的 run 会记 exit 86 → 再跑一次上面的命令会自动降到 K ≤ 2 重交（合包的 OOM 对三个格子都生效）。
- 回传后每个 metrics.json 照旧核对：`run.edge_shared_blocks` 与臂一致、`client_failures == []`、跑满 60 个云轮。
- **G6 不读 GM 精度与 global 层 ASR**（(b)(c) 下全局模型的 edge 段是初值；FedRep 下 head 本来就是初始化 head，F-007）。
  判定读受害 edge（E1–E3）的 fresh-PM benign ASR（逐 edge 行）与 fresh / 陈旧 pm_acc（D-059）。判定代码属 S7，还没写。
- 不想合包：`PACK_MIX=0`（= 3 个 K=1 作业，D-052 原规则）。
- G7 显示 stale 是预期（base.yaml 加了评估降频之后重新 materialize；D-053 默认不重交）。

## L2 替身（本地 CPU，只证明接线；F-054）

N-005 的做法（随机数据），G6 配置缩到 20 端 / 4 edge / [2,0,0,0] / R2 / 2 云轮，驱动脚本在 scratch、不入库。

| 臂 | exit | `run.edge_shared_blocks` | `tier_split.n_edge_tensors` | `client_failures` | 攻击者参与 | checksum R1 / R2 |
|---|---|---|---|---|---|---|
| (a) | 0 | 0 | 0 | [] | 第 2 轮 | `eeaae44b41bd` / `f635462ac051` |
| (b) | 0 | 1 | 15（stage4，约 75% 的值） | [] | 第 2 轮 | `d30992d5bbe9` / `a0472707fa90` |

- 两份 `errors[]` 各有 5 条 `MessageFactory … GetPrototype`：本地 venv 的 protobuf 7.x 与 TF 2.15 不配（import 期，`[Config]` 之前），
  换 protobuf 4.25 后消失 —— 环境问题，不是代码。集群上已回传的 6 个 G7 run `errors[]` 全为空；G6 回传若不为空，是真问题。
- (c) 没有在本地跑（与 (b) 同一条代码路径，只差前缀数；索引由 L1 覆盖）。

## 功能会话一览

| 功能会话 | 解锁 | run 数 | 说明 |
|---|---|---|---|
| ~~S8~~ 三层个性化 | G6（3-E，可选） | 9 | ✅ 本会话完成；**G6 待交**（上面的命令） |
| S5 逐 edge 轮评估 | G2（3-A）、G1 的前提之一 | 55 | **暂缓**（D-056）；预案已拍板（D-055）：`eval_grid: 5`、轻评估只算主列并喂停止判据（横轴改网格序号，F-052）、GM / EM 只在网格点上算 |
| ~~S3~~ 新划分 | G3（3-B） | 24（含 D-062 补的 C1） | ✅ 本会话完成；**G3 待交**（上面的命令）；另是 G0 / G1 的前提 |
| S4 影子攻击者 + 攻击起始轮 | G5（3.3） | 15 | 另是 G0 的前提（G0 规模等 FLR，D-061）；ρ=0 本身只要改配置（FLR 已用），S4 的活是 L1 守卫、ξ-only 是否做（D-051 与 PLAN 的 S4 行写法不一致，开 S4 时确认）、`attack_start_round`、G5 的拓扑与窗口单位 |
| S6 更新日志 | G4（3.2，**搁置**，D-047） | 12 | 3.2 的假设要先按 N-003 重新表述（用户）；F-051「私有 head 挡不住 ξ」是相关证据 |
| S7 判定代码 + 出图 | — | — | 3-E 的判定（D-059 口径）属于这里，本会话没写 |
| —（无需会话） | **G7**（预处理对比，D-025） | 6 | ✅ 已跑完并判定（`5edd4df`，事后判据「是」，D-049 / F-050）；重新 materialize 后显示 stale（预期） |

- G2 的规模还没定（G2P 已回来：`consistent`，F-049）——**由用户定**，定之前不做 S5。
- 在 S5 之前，G2 的跨 R 比较受 F-052 影响：停止判据的斜率横轴是云轮号，flat 比 R5 宽松 5 倍。

## 挂着的事

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`；**Claude 没有合并** |
| G2 的规模 | 用户 | D-056；定了再开 S5 |
| 一卡多跑的参数校准 | K=3 满长包回来后 | 第一个校准点已有：整卡读数 ≈ 真实峰值 × 1.98（F-055）；`PACK_MEM_PCT` / `PACK_CTX_MIB` 仍无 OOM 边界的证据 |
| 攻击接近饱和（F-045 / F-049） | 用户 | G 组的终值类比较可能撞天花板；设计 / 解读时考虑 |
| G7 的混杂 | 用户 | 官方预处理下干净精度低约 0.10，「攻击更容易」与「模型更弱」分不开（F-050） |
| fresh-PM 低估干净精度（F-051） | 用户 | 随 edge 数增大（10edge +0.094）；跨拓扑的精度结论同时报陈旧 pm_acc（D-054 后隔点算） |
| δ-only / ξ-only 消融 | 需要时 | 解释「白盒 ≈ 主列」；单独开实验（D-051） |
| `git_dirty` 排除结果文件 | 需要时 | 结果写在仓库里 → 同批后提交的 run 都会 `dirty=1`（F-045） |
| 3.2 的假设重新表述 | 用户 | N-003；S6 / G4 之前定；也影响 3-E 的解读 |
| `experiments/METRICS.md`「ξ 用 mal[0]」一句 | 用户 | 与 Bad-PFL 库双份同步（`test_metrics_doc.py` 守着）；按 D-015 + D-033 改 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007 |
| cifar100 静态触发器的标准化常数 | 需要时 | N-004：只记录，没改 |
| 交 G6 的 3 个 s44 并回传 | 用户（集群） | 上面「下一步」；探路包已回（F-055），剩 1 个 K=3 合包（D-060） |
| 3-E 的判定代码（D-059 口径） | S7 / 需要时 | 受害 edge 的原始 benign ASR 配对差 + MTA；「三臂 floor 相同」无证据 |
| 3-E 的 floor 格（ρ=0 × 三划分） | 用户，需要时 | 臂 (a) 的 floor = FLR（D-061）；(b)(c) 要 excess ASR 才需要，不用重跑 G6（D-059） |
| G3 的停轮：自适应还是固定长度 | 用户，需要时 | 现在是 base 的自适应（150–300 有效轮）；3-B 的差中差跨 run（C3 vs C1）比终值，run 长度不同时终值窗口落在不同轮上。有影响再改成固定 300（同 G6 的 D-058） |
| hdir 四档要不要 floor | 用户，FLR 回来后 | D-066；FLR 判 `negligible` 则不需要 |
| MTA 门槛「≤ 0.02」 | 用户 | 仍 ⚠待确认（PLAN §3） |

## 容易踩的坑

- **新评估开关**（预算旋钮）放 `EXTRA_SWITCHES`、值写进 `base.yaml`；**对齐开关**放 `SWITCHES` + 模板。两类都只经 `get_switch` 读。
- **副列的终值用 `runs_table.window_mean`**（末 10 个评估点窗口），不要用 `last_k_mean`：后者先丢 None 再往回够，隔点的列会够到 20 个点。
- **陈旧 ASR 与陈旧 pm_acc 共用 `CloudServer._eval_seq`**：隔点时 `config_validate` 要求 `backdoor.eval_interval == evaluation.eval_interval`。
- **`[GPUMem]` / `[TimingAcc]` / `[TimingASR]` 都是独立 kv 行**，不要往 `[Cloud]` / `[Timing]` 里加字段（全或无的正则）。
- **改 `base.yaml` 会改所有 P2 组的 sha**：已判完的组显示 stale，提交脚本默认不重交（D-053）。
- **新开关**：加进 `fedavg/alignment.py` 的开关表（进模板的放 `SWITCHES`，不进的放 `EXTRA_SWITCHES`），代码里只经 `get_switch(config, "…")` 读；`tests/test_alignment_switches.py` 会检查模板键集、P2 值 ≠ 旧值、以及每个键真的被读到。
- **P2 配置不能少开一项**：想做消融就登记在 pilot 表（P1 口径），不要在 P2 表里改回旧值（`config_validate` 会拒绝）。
- **`per_epoch` 管线只接了 Bad-PFL**：静态投毒（vanilla / neurotoxin）会被它绕过，`config_validate` 已拒绝这种组合。
- **fresh-PM 的草稿模型会被下一次同 slot 调用覆盖**（`CloudServer.pm_model`）：取一个、用完、再取下一个。
- **pilot 用 `pilot/submit_pilot.sh`，不是上一级的 `submit.sh`**。pilot 登记表是 P1 口径：升 P2 之后再重跑 pilot，`status.py` 会把它们标成 `stale`（口径不符）—— 那是对的，pilot 已经判完了。
- **`run.provenance.protocol` 是代码版本**：此后连冻结的 P1 配置重跑也记 P2；配置是不是 P2 口径看 `run.alignment.template`（p2 / legacy / mixed）。
- **D-029 的 pm_acc 门槛读 `pm_acc_stale`**（陈旧列，与 P1 同一定义）；主列 `pm_acc` 现在是 fresh-PM，不能拿它比 P1。
- **AUDIT 的行号只能是 `A##` / `D##`**，每行恰好一个状态词，格子里不写 `|`；`test_real_audit_parses_and_is_closed` 写死了各行状态，改状态要同步改它。
- **`resnet10` 是冻结配置用的**，P2 写 `resnet10_torch`（模板里已经是）。
- 旧方案的登记表在 `hfl-propagation/registry/v1.yaml`，不能放到那个目录顶层（`run_exp3.sh` 会把它当格子提交）；`submit.sh` 用 `RUN_GROUPS`，不能叫 `GROUPS`。
- 用 `git archive` 解到别处跑 L1 时，`test_provenance.py::test_git_commit_matches_git_rev_parse_on_this_repo` 会假红（没有 `.git`）。

- **S8 的键 `federation.edge_shared_blocks` 只对 `hier_fedrep` + ResNet-10（resnet10 / resnet10_torch）开放**：别的方法的 edge server
  会被 cloud 广播整体覆盖、edge 段每轮被冲掉。要给别的方法开，先在它的 edge server 里覆写 `set_weights` 走 `keep_segment`，
  再把它加进 `utils/tier_split.TIER_METHODS`（`test_tier_split.py` 守着这张表）。
- **(b)(c) 下全局模型不完整**（edge 段是初值）：GM 精度、global 层 ASR、以及读它们的停止判据都不能用 → G6 固定长度（D-058）。
- **`runs_table` 把 `edge_shared_blocks=0` 记成 None**（`FACTOR_DEFAULTS`）：S8 之前的文件没有 `[设定6]`，与 (a) 是同一种 run，要同格。
- **`[设定6]` 在 `CloudServer.__init__` 里打**（真正算出索引处），不在 `config_validate`；每个 run（含 k=0）都有这一行。
- **collect_metrics 是 schema 5**（S8：`run.edge_shared_blocks` / `run.tier_split`）。
- **S3 的新划分只经 `federation.partition ∈ {designed, hdir, equal_random}` + `federation.design.*` 生效**，写在组的 `set:` 里；
  **不要改 `base.yaml` 的 `partition: noniid`**（G6 / G7 / FLR 是旧划分，改了会整体重新生成）。
- **`status` 核对划分只挂 `federation.design.*`**（`partition_condition` / `partition_alpha_edge` / `partition_n`）：
  所有配置都声明 `federation.partition`，旧 run 的日志里又没有 `[Partition]` 行 —— 挂它会让 G6 / G7 全变 mismatch。
- **自描述行叫 `[Partition]` / `[PartitionEdge]`**，不能叫 `[Data]`（`data/dataset.py` 已占用）。collect_metrics 是 **schema 6**。
- **改 `build_clients` 的旧分支、`split_client_train_test`、`noniid_partition` 会让 `test_designed_partition.py` 的 AST 指纹变红**：
  那是故意的（G6 s44 / FLR 要与已跑完的同配置 run 配对）。确有必要改旧路径时，先想清楚已跑完的 run 怎么办，再更新指纹。
- **合包的 gpu.json 文件名是 `<组>__mix-a+b+c__pack-…`**（D-060）：各格子定 K 时按文件名里的格子列表认领它；
  改 tag 格式要同步改 `submit_lib.sh:_cell_k`。满包永远同格子，只有余数会跨格子。

## 历史：一卡多跑接入 + 评估降频（2026-09-27，S8 之前的一个会话）

| 提交 | 内容 | 决定 |
|---|---|---|
| `c88a023` | **一卡多跑接进** `submit.sh` / `pilot/submit_pilot.sh`（共用 `submit_lib.sh`）：按格子分包、每格先交 K=2 的探路包、之后按**真实显存峰值**定 K、OOM 自动降档；`pack.sbatch` 把被吞掉的 OOM 判成 exit 86；服务器新打 `[GPUMem]`；stale 不重交 | D-052 / D-053 |
| `68f865d` | **评估降频**：白盒 ASR 关、陈旧 ASR 与陈旧 pm_acc 隔点（同一批点），终值按「末 10 个评估点窗口」；`[TimingAcc]` 分项计时 | D-050 / D-054 |
| 文档提交（紧随其后） | 文档与交接：D-052 … D-056、F-052 / F-053、PLAN / registry 标注 G2 与 S5 暂缓 | D-055 / D-056 |

- L1：本地无 TF 1072 passed / 38 skipped / 3 xfailed；TF 2.15.1 CPU venv 只有陷阱 #4 的 2 条红。
- **反向锚点**：G7 与 pilot 的 `runs_table` / `pilot_a4` / `g7_posthoc` 输出与改动前逐字节相同。
- **没做**：S5、G2 materialize、G7 重新 materialize（`configs/INDEX.tsv` 仍是旧 sha，G7 仍显示 done）。

## 历史：A4（2026-09-26 / 27）


| 提交 | 内容 |
|---|---|
| `4c70f05` C1 | 开关底座 `fedavg/alignment.py` + 模板 `fedavg/config/alignment_p2.yaml` + `config_validate` 的 P2 核对 + `[设定4]` + 登记表 `overlays` |
| `c54a209` C2 | A01（官方 PGD）、A03（伯努利）、A05 训练侧、A14（官方生成器）；G7 前提 `data/pixel_space.py` + `data.normalize` / `data.augment`；F-041 的 6 条测试 |
| `943bbf9` C3 | A29 `resnet10_torch`、A27 BN 统计量私有、A24 只在 body 投毒、A26 `fedrep_order`、A25 `data/epoch_pipeline.py` |
| `d08bdea` C4 | A28 fresh-PM（`utils/pm.py`、`CloudServer.main_pm`）、A02 固定攻击者 + `[设定5]`、A06 `[ASR4]`、陈旧列 `[Stale]` / `[StaleASR]`、白盒 `[ASRwb]`；collect_metrics schema 3 |
| `7e43a42` C5 | A08 / D02 有效轮、D01 交错执行、A15 确定性 + `[Checksum]` |
| `e820c00` C6 | P2 `base.yaml`、登记表 base + overlays + G7、pilot 表 + `submit_pilot.sh` + `harness/pilot_a4.py`、AUDIT / DECISIONS / FINDINGS / CLAUDE.md |
| `299afe6` 收口 1 | `TorchBatchNorm`（D-043）、pilot 有效性闸（D-044）、`tests/test_bn_inference_determinism.py`（CPU 上模拟 GPU 检查） |
| `cd928bf` 收口 2 | AUDIT 最后 4 行关闭、`PROTOCOL_VERSION = "P2"`、D-045 / D-046、F-045 |
| `726d9d5` 预算 | `pack.sbatch` + `pilot/submit_pack_test.sh` + `harness/pack_test.py`；`[TimingASR]`；G2P 组 + `judge_g2p`；G2 登记表补齐；G4 搁置；G7 配置生成；D-047、F-046 |

- AUDIT：`done` 新增 A01 A02 A03 A05 A06 A14 A16 A22 A24 A27 A28 A29 D01 D02；收口时 A15 A25 `done`、A08 A26 `deviate` → **全部关闭**。
- 用户拍板：D-039（开关 + 一套模板）、D-040（评估尾块循环补足）、D-041（陈旧列 ξ 用攻击者陈旧模型）、D-042（pilot 完整 P2 代码路径 + DET 组）；收口：D-043 … D-046。
- L1（收口时）：本地无 TF 974 passed / 33 skipped / 3 xfailed；TF 2.15.1 CPU venv 只剩陷阱 #4 的 2 条红。

## 升 P2 时做了的 / 没做的

- 做了：`PROTOCOL_VERSION = "P2"`；`test_registry.py::test_real_audit_parses_and_is_closed`（原 `…_is_open`）断言 `audit_open_rows == []`；
  `test_status.py::test_v2_after_the_audit_only_feature_sessions_block`（G7 = todo，其余 151 只剩功能会话；S8 之后改为 G6 + G7 不被挡、其余 142）。
- 没做（用户的事）：`experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」一句，按 D-015 + D-033 改为「按 seed 固定选的一个恶意端的 fresh-PM」
  —— 该文件与 Bad-PFL 库双份同步，由用户改（`test_metrics_doc.py` 守着）。
