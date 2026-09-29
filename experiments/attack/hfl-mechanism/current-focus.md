# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-29**（S4 会话结束时；S9 会话的交接内容保留在「历史：S9」）。**S9 已完成**（D-071 … D-075）：评估仪表（常开）+ logits / 快照两个开关 + **下一实验组 G8**（3-C 攻击停止版 = 1B-2 的 HFL 复现）与 **G6D 探针**，两组都已 materialize、**可交**。
> **2026-09-29：已登记的可交格子全部回传，集群上没有待交的作业**（status：todo 0 / done 42 / stale 6 = G7（预期）/ blocked 121）。
> - G6 / FLR / G3（`df4e98e`）：FLR 判 **floor 不可忽略** → G0 逐划分测（F-065）；G3 初读见 F-066。
> - G8 / G6D（`e61af6b`）：仪表在 GPU 上不改数（F-067）；**G8 判 `user_decides`**，后门大部分褪去（F-068）；**G6D 止步**（F-069）。
> - **2026-09-29 S4 已完成**（D-076 … D-079）：投毒窗口起点 + 生成器语义开关（A = window / B = always）；
>   登记 **G8F**（G8 的 flat 对照，3 run）、**G5AB**（生成器语义 A/B 对比，8 run），G0 改固定 60 轮、**可交**；
>   G5 挂着等 G5AB。G2 / S5 仍暂缓（D-056）；G3 搁置。
> - **G8F / G5AB / G0 已回传**（`92e126a`）：flat 也衰减、没有 0.45 平台（F-071，判 `user_decides`）；
>   G5AB `insensitive` → **G5 用 A、已放行**（F-072 / D-080）；B 臂窗口前 = G0-random 在 GPU 上逐位成立；
>   G0 的 floor 跟着本 edge 的目标类占比走，3-B 差中差 CI 全 > 0 但是天花板下的机械结果（F-073）。
>   **下一步 = 集群交 G5（15 run，约 7 GPU-h）**；G8F 与 3-B 的解读待用户定。

## 几套编号（容易混，先看这里）

| 写法 | 是什么 | 在哪 |
|---|---|---|
| **A1–A4** | 审计会话的名字：A1 攻击、A2 训练协议、A3 FedRep / ResNet / HFL、**A4 = 按拍板改代码的实现会话** | PLAN §5 |
| **S1–S8** | 功能会话的名字：S3 新划分、S4 影子攻击者、S5 逐 edge 轮评估、S6 更新日志、S8 三层个性化、S9 评估仪表…… | PLAN §5 |
| **A01–A29** | `AUDIT.md` 的「对齐差异」行号 | AUDIT 第一、二节 |
| **D01–D06** | `AUDIT.md` 的「有意偏离登记」行号 | AUDIT 第三节 |
| **D-001 … D-075** | `DECISIONS.md` 的决策日志（带连字符、三位数），**与登记行 D01–D06 是两套东西** | DECISIONS |
| **F-001 … F-064 / N-001 … N-007** | `FINDINGS.md` 的证据条目 / 设计备注 | FINDINGS |
| **P0 / P1 / P2** | 数据批次的口径版本；只有 P2 进结论 | PLAN §0 |

## 本会话做了什么（2026-09-29，S4：G8 / G6D 回传核对 → G1 / flat 对照拍板 → S4 实现）

**问题**：G8 判 `user_decides` 之后 G1 怎么改、要不要 flat 对照；S4 的四个拍板项（用户：「我现在决定」）。

| 决定 | 内容 |
|---|---|
| D-076 | G8 按「退回 floor」一支处理 G1（用户判定：主体衰减完成、s42 / s44 残留长尾）；G1 保留 3-C，主量 margin，另报尾部 |
| D-077 | **G8F** = G8 的 flat 版，只做 F-std；`decay_verdict.py --flat`：池化原始 ASR 第 255–300 有效轮，全部 ≥ 0.35 → `flat_plateau`，全部 ≤ 0.25 → `flat_decays`。**阈值在写判定时从计划里的 0.20 / 0.30 改的**：G8 自己的 s44 池化值是 0.2001，「与 HFL 一样」也会落进 user_decides。恶意端 id 与 G8 不同（按 edge 布点在 1 个 edge 下从 100 端里抽） |
| D-078 | S4 ②③④：G5 = G0-random 配置、cloud 轮为单位、固定到 t0+75；G0 固定 60 轮；floor_ξ 不做；迁移 baseline 推迟；G5 判定口径与 3 seed |
| D-079 | S4 ①：两种生成器语义都实现，先做 **G5AB**（t0 {20, 140} × {A, B} × 2 seed）；`g5ab_verdict.py`；G5 挂 `g5-schedule` |

**实现**（批准后「开始改」）：`fedavg/attack/attack_window.py`（不 import TF）、`FLClientBase.attacking / generating` + `_gen_active`、
Bad-PFL 的生成器闸门改读 `_gen_active`、`config_validate` 的起点 / 语义校验 + `[设定7]`、collect_metrics **schema 8**、
`status` / `runs_table` / `figures` / `verdicts` 的因素键、`g5ab_verdict.py`、`decay_verdict.py --flat`、登记表（G0 / G5 / G5AB / G8F）。
已有 48 个配置的 sha 不变；新生成 26 个（G0 15、G5AB 8、G8F 3），全部过 `config_validate`、0 警告。

**验证**：本地 L1 **1346 passed / 41 skipped / 3 xfailed**；TF CPU 全量（`scratchpad/tfvenv`，TF 2.15.1）**1499 passed / 23 skipped / 3 xfailed / 2 failed**（只有陷阱 #4 的 2 条）。
CPU 替身（F-070）：默认配置（base / g8）与 S9 锚点逐位相同；**B 臂窗口前 = ρ=0 影子攻击者**（第 1–2 轮 checksum 与 152 个字段逐位相同，投毒率与 n_rounds 都不同）；A 臂窗口前 = 停手后的攻击者（= g8 第 2 轮）；A ≠ B 从第一个有攻击者的轮起。GPU 上没有证据。

## 历史：S9（2026-09-28：讨论 S4 / S6 → 记录项的取舍 → 下一实验组 → 仪表实现）

**问题**：用户要「讨论 S4、S6 的计划」，随后把问题收窄为「记录哪些数据、重跑不重跑，都要导向防御设计」，并要求本会话「敲定下一个实验组、实现这几个开关」。

| 决定 | 内容 |
|---|---|
| D-071 | 3-E 的 MTA 判定用 **fresh 列**，门槛维持 0.02（(a) 臂 fresh pm_acc 的 seed 间 SD ≈ 0.004；没有支持「更高」的数据），另报 ΔASR–ΔMTA 权衡 |
| D-072 | 仪表：**常开**（无配置键）= 逐客户端 ASR / 干净精度、margin 分位数、y_t 偏置、按类 ASR、非目标翻转率；**开关**（默认关）= logits 存盘、分析快照；δ / ξ 逐点消融仍不做（D-051） |
| D-073 | **何时开**：登记表写明了消费它的离线分析才开；logits ≤ 50 MB / run、快照 ≤ 3 次 / run、fp32、只供评估；项目总预算 **20 GB** |
| D-074 | **G1 以 FLR + G8 为条件重新规划**（`persists` → 3-C 缩为最小确认、主用途 3-D；`decays_to_floor` → 保留 3-C、主量用 margin） |
| D-075 | **下一实验组 = G8（只跑臂 a，3 seed）+ G6D 探针（三臂 × s42），同批交**；预注册判定 `harness/decay_verdict.py` |

**代码**：

| 改动 | 内容 |
|---|---|
| `fedavg/attack/eval_detail.py`（新，纯 numpy） | `client_record` / `summarize` / `pack_logits`；margin = log p_t − max log p_k（softmax 下与 logit 差相等，截断在 float32 tiny） |
| `fedavg/attack/backdoor_eval.py` | 三个前向函数加**仅关键字** `detail=None`（返回值、argmax 一字不改）；`evaluate_hierarchical_asr` 返回 `client_detail` / `probe_order`，新参数 `keep_probs` / `n_classes` |
| `fedavg/server/backdoor_server.py` | `_emit_detail`（在 `t_asr` 之后）打 `[EvalDetail]` / `[EvalDetailEdge]` / `[ClientEval]`；`_logits_due` / `_write_dump`；`_snapshot`（`run_round` 里、评估之后） |
| `fedavg/utils/dumps.py`（新）+ `utils/kvline.py` 的 `fmt_list` / `parse_list` | 落盘位置、`[Dump]` manifest、`snapshot_rounds` 解析 |
| `fedavg/alignment.py` / `config_validate.py` §4d' | 两个 EXTRA_SWITCHES；拒绝 bool / 负数 / YAML 列表 / 乱序 / 越界 / 超过 3 个；快照 > 1 GiB 警告 |
| harness | `collect_metrics` **schema 7**；`runs_table` 多 4 个末 10 点列；新 `instrumentation_check.py`、`decay_verdict.py`；`registry.EXPECT_KEYS` 加 `attack_stop_round` |
| 登记表 | `available: [S8, S3, S9]`；G8（3 run）、G6D（3 run）已 materialize（169 run）；已有 42 行 sha 不变 |
| 顺带修 | `test_designed_partition` 的 TF 集成测试夹具少 `lr_decay`（F-063） |

- **L1**（本地无 TF，`bash run_l1.sh`）：改动前 1191 passed / 40 skipped / 3 xfailed → 改动后 **1272 passed / 41 skipped / 3 xfailed**（PASS）。
- **本地 TF 2.15.1 CPU**（scratch venv，`pytest tests/`）：改动前 1331 passed / 23 skipped / 3 xfailed / 3 failed（F-063 那条 + 陷阱 #4 的 2 条）
  → 改动后 **1415 passed / 23 skipped / 3 xfailed / 2 failed**（只剩陷阱 #4）。
- **反向锚点**：`compute_asr_four_way` 里多调一次触发器 → `test_each_probe_calls_the_trigger_exactly_once` 变红；`instrumentation_check` 对 G6 a vs b 判不一致、对 F-045 的 DET 两次与 F-050 的跨提交对判一致。
- **CPU 替身前后对照**（F-064）：见下「L2 替身（S9）」。


## 历史：S3（2026-09-28：讨论 → FLR 登记 → 新划分实现）

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

## 下一步（2026-09-29，G8F / G5AB / G0 回传之后）

**⓪ 用户（集群）：拉代码、跑 L1**（GPU 节点 5 条红、无 GPU 节点 2 条红，F-067；多出来的才是回归）

**① 用户（集群）：交 G5**（在仓库根目录；约 7 GPU-h，按 G5AB 实测每云轮约 140–155 s 外推）

```bash
PACK=3 RUN_GROUPS="G5" bash experiments/attack/hfl-mechanism/submit.sh --dry-run   # 期望：5 个 k=3 包（t20 … t180，各 3 seed）
PACK=3 RUN_GROUPS="G5" bash experiments/attack/hfl-mechanism/submit.sh
```

- G5 的 t20 / t140 × s42 / s43 与 G5AB 的 A 臂配置**只差 meta**（group / run_id）→ 回来后与 G5AB-A 逐轮 checksum 应相同
  （跨作业的 GPU 复现检查，免费）。
- 3.3 的预注册判定只用**峰值**（窗口内 4 点最大值减同轮 G0-random floor，随 t0 的秩相关）；稀释点已贴 floor（F-072），只报告。
- G5 的判定代码还没写（属于 S7，或下个会话在数据回来之前写定）。

**② 用户拍板**

- **G8F 的解读**（F-071，预注册 `user_decides`，只因 s43 = 0.286）：两个 seed ≤ 0.25、末窗口三个都 ≤ 0.20、保留比例与 HFL 相同
  → 我的读法：flat 在本管线里也衰减，与 1B-2 的差别在训练协议、不在 HFL 结构。要不要据此进一步拆协议（例如 G7 那种官方预处理臂的衰减版），用户定。
- **3-B 在天花板下判不出**（F-073）：预注册差中差 +0.077 / +0.067 / +0.068，CI 全 > 0，但来自 floor 差（C3 的 E3 floor ≈ 0）。
  选项：降低攻击强度离开天花板，或 G3 带仪表重跑看 margin（此前用户搁置）。
- G8 的存盘文件（约 0.81 GB）：分析做完再删（用户）；候选分析见 F-068 / D-078（长尾客户端、迁移 baseline）。
- 合并 main：先不合并（用户）。

## 历史：G8 / G6D 的提交（2026-09-28 / 29）

- **原计划的 DET 重交不再需要**：`instrumentation_check G6(a) vs G8 --upto 30` 三个 seed 都 ✅（F-067），比 DET 更强。
- **单 seed 的新格子不要用 `submit.sh` 交**：每格都没有显存记录 → 各交一个 k=1 的探路包（`pack_flush` 的 `kc == 0` 分支）；
  跨格子合包（D-060）只对有真实峰值的格子生效。G6D 当时手工交了一个与 `submit_lib._submit_pack` 参数相同的 K=3 包（在仓库根目录执行）：

  ```bash
  M=experiments/attack/hfl-mechanism
  sbatch -c 12 --mem=72G --job-name=exp3v2-pack3 \
      --comment=exp3v2:G6D__a__s42,G6D__b__s42,G6D__c__s42 \
      $M/pack.sbatch G6D__mix-a+b+c__pack-k3-s42 \
      $M/configs/G6D__a__s42.yaml G6D__a__s42 $M/results/P2/G6D/G6D__a__s42.metrics.json \
      $M/configs/G6D__b__s42.yaml G6D__b__s42 $M/results/P2/G6D/G6D__b__s42.metrics.json \
      $M/configs/G6D__c__s42.yaml G6D__c__s42 $M/results/P2/G6D/G6D__c__s42.metrics.json
  ```

  tag 用合包格式，`_cell_k` 认得这份显存记录。
- **72G 下的 K=3 已实测 11 个包**（FLR 1 + G3 8 + G8 1 + G6D 1），全部没有 OOM（F-065 ② / F-067）。
- **OOM 的代价**：计费按作业（墙钟 × 1 张卡），被杀的 run 只多花它自己的重跑；`pack.sbatch` 记 exit 86 / 137，下次自动降档。


## L2 替身（S9；本地 CPU、随机数据，N-005 的做法；只证明接线，数字无意义）

驱动脚本在 scratch、不入库：把 `cifar10.load_data` 换成同形状随机数组，G6(a) 配置缩到 20 端 / 4 edge / [2,0,0,0] / R2；`taskset -c 0-3`（F-047）。
改动前用 HEAD 的冻结副本（`git archive`）跑，改动后用工作树跑。

| 变体 | 改动前 `[Checksum]` | 改动后 | `instrumentation_check` |
|---|---|---|---|
| base（2 轮） | `00527830725a` / `628233ae41d3` | 同 | ✅ 2 个 checksum、130 个已有字段逐位相同 |
| g8（3 轮、`attack_stop_round: 2`） | `00527830725a` / `2ced03f82896` / `d309897daaf5` | 同 | ✅ 3 个 checksum、195 个已有字段逐位相同 |
| g8dump（g8 + logits 每点 + 快照 "1/3"） | —（开关改动前不存在） | `00527830725a` / `2ced03f82896` / `d309897daaf5`（= 改动前的 g8） | ✅ 同上（195 个字段）；`dumps`：logits 3 个（每个 49 KB，893 个探针样本）、快照 r1 / r3 各约 100 MB，`errors == []`；快照 `evaluated` / `edge_matches_eval` 都是 true、`resumable` false |

- 替身里攻击者只在第 2、3 轮被选中（都在停止轮 2 之后）→ 生成器一次都没训，快照里 Adam 只有 `iteration = 0`（**预期**，不是 bug；G8 里攻击者在第 1–30 轮参与）。
- logits 的 fp16 对数概率会出现并列（893 张里 1 张，最大两类都是 −2.09375）→ **以同文件的 uint8 `trig_pred` / `clean_pred` 为准**，不要用 fp16 的 argmax 重算。
- 墙钟（只作参考）：`[TimingASR].main` 改动前后 37.9 → 35.8 s（base）、37.4 → 34.7 / 38.3 s（g8 / g8dump）—— 在噪声之内，看不出常开汇总的开销。



## 历史：S8 的 L2 替身（本地 CPU，只证明接线；F-054）

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
| ~~S9~~ 评估仪表 + 存盘开关 | **G8**（3-C 攻击停止版）、**G6D**（3-E 分散布点探针） | 3 + 3 | ✅；两组已回传：G8 `user_decides`（F-068）、G6D 止步（F-069） |
| ~~S8~~ 三层个性化 | G6（3-E，可选） | 9 | ✅；9 个 run 全部回传（F-061 / F-065 ③） |
| ~~S3~~ 新划分 | G3（3-B） | 24 | ✅；G3 已回传（F-066）；另是 G0 / G1 的前提 |
| ~~S4~~ 影子攻击者 + 攻击起始轮 | G0（3.1）、G5AB → G5（3.3）；另 G8F | 15 + 8 (+15) + 3 | ✅（2026-09-29，D-076 … D-079）：起点 + 生成器语义开关；G0 / G5AB / G8F 可交，G5 等 G5AB |
| S5 逐 edge 轮评估 | G2（3-A）、G1 的前提之一 | 55 | **暂缓**（D-056）；预案 D-055 |
| S6 更新日志 | G1（3-D）、G4（3.2，搁置） | 24 / 12 | **S9 已拿走其中的「恶意端干净精度」**；剩余：逐更新几何分数（S9 会话讨论推荐：每个上传的 body Δ 做 CountSketch + 在线精确余弦对拍）、周期全量转储给 c_k、c_k 的 head（推荐 edge 干净集上的类均值原型 NCM）。G1 已定按「退回 floor」一支（D-076：保留 3-C、主量 margin）→ **S6 的范围可以开始定** |
| S7 判定代码 + 出图 | — | — | 3-E 的判定（D-059 / D-071 口径）属于这里 |
| —（无需会话） | G7 | 6 | ✅ 已判完 |


## 挂着的事

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`；**Claude 没有合并** |
| ~~G1 重新规划~~ | — | ✅ 用户定按「退回 floor」一支（D-076）；具体设计随 S5 / S6 |
| ~~G6D go / no-go~~ | — | ✅ 止步：(b)/(c) 只比 (a) 低 0.033 / 0.024（F-069） |
| G3 要不要带仪表重跑 | 用户 | **搁置**（用户，2026-09-29）。C3 受害 edge 0.93 – 0.99，差中差判不出（F-066）；重跑 = C1 / C3 各 3 seed，约 6 GPU-h，config_sha 不变 → 要在登记表里另开一个组 |
| ~~G5 用哪种生成器语义~~ | — | ✅ G5AB `insensitive` → A（D-080），G5 已放行、已 materialize |
| G8 的存盘文件（约 0.81 GB） | 用户 | **分析做完再删**；候选分析见「下一步」③ |
| 3.2 的假设重新表述 | 用户 | N-003；本会话给过一版草案（D-021 下 ρ=1 时 body 学到的是与触发器无关的塌缩；预测：恶意端干净精度 ≈ 本地 y_t 占比、body ‖Δ‖ 更大、无触发器时判 y_t 的比例 ≈ 有触发器时 —— **只是推理**）；y_t 偏置与恶意端精度现在常开 |
| 磁盘预算 20 GB | 每批回传后 | 加总各 metrics.json 的 `dumps.logits.bytes` 与 `dumps.snapshots[].bytes`；目前 G8 约 0.81 GB；logits 54.7 MB / run 超了 D-073 的 50 MB 上限约 9%（F-067） |
| `.git` 已 360 MB | 需要时 | 红线 500 MB；每个 schema 7 的 metrics.json 约大 50 KB（`test_collect_eval_detail.py` 的体积守卫：≤ 70 KB） |
| G2 的规模 | 用户 | D-056；定了再开 S5 |
| 攻击接近饱和（F-045 / F-049 / F-061） | 用户 | 终值类比较可能撞天花板 → 看 margin 列（D-072） |
| G7 的混杂 | 用户 | 官方预处理下干净精度低约 0.10（F-050） |
| fresh-PM 低估干净精度（F-051） | 用户 | 跨拓扑的精度结论同时报陈旧 pm_acc |
| `experiments/METRICS.md`「ξ 用 mal[0]」一句 | 用户 | 与 Bad-PFL 库双份同步（`test_metrics_doc.py` 守着）；按 D-015 + D-033 改。S9 的新字段也没写进去 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007 |
| `git_dirty` 排除结果文件 | 需要时 | F-045 |
| G3 的停轮：自适应还是固定长度 | 用户，需要时 | 有影响再改成固定 300 |
| hdir 四档要不要 floor | 用户，FLR 回来后 | D-066 |


## 容易踩的坑

- **S9 的常开仪表只读主列那一次前向**：往 `compute_asr_four_way` / `compute_asr_on_dataset` 里再调一次触发器，
  会让同一列后面所有探针的 ξ 随机数错位（并改动生成器的 BN 统计）→ AST 守卫 `test_each_probe_calls_the_trigger_exactly_once`。
  新细节要么从 `detail` 里已收的概率算，要么单独开实验（D-051）。
- **细节打印在 `t_asr` 之后**：放到计时里面会让 `[TimingASR].main` 与改动前不可比（`test_detail_lines_are_emitted_after_the_asr_timer_stops`）。
- **`[EvalDetail]`（池化）与 `[EvalDetailEdge]`（每 edge）是两个 tag**：同一个 tag 带 edge 段会被 `_merge_side_columns` 按轮覆盖成 edge 行。
- **"/" 连接的列表里 None 写 `na`**（`fmt_list`），不是 `n/a`：后者自己含 "/"。
- **`snapshot_rounds` 写成字符串 `"30/70"`**，不要写 YAML 列表（`[设定4]` 往返会变形，`config_validate` 拒绝）。
- **两个存盘开关只在组的 `set:` 里开**：写进 `base.yaml` 会让所有 P2 组的 sha 变（D-053 的 stale）。
- **`[Dump]` 的 sha 用正则取**（同 `[Checksum]`）：十六进制串可能被 kvline 读成数（如 `123456e78901`）。
- **快照不能续训**：没有客户端 / edge / 数据的随机状态与陈旧 client.model。要分叉续跑得另做（原文 §11 的 checkpoint），本会话没做。
- **`instrumentation_check` 只比 checksum 与改动前就有的字段**；`check_reproducible` 比全部字段含计时，带仪表的新文件一定「不一致」。
- **本地跑 TF 测试**：scratch 里的 venv（TF 2.15.1 CPU + protobuf 4.25 + wandb）；`--deselect` 不需要，陷阱 #4 的 2 条红照旧。


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
- **首包缺省 K=3**（`PROBE_K`，D-069）；显存没测过的新配置类型（10 edge / R20 等）第一次交时写 `PROBE_K=2`。
- **主机内存也要随 K 放大**（D-070）：每个包 `--mem = PACK_MEM_PER_RUN_GB × K`（缺省 24）。每 run 实测约 16.7 GiB（F-060）；
  单卡作业内存 ≤ 102.6 GB 不多计费（MAX_TRES）。exit 137 = 主机 OOM，自动降档。
- **提交前会查队列**（D-070）：作业带 `--comment=exp3v2:<run_id,…>`，已在队列的 run 打印 `queued`；D-070 之前交的作业查不到。
- **日志文件名带作业号**：`tfdpfl-logs/exp3v2_<run_id>.<job>.log`（F-060：只按 run_id 命名时，重复提交的作业会截断前一个的日志）。
- **一卡多跑确实并行**（F-059）：看包的 `wall_s` ≈ 单个 run 的耗时，不是之和。日志里的「结束」顺序与 `run.provenance.start`
  都证明不了并行（前者按 wait 顺序打印，后者是提交时刻）。G6 每个 run 约 3 h 是因为固定跑满 300 有效轮。
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
