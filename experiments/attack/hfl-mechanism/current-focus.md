# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-26**（A3 会话结束时）。下一会话 = **A4**（实现会话）。

## 几套编号（容易混，先看这里）

| 写法 | 是什么 | 在哪 |
|---|---|---|
| **A1–A4** | 审计会话的名字：A1 攻击、A2 训练协议、A3 FedRep / ResNet / HFL、**A4 = 按拍板改代码的实现会话** | PLAN §5 |
| **S1–S8** | 功能会话的名字：S3 新划分、S4 影子攻击者、S5 逐 edge 轮评估、S6 更新日志…… | PLAN §5 |
| **A01–A29** | `AUDIT.md` 的「对齐差异」行号：本仓库与官方 Bad-PFL / FedRep / HierFAVG 不一样的地方 | AUDIT 第一、二节 |
| **D01–D06** | `AUDIT.md` 的「有意偏离登记」行号：HFL 特有、官方没有对应物的设计 | AUDIT 第三节 |
| **D-001 … D-038** | `DECISIONS.md` 的决策日志（带连字符、三位数），**与登记行 D01–D06 是两套东西** | DECISIONS |
| **F-001 … F-040 / N-001 …** | `FINDINGS.md` 的证据条目 / 设计备注 | FINDINGS |
| **P0 / P1 / P2** | 数据批次的口径版本；只有 P2 进结论 | PLAN §0 |

- 会话名 A4 ≠ 行号 A04；A4 没有改名，因为 `harness/registry.py:256` 的报错文案与两条测试匹配「A4」。
- AUDIT 状态词：`open` 未决定 · `align` 已决定改、待实现 · `deviate` 有意保留不同（已签字）· `done` 已对齐且有 L1。35 行全部是 `done` / `deviate` 之前，P2 一律 blocked（D-006）。

## 下一会话唯一要回答的问题

**A4：按 A1–A3 的拍板改代码并配 L1，跑 D-029 可行性实验（D-038 扩大后）与 A26 的对照 pilot，让 AUDIT 全部关闭、口径升 P2。**

## 客观判据

1. 下表每个 `align` 行都有 L1 测试（含反向锚点：改动前失败、改动后通过），行状态改为 `done`。L1 的设计写在 AUDIT 该行的「怎么验证」列，或第三节的「理由 / 决定」列。
2. D-029（D-038 扩大后）通过 → A08 改 `deviate`；没通过 → 逐个开关消融、回审计。
3. A26 pilot 按 D-031 的预注册判据判定：相同 → A26 改 `deviate`；不同 → 带回来由用户定。
4. `PROTOCOL_VERSION` 升 P2；2 个 smoke 复核标定；`status.py` 解除 blocked。

## 开工前

1. `git log --oneline -3`，确认在 `claude/federated-learning-experiment-review-pt5j1b` 上，最后一个是 A3 的提交。
2. 跑 L1，记下当天的基线：
   ```bash
   bash run_l1.sh 2>&1 | tail -3
   # 本地没有 pytest 时：python3 -m venv <scratch>/venv && <scratch>/venv/bin/pip install pytest numpy pyyaml matplotlib pypdf
   #                     然后 TFDPFL_PY=<scratch>/venv/bin/python bash run_l1.sh
   ```
   A3 结束时本地（无 TF）是 **853 passed / 22 skipped / 3 xfailed**。
3. 状态核对：
   ```bash
   python3 harness/status.py experiments/attack/hfl-mechanism/registry.yaml --quiet    # 应为 blocked=151
   python3 harness/status.py experiments/attack/hfl-propagation/registry/v1.yaml --quiet # mismatch=6 failed=1 done=18 orphan=1
   ```
4. 需要 TF 的 L1（A27 / A29 等）本地会 skip，要在集群上跑。A3 用 scratch 里的 TF 2.15.1 CPU venv 做过数值探针（F-033），本地想看 TF 行为时也可以这样装一个。

## A4 的实现清单（全部 `align` 行 + 两个 pilot）

| 行 | 决定 | 要点 |
|---|---|---|
| A01 | D-014 | ξ 改为官方的单步 PGD（随机起点 → clamp → 起点处求梯度 → 投影 → clamp），生成器训练 / 投毒 / 评估三处共用 |
| A02 | D-004、D-015，**D-033 修订** | 主 ASR 的 ξ 用 setup 时按 seed 固定选中的**一个**攻击者；它的模型 = **它自己的 fresh-PM**（不再是陈旧模型）；白盒列保留 |
| A03 | D-016 | 投毒改为逐样本伯努利(ρ) |
| A05 | D-018 | BN 模式：投毒 / 评估时生成器用 batch 统计，评估按 32 张分块 |
| A06 | D-019 | ASR 四列（过滤 / 不过滤 × 全体 / 仅良性），主指标 = 过滤 × 仅良性 |
| A14 | D-020 | 生成器结构对齐官方代码 |
| A15 | D-028 | `enable_op_determinism()` + `[Checksum]` 行 |
| A16 | D-028 | P2 配置删 `label_smoothing` |
| A22 | D-028 | P2 基配置 `target_label == 0` |
| A24 | D-021 | FedRep 只在 body 阶段投毒 |
| A25 | D-026 | 取数：每 epoch 重洗 + 重增强、`drop_last` 且尾批轮换；三处走同一个函数（开关，默认 = 现行为） |
| **A26** | **D-031** | 开关 `training.fedrep_order: head_first｜body_first`（默认 head_first）；body_first = 先 body（投毒）后在 w_k 上训 head。**pilot**：flat + 2edge_distributed、seed42 各 1 个 body_first run；head_first 臂复用 D-029 的 run |
| **A27** | **D-032** | BN 的 γ/β 共享，moving 统计量私有（像 head 一样保留，首次接收时采用广播值）；PM 的统计量取 head 阶段结束时的快照；edge / cloud 聚合的统计量只供 EM / GM 评估 |
| **A28** | **D-033** | **主 ASR 与主 pm_acc 在同一个模型上测**，都用 fresh-PM = [当前 edge body, 自己的 head, 自己的统计量]；陈旧 PM（`client.model`）的两者作副列；`pm_acc_plateau` 读 fresh-PM 的 pm_acc。`compose_pm` 放进不依赖 TF 的模块 |
| **A29** | **D-034** | 新 arch `resnet10_torch`：stride-2 的 3×3 卷积改为 `ZeroPadding2D(1)` + `valid`；BN momentum 0.9、eps 1e-5；torch 默认初始化。冻结的 `resnet10` 不动 |
| **D01** | **D-036** | edge 按 edge 轮交错执行（cloud 驱动：外层 edge 轮、内层 edge）；共享生成器仍是同一对象 |
| **D02** | **D-036** | `edge_quota` 按有效轮 t_eff 轮转（1 / 2 / 10 edge 逐元素不变，4 edge 由 30/20 变为 25/25） |
| A08 | D-023（`open`） | LR 按有效轮衰减的开关；D-029 通过后改 `deviate` |
| G7 | D-025 | 先让 `client_badpfl.py:53-56` 的 ε/σ 换算与 `attack/triggers.py` 跟随标准化开关（F-027），再在 `registry.yaml` 登记 G7 |

### 可行性实验（D-029，D-038 扩大后）

- 格子：`flat_baseline` + `2edge_distributed`，seed42 各 1 个；登记在单独的 pilot 登记表（P1 口径，不进 P2）。
- 开关：A08 + A25 + A27 + A29 **一起开**；2edge 那格同时带上 D01 / D02。
- 判据（先定后跑）：`stop_reason=converged`，且**陈旧 PM 列**的 pm_acc 末 10 点 ≥ P1 同 seed 重复的最小值 − 0.006（flat ≥ 0.7367、2edge ≥ 0.7297）；ASR 只记录。
- 没通过 → 逐个开关消融，回审计。
- 同批：A26 的 2 个 body_first run。判据（D-031）：两格都满足 fresh-PM pm_acc 末 10 点均值之差的绝对值 ≤ 0.006，且 local_benign_asr 之差的绝对值 ≤ 0.07。

---

## A3 做完了什么（2026-09-25/26）

| 行 | 状态 | 决定 |
|---|---|---|
| A12 余项 | `deviate`（不变） | D-030：head = 末层 Dense 的 kernel + bias |
| A17 | `done` | D-035：结论转入 A29；AUDIT 原来写错了文件（F-034） |
| A18 | `done` | D-035：事件系统与网格触发器不在执行路径上（F-040） |
| A21 | `deviate` | D-035：HierFAVG 的两层聚合权重与 TF 一致；偏离项写在行内（F-037） |
| **A26（新）** FedRep 训练顺序 | `open` | D-031：两种顺序都跑 pilot，按预注册判据定 |
| **A27（新）** BN 归属 | `align` | D-032：γ/β 共享、统计量私有 —— 是实现选择，不是框架差别（F-036） |
| **A28（新）** 评估用的 PM | `align` | D-033：ASR 与 acc 同模型，都用 fresh-PM（F-035） |
| **A29（新）** ResNet-10 | `align` | D-034：padding / BN / 初始化三项对齐（F-033） |
| D01 / D02 | `align` | D-036：交错执行（F-038）/ 按有效轮轮转（F-039） |
| D03–D06 | `deviate` | D-037：签字 |
| D-029 | — | D-038：范围扩大 |

- 取证：HierFAVG 官方代码 `LuminLiu/HierFL`、FedRep 原作者 fork、PFLlib、Bad-PFL 官方 `resnet.py` / `trigger.py` / `event_emitter.py` / `pfl.py`。
- **论文原文 HierFAVG（arXiv 1905.06641）与 FedRep 本会话都没读到**（arXiv / OpenReview 被网络策略 403），相关结论只以官方代码为证。

用户已拍板的决定：D-001 至 D-038（D-010 → D-023、D-012 → D-024 已被取代；D-015 被 D-033 部分修订；D-029 被 D-038 扩大）。

## 挂着的事（不属于 A4，但别忘了）

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`；CLAUDE.md 要求会话结束、L1 绿后合回 main；**Claude 没有合并** |
| 3.2 的假设重新表述 | 用户 | N-003：只在 body 投毒后，「私有 head 吸收」的机制要改写；S6 / G4 之前定 |
| G4 的 FedAvg 一侧 | S6 | fresh-PM 对 FedAvg 就是当前 edge 模型（D-033），G4 的两边要按同一定义比 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007。只有还要用旧方案的防御格时才需要 |
| `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」 | 用户（两库同步） | 与官方和 TF 都不符（F-004）。A4 实现 D-015 + D-033 之后，正确写法是「按 seed 固定选的一个恶意端的 fresh-PM」；该文件与 Bad-PFL 库双份同步，由 `test_metrics_doc.py` 守着 |

## 容易踩的坑

- **登记行 D01–D06 ≠ 决策 D-001 …**；**会话 A4 ≠ 行 A04**（见文件顶部）。
- **AUDIT 的行号只能是 `A##` / `D##` 两位数字**：`harness/registry.py:73` 的正则是 `^\|\s*([AD]\d{2})\s*\|`。写成 `A05a` 这种子行会被**静默忽略**。每行必须**恰好一个**单独成格的状态词。
- **格子里不要写 `|`**（哪怕转义成 `\|`；绝对值写成「绝对值」或 abs）：`audit_rows` 按 `|` 直接切分。
- **`tests/test_registry.py::test_real_audit_parses_and_is_open` 写死了 AUDIT 的行集合与各行状态**：现为 A01–A29 + D01–D06；A4 把 `align` 改 `done`、A08 / A26 改 `deviate` 时要同步改这里（故意这么设计的）。
- **D-029 的 pm_acc 门槛按陈旧 PM 列判**（与 P1 同一定义）；主列换成 fresh-PM 之后不能拿主列去比 P1。
- **`resnet10` 是冻结配置用的**，新 P2 配置要写 `resnet10_torch`；不要改 `build_resnet10`。
- **A27 的统计量快照取 head 阶段结束时**（body 还冻结在 φ_e），不是 body 阶段之后。
- **官方仓库里定义了不等于被用了**：`PMClient` / `PMPoisonClient` 从未被 import（F-022），事件处理器一个都没注册（F-032 / F-040）。
- **官方的「确定性」只覆盖 torch**：划分和客户端顺序没播种（F-024）。
- **比训练量要比总量，不要只比每轮步数**（F-029）。
- 旧方案的登记表在 `hfl-propagation/registry/v1.yaml`，**不能**放到那个目录顶层叫 `registry.yaml`：`run_exp3.sh` 的 `ls *.yaml` 会把它当成第 26 个格子提交。
- `submit.sh` 用 `RUN_GROUPS` 筛组，**不能叫 `GROUPS`**：那是 bash 的内置变量。
- 老 metrics.json 没有 `[Provenance]` 行 → `runs_table.py` 要加 `--legacy-protocol P1`。
- 按实际因素分组时，`3c_R5` 就是 `2edge_distributed`、`stoptest_10edge` 就是 `10edge_distributed`（F-002）；`figures.py` 拒绝混格，**不要用 `--allow-mixed` 绕过去**。
- 用 `git archive` 解到别处跑 L1 时，`test_provenance.py::test_git_commit_matches_git_rev_parse_on_this_repo` 会假红（没有 `.git`）。
