# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-25**（A2 会话结束时）。下一会话 = **A3**。
>
> **术语**：A1–A4 是**会话名**（PLAN §5），A01–A25 是 AUDIT 的**行号**，两者无关。
> A4 = 按拍板改代码的实现会话（与行 A04 撞名；没有改名，因为 `harness/registry.py:256` 的报错文案与两条测试匹配「A4」）。

## 下一会话唯一要回答的问题

**A3：FedRep 实现细节、ResNet-10、HFL 形式化与官方 / 文献对得上吗？另外把 D01–D06 签字。**

| 行 | 内容 | 备注 |
|---|---|---|
| A12 余项 | head 是哪几层？BN 算不算 body？ | A12 的训练量与 lr 已由 D-024 定（维持 P1 现状，取代 D-012），这里只剩层的归属。官方 FedBN 把 BN 留在本地（`pfl.py:3-24`）；本仓库 FedRep 的 BN 进聚合 |
| FedRep 训练顺序 | 论文 Alg.1 先 local 后 personalized；TF 先 head 后 body | A24（D-021）只定了「哪个阶段投毒」 |
| A17 | ResNet-10 逐层对比（官方 `resnet.py` 152 行 vs `models/resnet.py`） | A2 会话已下载 raw，**没读** |
| A18 | `trigger.py`、`event_emitter.py` 是否影响攻击 / 评估流程 | **A2 已预读**（FINDINGS F-032）：`emit` 全是空操作、`grid_trigger_adder` 没人调用。A3 复核后即可关 |
| A21 | HFL 的形式化（Liu et al. HierFAVG：edge / cloud 聚合权重、κ₁/κ₂） | A07 已定保留样本加权（D-027），与 HierFAVG 一致 |
| D01–D06 | HFL 特有的有意偏离，逐条签字 | D04「探针 = 客户端留出分片」与 A11 / D-027 相关 |

**本会话只做审计、给语义 diff 表，由用户逐行拍板。不改训练代码，不跑任何实验**（D-006）。
按 **D-022** 的原则判：结构类（层归属、训练顺序、HFL 形式化）影响「我们测的是不是 Bad-PFL × FedRep」，要对齐或写明偏离理由；纯训练超参不为对齐而改。

## 客观判据

1. A12 余项、FedRep 训练顺序、A17、A18、A21 都有用户拍板的结论，DECISIONS 从 D-030 编号。
2. D01–D06 每条是 `deviate`（签字）或给出修改方案。
3. 每个 `align` 写出 L1 断言设计（什么输入、得到什么解析值）；测试本身在 A4 实现。

## 开工前

1. `git log --oneline -3`，确认在 `claude/federated-learning-experiment-review-pt5j1b` 上，最后一个是 A2 的提交。
2. 跑 L1，记下当天的基线：
   ```bash
   bash run_l1.sh 2>&1 | tail -3
   # 本地没有 pytest 时：python3 -m venv <scratch>/venv && <scratch>/venv/bin/pip install pytest numpy pyyaml matplotlib pypdf
   #                     然后 TFDPFL_PY=<scratch>/venv/bin/python bash run_l1.sh
   ```
   A2 结束时本地（无 TF）是 **853 passed / 22 skipped / 3 xfailed**。
3. 参考源：
   - 官方代码：`https://raw.githubusercontent.com/fmy266/Bad-PFL/main/<file>`。A2 时 `fba.py` `d65deddbd62a…`、`client.py` `69e2b095c234…`、`main.py` `93855051ac20…`、`resnet.py` `a280f45176c9…`；变了就先重读。
   - **论文 PDF 不在仓库里**（D-013），要看原文请用户上传。Read 工具读不了 PDF（本机没有 `pdftoppm`）→ 在 scratch venv 里用 `pypdf` 逐页 `extract_text()`。A2 已抽过 p.4、7、8、13、14、18（摘录在 FINDINGS F-031）。
4. 状态核对：
   ```bash
   python3 harness/status.py experiments/attack/hfl-mechanism/registry.yaml --quiet    # 应为 blocked=151
   python3 harness/status.py experiments/attack/hfl-propagation/registry/v1.yaml --quiet # mismatch=6 failed=1 done=18 orphan=1
   ```

## 之后的顺序（PLAN.md §5）

A3（本表）→ **A4 实现会话**：按拍板改代码 + L1、跑 D-029 可行性实验、`PROTOCOL_VERSION` 升 P2、重新标定 → S3…S8。

### A4 从 A2 继承的清单

| 事 | 依据 | 要点 |
|---|---|---|
| LR 按有效轮衰减的开关 | D-023 | `0.1·0.992^t_eff`、head `0.005·0.992^t_eff`；R_edge=1 时与现在逐字节相同（L1 要证明这一条） |
| 取数修复的开关 | D-026（A25） | 每 epoch 重洗 + 重增强、`drop_last` 且尾批轮换；FedAvg / FedRep / 生成器走同一个函数 |
| **可行性实验** | D-029 | `flat_baseline` + `2edge_distributed`，各 seed42 × 1，判据已预先定好；登记在单独的 pilot 登记表（不进 P2）；通过 → A08 改 `deviate`、A25 配 L1 后改 `done` |
| 确定性 | D-028（A15） | `enable_op_determinism()` + `[Checksum]` 行；同 seed 两跑前 5 轮哈希相同。有算子不支持就回审计 |
| 删 `label_smoothing` | D-028（A16） | P2 配置不含它；守卫：`fedavg/` 中 0 处读取 |
| 目标标签守卫 | D-028（A22） | P2 基配置 `target_label == 0` |
| G7 预处理对比 | D-025 | 先让 `client_badpfl.py:53-56` 的 ε/σ 换算与 `attack/triggers.py` 跟随标准化开关（F-027），再在 `registry.yaml` 登记 G7 |
| 等大小划分 | D-027 | **不是 A4 的事**，是 S3 的硬要求 |

---

## A2 做完了什么（2026-09-25）

用户先否了第一版（逐项对齐官方训练协议），定下原则 **D-022**：攻击定义必须对齐；训练协议不为对齐而对齐，看内部效度、bug、与论文量级是否可比；改调过参的值要先做可行性实验。

| 行 | 状态 | 决定 |
|---|---|---|
| A07 聚合权重 | `deviate` | D-027：保留样本加权；S3 等大小后与官方等权数值等价 |
| A08 学习率 | `open` | D-023：保留 0.992 / head 0.005，改按有效轮衰减；D-029 通过后关 |
| A09 本地训练量 | `deviate` | D-024：维持 ep5（总本地训练量已与论文相当，F-029） |
| A10 预处理 | `deviate` | D-025：保留标准化 + 增强；官方预处理作 G7 对比 |
| A11 划分 | `deviate` | D-027：不照抄官方；S3 要求等大小（F-028） |
| A12 FedRep | `align` → `deviate` | D-024 取代 D-012：head 1 epoch + lr 0.005、body 5 epoch |
| A13 精度 | `deviate` | D-028：fp32 |
| A15 确定性 | `align` | D-028：op determinism + `[Checksum]` |
| A16 死配置 | `align` | D-028：删 `label_smoothing` |
| A20 论文超参 | `done` | 按原文逐项分派（F-031） |
| A22 目标标签 | `align` | D-028：固定 0（用户拍板） |
| A23 轮数 | `deviate` | D-028：保留 floor 150 / cap 300 |
| **A25（新）** 每 epoch 取数 | `align` | D-026：修 F-025 的缓存冻结 |

新证据：FINDINGS F-024 … F-032。其中 **F-025**（seed42 下 3.2% 的训练样本从未参与训练）与 **F-028**（恶意端数据占比随布点在 0.092–0.142 之间变化，provisional）会影响对 P1 结果的解读。

用户已拍板的决定：D-001 至 D-029（D-010 → D-023、D-012 → D-024 已被取代）。

## 挂着的事（不属于 A3，但别忘了）

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`、落后 0。CLAUDE.md 要求会话结束、L1 绿后合回 main；**Claude 没有合并** |
| 3.2 的假设重新表述 | 用户 | N-003：只在 body 投毒后，「私有 head 吸收」的机制要改写；S6 / G4 之前定 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007。只有还要用旧方案的防御格时才需要 |
| `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」 | 用户（两库同步） | 与官方和 TF 都不符（F-004）。A4 实现 D-015 之后，正确写法是「按 seed 固定选的一个恶意端的模型」；该文件与 Bad-PFL 库双份同步，由 `test_metrics_doc.py` 守着 |

## 容易踩的坑

- **会话名 A4 ≠ AUDIT 行 A04**（见文件顶部）。「在 A4 实现」指实现会话。
- **AUDIT 的行号只能是 `A##` / `D##` 两位数字**：`harness/registry.py:73` 的正则是 `^\|\s*([AD]\d{2})\s*\|`。写成 `A05a` 这种子行会被**静默忽略**，门槛就看不到它。每行也必须**恰好一个**单独成格的状态词（`` `open` `` / `` `align` `` / `` `deviate` `` / `` `done` ``）。
- **格子里不要写 `|`**（哪怕转义成 `\|`）：`audit_rows` 按 `|` 直接切分，会多切出格子。
- **`tests/test_registry.py::test_real_audit_parses_and_is_open` 写死了 AUDIT 的行集合与 A1 / A2 各行的状态**：
  - 行集合现为 A01–A25 + D01–D06；
  - 新增行、改状态时要同步改这里（故意这么设计的）。
- **官方仓库里定义了不等于被用了**：`PMClient` / `PMPoisonClient` 从未被 import（F-022），事件处理器一个都没注册（F-032）。引用官方代码作证据前，先确认它在 `main.py` 的执行路径上。
- **官方的「确定性」只覆盖 torch**：划分和客户端顺序没播种（F-024）。不要拿「官方可复现」当论据。
- **比训练量要比总量，不要只比每轮步数**（F-029 的教训）。
- 旧方案的登记表在 `hfl-propagation/registry/v1.yaml`，**不能**放到那个目录顶层叫 `registry.yaml`：`run_exp3.sh` 的 `ls *.yaml` 会把它当成第 26 个格子提交。
- `submit.sh` 用 `RUN_GROUPS` 筛组，**不能叫 `GROUPS`**：那是 bash 的内置变量（当前用户的 gid 数组）。
- 老 metrics.json 没有 `[Provenance]` 行 → `runs_table.py` 要加 `--legacy-protocol P1`，否则口径版本记成 `unknown`。
- 按实际因素分组时，`3c_R5` 就是 `2edge_distributed`（共 4 份重复）、`stoptest_10edge` 就是 `10edge_distributed`（F-002）。画图时组内混了不同格子，`figures.py` 会拒绝画，**不要用 `--allow-mixed` 绕过去**。
- 用 `git archive` 解到别处跑 L1 时，`test_provenance.py::test_git_commit_matches_git_rev_parse_on_this_repo` 会假红（没有 `.git`），在真仓库里是绿的。
