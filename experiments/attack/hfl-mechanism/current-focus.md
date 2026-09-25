# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-25**（A1 会话结束时）。下一会话 = **A2**。

## 下一会话唯一要回答的问题

**A2：训练协议与 Bad-PFL（官方代码 + 论文）逐项对齐了吗？**

范围：`AUDIT.md` 的 **A07–A11、A13、A15、A16、A20、A22、A23**：

| 行 | 内容 | 备注 |
|---|---|---|
| A07 | 聚合权重 | 官方不加权；HFL 下 cloud 权重的定义本身要决定 |
| A08 | 学习率 | **重新决定 LR 日程**（D-005 已作废 → D-010 待决）；官方与论文都是常数 0.1 |
| A09 | 本地训练量 | 官方与论文都是每轮 15 步 × batch 32；本仓库是 `local_epochs: 5`。影响最大的一条。注意 D-012 已定 FedRep 的 head、body 各 15 步 |
| A10 | 预处理 / 增强 | 官方只有 `ToTensor()`；本仓库标准化 + 增强，而且增强只采样一次就冻结（F-014） |
| A11 | 数据划分 | 客户端等大小可以对齐；HFL 的划分由 S3 负责，部分 `deviate` |
| A13 | 数值精度 | 官方 autocast 混合精度，本仓库 fp32 |
| A15 | 确定性 | 同 seed 重跑第 1 轮就分叉（F-002） |
| A16 | 死配置 | `label_smoothing: 0.1` 没人读（F-015） |
| A20 | 论文超参汇总 | 逐项核对进对应行后关闭 |
| A22 | 目标标签 | 论文说随机生成；官方代码默认 0；本仓库固定 0 |
| A23 | 训练轮数 | 论文 1000、官方 300、本仓库有效轮 floor 150 / cap 300；A09 改完后要重新标定 |

**本会话只做审计、给语义 diff 表，由用户逐行拍板。不改训练代码，不跑任何实验**（D-006）。

## 客观判据

1. 上表 11 行都从 `open` 变成 `align` / `deviate` / `done`（A20 查完即 `done`），每行都由用户拍板，并在 `DECISIONS.md` 追加一条（接着 D-022 编号）。
2. 每个 `align` 都写出 L1 测试的**断言设计**（什么输入、得到什么解析值）；测试本身在 A4 实现。
3. A08：LR 日程有明确答案（常数还是按某种轮数衰减），并写明它在 flat 与 HFL 之间是否可比。
4. A09 + A23：给出 A4 重新标定预算的方法（改成 15 步后收敛速度会变，floor / cap 要重新定）。

## 开工前

1. `git log --oneline -3`，确认在 `claude/federated-learning-experiment-review-pt5j1b` 上，最后一个是 A1 的提交。
2. 跑 L1，记下当天的基线：
   ```bash
   bash run_l1.sh 2>&1 | tail -3
   # 本地没有 pytest 时：python3 -m venv <scratch>/venv && <scratch>/venv/bin/pip install pytest numpy pyyaml matplotlib
   #                     然后 TFDPFL_PY=<scratch>/venv/bin/python bash run_l1.sh
   ```
   A1 结束时本地（无 TF）是 **853 passed / 22 skipped / 3 xfailed**。
3. 参考源：
   - 官方代码：`https://raw.githubusercontent.com/fmy266/Bad-PFL/main/<file>`。A1 时的 etag：`fba.py` `d65deddbd62a…`、`client.py` `69e2b095c234…`、`main.py` `93855051ac20…`；变了就先重读。
   - **论文 PDF 不在仓库里**（D-013）。A2 要核对 p.7 §4.1 与附录 A（p.13–14）；AUDIT A20 已有摘录，要看原文就**请用户重新上传**。
     读 PDF：本机没有 poppler，Read 工具读不了 PDF → 在 scratch venv 里装 `pypdf`，逐页 `extract_text()`。
4. 状态核对：
   ```bash
   python3 harness/status.py experiments/attack/hfl-mechanism/registry.yaml --quiet    # 应为 blocked=151
   python3 harness/status.py experiments/attack/hfl-propagation/registry/v1.yaml --quiet # mismatch=6 failed=1 done=18 orphan=1
   ```

## 之后的顺序（PLAN.md §5）

A2 训练协议 → A3（A12 的「head 是哪几层、BN 算不算 body」、**FedRep 训练顺序**、A17 ResNet-10、A18、A21 HFL 形式化）+ D01–D06 签字 →
A4 按拍板改代码，`PROTOCOL_VERSION` 升 P2，重新标定预算 → S3…S8。

---

## A1 做完了什么（2026-09-25）

| 行 | 状态 | 决定 |
|---|---|---|
| A01 ξ 构造 | `align` | D-014：对齐官方单步 PGD（随机起点、clamp、投影）；论文 Eq.6 的 FGSM 不采纳 |
| A02 评估 ξ 的模型 | `align` | D-015：按 seed 固定选一个恶意端，全程不变，写进 `[设定]` |
| A03 投毒量 | `align` | D-016：逐样本伯努利 |
| A04 生成器数据 | `deviate` | D-017：干净数据（论文 Eq.7），偏离官方代码（疑似 bug）= 现状 |
| A05 BN 模式 | `align` | D-018：投毒 / 评估时生成器用 batch 统计；训练期求 ξ 时 F 用训练模式；评估时攻击者 F 用推理模式（偏离官方的顺序依赖） |
| A06 ASR | `align` | D-019：四列都报，主指标 = 过滤 × 仅良性 |
| A14 生成器结构 | `align` | D-020：以官方代码为准（bias、eps 1e-5、torch 默认 init、输入反标准化到 [0,1]、末层无 BN） |
| **A24（新）** FedRep 哪个阶段投毒 | `align` | D-021：只在 body 阶段投毒（论文 Alg.1） |

另外：
- **集群 smoke 验证了 `[Provenance]`**（F-018）：`git=76c72a752f82`、`dirty=0`、`host=n26`。
- FINDINGS F-019（论文与代码冲突的三处）、F-020（取整偏差）、F-021（生成器 BN 失配）、F-022（官方 PM* 是死代码）、F-023（smoke 的 spread 布点失效）、N-003（3.2 要重新表述）。
- `tests/test_registry.py` 把 A24 加进 AUDIT 行集合，并把 A1 八行的状态写成断言钉住。

S1 做完了什么（2026-09-24/25）：见 `git log`（`9f11865` 计划入库、`4cee173` 溯源与登记表、`b5f4bc1` 分析层、`76c72a7` 交接）。
用户已拍板的决定：D-001 至 D-021（见 `DECISIONS.md`）。

## 挂着的事（不属于 A2，但别忘了）

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main` 41 个提交、落后 0。CLAUDE.md 要求会话结束、L1 绿后合回 main；**Claude 没有合并** |
| 3.2 的假设重新表述 | 用户 | N-003：只在 body 投毒后，「私有 head 吸收」的机制要改写；S6 / G4 之前定 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007。只有还要用旧方案的防御格时才需要 |
| `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」 | 用户（两库同步） | 与官方和 TF 都不符（F-004）。A4 实现 D-015 之后，正确写法是「按 seed 固定选的一个恶意端的模型」；该文件与 Bad-PFL 库双份同步，由 `test_metrics_doc.py` 守着 |

## 容易踩的坑

- **AUDIT 的行号只能是 `A##` / `D##` 两位数字**：`harness/registry.py:73` 的正则是 `^\|\s*([AD]\d{2})\s*\|`。写成 `A05a` 这种子行会被**静默忽略**，门槛就看不到它。每行也必须**恰好一个**单独成格的状态词（`` `open` `` / `` `align` `` / `` `deviate` `` / `` `done` ``）。
- **`tests/test_registry.py::test_real_audit_parses_and_is_open` 写死了 AUDIT 的行集合**：新增行时要同步改（故意这么设计的）。
- **官方仓库里定义了不等于被用了**：`PMClient` / `PMPoisonClient` 从未被 import（F-022）。引用官方代码作证据前，先确认它在 `main.py` 的执行路径上。
- 旧方案的登记表在 `hfl-propagation/registry/v1.yaml`，**不能**放到那个目录顶层叫 `registry.yaml`：`run_exp3.sh` 的 `ls *.yaml` 会把它当成第 26 个格子提交。
- `submit.sh` 用 `RUN_GROUPS` 筛组，**不能叫 `GROUPS`**：那是 bash 的内置变量（当前用户的 gid 数组）。
- 老 metrics.json 没有 `[Provenance]` 行 → `runs_table.py` 要加 `--legacy-protocol P1`，否则口径版本记成 `unknown`。
- 按实际因素分组时，`3c_R5` 就是 `2edge_distributed`（共 4 份重复）、`stoptest_10edge` 就是 `10edge_distributed`（F-002）。画图时组内混了不同格子，`figures.py` 会拒绝画，**不要用 `--allow-mixed` 绕过去**。
- 用 `git archive` 解到别处跑 L1 时，`test_provenance.py::test_git_commit_matches_git_rev_parse_on_this_repo` 会假红（没有 `.git`），在真仓库里是绿的。
