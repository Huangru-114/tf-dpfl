# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-25**（S1 会话结束时）。下一会话 = **A1**。

## 下一会话唯一要回答的问题

**A1：攻击部分与 Bad-PFL（官方代码 + 论文）逐行对齐了吗？**

范围：`AUDIT.md` 第一节的 **A01–A06 与 A14**：

| 行 | 内容 |
|---|---|
| A01 | ξ 的构造 |
| A02 | 评估时 ξ 用哪个模型（已定对齐，只剩「用哪个恶意端」） |
| A03 | 投毒量 |
| A04 | 生成器训练数据 |
| A05 | BN 的训练 / 推理模式 |
| A06 | ASR 定义（过滤 / 不过滤、全体 / 仅良性） |
| A14 | 生成器结构 |

**本会话只做审计、给语义 diff 表，由用户逐行拍板。不改训练代码，不跑任何实验**（D-006）。

## 客观判据

1. 这 7 行的状态都从 `open` 变成 `align` 或 `deviate`，用户逐行拍板，每条在 `DECISIONS.md` 追加一行。
2. 每个 `align` 都写出 L1 测试的**断言设计**（写成什么输入、得到什么解析值）；测试本身在 A4 实现。
3. A02：「用哪个恶意端的模型算 ξ」有明确答案。官方用的是循环里**最后一个**恶意端（`fba.py:64` 被反复覆盖）；本仓库 `shared_generator=true`。
4. A06：主指标定下来。四种组合 = 过滤 / 不过滤 × 全体（官方含恶意端）/ 仅良性。

## 开工前

1. `git log --oneline -3`，确认在 `claude/federated-learning-experiment-review-pt5j1b` 上，并且包含本会话的提交（最后一个是交接提交）。
2. 跑 L1，记下当天的基线：
   ```bash
   bash run_l1.sh 2>&1 | tail -3
   # 本地没有 pytest 时：python3 -m venv <scratch>/venv && <scratch>/venv/bin/pip install pytest numpy pyyaml matplotlib
   ```
   S1 结束时本地（无 TF）是 **853 passed / 22 skipped / 3 xfailed**。
3. 参考源：
   - 官方代码：`https://raw.githubusercontent.com/fmy266/Bad-PFL/main/<file>`（`fba.py` 的 etag 应当仍是 `d65deddbd62a…`，变了就先重读）。
   - **论文 PDF 不在仓库里**（D-013）。新会话看不到上一会话上传的文件 → **请用户重新上传**。读 PDF：在 scratch venv 里装 `pypdf`，逐页 `extract_text()`；设置在 p.7，附录 A 在 p.13–14。
4. 状态核对：
   ```bash
   python3 harness/status.py experiments/attack/hfl-mechanism/registry.yaml --quiet    # 应为 blocked=151
   python3 harness/status.py experiments/attack/hfl-propagation/registry/v1.yaml --quiet # mismatch=6 failed=1 done=18 orphan=1
   ```

## 之后的顺序（PLAN.md §5）

A1 攻击 → A2 训练协议（A07–A11、A13、A15、A16、A20、A22、A23，其中 **LR 重新决定**）→
A3 FedRep 实现细节 / ResNet-10 / HFL 形式化（A12 的「head 是哪几层」、A17、A18、A21）+ D01–D06 签字 →
A4 按拍板改代码，`PROTOCOL_VERSION` 升 P2，重新标定预算 → S3…S8（划分、下限、逐 edge 轮评估、更新日志……）。

---

## S1 做完了什么（2026-09-24/25）

| 提交 | 内容 |
|---|---|
| `9f11865` | 计划入库：PLAN / AUDIT / DECISIONS / FINDINGS / 原稿逐字副本 |
| `4cee173` | `[Provenance]` 溯源行、登记表 `harness/registry.py`、对账 `harness/status.py`、`submit.sh` + `cell.sbatch` |
| `b5f4bc1` | `runs_table.py` / `verdicts.py` / `figures.py`；CLAUDE.md 陷阱 #19–#22；旧方案冻结横幅 |
| 交接提交 | 读论文与两份 FedRep 实现 → AUDIT A12/A19/A20/A22/A23、D-012/D-013、F-016/F-017；本文件 |

用户已拍板的决定：D-001 至 D-013（见 `DECISIONS.md`）。最新两条：
- **D-012**：FedRep 与 Bad-PFL 论文一致，head、body 各 15 步，lr 同为 0.1，batch 32。
- **D-013**：论文 PDF 不入库。

## 挂着的事（不属于 A1，但别忘了）

| 事 | 谁 | 说明 |
|---|---|---|
| 集群 smoke 核对 `[Provenance]` | 用户 | `sbatch run_smoke.sh attack hfl-propagation badpfl none smoke_prov hier_fedavg_fedrep`，看日志里 `[Provenance] … git=` 是否非空。这一点本地验证不了 |
| 合并回 main | 用户决定 | 本分支领先 `origin/main` 38 个提交、落后 0。CLAUDE.md 要求会话结束、L1 绿后合回 main；**Claude 没有合并** |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007。只有还要用旧方案的防御格时才需要；新方案的 `cell.sbatch` 已经只传 `--config` |
| `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」 | 用户（两库同步） | 与官方和 TF 都不符（F-004）；该文件与 Bad-PFL 库双份同步，由 `test_metrics_doc.py` 守着 |

## 容易踩的坑（本会话踩过或差点踩的）

- 旧方案的登记表在 `hfl-propagation/registry/v1.yaml`，**不能**放到那个目录顶层叫 `registry.yaml`：`run_exp3.sh` 的 `ls *.yaml` 会把它当成第 26 个格子提交。
- `submit.sh` 用 `RUN_GROUPS` 筛组，**不能叫 `GROUPS`**：那是 bash 的内置变量（当前用户的 gid 数组）。
- 老 metrics.json 没有 `[Provenance]` 行 → `runs_table.py` 要加 `--legacy-protocol P1`，否则口径版本记成 `unknown`。
- 按实际因素分组时，`3c_R5` 就是 `2edge_distributed`（共 4 份重复）、`stoptest_10edge` 就是 `10edge_distributed`（F-002）。画图时组内混了不同格子，`figures.py` 会拒绝画，**不要用 `--allow-mixed` 绕过去**。
