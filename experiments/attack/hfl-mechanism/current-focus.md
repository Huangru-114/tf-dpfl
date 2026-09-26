# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-26**（A4 会话结束时）。下一会话 = **A4 收口**：读 pilot 结果、关最后 4 行、升 P2。

## 几套编号（容易混，先看这里）

| 写法 | 是什么 | 在哪 |
|---|---|---|
| **A1–A4** | 审计会话的名字：A1 攻击、A2 训练协议、A3 FedRep / ResNet / HFL、**A4 = 按拍板改代码的实现会话** | PLAN §5 |
| **S1–S8** | 功能会话的名字：S3 新划分、S4 影子攻击者、S5 逐 edge 轮评估、S6 更新日志…… | PLAN §5 |
| **A01–A29** | `AUDIT.md` 的「对齐差异」行号 | AUDIT 第一、二节 |
| **D01–D06** | `AUDIT.md` 的「有意偏离登记」行号 | AUDIT 第三节 |
| **D-001 … D-042** | `DECISIONS.md` 的决策日志（带连字符、三位数），**与登记行 D01–D06 是两套东西** | DECISIONS |
| **F-001 … F-042 / N-001 … N-005** | `FINDINGS.md` 的证据条目 / 设计备注 | FINDINGS |
| **P0 / P1 / P2** | 数据批次的口径版本；只有 P2 进结论 | PLAN §0 |

## 下一会话唯一要回答的问题

**pilot 回来之后：D-029 过没过、D-031 的两种顺序是否相同、GPU 上是否确定 —— 据此关掉 AUDIT 最后 4 行，把口径升到 P2。**

## 客观判据

1. `python3 harness/pilot_a4.py experiments/attack/hfl-mechanism/pilot/registry.yaml --json <out>` 的输出就是判定，**不许手算**。
2. D-029 `pass` → A08 改 `deviate`、A25 改 `done`；`fail` → 逐个开关消融（在 pilot 表里加组：模板 + 一条 `set` 把某个开关改回旧值），回审计。
3. D-031 `same` → A26 改 `deviate`（维持 head_first）；`different` → 带回来由用户定。
4. DET `pass` → A15 改 `done`；`fail` → 看第一个分叉轮，回审计（D-028 写了「算子不支持或慢得不可接受就重议」）。
5. 4 行都关 → `fedavg/utils/provenance.py:PROTOCOL_VERSION = "P2"`；同步 `tests/test_registry.py::test_real_audit_parses_and_is_open`；2 个 smoke 复核标定（见下）；`status.py` 解除 blocked。

## 用户要在集群上做的

```bash
git pull                                   # 本分支：claude/federated-learning-experiment-review-pt5j1b
bash run_l1.sh 2>&1 | tail -3              # 预期：只有陷阱 #4 的 2 条红（F-041 修掉了另外 6 条）
bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh --dry-run    # 应列出 6 个 run
bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh              # 提交
# 回传 experiments/attack/hfl-mechanism/pilot/results/P1/*/*.metrics.json（6 个）
```

预算：D029 + A26 共 4 个整 run（D-029 估 2 个约 2.5 GPU-h），外加 DET 两个 5 云轮的短 run。
**模板全开时每轮后门评估约是原来的 2–3 倍**（fresh + 陈旧 + 白盒三套，N-005 在 CPU 上量到约 28 s / 轮）；
`metrics.json` 的 `timing_summary` 会给出实数，4 个整 run 如果超出 sbatch 的 24 h 要先告诉用户。

## A4 做完了什么（2026-09-26）

| 提交 | 内容 |
|---|---|
| `4c70f05` C1 | 开关底座 `fedavg/alignment.py` + 模板 `fedavg/config/alignment_p2.yaml` + `config_validate` 的 P2 核对 + `[设定4]` + 登记表 `overlays` |
| `c54a209` C2 | A01（官方 PGD）、A03（伯努利）、A05 训练侧、A14（官方生成器）；G7 前提 `data/pixel_space.py` + `data.normalize` / `data.augment`；F-041 的 6 条测试 |
| `943bbf9` C3 | A29 `resnet10_torch`、A27 BN 统计量私有、A24 只在 body 投毒、A26 `fedrep_order`、A25 `data/epoch_pipeline.py` |
| `d08bdea` C4 | A28 fresh-PM（`utils/pm.py`、`CloudServer.main_pm`）、A02 固定攻击者 + `[设定5]`、A06 `[ASR4]`、陈旧列 `[Stale]` / `[StaleASR]`、白盒 `[ASRwb]`；collect_metrics schema 3 |
| `7e43a42` C5 | A08 / D02 有效轮、D01 交错执行、A15 确定性 + `[Checksum]` |
| 本提交 C6 | P2 `base.yaml`、登记表 base + overlays + G7、pilot 表 + `submit_pilot.sh` + `harness/pilot_a4.py`、AUDIT / DECISIONS / FINDINGS / CLAUDE.md |

- AUDIT：`done` 新增 A01 A02 A03 A05 A06 A14 A16 A22 A24 A27 A28 A29 D01 D02；仍开着 **A08（open）A15（align）A25（align）A26（open）**。
- 本会话的用户拍板：D-039（开关 + 一套模板）、D-040（评估尾块循环补足）、D-041（陈旧列 ξ 用攻击者陈旧模型）、D-042（pilot 完整 P2 代码路径 + DET 组）。
- L1：本地无 TF 950+ passed；TF 2.15.1 CPU venv 只剩陷阱 #4 的 2 条红。本地接线 smoke（N-005）两种模板状态都跑通。

## 升 P2 时要一起做的

- `PROTOCOL_VERSION = "P2"` 之后，`registry.yaml` 的 G7 才能真跑（`materialize` 现在就能生成它的配置，但 `submit.sh` 在 AUDIT 关完之前拒绝提交）。
- 「2 个 smoke 复核标定」：P2 模板下 flat 与 2edge 各一个短 run，看 `timing_summary` 里 `bd_eval_fraction`、`[Checksum]`、`stop_reason`；评估开销若不可接受，考虑让白盒 / 陈旧副列隔几个评估点才算一次（要写进 DECISIONS）。
- `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」一句，按 D-015 + D-033 改为「按 seed 固定选的一个恶意端的 fresh-PM」—— 该文件与 Bad-PFL 库双份同步，由用户改（`test_metrics_doc.py` 守着）。

## 挂着的事（不属于收口，但别忘了）

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`；**Claude 没有合并** |
| 3.2 的假设重新表述 | 用户 | N-003：只在 body 投毒后「私有 head 吸收」的机制要改写；S6 / G4 之前定 |
| G4 的 FedAvg 一侧 | S6 | fresh-PM 对 FedAvg 就是当前 edge 模型（D-033），`private_state()` 为空 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007 |
| cifar100 静态触发器的标准化常数 | 需要时 | N-004：只记录，没改 |

## 容易踩的坑

- **新开关**：加进 `fedavg/alignment.py` 的开关表（进模板的放 `SWITCHES`，不进的放 `EXTRA_SWITCHES`），代码里只经 `get_switch(config, "…")` 读；`tests/test_alignment_switches.py` 会检查模板键集、P2 值 ≠ 旧值、以及每个键真的被读到。
- **P2 配置不能少开一项**：想做消融就登记在 pilot 表（P1 口径），不要在 P2 表里改回旧值（`config_validate` 会拒绝）。
- **`per_epoch` 管线只接了 Bad-PFL**：静态投毒（vanilla / neurotoxin）会被它绕过，`config_validate` 已拒绝这种组合。
- **fresh-PM 的草稿模型会被下一次同 slot 调用覆盖**（`CloudServer.pm_model`）：取一个、用完、再取下一个。
- **pilot 用 `pilot/submit_pilot.sh`，不是上一级的 `submit.sh`**（后者在 AUDIT 关完之前拒绝提交，D-006）。
- **D-029 的 pm_acc 门槛读 `pm_acc_stale`**（陈旧列，与 P1 同一定义）；主列 `pm_acc` 现在是 fresh-PM，不能拿它比 P1。
- **AUDIT 的行号只能是 `A##` / `D##`**，每行恰好一个状态词，格子里不写 `|`；`test_real_audit_parses_and_is_open` 写死了各行状态，改状态要同步改它。
- **`resnet10` 是冻结配置用的**，P2 写 `resnet10_torch`（模板里已经是）。
- 旧方案的登记表在 `hfl-propagation/registry/v1.yaml`，不能放到那个目录顶层（`run_exp3.sh` 会把它当格子提交）；`submit.sh` 用 `RUN_GROUPS`，不能叫 `GROUPS`。
- 用 `git archive` 解到别处跑 L1 时，`test_provenance.py::test_git_commit_matches_git_rev_parse_on_this_repo` 会假红（没有 `.git`）。
