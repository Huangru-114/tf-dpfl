# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-27**（A4 收口会话结束时）。**A4 已收口**：AUDIT 全部关闭，口径升 P2。
> 下一会话 = **用户从下面「下一步」里挑一个功能会话**（本文件不替用户定）。

## 几套编号（容易混，先看这里）

| 写法 | 是什么 | 在哪 |
|---|---|---|
| **A1–A4** | 审计会话的名字：A1 攻击、A2 训练协议、A3 FedRep / ResNet / HFL、**A4 = 按拍板改代码的实现会话** | PLAN §5 |
| **S1–S8** | 功能会话的名字：S3 新划分、S4 影子攻击者、S5 逐 edge 轮评估、S6 更新日志…… | PLAN §5 |
| **A01–A29** | `AUDIT.md` 的「对齐差异」行号 | AUDIT 第一、二节 |
| **D01–D06** | `AUDIT.md` 的「有意偏离登记」行号 | AUDIT 第三节 |
| **D-001 … D-046** | `DECISIONS.md` 的决策日志（带连字符、三位数），**与登记行 D01–D06 是两套东西** | DECISIONS |
| **F-001 … F-045 / N-001 … N-006** | `FINDINGS.md` 的证据条目 / 设计备注 | FINDINGS |
| **P0 / P1 / P2** | 数据批次的口径版本；只有 P2 进结论 | PLAN §0 |

## A4 收口的结论（2026-09-27）

| pilot | 结果 | 去向 |
|---|---|---|
| 第一轮 `e2ee9ca` | 6/6 `invalid`：GPU 确定性 × 推理模式 BN 求梯度（F-043） | D-043 `TorchBatchNorm`、D-044 有效性闸 |
| 第二轮 `2853433` | 6/6 有效；**D-029 `pass`**、**A15 `pass`**、**D-031 `different`**（F-045） | A08 `deviate`、A25 `done`、A15 `done`；A26 由用户定 head_first → `deviate`（D-045） |

- **AUDIT 全部关闭**，`fedavg/utils/provenance.py:PROTOCOL_VERSION = "P2"`。标定复核由 D029 两格抵扣（D-046）：
  bd_eval_fraction 0.28 / 0.38，单格约 2.25–2.3 h，停轮 `converged`。
- 旁注（F-045）：head_first 下 global_asr ≈ 0.998、fresh local_benign ≈ 0.91–0.95 —— **攻击接近饱和**。
  终值类比较可能撞天花板；G2 主结论读的是 T_θ（到达阈值的轮数），受影响小一些，但 θ=0.75 附近的分辨率要看实测。
- `status.py experiments/attack/hfl-mechanism/registry.yaml`：`todo=6`（G7）、`blocked=151`（只剩功能会话依赖）。

## 下一步（用户挑一个作为下一会话的唯一问题）

| 功能会话 | 解锁 | run 数 | 说明 |
|---|---|---|---|
| **S5** 逐 edge 轮评估 | **G2**（3-A 结构扫描，**主结论**） | 55 | 单独就能解锁最大、最核心的一组；另是 G1 的三个前提之一 |
| S3 新划分（C1–C4、层级 Dirichlet） | G3（3-B） | 21 | 另是 G0 / G1 的前提 |
| S4 影子攻击者 + 攻击起始轮 | G5（3.3） | 15 | 另是 G0 的前提 |
| S6 更新日志 | G4（3.2） | 12 | 3.2 的假设要先按 N-003 重新表述（用户） |
| S8 三层个性化 | G6（3-E，可选） | 9 | |
| —（无需会话） | **G7**（预处理对比，D-025） | 6 | 现在就能跑：先 `harness/registry.py …/registry.yaml --materialize`（只会写出 G7），提交后 `RUN_GROUPS=G7 bash …/submit.sh` |

> `materialize` 只写 requires 里功能会话都具备的组，所以 `INDEX.tsv` 里现在只会有 G7，
> `submit.sh`（只查 AUDIT 门槛）不会误交被功能会话挡住的组。

## A4 做完了什么（2026-09-26 / 27）

| 提交 | 内容 |
|---|---|
| `4c70f05` C1 | 开关底座 `fedavg/alignment.py` + 模板 `fedavg/config/alignment_p2.yaml` + `config_validate` 的 P2 核对 + `[设定4]` + 登记表 `overlays` |
| `c54a209` C2 | A01（官方 PGD）、A03（伯努利）、A05 训练侧、A14（官方生成器）；G7 前提 `data/pixel_space.py` + `data.normalize` / `data.augment`；F-041 的 6 条测试 |
| `943bbf9` C3 | A29 `resnet10_torch`、A27 BN 统计量私有、A24 只在 body 投毒、A26 `fedrep_order`、A25 `data/epoch_pipeline.py` |
| `d08bdea` C4 | A28 fresh-PM（`utils/pm.py`、`CloudServer.main_pm`）、A02 固定攻击者 + `[设定5]`、A06 `[ASR4]`、陈旧列 `[Stale]` / `[StaleASR]`、白盒 `[ASRwb]`；collect_metrics schema 3 |
| `7e43a42` C5 | A08 / D02 有效轮、D01 交错执行、A15 确定性 + `[Checksum]` |
| `e820c00` C6 | P2 `base.yaml`、登记表 base + overlays + G7、pilot 表 + `submit_pilot.sh` + `harness/pilot_a4.py`、AUDIT / DECISIONS / FINDINGS / CLAUDE.md |
| `299afe6` 收口 1 | `TorchBatchNorm`（D-043）、pilot 有效性闸（D-044）、`tests/test_bn_inference_determinism.py`（CPU 上模拟 GPU 检查） |
| 本提交 收口 2 | AUDIT 最后 4 行关闭、`PROTOCOL_VERSION = "P2"`、D-045 / D-046、F-045 |

- AUDIT：`done` 新增 A01 A02 A03 A05 A06 A14 A16 A22 A24 A27 A28 A29 D01 D02；收口时 A15 A25 `done`、A08 A26 `deviate` → **全部关闭**。
- 用户拍板：D-039（开关 + 一套模板）、D-040（评估尾块循环补足）、D-041（陈旧列 ξ 用攻击者陈旧模型）、D-042（pilot 完整 P2 代码路径 + DET 组）；收口：D-043 … D-046。
- L1（收口时）：本地无 TF 974 passed / 33 skipped / 3 xfailed；TF 2.15.1 CPU venv 只剩陷阱 #4 的 2 条红。

## 升 P2 时做了的 / 没做的

- 做了：`PROTOCOL_VERSION = "P2"`；`test_registry.py::test_real_audit_parses_and_is_closed`（原 `…_is_open`）断言 `audit_open_rows == []`；
  `test_status.py::test_v2_after_the_audit_only_feature_sessions_block`（G7 = todo，其余 151 只剩功能会话）。
- 没做（用户的事）：`experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」一句，按 D-015 + D-033 改为「按 seed 固定选的一个恶意端的 fresh-PM」
  —— 该文件与 Bad-PFL 库双份同步，由用户改（`test_metrics_doc.py` 守着）。

## 挂着的事（不属于收口，但别忘了）

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`；**Claude 没有合并** |
| 攻击接近饱和（F-045） | 用户 | G 组的终值类比较可能撞天花板；设计 / 解读时考虑 |
| `git_dirty` 排除结果文件 | 需要时 | 结果写在仓库里 → 同批后提交的 run 都会 `dirty=1`，区分不了代码脏还是结果脏（F-045） |
| 3.2 的假设重新表述 | 用户 | N-003：只在 body 投毒后「私有 head 吸收」的机制要改写；S6 / G4 之前定 |
| G4 的 FedAvg 一侧 | S6 | fresh-PM 对 FedAvg 就是当前 edge 模型（D-033），`private_state()` 为空 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007 |
| cifar100 静态触发器的标准化常数 | 需要时 | N-004：只记录，没改 |

## 容易踩的坑

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
