# current-focus —— Experiment 3（改版）· 交接

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-27**（一卡多跑接入 + 评估降频会话结束时，`c88a023` / `68f865d`）。
> **下一会话 = S8（三层个性化 → G6）**（D-056）。G2 与 S5 暂缓；S5 的设计已拍板为预案（D-055）。

## 几套编号（容易混，先看这里）

| 写法 | 是什么 | 在哪 |
|---|---|---|
| **A1–A4** | 审计会话的名字：A1 攻击、A2 训练协议、A3 FedRep / ResNet / HFL、**A4 = 按拍板改代码的实现会话** | PLAN §5 |
| **S1–S8** | 功能会话的名字：S3 新划分、S4 影子攻击者、S5 逐 edge 轮评估、S6 更新日志、**S8 三层个性化**…… | PLAN §5 |
| **A01–A29** | `AUDIT.md` 的「对齐差异」行号 | AUDIT 第一、二节 |
| **D01–D06** | `AUDIT.md` 的「有意偏离登记」行号 | AUDIT 第三节 |
| **D-001 … D-056** | `DECISIONS.md` 的决策日志（带连字符、三位数），**与登记行 D01–D06 是两套东西** | DECISIONS |
| **F-001 … F-053 / N-001 … N-006** | `FINDINGS.md` 的证据条目 / 设计备注 | FINDINGS |
| **P0 / P1 / P2** | 数据批次的口径版本；只有 P2 进结论 | PLAN §0 |

## 本会话做了什么（2026-09-27）

| 提交 | 内容 | 决定 |
|---|---|---|
| `c88a023` | **一卡多跑接进** `submit.sh` / `pilot/submit_pilot.sh`（共用 `submit_lib.sh`）：按格子分包、每格先交 K=2 的探路包、之后按**真实显存峰值**定 K、OOM 自动降档；`pack.sbatch` 把被吞掉的 OOM 判成 exit 86；服务器新打 `[GPUMem]`；stale 不重交 | D-052 / D-053 |
| `68f865d` | **评估降频**：白盒 ASR 关、陈旧 ASR 与陈旧 pm_acc 隔点（同一批点），终值按「末 10 个评估点窗口」；`[TimingAcc]` 分项计时 | D-050 / D-054 |
| 文档提交（紧随其后） | 文档与交接：D-052 … D-056、F-052 / F-053、PLAN / registry 标注 G2 与 S5 暂缓 | D-055 / D-056 |

- L1：本地无 TF 1072 passed / 38 skipped / 3 xfailed；TF 2.15.1 CPU venv 只有陷阱 #4 的 2 条红。
- **反向锚点**：G7 与 pilot 的 `runs_table` / `pilot_a4` / `g7_posthoc` 输出与改动前逐字节相同。
- **没做**：S5、G2 materialize、G7 重新 materialize（`configs/INDEX.tsv` 仍是旧 sha，G7 仍显示 done）。

## 下一会话唯一要回答的问题（D-056）

**「实现 S8（3-E 三层个性化：cloud 只聚合全局共享层、edge 内共享一段中间层、客户端私有 head），让 G6 能跑。」**

原文 §8：三种划分（ResNet-10）——(a) FedRep 基线（body 全部云端共享、head 私有）；(b) 最后一个残差块只在 edge 内共享；
(c) 最后两个残差块只在 edge 内共享。指标：逐 edge 的 benign ASR 与 MTA、跨 edge 迁移的 T50。

**开工前先和用户对齐（按 CLAUDE.md：先出语义 diff 表，用户说「开始改」再动代码）**，下面四件本会话看到了、没有替用户决定：

1. **3-E 的前提被 F-051 动摇了**：原文说「如果 3.2 的私有头吸收成立……」；而 F-051 发现**私有 head 挡不住 ξ**（白盒 ≈ 主列），
   3.2 本身也等着按 N-003 重新表述（G4 搁置）。S8 照做没问题，但结论怎么读要先说清楚。
2. **判定用的是 excess ASR = ASR − floor**（PLAN §3），floor 要 S4 / G0（ρ=0 影子攻击者）才有；
   而 `registry.yaml` 里 G6 的 `requires` 只有 `[audit, S8]`。要么补依赖，要么判定改用原始 ASR —— 由用户定。
3. **MTA 门槛「≤ 0.02」还是 ⚠待确认**；F-051：fresh-PM 的干净精度定义偏差在 10edge 达 0.094，远大于 0.02
   → 精度判定至少要同时报陈旧 pm_acc（D-054 后它隔点算，终值窗口约 5 点）。
4. **陷阱 #8**：cloud 层聚合入口 `CloudServer.aggregate_edges` 现在是「永远朴素 FedAvg」。S8 要 cloud 只聚合共享段、
   下行广播不覆盖 edge 内共享段 —— 会碰到这里；fresh-PM（D-033）的定义也要随之扩成 [edge 共享段 + 全局段, 私有 head, 私有统计量]。

代码入口（PLAN §2）：`get_base_head_indices` 加第三组；cloud / edge 各自只聚合对应层；登记表 G6 的 `set:` 要补
（布点 / 轮数 / 评估网格三件，见 registry.yaml 页首）。G6 是 9 个 run。

## 交作业怎么用（本会话接好的）

```bash
bash experiments/attack/hfl-mechanism/submit.sh --status                 # done / stale / todo
PACK=3 RUN_GROUPS=G6 bash experiments/attack/hfl-mechanism/submit.sh --dry-run
PACK=3 RUN_GROUPS=G6 bash experiments/attack/hfl-mechanism/submit.sh     # 第一次：每格只交一个 K=2 的探路包
# 探路包回传后（results/P2/<组>/<组>__<格>__pack-k2-s<seed>.gpu.json）再跑同一条命令 → 按实测峰值定 K
```

- **交完等回传再跑下一次**：脚本不查 SLURM 队列，连着跑两遍会重复提交。
- 探路包回传后先看 gpu.json 的 `run_peak_max_mib`（真实峰值）与 `mem_max_mib / k`（整卡读数）差多少 ——
  这是校准 `PACK_MEM_PCT=85` / `PACK_CTX_MIB=1024` 的唯一数据（F-053，**目前没有证据**）。
- stale（exit 0 但 sha 不符）默认不重交；确实要重跑设 `RESUBMIT_STALE=1`。
- 下一次 materialize（S8 之后）G7 会显示 stale —— 预期（base.yaml 加了评估降频）。

## 功能会话一览

| 功能会话 | 解锁 | run 数 | 说明 |
|---|---|---|---|
| **S8** 三层个性化 | G6（3-E，可选） | 9 | **下一会话**（D-056）；开工前的四件事见上 |
| S5 逐 edge 轮评估 | G2（3-A）、G1 的前提之一 | 55 | **暂缓**（D-056）；预案已拍板（D-055）：`eval_grid: 5`、轻评估只算主列并喂停止判据（横轴改网格序号，F-052）、GM / EM 只在网格点上算 |
| S3 新划分（C1–C4、层级 Dirichlet） | G3（3-B） | 21 | 另是 G0 / G1 的前提 |
| S4 影子攻击者 + 攻击起始轮 | G5（3.3） | 15 | 另是 G0 的前提；3-E 的 excess ASR 也要它的 floor |
| S6 更新日志 | G4（3.2，**搁置**，D-047） | 12 | 3.2 的假设要先按 N-003 重新表述（用户）；F-051「私有 head 挡不住 ξ」是相关证据 |
| —（无需会话） | **G7**（预处理对比，D-025） | 6 | ✅ 已跑完并判定（`5edd4df`，事后判据「是」，D-049 / F-050） |

- G2 的规模还没定（G2P 已回来：`consistent`，F-049）——**由用户定**，定之前不做 S5。
- 在 S5 之前，G2 的跨 R 比较受 F-052 影响：停止判据的斜率横轴是云轮号，flat 比 R5 宽松 5 倍。

## 挂着的事

| 事 | 谁 | 说明 |
|---|---|---|
| 合并回 main | 用户决定 | 本分支领先 `origin/main`；**Claude 没有合并** |
| G2 的规模 | 用户 | D-056；定了再开 S5 |
| 一卡多跑的参数校准 | 第一批满长包回来后 | `PACK_MEM_PCT` / `PACK_CTX_MIB` 没有证据（D-052 / F-053） |
| 攻击接近饱和（F-045 / F-049） | 用户 | G 组的终值类比较可能撞天花板；设计 / 解读时考虑 |
| G7 的混杂 | 用户 | 官方预处理下干净精度低约 0.10，「攻击更容易」与「模型更弱」分不开（F-050） |
| fresh-PM 低估干净精度（F-051） | 用户 | 随 edge 数增大（10edge +0.094）；跨拓扑的精度结论同时报陈旧 pm_acc（D-054 后隔点算） |
| δ-only / ξ-only 消融 | 需要时 | 解释「白盒 ≈ 主列」；单独开实验（D-051） |
| `git_dirty` 排除结果文件 | 需要时 | 结果写在仓库里 → 同批后提交的 run 都会 `dirty=1`（F-045） |
| 3.2 的假设重新表述 | 用户 | N-003；S6 / G4 之前定；也影响 3-E 的解读 |
| `experiments/METRICS.md`「ξ 用 mal[0]」一句 | 用户 | 与 Bad-PFL 库双份同步（`test_metrics_doc.py` 守着）；按 D-015 + D-033 改 |
| S1b：修 `exp3_cell.sbatch` 写死的 `--defense none` | 需要时 | D-007 |
| cifar100 静态触发器的标准化常数 | 需要时 | N-004：只记录，没改 |

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
  `test_status.py::test_v2_after_the_audit_only_feature_sessions_block`（G7 = todo，其余 151 只剩功能会话）。
- 没做（用户的事）：`experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」一句，按 D-015 + D-033 改为「按 seed 固定选的一个恶意端的 fresh-PM」
  —— 该文件与 Bad-PFL 库双份同步，由用户改（`test_metrics_doc.py` 守着）。
