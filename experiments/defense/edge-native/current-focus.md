# 交接 —— 阶段三（edge 原生防御）· 2026-11-24 晚：**D3 全部收尾**（P-a 追加探针判 `not_removable`，F-137；见〇·〇·〇·〇·〇·〇）；**待用户决定**：D3 结论怎么写进论文 6.1 / 6.3；之后 = 攻击基线接线（时间盒 2026-12-06）

> 过程与数字：本目录 `FINDINGS.md`（到 F-137；预注册 N-008 … N-016 都已生效）与 `DECISIONS.md`（到 D-140）。
> 阶段性总结报告：本目录 `REPORT.md`（11-07；结果、CCS 线、术语表）；文献：`LITERATURE.md` 第 3 部分（11-17 新颖性核查 + 三篇全文读数）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（用户 2026-10-10）。分支 `claude/eloquent-faraday-b9kmop`。
> ⚠️ 用户要求（2026-11-01、11-17 重申）：**文档和回复里不要用不加解释的缩写、编号、专有名词**；第一次出现时说清楚是什么。用户隔了一段时间回来，偏好「把前因后果讲清楚」的说明。
> ⚠️ **CCS 组件拆分（只开对抗训练 / 只开聚类剔除）已经做完**：组名 **CCSD**（种子 42）/ **CCSDA3**（只开对抗训练 43–44）/ **CCSDC5**（只开聚类 43–46）/ **CCSDP**（只开聚类 + 缓存池 43、46），结论见第一节与 F-117 … F-120。
>    计划里叫 **CCSP** 的是「分层设置下的三臂」，还没做；名叫「消融」的 **CCSA** 比的是 FedPer / FedRep 与训练量，**不是**组件拆分。`CLAUDE.md` 与 `PLAN.md` 只是摘要、可能落后，以本文件与 FINDINGS 为准。
> ⚠️ 11-17 会话开头用了**过期的本地副本**（停在 11-01），先给出了一版错误的现状评估 —— **开场第一步一定 `git fetch` + 看远程最新**（见〇）。

---

## 〇·〇·〇·〇·〇·〇、11-24 晚：P-a 追加探针标定回传 → `not_removable`，D3 全部收尾（**新会话先读这一节**）

标定作业（`3889c0a`）的闸都过；按 N-016 机械运行 `select` → **`not_removable`**（F-137）：12 档里没有一档能在「干净精度降幅 ≤ 0.05」的约束下让「训全部主干」把后门清掉一半。
- 学习率 0.01 / 0.03：模型几乎不动、后门一点不掉（f ≤ 0.013）；
- 学习率 0.1 + 临时线性头：后门确实消失（f 0.70–0.99），但 E0 良性端干净精度同时掉 0.18–0.43（第 15 轮从 0.843 掉到 0.418）；
- 学习率 0.1 + 重新初始化头：清不掉（f ≤ 0.18），精度也降 0.06–0.10。
按 N-016 **不交正式作业**。冻结清单 `analysis/d3pa2_frozen.json`、判定 `analysis/d3pa2_verdict.json`（都已入库，测试钉住）。

**D3 三项的最终读法**（F-134 … F-137）：
| 探针 | 结论 | 对论文的含义 |
|---|---|---|
| P-a + 追加 | 定位不了：不打坏模型就清不掉后门 | 6.1 的「后门存在最后一块」写成**未验证**；如实报告两次探针 |
| P-b | 后门没传过来（白盒 − 主列 −0.07 / −0.01 / −0.07） | **支持** 6.1 的「隔离传播」 |
| P-c | 统计量偏移随后门强度变化（Spearman 0.91） | 6.3 的反馈循环升为「中」 |

⚠ F-137 的精度崩可能部分来自「新主干 + 没重训的私有分类头」对不上，**不能**读成「后门与主任务缠在一起」。

**待用户决定**：D3 结论怎么写进 6.1 / 6.3、要不要出图（OUTLINE 对照表、FIGURE-NOTES 都还没更新）；`ISSUE-sharing-value.md` 的「块内切分」P-a 没给出依据。
**之后的顺序不变**：攻击基线接线（时间盒 2026-12-06）。

---

## 〇·〇·〇·〇·〇、11-24 晚：P-a 追加探针（加大训练量）—— 预注册、代码、作业脚本都已提交，等集群（代码与预注册；已回传，见上一节）

**为什么**：D3 的 P-a（F-134）三臂清除比例全是 0（lr 只有客户端的 0.1 倍、75 步），没有分辨力 → 6.1 的「只改最后一块就足以让后门失效」仍没有证据。
用户 11-24 决定补做；规则写成预注册 **N-016**（FINDINGS 末尾）、决定 **D-140**，**在交任何作业之前**提交。这是看过 F-134 之后设计的追加分析，论文里如实标注。

**两步**（用户定：分两步交）：
1. **标定**：SNAP 集中 s42 × 第 15 / 60 轮，**只跑「训全部主干」**，12 档 = 学习率 {0.01, 0.03, 0.1}（固定）× epoch {5, 20}（75 / 300 步）× 分类头 {临时线性头, 随机重新初始化}；
   规则选「两轮都清除比例 ≥ 0.5、精度降幅 ≤ 0.05」的最小一档并冻结；没有合格的 → `not_removable`，P-a 到此为止。
2. **正式**：选定那一档 × 三臂 × s42–44 × 两轮；阳性对照（训全部主干 6 个都 ≥ 0.5）不过 → `uninformative`；
   只训最后一块 6 个都 ≥ 0.5 → `last_block_suffices`（支持 6.1），都 < 0.25 → `last_block_insufficient`（6.1 要改写）。

**新文件**：`fedavg/analysis/d3_pa2.py`（TF）、`d3_spec.py` 加网格函数、清单 `d3pa2_configs.yaml`（内容 sha `79d915ec7703b4ca`，测试锁住）、
作业 `d3pa2.sbatch` / `d3pa2_submit.sh`、判定 `harness/d3pa2_verdict.py`（`select` / `judge`）；测试 `tests/test_d3pa2_verdict.py`、`test_d3_spec.py` / `test_d3_probes_tf.py` 增补。
`d3_probes.harden_subset` 只多了缺省为 None 的 `head_init`（D3 已有行为不变）。

**集群上（仓库根目录）—— 第 1 步**：

```bash
git pull
```

```bash
bash experiments/defense/edge-native/d3pa2_submit.sh calibrate
```

1 个作业（1 张卡、限时 2 小时、32G），估计 15 分钟左右，**没实测过**。回传 `experiments/defense/edge-native/analysis/d3pa2/SNAP__collocated__s42.calibrate.json`。

**回传后（本地）**：

```bash
python3 harness/d3pa2_verdict.py select experiments/defense/edge-native/analysis/d3pa2/SNAP__collocated__s42.calibrate.json
```

→ `analysis/d3pa2_frozen.json`：`selected` 就提交、推送，然后集群上 `git pull` + `bash experiments/defense/edge-native/d3pa2_submit.sh main`（3 个作业，各约 5–10 分钟）；
`not_removable` 就停在这里、写 FINDINGS。第 2 步回传后 `python3 harness/d3pa2_verdict.py judge` → FINDINGS F-137。

**本会话的本地测试**（Python 3.11）：不装 TF **2301 passed / 62 skipped / 3 xfailed**（比上一批多的全是新测试）；装 tensorflow-cpu 2.15.1 全量 **2623 passed / 31 skipped / 3 xfailed / 2 failed**（2 条 = 陷阱 #4）→ 无回归。

**顺带记下的小问题（不影响已有结论）**：`harness/d3_verdict.py` 先把 f / V / 每种子 Spearman 舍入到 4 位再和阈值比；D3 的实际数离阈值都很远（最近的是 P-b 差值 0.07 对 0.10），标签不受影响。
新的 `d3pa2_verdict.py` 已改成用原值比（带反向锚点测试）；D3 的脚本是数据之后用过的，没有改。

---

## 〇·〇·〇·〇、11-24：D3 离线探针回传并判定（11-24 上午）

用户在 11-24 确认了 D-138 里待定的三处（D-139；确认时数据已回传、Claude 没有打开过），之后才运行判定 `python3 harness/d3_verdict.py` → `analysis/d3_verdict.json`。
9 个 JSON 的参数清单 sha 都对；**重现闸 27 / 27 全过**（离线重算的主列与记录最大差 0.0005）。

| 探针 | 判定（预注册规则的输出） | 一句话 | FINDINGS |
|---|---|---|---|
| P-a 重训最后一块 | `mainly_earlier_layers` | **不能照字面读**：只训 stage4、只训前面几层、训全部主干，三臂的清除比例**全是 0**（权重几乎没动：lr 只有客户端的 0.1 倍、干净损失本来就低）→ 这个预算下哪里都清不掉，定位不了；6.1 既不改写也不加强 | F-134 |
| P-b 受害模型上白盒求扰动 | `not_transmitted` | 在受害者自己的模型上求扰动并不更强（白盒 − 主列 = −0.07 / −0.01 / −0.07）→ 三层个性化下受害边缘攻击成功率低是**后门没传过来**，支持 6.1 的「隔离传播」 | F-135 |
| P-c 统计量偏移 vs 后门强度 | `feedback_moderate` | 108 个边缘模型 Spearman 0.91（每个种子 0.90–0.93、每轮内部 0.78–0.89、按轮分层置换 p = 0.0001）→ 参照打分测的是「模型已有后门」，6.3 的反馈循环升为「中」 | F-136 |

附带读数（探索性）：快照里的私有 BN 统计量是陈旧的 —— 只做精确重估就让 E0 良性端干净精度在第 15 轮升 0.075–0.14（F-134 末段，与 F-051 同方向）。

**待用户决定**：
1. P-a 要不要追加一个加大训练量的版本（看过数据之后设计 → 只能算探索性，或先写新预注册）；做的话先以「训全部主干能把清除比例拉到 ≥ 0.5」作阳性对照。只需 SNAP 集中 ×3、每个作业几分钟。
2. 论文 6.1 / 6.3 怎么引用这三条、要不要出图（OUTLINE 论点—证据对照表与 FIGURE-NOTES 还没更新）。
3. `experiments/thesis/ISSUE-sharing-value.md` 里「块内切分要先看 P-a」—— P-a 没给出依据，这一条暂时无法据此决定。

**之后的顺序不变**：攻击基线接线（BadNets / Blended / DBA 已有触发器与静态投毒，缺的是接进 `per_epoch` 取数；Neurotoxin 掩码方向待核实、CerP 补完；IBA 从草稿重写），时间盒 2026-12-06。

---

## 〇·〇·〇、11-23 晚：D3 离线探针 —— 代码、判定脚本、作业脚本都已提交，等集群跑（代码；11-24 已回传，见上一节）

**是什么**（不熟悉的话先读下面〇·〇·下一步的表）：三个「零训练」探针，只读 SNAP / TBS 已有的快照，给论文 6.1 / 6.3 补直接证据；规则 PREREG-v2 §五（N-015），
实现取舍 **D-138**（数据之前写定；其中 ④ ⑤ ⑨ 是 Claude 的读法、**待用户确认**，见 DECISIONS）。

**新文件**：
- `fedavg/analysis/d3_spec.py`（纯算术，不 import TF）、`fedavg/analysis/d3_probes.py`（TF：P-a / P-b / P-c + 重现闸）；
- `experiments/defense/edge-native/d3_configs.yaml`（参数清单，内容 sha `89160c6203b99416` 由测试锁住）、`d3.sbatch`（一个 run 一个作业）、`d3_submit.sh`（9 个 run 一次交完）；
- `harness/d3_verdict.py`（纯标准库；没有数据时三个探针都报 insufficient，底线复算 = 0.0974 / 0.0633 / 0.0731）；
- 测试 `tests/test_d3_spec.py`、`tests/test_d3_verdict.py`（本地）、`tests/test_d3_probes_tf.py`（TF，CPU 小联邦真跑；「写回统计量 / 写回生成器」两个反向锚点实测会红）。

**集群上（仓库根目录）**：

```bash
git pull
```

```bash
bash experiments/defense/edge-native/d3_submit.sh
```

9 个作业（每个 1 张卡、限时 3 小时、32G）：SNAP 集中 ×3（P-a + P-b 参照 + P-c）、TBS k1-snap ×3（P-b + P-c）、SNAP 分散 ×3（P-c）。估计合计约 1.5 GPU 小时，**没有实测过**。
想先探路：`sbatch experiments/defense/edge-native/d3.sbatch SNAP__collocated__s42 --rounds 15`（结果进 `analysis/d3_probe/`，不进判定）。
exit 3 = 有快照轮没过重现闸（离线重算的第 t 轮主列与 run 记录对不上）→ 先排查，不要判定。

**回传后（按顺序）**：
1. 回传 `experiments/defense/edge-native/analysis/d3/*.json`（9 个）。
2. 本地：`python3 harness/d3_verdict.py` → `analysis/d3_verdict.json`；结果写成 FINDINGS 新条目（F-134 起），并给 `tests/test_d3_verdict.py` 加「入库 JSON == 脚本输出」的锚点。
3. P-a 的结果决定 `experiments/thesis/ISSUE-sharing-value.md` 里「块内切分」方案值不值得做。

**本会话的本地测试**（Python 3.11）：不装 TF 改动前 **2189 passed / 57 skipped / 3 xfailed** → 改动后 **2254 / 60 / 3**（多出的全是新测试）；装 tensorflow-cpu 2.15.1 的全量（CPU，约 12 分钟）**2568 passed / 29 skipped / 3 xfailed / 2 failed**，2 条红 = 一直存在的 Neurotoxin（陷阱 #4）→ 无回归。

---

## 〇·〇·下一步、下个会话：D3 离线探针（11-23 下午写的任务说明；代码已按它做完）

**开场**：`git fetch origin claude/eloquent-faraday-b9kmop` 并快进（这条分支上还有别的会话在写 `experiments/thesis/` 与 `experiments/overview/` 的文档，推送前先合并）；
本地建 Python 3.11 环境跑 L1 基线（不装 TF 时 11-23 是 **2187 passed / 57 skipped / 3 xfailed**；做 P-a / P-b 要装 `tensorflow-cpu==2.15.1 wandb`）。
**先出语义对照表、用户确认后再写代码**（CLAUDE.md 交互约定）。

**做什么**：三个「零训练」探针，只读集群上已有的快照，给第 6 章两处只有「数据一致」的解释加直接证据；读法已在数据之前写定（`experiments/thesis/PREREG-v2.md` §五，预注册 N-015），不要改阈值。

| 探针 | 回答什么 | 快照 | 判定（三个种子都满足才算） |
|---|---|---|---|
| P-a 重训最后一块 | 后门是否主要存在最后一块（6.1 的机制解释） | SNAP 集中 s42–44 第 15 / 60 轮，攻击者所在边缘 E0 | 清除比例 f 都 ≥ 0.5 →「主要在最后一块」；都 < 0.25 →「主要在前面的层，6.1 要改写」；闸：干净精度降 ≤ 0.05 |
| P-b 受害模型上白盒求扰动 | 受害边缘低攻击成功率是「后门没传过来」还是「触发器迁移不过去」 | TBS k1-snap s42–44 第 60 轮（SNAP 作参照） | 白盒 − 主列都 ≥ 0.30 且白盒 ≥ 0.5 →「迁移不过去」；都 < 0.10 →「没传过来」 |
| P-c 统计量偏移 vs 后门强弱 | 参照过滤的反馈循环是否存在（6.3） | SNAP 集中 + 分散、TBS k1-snap，第 6 / 15 / 60 轮，约 108 个边缘模型 | 合并秩相关 ≥ 0.5、每种子内 > 0、置换 p < 0.01 → 证据升为「中」；≤ 0.2 → 不支持 |

**已查到的代码事实**（11-22 探查）：
- 快照复原 / 评估：`fedavg/analysis/p0_snapshot.py`（`build_world` / `load_snapshot` / `restore` / `eval_post`；第 60 轮要 `--ignore-v0`，因为没有第 61 轮的云聚合后点）；
  ⚠ 它的 `fedavg()` 不处理三层个性化的边缘段，用于 TBS 快照时不能直接调。
- 重训：`fedavg/hardening/core.py` 的 `EdgeHardener` 只能训全部主干、`objective="cft"` 是普通交叉熵 → P-a 要在探针里把 `body_vars` 换成 stage4 子集
  （前缀 `utils/tier_split.edge_shared_prefixes(1)`）；重训后各客户端私有 BN 统计量与新主干失配 → 按 PREREG 在各端训练分片上重估统计量（基线也做同样重估）。
- 攻击者在 E0 上 → 改 E0 会连带改攻击者的扰动 ξ；主读数用主列语义（重求 ξ），另报固定 ξ。E0 在这些快照上已饱和（0.82–1.00）→ ASR 与 margin 都报；
  底线 = G0-C1 同种子 E0 末 10 点（0.097 / 0.063 / 0.073）。
- 白盒扰动：`client_badpfl.eval_trigger(victim_PM, …)`；按边缘的白盒评估现在没有现成代码（`evaluate_hierarchical_asr` 传 `trigger_fn` 即可）。
- 统计量偏移：`fedavg/defense/edge_ref.py` 的 `_moments`（整批训练模式前向）+ `edge_ref_spec.gauss_kl`；带触发器的图照 `p0_snapshot.trigger_norms` 的写法生成。
- 作业脚本照 `experiments/defense/edge-native/p0.sbatch`（只传参数、回传一个 JSON 到 `analysis/`）；`p0_configs.yaml` 被测试锁定，**新配置另起清单**。

**之后的顺序**（攻击基线接线时间盒 **2026-12-06**，到期没接通的写进局限）：静态三种攻击（BadNets / Blended / DBA）「每批投毒」接线 + 探路 → IBA → 自适应攻击者最小版 → Neurotoxin / CerP。

**待议、未排期**：云端共享本身的价值在本设定下偏小（几乎断开只差约 0.02 精度）→ `experiments/thesis/ISSUE-sharing-value.md`
（数据稀缺实验、块内切分出 52% / 73% 上云档、跨边缘评估、只写进论文四个方案；块内切分要先看 P-a 的结果）。

---

## 〇·〇、11-23：补充实验第一批回传与判定

27 个 run 全部有效（`status`：done 112、无 mismatch）；判定脚本回传后没改，结果 `analysis/supp_verdict.json`；用户拍板记 D-136。

| 实验 | 标签（N-015） | 一句话 |
|---|---|---|
| D1 块数扫描（F-130） | `inconclusive` | B=1 比几乎不上云（B=4）多保住 0.018–0.022 精度，种子 44 差一点没到 0.02 → 报数字不报方向（不补种子，D-136）。受害边缘 0.996 / 0.281 / 0.046 / 0.027（B = 0 / 1 / 2 / 4），单调 |
| D2 威胁边界 [9,1,0,0]（F-131） | `clean_protected` + `e1_breached` | 干净边缘降 0.61–0.68；只有 1 个攻击者的边缘照样失守（0.93–1.00）→ 信任域解释「事先预测、被检验」 |
| D4 逐坐标中位数（F-132） | 三格 `blocks`、云层 · 分散 `not_blocks` | 效应量只有 0.004–0.10，受害边缘仍 0.89–0.99；集中布点下 `weaker_than_b1`；中位数不砍灌入 |

- D1 的闸 ② 通过：B=1 存快照重跑与已有的只记录模式 run 逐位相同 → 「k = 1 下只记录模式不改训练」有了直接证据。
- 统一对比表 `experiments/thesis/analysis/defense_balance.{json,md}` 已加新臂（块数 2 / 4、两个中位数臂；已有行逐字不变）。
- 论文图（`3f28fbe`）：6.2 共享深度权衡、6.3 威胁边界、6.9 加中位数两行 → `experiments/thesis/figures/`；其余 14 张重画逐字节不变。`OUTLINE.md` 论点—证据对照表与 M1–M3 已更新。
- 本地测试（Python 3.11，不装 TF）：2184 passed / 57 skipped / 3 xfailed（无新红）。本批没改训练代码，没重跑含 TF 的全量。

**D0b 快照结构核查：完成，全过**（F-133 `confirmed`）。第一次（`6627ae8`）四条全过，但核查本身有两个盲点 → 补强（D-137：拟合矩阵满秩、四台边缘上云段两两不同、拟合用全部 1 227 712 个坐标）后重跑（`9eb772f`）三个种子 × 三个快照轮仍全过（满秩、σ_min / σ_max 0.020–0.044、残差约 4e-8）。→ 6.1 的结构前提成立，不需要重审。
`experiments/overview/` 三个 .md 已同步改为「已完成」；**导师简报 PDF 没有重生成**（用户 11-23：只改 .md），PDF 里的待办表仍停在 11-23 早上的版本。

**待议（未排期）**：云端共享本身的价值在本设定下偏小（几乎断开只差约 0.02 精度）—— 问题、推测与四个可选方案见 `experiments/thesis/ISSUE-sharing-value.md`（用户 11-23：先记录，不是最需要做的）。

**后续会话顺序**（不变；攻击基线时间盒 2026-12-06）：D3 离线探针（P-a / P-c 用 SNAP 快照，P-b 用 TBS k1-snap 第 60 轮快照；读法 PREREG-v2 §五）→
静态三种攻击接线 + 探路 → IBA → 自适应攻击者最小版 → Neurotoxin / CerP。

---

## 〇·〇（11-22）、补充实验第一批 —— 规则定稿、代码、登记、判定脚本都已提交

**来历**：用户上传《补充实验预注册规则 v2》，要求完成收尾实验。本会话把原文的「待填」从仓库查实、「待拍板」问用户，
定稿为 `experiments/thesis/PREREG-v2.md`（预注册 **N-015**，commit `25163a5`，**先于任何判定脚本与新数据**），拍板记 **D-135**。
用户拍板：D4 加云层中位数臂；精度口径同 D-071（全部 100 个客户端的 fresh 个性化精度）；阈值全部照草案；攻击基线时间盒 **2026-12-06**；
知道最后一块不上云的自适应攻击者**做最小版**；A2 不做分散布点；**本会话只做第一批**。

**查实后改写原文的三处**（规则文件里标 ✎）：
- 现有 B=1 run 都没有快照 → D0 拆成 **D0a**（CPU 小联邦真跑，已做：`tests/test_three_tier_personalization.py::test_d0a_structure_after_one_cloud_round`，k = 1 / 2 / 4 都过）
  与 **D0b**（TBS k1-snap 回传后在快照上核对，脚本 `fedavg/analysis/tier_structure.py` + `tier_structure.sbatch`）；
- D4 分散布点没有受害边缘 → 判定量用全部良性端池化攻击成功率；
- B=1 的复用对照（ERP / ERC main-t1）是 observe 模式 → TBS k1-snap（不设防）与它们逐位相同就是闸 ②，同时补上「k = 1 下 observe 不改训练」的直接证据。

**做完的（commit）**：
1. `25163a5` 规则定稿（PREREG-v2 + N-015 + D-135）。
2. `8607812` `federation.edge_shared_blocks` 放宽到 0–4（`fedavg/utils/tier_split.py`；k = 4 上云只剩 stem）+ 测试（含 D0a、元素数钉住 [设定6] 实测值）。
3. `2f8c7c0` `collect_metrics` 记 `run.defense_layers`（中位数两臂只差作用层）；`defense_balance.arm_of` 认中位数 / k = 2、4 / 非标准布点（以前中位数会被当成「不设防」、[9,1,0,0] 会被当成「分散」）；现有统一对比表重算逐字节不变。
4. `816feae` 登记 **TBS**（D1 块数扫描：集中 × {k1-snap, k2, k4} × 3 种子 = 9）、**TBH**（D2 威胁边界：[9,1,0,0] × {k0, k1} × 3 = 6）、**MED**（D4 中位数：{边缘层, 云层} × {集中, 分散} × 3 = 12）；27 个配置都过启动校验。
5. `1ca1b5a` 判定脚本 `harness/supp_verdict.py`（D1 / D2 / D4；现有数据上三个都是 insufficient，已有对照全部过闸）+ D0b 脚本。

**本地测试**（Python 3.11）：不装 TF 改动前 2089 passed / 57 skipped / 3 xfailed → 改动后 **2177 / 57 / 3**（多出的 88 条全是新测试）；装 tensorflow-cpu 2.15.1 的全量（CPU，约 11 分钟）**2472 passed / 27 skipped / 3 xfailed / 2 failed**，2 条红 = 一直存在的 Neurotoxin（陷阱 #4）→ 无回归。（第一次全量跑出 `test_edge_ref_tf` 两条红：它复用了三层个性化测试的小模型，被本批改了；`fa79282` 恢复原夹具、D0a / D0b 改用单独的五段模型。）GPU 节点上照旧会多出 F-127 的那几条。

**集群上（仓库根目录）**：

```bash
git pull
```

```bash
PACK=3 RUN_GROUPS="TBS TBH MED" bash experiments/defense/edge-native/submit.sh
```

9 个三跑包（每格 3 个种子一包），约 9 × 4.3 ≈ 39 GPU 小时；TBS k1-snap 存快照约 3 × 0.32 GB。新配置类型（k = 4、中位数）没有显存数据，提交工具会先按 K = 3 探路，OOM 会自动降档。

**回传后（按顺序）**：
1. 对账：`python3 harness/status.py experiments/defense/edge-native/registry.yaml`（27 个应为 done、无 mismatch）。
2. D0b（集群，约几分钟）：`sbatch experiments/defense/edge-native/tier_structure.sbatch` → 回传 `analysis/d0_structure.json`；**任何一条不过 → 停下，6.1 需要重新审视**。
3. 三个判定（本地或集群，纯标准库）：`python3 harness/supp_verdict.py` → `analysis/supp_verdict.json`；结果写成 FINDINGS 新条目（F-130 起）。
4. 把新臂加进统一对比表：`harness/defense_balance.py` 的 `GROUPS` 加 `TBS` / `MED`（TBH 是非标准布点，本表不收），出图前按 PREREG-v2 §八 写定口径。

**已知的小问题（没改，不影响训练）**：`fedavg/config_validate.py` 里「backdoor.enabled=false 但 defense.name=…」那条警告的 `elif` 挂在了 `if stop_cfg:` 上 →
只要 `stopping` 关、又开了任何防御，就会打出这条**不实**的警告（攻击其实开着）。MED 的 12 个配置、以前的 ERP / ERF / CCS 格子都会打。只是一条警告，不改任何行为。

**后续会话顺序**（一个会话一件事；攻击基线时间盒 2026-12-06，到期没接通的写进局限）：
D3 离线探针脚本（P-a / P-c 可先用 SNAP 快照做，P-b 等 TBS k1-snap 的快照；读法已在 PREREG-v2 §五写定）→
静态三种攻击（BadNets / Blended / DBA）的「每批投毒」接线 + 探路 → IBA（主线没有实现；`origin/claude/iba-attack-integration-vt3jjm` 有 9 月草稿，早于 P2，只能作起点）→
自适应攻击者最小版（攻击 mixin 里屏蔽 stage4 梯度；读法 PREREG-v2 §七末）→ Neurotoxin / CerP。每个攻击先出语义对照表再写代码。

---

## 〇·〇′、11-19 晚：转入论文整合

用户决定：**不再开新方向**，把三个阶段的已有实验整合进一篇学位论文。整合方案已确认：`experiments/thesis/INTEGRATION-PLAN.md`；
提纲 + 论点—证据对照表 + 术语对照：`experiments/thesis/OUTLINE.md`。

**本会话做完的（第一步，零训练）**：`harness/defense_balance.py`（阶段三全部防御臂放进同一把尺子：分边缘的终值、相对不设防的下降、
每个云周期的「灌入 / 洗掉 / 净增量」，攻击成功率与 margin 两个口径；描述性分析，不是预注册判定）+ `tests/test_defense_balance.py`；
输出 `experiments/thesis/analysis/defense_balance.{json,md}`。

**新读数（描述性）**：集中布点、前半程，三层个性化把每次云聚合的灌入按攻击成功率砍到 1/3–1/6、按 margin 砍掉约一半；
上传前加固按攻击成功率几乎不砍、按 margin 也砍掉约三分之一 → F-124「加固不砍灌入」的措辞要改成「主要不是靠砍灌入」；论文以哪个口径为主待定。

**用户决定（11-19）**：
- 攻击侧**一定要补并做实验**，规模待定：**先做完 Bad-PFL，再做一组比 Bad-PFL 弱的基线攻击**，之后再定规模。
  ⚠ 正式口径下只有 Bad-PFL 接了按 epoch 重洗的取数路径，静态投毒 / Neurotoxin / CerP 都会被 `config_validate.py` 拒绝 → 较弱的基线攻击要先写接线代码；选哪几种待用户定。
- 阶段一（另一个仓库）放背景章简述，数字与图由用户提供。

**下一步（每步等用户点头，一个会话一件事）**：见 `INTEGRATION-PLAN.md` 第七节 —— ② 防御侧补充包的代码与登记（三层个性化块数放宽到 0–4、存快照重跑、[9,1,0,0] 布点、预注册）→
③ 集群约 22 GPU 小时 → ④ 离线探针 → ⑤ 攻击侧基线攻击 → ⑥ 写作。

本会话 L1 基线（Python 3.11、不装 TF）：改动前 **2049 passed / 57 skipped / 3 xfailed**。

**11-20 更新**：用户上传提纲 v2，已存为 `experiments/thesis/OUTLINE.md`（取代 11-19 初稿）。两个决定：
- **两个口径都报**；正文标题数字默认攻击成功率，对照饱和（≥ 0.95）或贴着底线时改用 margin（规则一律照用，不逐图挑；OUTLINE 第七节第 1 条）；
- **六种基线攻击**：BadNets、Blended、DBA、Neurotoxin、CerP、IBA（IBA 要新写；先各跑 1 个探路实验；第七节第 2 条）。

**图表方案** `experiments/thesis/FIGURES.md`（批准）。**会话 A 已完成**：`harness/thesis_style.py`（正文宽 5.9 英寸 —— 等用户给模板的 `\the\textwidth`；8 / 7 磅；
实体颜色 / 线型 / 标记全书固定，配色已用 dataviz 校验）+ `harness/thesis_figures.py`（第 4、5 章 7 张：4.1 划分、4.2 灌入—洗掉示意、5.1 底线、5.2 收敛门控、5.3 锯齿、5.4 衰减、5.5 可观测性）
→ `experiments/thesis/figures/*.{pdf,png}`；守卫 `tests/test_thesis_figures.py`（数与判定同源、渲染文字不含内部名词且有反向锚点、无硬写颜色、宽度、PDF 字体嵌入）。
L1：**2081 passed / 57 skipped / 3 xfailed**（+12）。

**会话 B 已完成（11-20）**：第 6 章 8 张图（6.1 共享块轨迹、6.4 上传前加固、6.5 不分层聚类、6.6 参照打分可分性、6.7 在线过滤与反馈循环、6.8 良性端误判分布、6.9 综合比较、6.10 灌入—洗掉分解），数据由 `thesis_figures.digests()` 一次读完阶段三全部结果；测试钉住 REPORT §4.2 的筛选数、F-126 的 0.847、F-129 的 0.07 / 0.57、defense_balance 的逐种子值。L1：**2088 passed / 57 skipped / 3 xfailed**（+7）。**下一步 = 会话 C**：示意图 1.1 + 触发器图库第一版（集群短作业）；防御侧补实验（代价对照、[9,1,0,0]）回来后画 6.2 / 6.3。

---

## 〇、新会话开场

1. `git fetch origin claude/eloquent-faraday-b9kmop && git log --oneline -3`：看集群有没有推新东西（现在**没有待交、待回传的作业**）。
2. 本地测试环境：容器是新的，要现建 Python 3.11 环境（3.12 / 3.13 会有与代码无关的红）：

   ```bash
   python3.11 -m venv <scratchpad>/venv311 && <scratchpad>/venv311/bin/pip install -q pytest pyyaml numpy scipy scikit-learn matplotlib pandas
   ```

   ```bash
   TFDPFL_PY=<scratchpad>/venv311/bin/python bash run_l1.sh 2>&1 | tail -3
   ```

   基线：不装 TF 的数在 11-18 的改动后没有重测（11-17 时 1939 passed / 55 skipped / 3 xfailed）；
   再装 `tensorflow-cpu==2.15.1 wandb` 后跑全部 TF 测试（CPU，约 12 分钟）：**2332 passed / 26 skipped / 3 xfailed / 2 failed**（2 条 = 陷阱 #4）。多出来的红才是回归。
   集群 GPU 节点上是 11 条红、`CUDA_VISIBLE_DEVICES=` 藏起 GPU 后 3 条（F-127）。
3. 先读下面「〇·一」（方向 2 的结论 + 下个会话的议题：三层个性化）；需要细节再看「〇·二 … 〇·五」与 FINDINGS F-125 … F-129。

---

## 〇·一、11-19：方向 2 的结论 + 下个会话的议题

### 方向 2 做了什么、得到了什么（一段话版）
边缘服务器用自己的 500 张干净图片当参照，给每个客户端上传打「可疑分」（上传里的批归一化统计量 vs 干净图在同一模型上的统计量），分两段：
- **第一段（只打分）**：攻击者与正常客户端分得开（四格最小 AUROC：现主线 0.847、组内更同质 0.895），N-013 判可行（F-126）。
- **第二段（分数超线就扔掉上传）**（F-128）：现主线上攻击成功率从约 1.0 降到约 0.6（三层个性化开 + 集中时 0.40 → 0.29），精度代价约 0.02；
  按预注册只有「三层个性化开 + 分散」判 `protects`，三层个性化关的两格精度降 0.021 / 0.024、刚超门槛判 `accuracy_bound`。
  **组内更同质的划分全部死锁**（第 1 轮所有上传都超线 → 全扔 → 模型停在初始值 → 永远过不了线），对那个划分不提供任何证据。
  死锁原因是「选中的分数（逐客户端参照 KL）训练初期本来就是后期的 2.5 倍 + 阈值是全程固定的一个数 + 全扔则模型不变」，**与干净集无关**。
- **为什么过滤只能削弱**（F-129，相关性证据）：一过滤，攻击者的分数就降到和正常客户端一样；它跟着「E0 模型被污染的程度」走（相关 0.57 / 0.71）。
  解释：有界小扰动的触发器只有在模型学会后门之后才会让统计量明显偏移 → 分数测的是「已被植入后门」的状态 → 过滤与后门之间形成反馈循环，停在攻击成功率约 0.3–0.6。
  文献（LITERATURE 3.2）：单步原理已有（BNP、潜在可分性），「在线过滤 → 反馈循环 → 离线高估在线」没找到。
- **可选、不承诺的验证**：SNAP 快照（第 6 / 15 / 60 轮）上直接量「触发器引起的统计量偏移」随后门强度的变化（离线、约 1 个短 GPU 作业）。用户说做才做。

### 下个会话：讨论三层个性化（用户 11-19：「这个简单但有效的思路」）
**它是什么**：`federation.edge_shared_blocks: 1` —— ResNet-10 最后 1 个残差块只在边缘内平均、不上云（`fedavg/utils/tier_split.py`；CLAUDE.md「3-E 三层个性化」条）。

**已有证据**（全部是不设防的对照、三个种子均值；数据 = SNAP / ERP / ERC，脚本 `harness/d0_power.run_quantities`；早期 G6 见 hfl-mechanism F-077）：

| 划分 | 布点 | 三层个性化 | 良性端攻击成功率 | 受害边缘 | 攻击者所在边缘（E0）的良性端 | 个性化精度 |
|---|---|---|---|---|---|---|
| 现主线 | 集中 | 关 → 开 | 0.997 → **0.399** | 0.996 → **0.281** | 0.999 → 0.988 | 0.868 → 0.867 |
| 现主线 | 分散 | 关 → 开 | 0.986 → 0.956 | — | — | 0.869 → 0.867 |
| 组内更同质 | 集中 | 关 → 开 | 0.926 → **0.418** | 0.911 → **0.302** | 0.999 → 1.000 | 0.844 → 0.845 |
| 组内更同质 | 分散 | 关 → 开 | 0.977 → 0.990 | — | — | 0.844 → 0.844 |

- G6（旧划分、集中）：k = 1 / 2 时受害边缘降 0.524 / 0.730，精度代价 fresh 口径 ≤ 0.02（陈旧口径 k = 2 约 0.028）。
- 与参照过滤叠加（集中）：0.399 → 0.286（F-128，`additive`）。

**看起来有效的地方**：攻击者集中在一台边缘时，几乎不花精度就把其他边缘的后门压掉六七成；组内同质与否不影响这个效果。
**已知的盲区**：① 攻击者所在边缘的良性端完全不受保护（0.99–1.00）；② 攻击者分散到每台边缘时没有作用（0.96–0.99）。

**待讨论的问题（只列，不下结论）**
1. 它为什么有效：edge 段是否截住了跨边缘传播？后门有多少比例「存」在最后一个残差块里？（可用已有快照离线量，不必新训练）
2. 两个盲区能不能补：E0 的良性端（边缘内本地防御？）、分散布点（攻击者在每台边缘都有时，层级结构还能利用什么？）
3. k 的取舍：k = 1 / 2 的保护与精度代价（G6 有数，新划分上只有 k = 1）。
4. 知道 edge 段的适应性攻击者：只往上云的层写后门（交接〇·五「之后的顺序」里的 LP 思路），三层个性化还剩多少？这是它作为防御最需要回答的问题。
5. 作为贡献的定位：三层个性化的结构有先例（ESPerHFL / HPFL，只为精度，F-122），卖点只能是「后门隔离作用」及其边界。
6. 机时：用户要求下一步先充分论证再花机时（11-19）。

## 〇·二、11-18 晚：第二段代码完成 + 集群上的提交指令

**做完的**：`defense.edge_ref.mode: filter`（+ `score` / `threshold`；分数 > τ 的上传不进本次边缘平均，全被剔则边缘模型不变；其余与 observe 完全相同）；
`[EdgeRefScore]` 加 `admitted` 列，`[设定14]` 加 `score` / `threshold`（全精度打印）；判定脚本 `harness/edge_ref_filter_verdict.py`（N-014）；
登记 ERF（过滤臂 {main, homo} × {t0, t1} × {集中, 分散} × s42–44 = 24）与 ERC（不设防对照补 s43 / s44 = 12）。D-133 记实现取舍。
本地测试：新 TF 测试（τ 极大时与 observe 逐位相同；τ = 0 时边缘模型冻结；只聚合被接纳的上传、逐位等于 fedavg_aggregate）都过。
全量 L1（本地 CPU，Python 3.11 + tensorflow-cpu 2.15）：**2332 passed / 26 skipped / 3 xfailed / 2 failed**（2 条 = 陷阱 #4 的 Neurotoxin）→ 无回归。GPU 节点上跑会是 11 条红（F-127），`CUDA_VISIBLE_DEVICES= bash run_l1.sh` 应为 3 条。

**集群上（仓库根目录）**：

```bash
git pull
```

ERF 每格 3 个种子，用提交工具正常合包（8 个三跑作业）：

```bash
PACK=3 RUN_GROUPS=ERF bash experiments/defense/edge-native/submit.sh
```

ERC 每格只有 2 个种子，手工装成 4 个三跑作业（`sub` 函数见〇·四，没定义过就先贴一次）：

```bash
sub ERC__mix-main-t1-collocated+main-t1-distributed__pack-k3-s43 ERC__main-t1-collocated__s43 ERC__main-t1-collocated__s44 ERC__main-t1-distributed__s43
sub ERC__mix-main-t1-distributed+homo-t0-collocated__pack-k3-s44 ERC__main-t1-distributed__s44 ERC__homo-t0-collocated__s43 ERC__homo-t0-collocated__s44
sub ERC__mix-homo-t0-distributed+homo-t1-collocated__pack-k3-s43 ERC__homo-t0-distributed__s43 ERC__homo-t0-distributed__s44 ERC__homo-t1-collocated__s43
sub ERC__mix-homo-t1-collocated+homo-t1-distributed__pack-k3-s44 ERC__homo-t1-collocated__s44 ERC__homo-t1-distributed__s43 ERC__homo-t1-distributed__s44
```

约 12 个作业 × 4.3 小时 ≈ 52 GPU 小时。回传后：

```bash
R=experiments/defense/edge-native/results/P2; python3 harness/edge_ref_filter_verdict.py --filter $R/ERF/*.metrics.json --control $R/SNAP/SNAP__*__s4[234].metrics.json $R/ERP/ERP__main-t1-*__s42.metrics.json $R/ERP/ERP__homo-*__s42.metrics.json $R/ERC/*.metrics.json --out experiments/defense/edge-native/analysis/edge_ref_filter_verdict.json
```

## 〇·三、11-18：ERP / ERPB 回传与判定（F-126）

- 12 个 run 有效；闸通过（SNAP vs ERP main-t0，集中 / 分散都 60 轮逐位一致）。
- N-013：main（现主线）选 mix_res、四格最小 AUROC 0.847；homo（固定画像 λ = 0.4）选 kl_all、0.895 → **两个都 `feasible`**。
- 要带进第二段规则讨论的读数：分散布点漏网多（main 43% / 23%）；homo 良性误剔 17–28%（没有攻击者的边缘也高 → 开发种子的绝对阈值搬不过来，考虑逐边缘标定）；
  homo 下零分布不可用；三层个性化开、集中布点把良性 ASR 降到 0.436 / 0.582。
- GPU 节点上的 L1 基线是 11 条红（藏起 GPU 后 3 条），F-127。
- **下一步**：第二段规则 **N-014 已生效**（D-132：阈值 0.90 分位点、完整版 36 run、丢弃、门槛照原值）。等用户说「开始改」后：
  代码 = `defense.edge_ref.mode: filter`（`edge_ref_spec.MODES` 加 filter；`EdgeRefDefense.aggregate` 按 τ 剔除；τ 写进配置 `defense.edge_ref.threshold` + `score`，由登记表从 ERPB 的数值填）；判定脚本 `harness/edge_ref_filter_verdict.py`（N-014，先 commit 再交）；登记 ERF（24）/ ERC（12），两个种子的对照手工装包。

## 〇·四、11-17 晚做完的事 + 集群上的下一步

**做完的（4 个 commit，分支 `claude/eloquent-faraday-b9kmop`）**
1. **固定客户端画像划分**（D-130，F-125）。起因：用户指出「组内更同质」是核心假设，而现主线划分里同边缘两个客户端的差异是跨边缘的 0.79 倍、
   客户端偏离本边缘（总变差 0.41）比边缘之间（0.375）还大。新开关 `federation.design.client_profile: fixed` + `profile_lambda`：
   边缘层（C1 比例表）与干净集不变，边缘内第 c 个客户端 = (1 − λ)·本边缘分布 + λ·（第 c mod 5 对类各 0.5）；λ = 0.4 时比值降到 0.54。不写 = 原来的 Dirichlet，逐字节不变。
2. **参照打分**（D-131）：`defense.name: edge_ref` + `defense.edge_ref.mode: observe`（只记录、全部接纳 → 开 / 关训练逐位相同）。
   代码 `fedavg/defense/edge_ref_spec.py`（纯 numpy 算术）+ `fedavg/defense/edge_ref.py`（TF 前向）；每个边缘轮对每个上传记：
   逐客户端参照的高斯 KL（全部层 / stem / stage1…4；stage4 = 三层个性化里只在边缘内共享的那一块）、类混合残差、BNGuard 四数、缓存池多数原则；
   每轮另记 8 个「假客户端」的分数作零分布。日志 `[设定14]` / `[EdgeRefScore]` / `[EdgeRefNull]` / `[TimingEdgeRef]` → collect_metrics schema 16（`edge_ref_scores` / `edge_ref_null`）。
3. **判定脚本** `harness/edge_ref_probe.py` = N-013（已生效）：按划分分别执行；每个划分四格（布点 × 三层个性化）取 AUROC 最小值最大的分数；
   阈值 = 开发 run 的 90% 分位点；最佳四格最小 AUROC ≥ 0.75 → `feasible`，否则 `infeasible`。
4. **登记**：ERP（有攻击，s42）{main, homo} × {t0, t1} × {collocated, distributed} = 8；ERPB（投毒比例 0、攻击者在，s41）{main, homo} × {t0, t1} = 4。已放行、配置已生成。

**本地测试**：本会话在 scratchpad 里装了 tensorflow-cpu 2.15（Python 3.11），新写的 TF 测试在 CPU 上真跑过：
observe 开 / 关 `[Checksum]` 逐轮相同、去掉随机数围栏就不同（反向锚点）、TF 取出的矩 = 手写的整批训练模式前向、统计量挪开 3σ 时 KL = 4.5（解析值）、与三层个性化同开能跑。
全量 L1（Python 3.11 + tensorflow-cpu 2.15，CPU，含全部 TF 测试）：**2285 passed / 26 skipped / 3 xfailed / 2 failed**，2 条红 = 一直存在的 Neurotoxin（陷阱 #4）→ 无回归。不含 TF 的本地基线（划分 commit 时）1939 passed / 55 skipped / 3 xfailed。

**集群上的下一步（按顺序）**
1. `git pull`，跑有 TF 的 L1（预期只有一直存在的 Neurotoxin 2 条红；GPU 节点上另 3 条，F-067）。
2. **不要直接用 `submit.sh` 交**（11-17 晚更正）：提交工具只把**同一格子的不同种子**装进一个包，而 ERP / ERPB 每格只有 1 个种子，
   跨格子合包又要求每格先有自己的显存实测 → `submit.sh`（含 `PACK=3`）会交出 **12 个单跑作业**（本地 `--dry-run` 已确认），
   单跑约 3.9 小时（SNAP 实测）→ 约 47 GPU 小时。`PROBE_K` 对单种子格子不起作用；也**不需要先交探路**：
   GPU 上若训练模式前向会报错，第 1 个边缘轮的聚合（主线程）就会让 run 非 0 退出，几分钟内暴露。
   改为用 `pack.sbatch` 手工把不同格子装包（同模型同拓扑的 SNAP 已实测三个一包：每 run 约 17 GiB、checksum 不受影响）：
   12 个 run → 5 个作业（ERP 3 + 3 + 2、ERPB 2 + 2），约 22 GPU 小时（D-131 ③ 里的「约 20」按此更正）。在集群仓库根目录：

   ```bash
   sub() {  # sub <tag> <run_id …>：参数拼法同 submit_lib 的 _submit_pack（4 核 / run、24G 主机内存 / run、comment 防重交）
     local tag=$1; shift; local a=() r g E=experiments/defense/edge-native
     for r in "$@"; do g=${r%%__*}; a+=("$E/configs/$r.yaml" "$r" "$E/results/P2/$g/$r.metrics.json"); done
     sbatch -c $((4*$#)) --mem=$((24*$#))G --job-name=exp3v2-pack$# --comment="exp3v2:$(IFS=,; echo "$*")" \
       experiments/attack/hfl-mechanism/pack.sbatch "$tag" "${a[@]}"
   }
   ```

   ```bash
   sub ERP__mix-main-t0-collocated+main-t0-distributed+main-t1-collocated__pack-k3-s42 ERP__main-t0-collocated__s42 ERP__main-t0-distributed__s42 ERP__main-t1-collocated__s42
   sub ERP__mix-main-t1-distributed+homo-t0-collocated+homo-t0-distributed__pack-k3-s42 ERP__main-t1-distributed__s42 ERP__homo-t0-collocated__s42 ERP__homo-t0-distributed__s42
   sub ERP__mix-homo-t1-collocated+homo-t1-distributed__pack-k2-s42 ERP__homo-t1-collocated__s42 ERP__homo-t1-distributed__s42
   sub ERPB__mix-main-t0+main-t1__pack-k2-s41 ERPB__main-t0__s41 ERPB__main-t1__s41
   sub ERPB__mix-homo-t0+homo-t1__pack-k2-s41 ERPB__homo-t0__s41 ERPB__homo-t1__s41
   ```
3. 开跑约 15 分钟后看一个日志：`grep -m3 -E "EdgeRefScore|Traceback|Error" ../tfdpfl-logs/exp3v2_ERP__homo-t1-collocated__s42.*.log`
   —— 有 `[EdgeRefScore]` 行、没有 Traceback 即正常（`ERROR:absl:` 那条是已知噪音）。回传后 gpu.json 的 `run_peak_mib` 应约 17–20 GiB、`n_oom` = 0。
4. 回传后：`python3 harness/instrumentation_check.py <SNAP__collocated__s42> <ERP__main-t0-collocated__s42>`（分散同理）必须逐位相同；
   然后 `python3 harness/edge_ref_probe.py --attack <ERP 的 8 个> --dev <ERPB 的 4 个> --out experiments/defense/edge-native/analysis/edge_ref_probe.json`。
5. 某个划分判 `feasible` → 写第二段（过滤）的规则（在过滤数据之前写，用户确认）；判 `infeasible` → 那个划分上不做第二段，回来讨论。

**已知风险 / 没有证据的**
- 固定画像下攻击成功率、个性化增益、三层个性化的隔离作用会不会变：没测过（ERP / ERPB 的 homo 格子同时就是第一份数据）。
- 类混合残差可能把 Bad-PFL 的触发器（目标类的自然特征）解释成「多一点目标类」；受害边缘有 4 类在干净集里只有 12–13 张。
- metrics.json 每 run 约 +1.9 MB（12 run 约 23 MB），是选择「每个边缘轮都打分」的代价。

## 〇·五、11-17 会话定下的事 + 下个会话要做的事

### 用户的决定（D-129）
1. **先做「方向 2」= 边缘服务器用自己的 500 张干净图作参照，逐个过滤客户端上传；而且一开始就和三层个性化叠加**（三层个性化 = `federation.edge_shared_blocks: 1`，模型最后 1 个残差块只在边缘内平均、不上云）。
   分工：三层个性化管「云端平均那一刻的跨边缘灌入」；参照过滤管「边缘内部的攻击者」→ 对准两个开放情形（分散布点、攻击者所在边缘的良性客户端）。
2. **攻击侧先把 CerP 接好**（第二种攻击，可训练触发器；`client/client_cerp.py` 的触发器训练还读旧数据管线，PLAN 的 SA5），**放到方向 2 之后的那个会话**（一个会话一个模块）。
3. 用户的两个总要求（11-07）：防御要能覆盖其他攻击；方法要体现层级结构。

「方向 1 … 5」的编号来自 11-07 交接的第二节（下面保留）。为什么选 2 不选 1（用户问过，已解释）：每台边缘服务器每个边缘轮只收到 2–3 个上传，多数原则的聚类成不了簇；集中布点时第 0 台边缘里攻击者占四成（10 / 25），多数原则会失效；方向 1 是单层方法换个位置、已有先例。方向 2 做在分层 + 新划分上，等于已包含方向 1 的「搬进分层」；方向 3（会适应的攻击者）在方向 2 有效之后必须马上做。

### 本会话查清的事实（决定方向 2 的设计）
- **主线设定（统计量私有）下，上传里本来就带批归一化统计量**，值是主干训练段末尾的（攻击者这一段用投毒批次），边缘照样平均它们，只是客户端下次收到时换回私有副本 → 方向 2 **留在正式口径（P2）里做**，已有的 SNAP 不加防御对照（集中 / 分散 × 42–44）直接复用，**不需要**另开统计量共享的消融。旧交接里「主线统计量私有、不上传」是错的（F-123，有代码行号）。
- 每台边缘服务器每个边缘轮 2–3 个上传（`server/participation.py:29-31`）。
- **同一边缘内客户端异质**：每端自己的 Dirichlet(0.5) 先验（`data/designed_partition.py:274-279`），种子 42 上客户端与本边缘分布的平均总变差 0.41；边缘干净集按本边缘混合分布切出，受害边缘有 4 个类各只有 12–13 张 → 良性客户端的统计量本来就偏离参照，**误剔是主要风险**。
- 上传的统计量是动量 0.9 的指数平均（约最后 10 个批次，`models/cnn.py:268`）；Bad-PFL 攻击者算非定向扰动那一步还用投毒输入额外更新了一次统计量（`client/client_badpfl.py:144`）。
- 接口：防御的 `aggregate` 拿不到边缘模型与干净数据（`server/robust_aggregation.py:67-86`）；干净图只在 `update_ck` / `edge_harden` 开时挂到边缘（`main.py:787-794`）；三层个性化只支持 `resnet10` / `resnet10_torch`（`utils/tier_split.py:30`，不能用 CCS 的投影层模型）；没有现成的「在一份数据上算批归一化统计量」的工具；`update_ck.py:89-106` 是「每个上传装进草稿模型 + 随机数围栏」的最佳模板。
- 上传前加固为什么有效的探索性读数（看过数据之后才算，F-124）：每次云端灌入几乎没变，但每周期「灌入 − 洗掉」的净增量从 0.026–0.033 降到 0.012–0.030，受害边缘停在不饱和的平衡上。

### 方向 2 的设计（语义对照，开工前再给用户确认）
| 步骤 | 文献 | 本方案 | 差异 |
|---|---|---|---|
| 信任锚 | FLTrust（服务器约 100 张根数据算参考更新，按余弦打信任分）；FoggyTrust（雾节点各持 100 张根数据跑 FLTrust，**前提组内同质**，针对模型投毒、不个性化） | 每台边缘服务器的 500 张干净图（本机构分布） | 后门（优化型触发器）+ 个性化 + 统计量 + 组内异质下仍要成立 |
| 比对什么 | BNP（NeurIPS 2022：模型里记录的统计量 vs 干净数据上算的统计量，逐神经元 KL 找后门神经元） | 每个上传装进草稿模型，在边缘干净图上算各批归一化层输入的均值与方差（**逐客户端参照**），与它上传的统计量逐通道算高斯 KL | 从「找神经元」改为「给上传打分」 |
| 类别偏斜 | 无（BNGuard / CCS 用多数原则） | 候选分数：用干净图的**逐类**矩，单纯形非负最小二乘拟合上传统计量，残差当分数 | 新设计，无文献；**Bad-PFL 的 δ 是目标类自然特征，可能被混合拟合解释掉** |
| 判据 | BNGuard（第一层 4 个数 + K-means 大簇）；CCS（余弦 + HDBSCAN 大簇） | 绝对阈值，不靠多数 | 每轮 2–3 个上传也能判 |
| 阈值 | SURA（只在良性开发 run 上标定一次、之后固定） | 照 SURA：只在**不投毒**的开发 run 上取良性分数 90% 分位点 | 不用身份标签 |
| 威胁模型 | SURA：代理集泄露 + 方向幅度联合调整 → ASR 94.91% | 写明边缘干净集保密；「拿到干净集的攻击者」进后续自适应检验 | — |

### 计划
**第一段：记录阶段（只记录、全部接纳；6 个 run，约 9–11 GPU 小时）**
- 开关 `defense.name: edge_ref` + `defense.edge_ref.mode: observe`（名字暂定）。每个边缘轮对每个上传记全部候选分数：逐客户端参照 KL（全部层 / 每个残差阶段）、类混合残差、BNGuard 四数、CCS 式缓存池余弦（多数原则的对照）、Dirichlet(0.5) 抽的「假客户端」分数（不用 run 的参照）。
- 格子：有攻击 {三层个性化关, 开} × {集中, 分散} × 种子 42（组 ERP，4 个）；不投毒开发 run {关, 开} × 种子 41（组 ERPB，2 个）。
  「关」的两格必须与 SNAP 同种子逐位相同（「记录不改训练」的闸）；「开」的两格顺带是三层个性化在新划分上的复现。
- 预注册草案 N-013（**交作业前用户确认三个数**）：在种子 42 有攻击 run 上按两种布点下攻击者 vs 良性的 AUROC 取「最小值最大」的分数（只用于选分数、判可行性）；阈值 = 不投毒开发 run 里该分数的 90% 分位点；最佳 AUROC < 0.75 → 判「参照过滤不可行」，第二段不做。
**第二段：过滤阶段（约 16 个 run、25 GPU 小时；规则在第一段数据之后、过滤数据之前写）**
- {三层个性化关, 开} × {集中, 分散} × 种子 42–44 开过滤，加三层个性化单独开的 43、44。门槛：降 ≥ 0.15、最小种子 ≥ 0.10、精度降 ≤ 0.02；叠加 = 比两者较好者再低 0.10，冗余 = 差 < 0.05。

**代码（第一段）—— ✅ 已完成（11-17 晚，见〇·四；实际做法以 D-131 为准，与下面的草案有细节差别）**
- `fedavg/defense/edge_ref_spec.py`（新，不 import TF）：参数与校验、设定行字段；纯 numpy 的逐通道高斯 KL、单纯形 NNLS 混合残差、BNGuard 四数、余弦、阈值。
- `fedavg/defense/edge_ref.py`（新）：`EdgeRefDefense(BaseDefense)`，`layers={"edge"}`、`from_config`、`set_edge(edge)`；草稿模型第一次用到才建并包 `random.getstate()/setstate()`（F-078）；每个上传一次 500 张整批训练模式前向，取各批归一化层输入的整体 / 逐类矩；独立 `default_rng([seed, 标签, 轮, edge])`；observe 全部接纳。
- `server/edge_server_base.py`：防御有 `set_edge` 就注入（同 `set_stat_indices`；吸取 F-107：测试走生产路径）。`main.py:787`：开 `edge_ref` 也挂干净图。
- `defense/factory.py`、`config_validate.py`：登记；单独一节规则（hier_fedrep、交错调度、S3 划分、干净集 ≥ 1、层只能 edge）；**不进** `AGGREGATION_DEFENSES`（否则跑进全方法 × 全防御测试）。
- `[设定14]` → `run.edge_ref_*`；每次边缘平均一行 `[EdgeRef]` → `collect_metrics` schema 16 的 `edge_ref_decisions`；`harness/registry.py` 的 `EXPECT_KEYS`。
- `harness/edge_ref_probe.py`（新）：按 N-013 选分数与阈值（AUROC 复用 `analysis/functional_score.py`）。
- 登记表加 ERP / ERPB（挂预注册口令）；`test_edge_native_registry.py` 加差异守卫；全量生成配置，**不带 `--group`**。
- 测试：本地手算对拍、设定行往返、配置校验拒绝分支、登记差异守卫、解析；TF：observe 开 / 关 checksum 逐轮相同、去掉随机数围栏就不同（反向锚点）、注入走生产路径、与三层个性化同开能跑、人为污染统计量的上传分数变大。

### 之后的顺序
方向 2 第一段 → 第二段 → 下一个会话接 CerP → 会适应的攻击者（拿到干净集的；只往上云层写后门、专门绕过三层个性化的，LP 的思路）→ 修已知防御的缺陷、跑对比基线（中位数、剔除离群者类、FLAME、AdvPurge 等）。

### 还等用户的
- ~~说「开始改」；确认 N-013 的三个数~~ ✅（11-17 晚：三个数照草案、四格取最小、按划分分别执行，D-131）。
- 本机补查：Bad-PFL / CCS 在 Scholar 上 2026-09 以后的被引用、OpenReview 上 ICLR 2027 投稿（本环境打不开 arxiv / Scholar）。

---

## 一、现在站在哪里（一段话版）

- **上传前加固那条线已结案**：边缘服务器每个云轮末用 500 张干净图做对抗反学习（SAU），集中布点下 3 个种子都「保护受害客户端」（受害端攻击成功率平均降 0.41、精度降 ≤ 0.015），
  攻击者所在边缘服务器的良性端没被保护；分散布点精度降 0.030 超门槛，只记录。
- **CCS 那条线**（全部在**不分层**设置：1 个边缘服务器 × 100 个客户端、FedRep、BatchNorm 统计量**共享**、300 轮）：
  - CCS 两个组件拆开后：**只开聚类剔除**（按上传的统计量聚类、剔除离群上传）精度代价小且稳（降约 0.01），但**防御效果不可靠** ——
    5 个种子攻击成功率降幅 0.614 / 0.139 / 0.762 / 0.753 / 0.345，均值 0.523，按预注册规则刚过 0.5 的门槛，种子 43、46 失守（F-119）。
  - **对抗训练**让训练不稳定：6 个带对抗训练的 run 都从第 65–110 轮起剧烈震荡，崩不崩随机（F-118）。
  - **缓存池**（聚类时把其余客户端最近一次上传的统计量也放进来，池子从 10 个变成约 97 个）在两个失守种子上修好了防御（攻击成功率降 0.779 / 0.794），
    但主任务精度降 0.026 / 0.028 > 0.02 → 预注册判 `no_fix`（没修好）；代价来自误剔增加（约三分之一的良性端经常被排除）。**用户看过数据后认为这点代价可以接受**（D-128），标签不改（F-120）。
  - **先例**：「用 BatchNorm 统计量检测恶意客户端」已有 BNGuard（MobiHoc 2025）和 CCS（ICASSP 2026）两篇，都不分层 → 统计量聚类本身**不是本项目的贡献**（F-121）。
- **用户提出的两个要求**（2026-11-07）：防御**要能覆盖其他攻击**；方法**要体现层级结构**。

---

## 一·五、新颖性核查（2026-11-17，F-122；LITERATURE 第 3 部分）

- 网页检索没有找到同时占「三层结构 + 个性化 + 优化型触发器 + 边缘服务器主导防御」四样的工作；全部据摘要、未读全文。
- **方向 2 最近的先例 = FoggyTrust**（arXiv 2606.27622：分层、雾节点持根数据算信任分；据摘要针对模型投毒、不个性化、用更新余弦）→ 方向 2 的语义对照表要加这一行，定位收窄为「后门 + 个性化 + 统计量 + 逐客户端参照」。
- **三层个性化的结构有先例**（ESPerHFL、HPFL 等，只为精度）→ 卖点写「后门隔离作用」。
- 待办：
  - [x] AdvPurge、FoggyTrust、SURA 已读全文（2026-11-17，LITERATURE 3.1）：都不撞车；AdvPurge 的「层级」= 主干 / 分类头；FoggyTrust 是方向 2 的直接先例（雾节点根数据 + FLTrust，前提组内同质）；
  - 方向 2 计划据此改两处：① 阈值只在**良性开发 run** 上标定（SURA 做法），不用身份标签；② 威胁模型写明**边缘干净集保密**，并把「拿到干净集的攻击者」列入自适应检验（SURA 泄露后 ASR 94.91%）；语义对照表加 FoggyTrust 一行（中间层根数据 + 更新余弦 + 组内同质前提）；
  - [ ] 用户本机查 Bad-PFL / CCS 的 Scholar「被引用」（2026-09 以后）与 OpenReview 上 ICLR 2027 投稿；
  - [ ] 用户设 Scholar 提醒（关键词 + 两篇锚点论文的新引用）。
- 方向 2 的实施计划见〇·五（已按三篇文献改好），**等用户说「开始改」**。

---

## 二、11-07 列出的候选方向（「方向 1 … 5」的出处；11-17 用户选了 2，见〇·五）

| # | 方向 | 回答什么 | 需要什么 | 注意 |
|---|---|---|---|---|
| 1 | **搬到分层设置** | 边缘服务器上做统计量聚类，在 4 个边缘服务器 × 集中 / 分散布点下还有没有用；层级会不会稀释或放大漏网 | 登记即可（CCS 已支持 `defense.ccs.pool: edge_cache`，D-109 ③ 本来就是为分层设计的）；**先补统计量共享的分层不加防御对照**（主线统计量私有、不上传，聚类没东西可聚） | 集中布点时 E0 里攻击者占四成（10 / 25），按多数原则聚类可能失效 —— 阶段二就见过 E0 参照池被污染（hfl-mechanism FINDINGS F-085 更正） |
| 2 | **edge 原生变体** | 用边缘服务器 500 张干净图算参照统计量，代替「少数服从多数」（类似 FLTrust 用服务器干净数据当信任锚）；可加云端按边缘服务器的第二级过滤 | 新代码（`fedavg/defense/ccs.py` 加一种判据），先出语义对照表 | **CCS 这条线用的是旧 `noniid` 划分，没有边缘服务器干净集**（F-087；干净集只在 S3 划分下才有，`clean_per_edge`）→ 这个方向必须在 S3 划分（分层主线用的）上做，等于和方向 1 绑在一起 |
| 3 | **会适应的攻击者** | 攻击者上传前用干净数据重算统计量，聚类还能不能分出来；如果一下就绕过，统计量路线要重新考虑 | 新代码：攻击 mixin 的上传钩子（`FLClientBase.on_upload`）里替换统计量，先出语义对照表 | BNGuard 论文有没有做过这个**未核实**（arxiv 不可达，只读了它的代码）；可请用户上传 PDF |
| 4 | **其他攻击** | 统计量聚类是不是只对 Bad-PFL 这种对抗扰动式触发器有效 | 第二攻击要先写代码：CerP 的 mixin 在，但触发器训练还读旧数据管线（AUDIT A25）；IBA 没有实现；静态投毒在 P2 口径下被配置检查拒绝 | 推断：语义后门（自然图片、只改标签）统计量几乎不偏移，大概率检测不到 —— 没有实验 |
| 5 | 缓存池补跑 / 范数裁剪 | 缓存池在原来挡住的种子 42 / 44 / 45 上会不会弄坏；或用裁剪 / 阻尼限制漏网者的影响 | 补跑只需登记（约 6 GPU 小时；用户提过精度门槛 0.03，**未登记、不是生效的预注册**）；裁剪 / 阻尼要新代码 | 方案 A–E 的文献、优缺点见 REPORT §7.1′ 与 FINDINGS F-118 第四部分 |

用户此前定过、仍未开始的方向：
- **FedPer 排查**：后门进了共享模型（约 0.995），为什么到不了良性端的个性化模型（0.39–0.41）？能不能用作防御？切入点：FedPer 与 FedRep 的差别逐个换（每步联合训练 vs 分两阶段、有没有单独的分类头阶段）。
- **统计量共享能否叠加到其他防御上**：前提是上传前加固都在统计量私有下做的，叠加前要补统计量共享下的不加防御对照。
- 边缘服务器用干净数据只做前向、重估统计量（D-119 ②(b)）：可复用 `fedavg/hardening/online.py` 与 `fedavg/analysis/p0_snapshot.py`；`p0_configs.yaml` 被测试锁定（改它 = 改预注册），新配置另起清单。

更早的待定事项：用本区域数据产生的合成数据代替真实干净集（10-28 提出）；「每个边缘轮都加固」未实现；CCS 在 FedPer 下 300 轮精度降 0.21 原因未查；在线功能评分只测了飞机 / 汽车两类、存盘清理（PLAN §8）。

---

## 三、本阶段的判定文件与脚本（都先于数据提交、有测试）

| 判定 | 文件 | 规则 | 结果 |
|---|---|---|---|
| CCS（统计量共享，种子 42） | `analysis/ccsb_verdict.json` | N-008 CCSF 段 | `partial_repro`（防御过关、精度降 0.112） |
| 统计量私有 vs 共享（3 个种子） | `analysis/bns_verdict.json` | N-009 | `not_established` |
| CCS 拆分（种子 42） | `analysis/ccsd_verdict.json` | N-010 | 已被补种子推翻（F-118） |
| CCS 拆分（只开聚类 5 个种子，其余 3 个） | `analysis/ccsd5_verdict.json`（`.partial.json` 是 8 / 10 时的留档） | N-011 | 头条 `cluster_alone_sufficient`（刚过线）；归因 `seed_dependent` |
| 缓存池（种子 43、46） | `analysis/ccsdp_verdict.json` | N-012 | `no_fix`（防御修好、精度超门槛；用户事后接受代价） |

脚本：`harness/ccsf_verdict.py`、`bns_verdict.py`、`ccsd_verdict.py`、`ccsd5_verdict.py`、`ccsdp_verdict.py`；
描述性读数（不是判定）：`harness/ccs_leak_timeline.py <metrics.json …>`（每 25 轮放进聚合的攻击者上传数、历史剔除规则的反事实回放、共享模型精度的稳定性）。

---

## 四、本会话踩过的坑（下次照做）

- **提交 CCS 的格子一定带 `PACK=2 PACK_MEM_PER_RUN_GB=48`**：不带 `PACK` 会走单跑脚本、只申请 24G 主机内存，CCS 格子要约 31 GiB，会被杀。
- **`RUN_GROUPS` 里多个组用空格分隔并加引号**（`RUN_GROUPS="A B"`）；用逗号会一个组都匹配不上，什么也不交。
- **提交工具对没测过显存的格子只先交一个探路包，其余显示 held（暂缓）**。显存已知的配置（本线每 run 约 17 GiB）可以**交完立刻把同一条命令再跑一次**，已在队列的会被跳过，剩下的就交出去了。
  11-02 那次漏了这一步，只开聚类的种子 45、46 晚了一轮才交。
- **生成配置不要带 `--group`**：`python3 harness/registry.py experiments/defense/edge-native/registry.yaml --materialize` 全量生成（带 `--group` 会把 INDEX.tsv 重写成只含那一组）。
  组挂着未放行的 token 时不会生成配置，这是正常的；放行后再全量生成一次。
- 登记表的种子是**按组**写的；几个臂要的种子不同时，新开小组只放新种子，已有组不动（CCSOFF5 / CCSB3 / CCSDA3 / CCSDC5 就是这样）。
- 预注册规则一旦发现漏洞（如 N-011 没给「两个都开」臂设崩坏线），**照实记录、不事后改规则**；用户事后接受的门槛记为决定，不改标签。
- 本会话的网络访问不到 arxiv.org（网页检索可以，但读不了 arxiv 全文）；GitHub 可以读。

---

## 五、环境 / 机时

- 本地 TF 测试另需 tensorflow-cpu 2.15 + scikit-learn 的环境（本会话没建）；有 TF 的基线是 2026-10-30 的 **2093 passed / 26 skipped / 3 xfailed / 2 failed**（2 条是一直存在的 Neurotoxin 已知失败）。
- 本地拿不到 CIFAR-10（原站被代理拒绝），需要真实数据的检验只能在集群上做。
- 集群：先 `git pull` 再交；作业脚本只传参数；命令前加 `PYTHONUNBUFFERED=1`（被杀时日志不丢行）。
- 机时：上限约 100 GPU 小时 / 周；阶段三有记录的作业累计约 90 GPU 小时（下限，不含离线筛选与探路作业）。
