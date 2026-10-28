# current-focus —— 阶段三（edge 原生防御）· 交接给 SA-C（CCS）

> **写于 2026-10-28**（SA0 / P0 / SA1 会话收尾）。这一版取代之前叠在一起的各段状态；过程与数字都在本目录 `FINDINGS.md`（F-090 … F-100）与 `DECISIONS.md`（D-096 … D-108）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（CLAUDE.md 分支纪律，用户 2026-10-10）。分支 `claude/eloquent-faraday-b9kmop`。

## 一、下一个会话 = SA-C：CCS 的实现（只做语义对照表，用户说「开始改」之前不写代码）

**要回答的问题**（PLAN §4 CCSF / CCSP）：CCS（良性客户端本地训练时做对抗训练 + 用客户端上传的 BN 统计量聚类剔除可疑客户端）
在我们的分层系统里能不能挡住 Bad-PFL；两部分各贡献多少。

**会话开头要做的**：
1. 重新拉官方代码（这个容器里没有 `reference/`）：`https://github.com/chenjian0924/Paper-Code`（`CCS/` 子目录；FL 主干是 FL-bench），读后用完即删（`reference/` 在 gitignore 里）。
   原文 PDF 的要点已摘在 `LITERATURE.md` §2.2；官方代码的已知情况见 F-090。
2. 出**语义对照表**（论文公式 / 官方实现 / 本仓库拟实现 / 差异 / 怎么验证），逐条给用户定。已知要定的：
   - **对抗样本初始化**：D-097 定的是先照原文 / 官方代码用可变形贴片（DPR），普通 PGD 版本之后作为变体（用户 2026-10-28 讨论时倾向维持原文版本）；
   - **原文与官方代码不一致的三处**（F-090）：聚合按样本加权（代码）vs 无权平均（原文）；KL 系数 0.1（代码默认）vs 0.01（原文）；本地「两步」（原文）vs `local_epoch: 2`（代码）；
   - **MMD 与个性化投影网络**的代码没读到（F-093）—— 要读到或记为偏差；
   - **分层系统里聚类放在哪、聚多少端**（最大的设计问题）：中心看不到单个客户端，只能放在区域服务器；但每个区域每个 edge 轮只选 2–3 个客户端，
     HDBSCAN 的 `min_cluster_size = n // 2 + 2` 超过 n → 成不了簇（P0 实测，`fedavg/analysis/bn_cluster.py` 的 `degenerate`）。
     候选：每个云轮对本区域全部 25 个客户端「最近一次上传的统计量」聚一次（倾向这个），或累积最近若干轮。由用户定，记 DECISIONS；
   - **BN 统计量来源**：我们的 FedRep（P2 口径）里统计量是私有的（D-032）；CCS 要聚的是**上传物里**的统计量（body 阶段末、含投毒批次），
     **不是**快照里的私有统计量（那个只见干净批次，D-105 ④）。上传物里本来就带统计量（`hier_fedrep` 的 `upload_weights`），要核对取的是哪一份；
   - 本仓库与原文的设定差异（PLAN §4 CCSF 已列）：FedRep vs FedPer、ResNet-10 vs ResNet-18、本地 5 epoch、300 有效轮 vs 1000 轮。
3. 实现方式（CLAUDE.md 地基）：客户端对抗训练写成 **mixin + 钩子**（`client/compose.py`），不要写 Client 子类；聚类剔除经 `RobustAggregationMixin.robust_mean` 收口（陷阱 #10 的统一 `[Decision]` 行）；
   新开关登记到 `alignment.py`（缺省关 = 逐字节不变）；判定脚本 `harness/ccsf_verdict.py` 先 commit 再交作业（N-008 CCSF 段：`ccs_reproduces` / `ccs_not_reproduced` / `partial_repro`，外加「off 臂必须植入」的闸）。

**之后的顺序**（PLAN §4 / §5）：CCSF（不分层的复现检查，2 run，约 5 GPU-h；登记表里已占位，划分 = Dir 0.5，D-098）→ 复现成功才做 CCSP（分层里拆开三臂 × 两种布点，6 run，约 13–16 GPU-h）→ 有效的进主检验。
**还开着的**：D-096 —— 「CCS 消融里对抗训练约占 85%」的出处，用户在找；找到要在 **CCSP 交作业之前**补成预注册预测。

## 二、正在集群上跑的：P1（在线上传前加固，D-108）

- 1 个 run：`P1__col-pre__s42`（每个云轮各区域上交前用 500 张干净图按 SAU 洗一遍：8/255、BN 更新、5 遍；对照 = SNAP__collocated__s42）。用户 2026-10-28 已提交，预计约 7.5 GPU-h。
- 回传 `results/P2/P1/P1__col-pre__s42.metrics.json` 后：

  `python3 harness/p1_verdict.py --arm experiments/defense/edge-native/results/P2/P1/P1__col-pre__s42.metrics.json --control experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json --out experiments/defense/edge-native/analysis/p1_verdict.json`

  先看闸（D-108 ③）：第 1 云轮 `[HardenAgg] unhardened` = SNAP 的 `[Checksum] Round 1`；`[Harden]` 240 行。判定规则 = N-008 P1 段
  （对照 V = 0.995 → 天花板规则大概率生效：两边 V ≥ 0.95 时看受害 margin 降 ≥ 2.0；且精度损失 ≤ 0.02）。结果记 FINDINGS（F-101 起）。
- 这个结果**不阻塞** SA-C：两条线独立。

## 三、到现在为止的结论（细节见 FINDINGS）

| 步骤 | 结果 | 出处 |
|---|---|---|
| SNAP（无防御对照 + 快照） | 有效；集中布点与 G1R5 逐位相同 | F-094 |
| SA0（加固核心 + 离线回放工具） | 三轮对抗式审查后完成；离线回放在 GPU 上与原训练一致（V0） | F-096、F-097 |
| 加固 lr | 客户端 lr 在 BN 固定模式下发散；按预注册规则选 train ×0.1、frozen ×0.03 | F-098、F-099、D-106、D-107 |
| P0（离线回放，s42 全 36 个配置） | `accuracy_bound`：砍得动的都伤精度（PGD 4/255 砍 83–88%、精度降约 0.10；SAU 8/255 BN 更新 5 遍砍 44–58%、降 0.02–0.03）；只做干净微调砍不动；BN 固定模式多数更糟 | F-100、D-108；对比表 `analysis/p0_s42_screen_tables.md` |
| 「只吸收部分中心更新」（阻尼） | 一次性回放里看起来好，但只是延缓灌入，对持久 / 强攻击（如 IBA）无用 → 只作参照，不当候选 | 用户 2026-10-28 |
| P1（在线 SAU 一个 run） | 在跑 | D-108 |

**用户 2026-10-28 提出的另一条路线**（还没讨论完）：用本区域数据产生的**合成数据**代替区域服务器手里的真实干净集。
看法（会话里已说明）：可以去掉「区域服务器持有真实数据」这个强假设，但合成数据可能把后门特征一起反演出来，且依赖于区域侧加固本身先被证明有用 → 排在 P1 结果之后再议。

## 四、还等用户定的旧事项

- F-095：G1 / G1R5 的在线 c_k 只在飞机 / 汽车两类上测 —— `update_ck.py` 的取样要不要改、REPORT §5.13 要不要加注。
- PLAN §8 仍待定：G8 存盘删不删、F-089 的测试要不要改成与 Python 版本无关。

## 五、环境 / 基线

- 本地 L1：Python 3.11 venv（F-089）；本地 TF 测试用 tensorflow-cpu 2.15（Keras 2）+ scikit-learn 的 venv，`TFDPFL_PY=<venv>/bin/python bash run_l1.sh`。
  2026-10-28 基线（`1b22d10`）：无 TF 1719 passed / 53 skipped / 3 xfailed；有 TF 只有陷阱 #4 的 2 条红。
- 集群：先 `git pull`，看 `git log --oneline -1` 再交；作业脚本只传参数（陷阱 #19）；排队靠 fairshare（10-27 实测 `sprio` 全部来自 FAIRSHARE），短作业用 `sbatch -t …` 改短时限便于回填。
- 机时：上限约 100 GPU-h / 周（D-047）。
