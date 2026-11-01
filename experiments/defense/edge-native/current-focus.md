# 交接 —— 阶段三（edge 原生防御）· CCS 这条线：CCS 判「部分复现」；验证「统计量共享本身是不是防御」的补种子实验已登记

> **写于 2026-11-01**（取代 2026-10-31 版的第一、二节；第三节起照旧）。过程与数字在本目录 `FINDINGS.md`（到 F-115、预注册 N-009）与 `DECISIONS.md`（到 D-120）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（用户 2026-10-10）。分支 `claude/eloquent-faraday-b9kmop`。
> 用户要求（2026-11-01）：**文档和回复里不要用不加解释的缩写**；组名、标签第一次出现时说清楚是什么。

## 一、开 CCS 的那次实验（组名 CCSB：FedRep + BatchNorm 统计量共享 + CCS，300 轮）：已回传、已判定

判定（`analysis/ccsb_verdict.json`，事先定好的三个条件，F-115）：**部分复现**（标签 `partial_repro`）。

| | 不加防御 | 开 CCS | 差值 | 要求 |
|---|---|---|---|---|
| 良性端攻击成功率（末 10 点） | 0.796 | 0.112 | 降 0.684 | 降 ≥ 0.5 ✓ |
| 主任务精度（新鲜个性化模型，末 10 点） | 0.894 | 0.783 | 降 0.112 | 降 ≤ 0.02 ✗ |
| 共享的边缘模型精度（只报告） | 0.868 | 0.703 | 降 0.165 | — |

- 聚类每轮都剔掉 10 个上传里的 2–3 个：恶意上传剔中 61%，良性上传误剔 23%，300 轮里几乎不变。
- 主任务精度很不稳（末 10 点标准差 0.041，不加防御时 0.002）。
- **还不知道**精度损失来自「误剔良性上传」还是「良性端的对抗训练」。要拆开得跑 CCS 的三臂消融（只开对抗训练 / 只开聚类 / 两者都开），还没登记 —— 留给用户决定要不要做。

## 二、马上要交的：补种子实验（组名 BNS），检验「统计量共享让攻击成功率低了 0.13」是不是真的

- **设置**：不分层（1 个边缘服务器 × 100 端）、FedRep、不加防御、300 轮；统计量 {私有, 共享} × 种子 {43, 44}，4 个 run。
  种子 42 不重跑：私有 = `legacy/CCSF-fedrep/CCSF__off__s42`，共享 = CCSBOFF（两份 run 块核对过，只差统计量开关）。
- **判定规则已事先写死**（FINDINGS N-009，`harness/bns_verdict.py`，测试 `tests/test_bns_verdict.py`）：
  三个种子上「私有减共享」都 > 0 且平均 ≥ 0.10 → `real_effect`（真效应）；否则 `not_established`（没有确立）。
- **集群上**（先 `git pull`，看 `git log --oneline -1`）：

```bash
PACK=2 PACK_MEM_PER_RUN_GB=24 RUN_GROUPS=BNS bash experiments/defense/edge-native/submit.sh --dry-run
```

  应列出 2 个 GPU 作业（同一设置的两个种子一包，各 `--mem=48G`），预计合计约 6 GPU 小时（两个 run 并行跑在一张卡上，墙钟约等于单跑的 2.8 h）。
  确认后去掉 `--dry-run`，并在命令前加 `PYTHONUNBUFFERED=1`。不开 CCS，所以每 run 主机内存按 24G（开 CCS 时才要 48G）。

- **回传后**：

```bash
python3 harness/bns_verdict.py --private experiments/defense/edge-native/legacy/CCSF-fedrep/CCSF__off__s42.metrics.json experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnprivate-off__s43.metrics.json experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnprivate-off__s44.metrics.json --shared experiments/defense/edge-native/results/P2/CCSBOFF/CCSBOFF__fedrep-bnshared-off__s42.metrics.json experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnshared-off__s43.metrics.json experiments/defense/edge-native/results/P2/BNS/BNS__fedrep-bnshared-off__s44.metrics.json --out experiments/defense/edge-native/analysis/bns_verdict.json
```

- **判定之后**：真效应 → 做下面第 2 条的防御；没有确立 → 不把统计量当防御线索。

**还没做的（D-119 ②(b)，本会话没动）**：边缘服务器用自己手里的干净数据、只做前向传播、重新估计 BatchNorm 统计量，作为最便宜的防御臂。
需要新代码；可复用上传前加固的挂点（`fedavg/hardening/online.py`）和离线回放工具（`fedavg/analysis/p0_snapshot.py`），先离线在已存的模型快照上看能削掉多少后门。
注意：之前离线评估的配置清单 `p0_configs.yaml` 内容被测试锁定（改它 = 改预注册），新配置要另起一份清单。

## 三、CCS 这条线到现在的来龙去脉（按时间）

| 实验 | 设置 | 结果 | 结论 |
|---|---|---|---|
| FedRep + CCS（统计量私有） | 我们的主线设置 | 主任务从第 10 轮起停在 0.34；第 54 轮 NaN 崩溃 | 主任务崩坏 |
| FedPer + CCS 官方超参数 | 2 epoch、batch 64、lr 0.1 常数 | 不加防御时良性端只有 0.39 → 复现闸判「攻击没植入」 | FedPer 下后门传不到大多数良性端 |
| FedPer + 我们的训练量 | 5 epoch、batch 32 | 良性端仍只有 0.41 | 不是训练量的问题，是 FedPer 的私有层挡住了 |
| FedRep + CCS，投影网络两段都训 | 照用户 CCS 复现仓库的 FedRep 写法 | 主任务仍垮（0.29） | 不是「梯度到不了投影网络」 |
| FedRep + CCS，统计量共享 | 30 轮诊断 | 主任务 0.70，不再垮 | **主因：客户端私有统计量与新主干失配** |
| FedRep + 统计量共享，不加防御 | 300 轮（CCSBOFF） | 良性端 0.796，主任务 0.894 | 条件 1 通过 → 放行 CCSB |

其他确认过的事实：
- CCS 官方代码里没有可用的 Bad-PFL：训练时不投毒，评估用的生成器从没训练过。原文「FedPer 下 94%」找不到来源。
- 官方 Bad-PFL 只实现了 FedBN；它自己的剂量反应是每轮 15 / 45 / 75 步 → 攻击成功率 0.22 / 0.80 / 0.88。
- 用户补全的 CCS 复现仓库（`Huangru-114/CCS-reproduce`）与官方 Bad-PFL 有三处不同，都会让攻击显得更强：扰动加在标准化后的图上并截断到 [0,1]；评估时在被测模型上求扰动；良性端评估用未训练的生成器。

**代码层面新增的东西**（都有测试守着，缺省行为与以前逐字节相同）：
- FedPer 方法（`training.drift_correction: hier_fedper`）；
- FedRep 的投影网络训练阶段开关（`training.fedrep_proj_phase`）；
- CCS 剔除含 NaN 的上传并报告有效端数，有效端过少就停止训练；
- 登记表的组可以声明自己的口径（`protocol: P2-ablation`，用于刻意偏离对齐模板的实验）。

## 四、上传前加固（SAU）那条线：已结案

| 格子 | 判定 | 要点 |
|---|---|---|
| 集中布点（3 个种子） | 保护受害客户端 | 受害 edge 攻击成功率平均降 0.41，精度只降 0.012；攻击者所在 edge 的良性端没被保护 |
| 分散布点（1 个种子） | 边缘，只记录 | 攻击成功率降 0.29，但精度降 0.030 超门槛 |

## 五、还等用户定的旧事项
- 用本区域数据产生的合成数据代替区域服务器手里的真实干净集（2026-10-28 提出，未讨论完）。
- 「每个 edge 轮都加固」（唯一能碰到攻击者所在 edge 良性端的 edge 侧做法）未实现。
- CCS 在 FedPer 下的长期精度代价（300 轮精度降 0.21）原因未查。
- 在线功能评分只在飞机 / 汽车两类上测、存盘清理、与 Python 版本有关的测试（旧事项，见 PLAN §8）。

## 六、环境 / 基线
- 本地 L1：Python 3.11 venv（3.12 / 3.13 下有与代码无关的红）；本地 TF 测试用 tensorflow-cpu 2.15 + scikit-learn + wandb 的 venv，`TFDPFL_PY=<venv>/bin/python bash run_l1.sh`。
- 最新基线（2026-11-01，Python 3.11）：无 TF **1847 passed / 55 skipped / 3 xfailed**（= 10-31 的 1823 + 本会话新增 24 条）；有 TF（2026-10-30）**2093 passed / 26 skipped / 3 xfailed / 2 failed**（2 条是一直存在的 Neurotoxin 已知失败）。
- 本地拿不到 CIFAR-10（多伦多原站被代理拒绝），需要真实数据的检验只能在集群上做。
- 集群：先 `git pull` 再交；作业脚本只传参数；CCS 格子主机内存峰值约 31 GiB → `PACK_MEM_PER_RUN_GB=48`；命令前加 `PYTHONUNBUFFERED=1`。
- 机时：上限约 100 GPU-h / 周。
