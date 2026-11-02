# 交接 —— 阶段三（edge 原生防御）· CCS 这条线：拆分实验种子 42 已回传，分层补种子已放行、可交

> **写于 2026-11-02**（第二节方向 1 重写；其余同 11-01 版）。过程与数字在本目录 `FINDINGS.md`（到 F-117；预注册 N-009 / N-010 / N-011 均已生效）与 `DECISIONS.md`（到 D-124）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（用户 2026-10-10）。分支 `claude/eloquent-faraday-b9kmop`。
> 用户要求（2026-11-01）：**文档和回复里不要用不加解释的缩写**；组名、标签第一次出现时说清楚是什么。

## 一、现在的结果（用户 2026-11-01 的总结，经核对）

**FedRep**（下面全部是统计量共享、1 个边缘服务器 × 100 端、300 轮、种子 42，除非另说）
- 不加防御时后门能植入良性客户端（0.796）；开 CCS 后降到 0.112，但主任务精度降 0.112（门槛 0.02）→「部分复现」（F-115）。
  只在统计量共享下成立：统计量私有时开 CCS 主任务直接崩。共享模型的精度也降了 0.165；开 CCS 后精度波动大。
- 「统计量共享本身削弱后门」：三个种子都同向（低 0.130 / 0.109 / 0.017），平均 0.085，没到预注册的 0.10 →「没有确立」（F-116）。
  共享时攻击成功率 0.80–0.82，但私有时也有一个种子是 0.83，**不能**读成统计量能把攻击成功率控制在 0.8。

**FedPer**（统计量一直共享）
- 主任务正常；后门**进了共享模型**（共享模型攻击成功率约 0.995，攻击者自己 1.0），但**到不了良性客户端**（0.39–0.41），加大训练量也一样（F-110 / F-111）。
- FedPer 下开 CCS，300 轮主任务精度降 0.21，原因没查。
- 推断（没有直接证据）：统计量共享在 FedRep 上只解释约 0.085，FedPer 比 FedRep 低约 0.5 → 主因多半是每一步联合训练主干与私有部分、没有单独的分类头阶段。

## 二、用户定的三个方向

### 方向 1：CCS 拆分实验 —— 种子 42 已回传（F-117）；**分层补种子已放行、可交**（N-011 用户确认：头条只看均值，崩坏线 0.2，D-124）

种子 42 的结果（N-010 判定 `analysis/ccsd_verdict.json`）：

| | 都不开 | 两个都开 | 只开对抗训练 | 只开聚类剔除 |
|---|---|---|---|---|
| 良性端攻击成功率 | 0.796 | 0.112 | 0.133 | 0.182 |
| 主任务精度 | 0.894 | 0.783 | 0.419 | 0.884 |
| 共享边缘模型精度 | 0.868 | 0.703 | 0.145 | 0.858 |

- 只开聚类就满足复现三条件；精度代价来自对抗训练，只开对抗训练会把共享模型训坏（它的攻击成功率低不能算防御有效）。
- **只有种子 42**，用户：这是到现在最有价值的实验，补种子之后再判断、再谈路线。
- 前提都还没检验：统计量要随主干上传（主线是私有的）；攻击者不针对统计量做适应（下一步，见下）。

**分层补种子**（10 个 run，约 22 GPU 小时，D-123）：不加防御 + 只开聚类补到 5 个种子（42–46）；两个都开 + 只开对抗训练补到 3 个种子（42–44）。
新种子放在 4 个新组（已有组不动）：`CCSOFF5`（不加防御 45、46）、`CCSDC5`（只开聚类 43–46）、`CCSB3`（两个都开 43、44）、`CCSDA3`（只开对抗训练 43、44）。
判读规则 FINDINGS **N-011**（`harness/ccsd5_verdict.py`）。

- 配置已生成。集群上（先 `git pull`，看 `git log --oneline -1`）：

```bash
PACK=2 PACK_MEM_PER_RUN_GB=48 RUN_GROUPS="CCSOFF5 CCSDC5 CCSB3 CCSDA3" bash experiments/defense/edge-native/submit.sh --dry-run
```

  应列出 4 个作业 + 2 个 held（同格子两个种子一张卡，各 `--mem=96G`）。组名之间用**空格**、加引号（用逗号会一个组都匹配不上）。
  ⚠️ 第一次真交只会交 **4 个**作业：只开聚类那组有 4 个 run，提交工具对没测过显存的格子先交一个探路包、其余 2 个 held。
  这个配置的显存已知（种子 42：每 run 17.0 GiB，两 run 一卡约 34 GiB / 96 GiB），所以**交完立刻把同一条命令再跑一次**：
  已在队列里的 run 会被跳过（按作业注释查队列），剩下那 2 个就交出去了，不必等探路包回传。⚠️ `PACK=2` 不能省（不写只申请 24G，开 CCS 的格子约 31 GiB 会被杀）。

- **回传后**：

```bash
R=experiments/defense/edge-native/results/P2
```

```bash
python3 harness/ccsd5_verdict.py --off $R/CCSBOFF/CCSBOFF__fedrep-bnshared-off__s42.metrics.json $R/BNS/BNS__fedrep-bnshared-off__s43.metrics.json $R/BNS/BNS__fedrep-bnshared-off__s44.metrics.json $R/CCSOFF5/CCSOFF5__fedrep-bnshared-off__s45.metrics.json $R/CCSOFF5/CCSOFF5__fedrep-bnshared-off__s46.metrics.json --both $R/CCSB/CCSB__fedrep-bnshared-ccs__s42.metrics.json $R/CCSB3/CCSB3__fedrep-bnshared-ccs__s43.metrics.json $R/CCSB3/CCSB3__fedrep-bnshared-ccs__s44.metrics.json --at-only $R/CCSD/CCSD__fedrep-bnshared-at-only__s42.metrics.json $R/CCSDA3/CCSDA3__fedrep-bnshared-at-only__s43.metrics.json $R/CCSDA3/CCSDA3__fedrep-bnshared-at-only__s44.metrics.json --cluster-only $R/CCSD/CCSD__fedrep-bnshared-cluster-only__s42.metrics.json $R/CCSDC5/CCSDC5__fedrep-bnshared-cluster-only__s43.metrics.json $R/CCSDC5/CCSDC5__fedrep-bnshared-cluster-only__s44.metrics.json $R/CCSDC5/CCSDC5__fedrep-bnshared-cluster-only__s45.metrics.json $R/CCSDC5/CCSDC5__fedrep-bnshared-cluster-only__s46.metrics.json --out experiments/defense/edge-native/analysis/ccsd5_verdict.json
```

- **下一步（用户：补种子回来再做）**：会适应的攻击者 —— 恶意端上传前用干净数据重算 BatchNorm 统计量，看只开聚类还能不能分出攻击者。需要新代码，先出语义对照表。
- **开放问题**：为什么加上聚类能把对抗训练的损失拉回来？要逐上传的更新幅度 + 剔除记录。

### 方向 2：FedPer 排查 —— 下个会话先出方案
- 要回答：后门进了共享模型，为什么到不了良性端的个性化模型？这个机制能不能用作防御？
- 可能的切入点：FedPer 与 FedRep 的差别逐个换（每步联合训练 vs 分两阶段；有没有单独的分类头阶段；私有投影网络的训练方式），看哪一个让良性端攻击成功率从约 0.4 回到约 0.9。

### 方向 3：统计量共享能否叠加到其他防御上 —— 下个会话先出方案
- 想法：统计量共享作为简单的附加手段，与上传前加固（已结案，集中布点下保护受害客户端）叠加，看有没有增量效果。
- 前提：上传前加固的实验都在统计量私有、分层设置下做的；叠加前要先在统计量共享下补一个不加防御的对照。

### 仍未做（D-119 ②(b)）
边缘服务器用自己的干净数据、只做前向传播、重新估计 BatchNorm 统计量。需要新代码；可复用 `fedavg/hardening/online.py` 与 `fedavg/analysis/p0_snapshot.py`；
`p0_configs.yaml` 被测试锁定，新配置另起清单。可以并进方向 3。

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
- 最新基线（2026-11-02，Python 3.11）：无 TF **1892 passed / 55 skipped / 3 xfailed**；有 TF（2026-10-30）**2093 passed / 26 skipped / 3 xfailed / 2 failed**（2 条是一直存在的 Neurotoxin 已知失败）。
- 本地拿不到 CIFAR-10（多伦多原站被代理拒绝），需要真实数据的检验只能在集群上做。
- 集群：先 `git pull` 再交；作业脚本只传参数；CCS 格子主机内存峰值约 31 GiB → `PACK_MEM_PER_RUN_GB=48`；命令前加 `PYTHONUNBUFFERED=1`。
- 机时：上限约 100 GPU-h / 周。
