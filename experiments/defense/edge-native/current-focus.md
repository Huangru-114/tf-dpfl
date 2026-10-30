# current-focus —— 阶段三（edge 原生防御）· A-pre 结案；CCS：先查 FedPer 下的攻击强度（CCSL）

> **写于 2026-10-28**（SA0 / P0 / SA1 会话收尾；同日 SA-C 会话改写第一节）。这一版取代之前叠在一起的各段状态；过程与数字都在本目录 `FINDINGS.md`（F-090 … F-100）与 `DECISIONS.md`（D-096 … D-108）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（CLAUDE.md 分支纪律，用户 2026-10-10）。分支 `claude/eloquent-faraday-b9kmop`。

## 一、现在要做的：交 CCSL（D-115），检验「FedPer 下攻击起不来是因为训练量太小」

- CCSF（FedPer + CCS 官方超参数）判 `invalid_no_attack`：攻击植入了共享模型（GM ASR ≈ 1.0），但没传到大多数良性端（0.41）；
  官方 Bad-PFL 的剂量反应（15 / 45 / 75 步 → 0.22 / 0.80 / 0.88）提示训练量太小（约 12 步 / 轮）是首要嫌疑（F-110）。
- CCSL = FedPer + 我们的训练量（5 epoch、batch 32）、无防御、300 轮，1 个 run，约 3.5 GPU-h（外推）。

**集群上**（先 `git pull`、`git log --oneline -1`）：

```bash
PACK=2 RUN_GROUPS=CCSL bash experiments/defense/edge-native/submit.sh --dry-run
```
应列出 1 个作业（`CCSL__fedper-ours-off__pack-k1-s42`，`--mem=24G`）。确认后去掉 `--dry-run`，并在命令前加 `PYTHONUNBUFFERED=1`。

**回传后**：看良性端 ASR（末 10 点，过滤目标类）。≥ 0.5 → CCSF 改在这个训练量下重做（D-115）；仍在 0.4 左右 → 回头查 FedPer 私有层。

**还开着的**：CCS 的长期精度代价（FedPer 下 ΔMTA 0.21，F-110）；D-096；CCSP 每布点补同模型 off 臂；NaN（只在 FedRep 下出现过，F-109）。

## 二、上传前加固（A-pre）：已结案（D-112）

| 格子 | 判定 | 要点 | 出处 |
|---|---|---|---|
| 集中布点 H1（s42–44） | `protects_victims` | 受害 edge ASR 平均降 0.409（0.476 / 0.522 / 0.228）、精度平均降 0.012；E0 良性端 `no_effect`；每次云聚合灌入量不变 | F-104，`analysis/h1_pre_verdict.json` |
| 分散布点 P1（s42） | `marginal` → 只记录、不进 H3 | ΔP 0.285、Δmargin 3.83，但 ΔMTA 0.030 > 0.02；末段振荡 | F-105，`analysis/p1_dist_verdict.json` |

- 用户定：H1 停在 3 seed（不补 s45 / s46，SNAP5 继续挂 `h-main`）；数字都是第 51–60 云轮（加固了 50–60 次之后）的均值。
- 打包：两个加固 run 一张卡只多约 3% 墙钟（F-106）→ 以后加固格子默认 K=2。
- 下一步候选（用户定）：A-every（每个 edge 轮都加固，唯一能碰到 E0 / 分散布点里本 edge 良性端的 edge 侧做法，**未实现**）、CCSP（等 CCSF）。

## 三、到现在为止的结论（细节见 FINDINGS）

| 步骤 | 结果 | 出处 |
|---|---|---|
| SNAP（无防御对照 + 快照） | 有效；集中布点与 G1R5 逐位相同 | F-094 |
| SA0（加固核心 + 离线回放工具） | 三轮对抗式审查后完成；离线回放在 GPU 上与原训练一致（V0） | F-096、F-097 |
| 加固 lr | 客户端 lr 在 BN 固定模式下发散；按预注册规则选 train ×0.1、frozen ×0.03 | F-098、F-099、D-106、D-107 |
| P0（离线回放，s42 全 36 个配置） | `accuracy_bound`：砍得动的都伤精度（PGD 4/255 砍 83–88%、精度降约 0.10；SAU 8/255 BN 更新 5 遍砍 44–58%、降 0.02–0.03）；只做干净微调砍不动；BN 固定模式多数更糟 | F-100、D-108；对比表 `analysis/p0_s42_screen_tables.md` |
| 「只吸收部分中心更新」（阻尼） | 一次性回放里看起来好，但只是延缓灌入，对持久 / 强攻击（如 IBA）无用 → 只作参照，不当候选 | 用户 2026-10-28 |
| P1（在线 SAU 一个 run，s42） | `go_main_col`：受害 V 0.995 → 0.519、ΔMTA 0.009；jump 只降 21%、E0 不动、末段振荡 | F-103 |
| H1 A-pre（集中 3 seed） | `protects_victims`；E0 `no_effect` | F-104 |
| P1 A-pre 分散（s42） | `marginal`，只记录 | F-105 |

**用户 2026-10-28 提出的另一条路线**（还没讨论完）：用本区域数据产生的**合成数据**代替区域服务器手里的真实干净集。
看法（会话里已说明）：可以去掉「区域服务器持有真实数据」这个强假设，但合成数据可能把后门特征一起反演出来，且依赖于区域侧加固本身先被证明有用 → 排在 P1 结果之后再议（P1 已回传，F-103）。

## 四、还等用户定的旧事项

- F-095：G1 / G1R5 的在线 c_k 只在飞机 / 汽车两类上测 —— `update_ck.py` 的取样要不要改、REPORT §5.13 要不要加注。
- PLAN §8 仍待定：G8 存盘删不删、F-089 的测试要不要改成与 Python 版本无关。

## 五、环境 / 基线

- 本地 L1：Python 3.11 venv（F-089）；本地 TF 测试用 tensorflow-cpu 2.15（Keras 2）+ scikit-learn 的 venv，`TFDPFL_PY=<venv>/bin/python bash run_l1.sh`。
  2026-10-28 基线（`1b22d10`）：无 TF 1719 passed / 53 skipped / 3 xfailed；有 TF 只有陷阱 #4 的 2 条红。
  D-112 之后（2026-10-29 实测，`edge_server_base` 注入 CCS 统计量索引）：无 TF **1796 passed / 52 skipped / 3 xfailed**；有 TF（tensorflow-cpu 2.15 + wandb）**2058 passed / 26 skipped / 3 xfailed / 2 failed**（陷阱 #4）。
  SA-C 之后（2026-10-28 实测）：无 TF **1781 passed / 52 skipped / 3 xfailed**；有 TF（tensorflow-cpu 2.15）**2043 passed / 26 skipped / 3 xfailed / 2 failed**（陷阱 #4；改动前同环境 1972 / 26 / 3 / 2）。
- 集群：先 `git pull`，看 `git log --oneline -1` 再交；作业脚本只传参数（陷阱 #19）；排队靠 fairshare（10-27 实测 `sprio` 全部来自 FAIRSHARE），短作业用 `sbatch -t …` 改短时限便于回填。
- 机时：上限约 100 GPU-h / 周（D-047）。
