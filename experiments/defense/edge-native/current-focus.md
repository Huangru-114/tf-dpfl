# current-focus —— 阶段三（edge 原生防御）· SA0 完成，交接给 P0（GPU）

> **写于 2026-10-13**（SA0 会话）。上一版（2026-10-10，「D0 完成、SNAP 已回传，交接给 SA0」）的内容：D0 → `FINDINGS.md` F-091 … F-094、N-008（D-101 生效）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（CLAUDE.md 分支纪律，用户 2026-10-10）。

## SA0 做了什么

- **加固核心** `fedavg/hardening/`：`spec.py`（纯算术：网格、像素 ↔ 输入空间、分层子集、探针、记账、R_H / 精度过滤 / V0）+ `core.py`（TF：`EdgeHardener`，五种目标，只改 body 可训练变量，random 围栏）。SA1 在线复用。
- **P0 离线驱动** `fedavg/analysis/p0_snapshot.py`：按 `main.py` 重建世界 → 从 SNAP 快照复原 → V0 三项 → 每个配置「4 个 edge 都加固 → FedAvg → run 自己的云聚合后评估」→ 附加读数。
- **判定** `harness/p0_verdict.py`（select / judge，N-008 + D-105）；配置清单 `p0_configs.yaml`（**36 个**：28 AT + 8 C-ft）；作业 `p0.sbatch`。
- 两轮对抗式审查确认的问题都已改（F-096）；决定 D-103（SAU 两档 ε）、D-104（AT-ccs 不进 P0）、D-105（SAU 匹配的 C-ft、比例步长、AT-tgt 掩码、`bn_private_clu` 不是 CCS-clu、判定照 N-008 原文）。
- 顺带：F-095（G1 / G1R5 的在线 c_k 只在飞机 / 汽车上测）—— `update_ck.py` 改不改、REPORT 加不加注**还等用户定**。

## 状态（2026-10-27）：发散原因已定（F-098），等 lr 扫描（D-106）

- 诊断（2 × 2）：bn=frozen 下连干净微调都 3 步发散 → 原因是「推理模式 BN + 客户端 lr」；bn=train 不发散但干净 loss 被抬到 4–8 → 客户端 lr 对加固太大。
- **现在要交的是 lr 扫描**（只算精度、不读 ASR；预计 20–40 分钟）：

  `sbatch experiments/defense/edge-native/p0.sbatch SNAP__collocated__s42 screen --lr-scan --rounds 6`

  → 回传 `analysis/p0_probe/SNAP__collocated__s42.screen.lrscan.json`。`snapshots."6".lr_scan.pick` 就是按 D-106 规则选出的每个 BN 模式的 lr 倍数（`null` = 该模式退出网格）。
  之后：把倍数写进 `p0_configs.yaml` + `spec`（改预注册、留 DECISIONS）→ 测试 → 再交 s42 screen。

## 状态（2026-10-16）：GPU 探路回传 —— V0 过了，但加固发散（F-097）

- V0 在真实 SNAP 上成立（FedAvg 逐位、ASR 差 ≤ 4e-4）。`pgd-e8-frozen-low` 在 4 个 edge 上 15 步内发散成 NaN；驱动现在遇到发散不评估、判定当 `incomplete`。
- **现在要交的是诊断作业**（只加固、不评估、不读 ASR；约 10 分钟）：

  `sbatch experiments/defense/edge-native/p0.sbatch SNAP__collocated__s42 screen --harden-only --only cft-frozen-low cft-train-low pgd-e8-frozen-low pgd-e8-train-low --rounds 6`

  → 回传 `analysis/p0_probe/SNAP__collocated__s42.screen.diag.json`（每个 edge × 配置：`loss_trace`、`gnorm_body`、`gnorm_probe`、`diverged_at`、`dw_body_absmax`）。
  读法：cft-frozen 也发散 → 原因 ①（推理模式 BN + 客户端 lr）；只有对抗配置发散 → 看探针 / 对抗 loss 的尺度。修法定了改 `p0_configs.yaml` = 改预注册 → 先留 DECISIONS。
- 下面第 1 步（探路）已做完；第 2 步（s42 screen）等发散修好再交。

## 下一步（按顺序；每一步的产物先回传再走下一步）

1. **GPU 探路**（约 10 分钟，陷阱 #23：GPU 才会抛的东西先短作业探路）：
   `sbatch experiments/defense/edge-native/p0.sbatch SNAP__collocated__s42 screen --only pgd-e8-frozen-low --rounds 6`
   → 回传 `analysis/p0_probe/SNAP__collocated__s42.screen.json`。先看 `snapshots."6".v0`：`pass`、`fedavg_max_abs`（≤ 1e-5）、`post_abs` / `full_abs`（≤ 0.01）；
   再看 `configs.*.seconds` 与 `harden.*.seconds`（墙钟外推全配置）。V0 不过 → 停，**不交 screen**（N-008：容差要改得在打开 P0 数据之前、留 DECISIONS）。
   注意探路作业**会**读到 s42 第 6 轮一个配置的 R_H —— 选择规则与配置清单都已 commit（`p0_configs.yaml` + `p0_verdict.py`），读到也改不了它们。
2. **s42 screen**：`sbatch …/p0.sbatch SNAP__collocated__s42 screen` → 回传 `analysis/p0/SNAP__collocated__s42.screen.json`。
3. **选择并冻结**：`python3 harness/p0_verdict.py select experiments/defense/edge-native/analysis/p0/SNAP__collocated__s42.screen.json --out experiments/defense/edge-native/analysis/p0_frozen.json`
   → `status` 必须是 `ok`（`incomplete` = 结果缺东西，查作业日志）→ **commit `p0_frozen.json` 之后**才交第 4 步。
4. **冻结阶段 6 个作业**：`SNAP__collocated__s{42,43,44}` 与 `SNAP__distributed__s{42,43,44}` 各 `frozen`（s42 col 的 frozen 只为冻结阶段的附加读数）。
5. **判定**：`python3 harness/p0_verdict.py judge experiments/defense/edge-native/analysis/p0 --frozen experiments/defense/edge-native/analysis/p0_frozen.json --out experiments/defense/edge-native/analysis/p0_verdict.json`
   → `go_online`（进 SA1，取 `best`）/ `kill_pre` / `accuracy_bound`（用户定）/ `inconclusive` / `insufficient` / `invalid`。必报项在 `attribution_vs_cft` 与 `aux`。
- 机时（PLAN §4 P0 的估计，**没有实测**）：screen 约 3–4 GPU-h，冻结阶段约 1.5 GPU-h；探路回来后按 `seconds` 改写。

## 没有证据的（别当成已知）

- V0 在 GPU 上是否逐位（CPU 小 HFL 上是，F-096）；小预算 AT 的墙钟；探针 head 与受害端私有 head 的迁移缺口（`head_diag` 会给出）；比例步长与官方绝对步长的差别。

## 环境 / 基线

- 分支 `claude/eloquent-faraday-b9kmop`（起点 `de0e8b2`），**不合并 main**。
- 本地 L1 用 Python 3.11 venv（F-089）；本地 TF 测试用 tensorflow-cpu 2.15（Keras 2）+ scikit-learn 的 venv：`TFDPFL_PY=<venv>/bin/python bash run_l1.sh`。
  基线见 F-096 / 本次 commit 信息（有 TF 时只有陷阱 #4 的 2 条红）。
- 机时：上限约 100 GPU-h / 周（D-047）。
