# current-focus —— 阶段三（edge 原生防御）· 交接

> **写于 2026-10-03**（Experiment 3 收尾会话结束时）。本文件是下一个会话「新会话开场第 3 步」要读的那一份。

## 下一个会话要做什么

**讨论阶段三的设计**，部分以 `PLAN-draft.md` 为基础；**用户会在会话里继续提供更多材料**（论文 / 思路 / 约束）。
- 这是**讨论 + 定方案**的会话：在用户明确说「开始改」之前不动代码、不登记组、不交作业（CLAUDE.md 交互约定）。
- **D-008 仍生效**（「本计划只覆盖改版实验 3，阶段三不在范围内」）→ 拍板解除时写一条新的 DECISIONS（D-089 起），并决定阶段三的决策 / 证据台账放在哪
  （建议：本目录下另起 `DECISIONS.md` / `FINDINGS.md`，编号沿用 D- / F- 但另起序列前缀，或继续用 hfl-mechanism 的序列 —— 由用户定）。
- 用户给的材料一律先读完、再对照下面「已知证据」逐条看它能回答 / 改变什么，不要直接套进草案。

## 先读（按顺序）

1. `PLAN-draft.md`（本目录）—— 主线模块 A（edge 上传前加固）+ 实验序列 P0 / P1 / H1–H5 + 评审补充的 8 条缺口 + §7 待拍板的 5 件事。
2. `experiments/attack/hfl-mechanism/REPORT.md` §1（结论一览）/ §9.4（证据 → 防御线）/ §10（综合讨论、局限）。
3. FINDINGS（`experiments/attack/hfl-mechanism/FINDINGS.md`）：F-085（G1 判定 + 复核）、F-086（探索性）、F-077 / F-069（3-E）、F-068 / F-071（衰减）、F-081 及其更正（c_k；快照里各 edge 是各自的上传前 body）。
4. 原始规划 `experiments/attack/hfl-mechanism/PLAN-original-2026-09-24.md` §9（阶段三原文）。

## 已知证据（阶段二，非自适应攻击者；都已判定或有出处）

| 事实 | 出处 |
|---|---|
| 后门只经云聚合跨 edge 传播：每次聚合灌入 0.13–0.27，周期内只洗掉 0.11–0.22 → 受害 edge 仍饱和 | F-085 / F-086 |
| 没有安全时段（3.3 `not_gated`） | F-076 |
| edge 视角无检测优势；余弦、单更新 c_k 不可检测；`norm_s` 来自不进模型的 BN 统计量通道（可伪造，代码证据） | F-085 |
| 3-E edge 段：攻击者集中时 `blocks`（−0.52 / −0.73，fresh 代价 ≤ 0.02）；分散时无效；治不了 E0 良性端 | F-077 / F-069 |
| 攻击者走后后门自己褪去（约 200 有效轮），flat 一样 | F-068 / F-071 |
| 攻击接近饱和；R20 的 3-C 标签在噪声里；主结论只有 3 seed | REPORT §10.5 / F-085 |

**真正开放的威胁情形**（评审共识）：分散布点、攻击者 edge 内的良性端。只在「集中 × 受害 edge」上有效的东西与 3-E 重复。

## 待用户拍板（`PLAN-draft.md` §7，下个会话讨论）

1. 解除 D-008？
2. **G8 存盘（集群 `tfdpfl-dumps/G8__a__s4x.*`，约 0.81 GB）先别删** —— P0 离线探针要用第 30 轮快照。
3. 主线是否采用模块 A；是否把分散布点提前、加不饱和工作点与 flat 对照。
4. 抢救检测线（更好的 c_k）只作诊断，还是给独立额度（≤ 20 GPU-h）。
5. 下一个动手会话 = D0（只写登记表与判定规则，不改 `fedavg/`）还是 D0 + SA0。

另：Experiment 3 遗留的待定事项（random·R20 的 3-C 标签怎么记、在线 c_k 是否继续、3-B 出路、G2 规模、git 瘦身、合并 main）见 `REPORT.md` §9.2。

## 环境 / 基线

- 分支：`claude/federated-learning-experiment-review-pt5j1b`（用户定「先不合并」；`origin/main` 停在 `cf40b13`）。新会话照 CLAUDE.md 开场：`git fetch` + 读本文件；若用户要求新开分支，从本分支（不是 main）分出去要先问。
- L1（本地无 TF，装 numpy / pytest / pyyaml / matplotlib）：**1599 passed / 45 skipped / 3 xfailed**；TF 侧不变（陷阱 #4 的 2 条红）。
- `status.py`（`experiments/attack/hfl-mechanism/registry.yaml`）：todo 0 / done 117 / stale 6 / blocked 67。集群上没有在跑的作业。
- 本会话（2026-10-03）的提交：`1c3119e`（冻结的 G1 判定脚本）→ `c3f47b5`（判定结果 + 图 + REPORT 终版）→ `9b32856` / `5401253`（统计审查补注）→ `f76c6ae`（本草案）。
