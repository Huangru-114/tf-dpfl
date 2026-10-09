# current-focus —— 阶段三（edge 原生防御）· 交接给 D0

> **写于 2026-10-09**（阶段三计划定稿会话）。本文件是下一个会话「新会话开场第 3 步」要读的那一份。
> 上一版（2026-10-03，「讨论阶段三」）的内容已经落到 `PLAN.md` 与本目录的台账里。

## 上一个会话做了什么（只改文档）

- **计划定稿**：`PLAN.md`（D-089 … D-095）；草案 `PLAN-draft.md` 已被取代；hfl-mechanism 的 D-008 标为已解除。
- **文献**：`LITERATURE.md` = 用户综述原文 + Bad-PFL / CCS 原文的数字摘录（PDF 不进 git；以后要用原文请用户重新上传）。
- **台账**：本目录 `DECISIONS.md`（D-089 … D-095）、`FINDINGS.md`（F-087 G8 没有 edge 干净集 → SNAP；F-088 文献核对；F-089 L1 与 Python 版本）。
- **没有**登记任何组、**没有**改 `fedavg/`、**没有**交作业。

## 下一个会话 = D0（不改 `fedavg/`）

**要回答的问题**：把 `PLAN.md` 的阶段 0 变成可交、可判的东西 —— SNAP 登记好就能交；P0 / P1 / CCSF 的判定规则在数据之前冻结。

1. **阶段三登记表**：建议 `experiments/defense/edge-native/registry.yaml`，沿用 `harness/registry.py` / `status.py`（materialize **不带** `--group`，CLAUDE.md 有警告）。
   - 先登记 **SNAP**：`D-base`（`PLAN.md` §3.1）+ `evaluation.snapshot_rounds: "6/15/60"` × {col `[10,0,0,0]`, dist `[3,3,2,2]`} × s42–44。不需要改代码。
   - 按 CLAUDE.md 核对 `set:` 的三件事：`malicious_per_edge` 长度 = `n_edges`；`n_rounds × edge_rounds ≥ cap_effective`；各格评估网格一致。
   - P0 / CAL / CCSF 先占位，`requires:` 写上 SA0 / SA1 / SA-C。
2. **预注册**（写成 FINDINGS 的 N-008，冻结进 git 之后才交）：
   - P0：有效性闸 V0、精度过滤、`go_online` / `kill_pre` / `accuracy_bound` 的阈值与读法；
   - P1：`go_main` / `stop`；
   - CCSF：`ccs_reproduces`。
3. **功效分析**（纯 harness，不要 GPU）：用 G1 / G1R5 / G6 / G0 已有数据，算主量 V / B0 / P / margin_p50 / MTA 的 seed 间 SD 与按 seed 配对的差的 SD。
   据此确认 0.15 / 0.10 / 0.05 与 margin 2 logit 这些门槛（`PLAN.md` §8 第 1 条）。
4. **文献核对**：`PLAN.md` 里凭记忆引用的条目（TRADES、I-BAU、NAD、FLTrust、A3FL、EOT、Tsipras 2019、LP 等），以及综述里标「待核实」的条目。
   本环境的出站代理挡 arxiv / ICLR proceedings，要原文请用户上传。
5. **交 SNAP**：6 个 run，约 6 GPU-h（K=3 两个包）。回传后先过有效性闸 —— col 的 s42–44 `[Checksum]` 要与 G1R5 同 seed 逐位相同。

之后的顺序（`PLAN.md` §5–§6）：SA0（edge 侧对抗训练核心 + 快照离线探针）→ P0 → SA1 → P1 + CAL；SA-C（CCS）→ CCSF → CCSP；其余按闸门走。

## 待用户拍板（`PLAN.md` §8）

0. ~~CCS 的消融出处~~ → 搁置（D-096），CCSP 交作业前补。
1. 各组门槛：D0 的功效分析之后再确认。
2. ~~头条结论~~ → H1 / H3，5 seed（D-099）。
3. ~~DPR~~ → 照原文，PGD 之后做（D-097）。官方代码 `github.com/chenjian0924/Paper-Code`（F-090）。
4. ~~CCSF 的划分~~ → 主实验 Dir 0.5 = 旧 `noniid`，equal_random 之后做（D-098）。
5. ~~H5 筛选~~ → 接受（D-100）。
6. G8 存盘（约 0.81 GB）删不删：阶段三不再依赖它（F-087）。
7. F-089 的几条测试要不要改成与 Python 版本无关。

## 先读（按顺序）

1. `PLAN.md`：
   - §0 范围、§1 决定性的事实、§2.3 证据追溯；
   - §4 的 SNAP / P0 / CAL / CCSF；
   - §5 执行顺序与机时、§8 待讨论。
2. 本目录 `DECISIONS.md` / `FINDINGS.md`。
3. `LITERATURE.md` 第 2 部分（Bad-PFL / CCS 的数）。

## 环境 / 基线

- **分支**：`claude/eloquent-faraday-b9kmop`。2026-10-09 开场时 `origin/main` = `ae425cb`，与本分支起点相同。
- **L1（本地无 TF）**：用 **Python 3.11 的 venv** 跑，结果 1599 passed / 45 skipped / 3 xfailed（PASS）。
  本容器默认的 `python3` 是 3.13，会多 4 条与代码无关的红（F-089）。做法：
  `uv venv -p python3.11 <dir> && uv pip install -p <dir>/bin/python pytest numpy pyyaml matplotlib && TFDPFL_PY=<dir>/bin/python bash run_l1.sh`
- **`status.py`**（hfl-mechanism 登记表）：todo 0 / done 117 / stale 6 / blocked 67。集群上没有在跑的作业。
- **机时**：上限约 100 GPU-h / 周（D-047）。阶段三全部分支放行约 150–300 GPU-h；最短路径约 40 GPU-h（`PLAN.md` §5）。
